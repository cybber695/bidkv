"""BidKV baseline — 完整 bid 机制 + utility greedy。

这是 BidKV 的完整策略包装器（作为 baseline 接口的适配器）。
scorer-agnostic：支持任意实现 ScoringStrategy 的评分器，
默认使用 H2OScoring。

选择公式：U = r / (δ + ε)，greedy by U（Algorithm 1）。

Mode A 语义（request-level preemption）
--------------------------------------
δ = 1 + 0.5·completion + starvation_penalty
- freed 数量为主信号（效率优先：大请求释放更多 KV）
- completion 为轻量次信号（max ratio 1.5×，freed 强主导）
- anti-starvation：被反复 preempt 的请求分母递增

max ratio = 1.5×（completion 0% 与 100% 之间），确保 freed 强主导排序。
低 completion weight 使 victim ordering 接近最优的 freed-first，同时保留
anti-starvation 以防止 cascading preemption（区别于 h2o-style）。
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.pool import BidPoolManager
from bidkv.protocol.bid import CompressionBid, make_bid_id
from bidkv.scoring import H2OScoring, ScoringStrategy
from bidkv.solver import GreedyBidSolver, SolverConfig


class BidKVStrategy(BaselineStrategy):
    """BidKV 完整策略：scoring → bid → pool → solver。

    Mode A 使用 U = freed / (1 + 0.5·completion + starvation + ε) 排序，
    freed 强主导、completion 提供 ≤1.5× 轻量保护。

    Parameters
    ----------
    scoring:
        ScoringStrategy 实例。若为 None，使用 H2OScoring 默认配置创建。
    delta_budget:
        质量损失上限。默认 0.15。
    """

    def __init__(
        self,
        *,
        scoring: ScoringStrategy | None = None,
        delta_budget: float = 0.15,
    ) -> None:
        self._scoring: ScoringStrategy = scoring or H2OScoring()
        self._delta_budget = delta_budget
        self._solver = GreedyBidSolver(SolverConfig(enabled=True, delta_budget=delta_budget))

    @property
    def name(self) -> str:
        return "bidkv"

    @property
    def scoring(self) -> ScoringStrategy:
        """当前使用的评分策略实例。"""
        return self._scoring

    @staticmethod
    def _completion_factor(req: RequestState) -> float:
        """Compute recompute-cost penalty multiplier for a candidate.

        Near-completion requests get a multiplier > 1 that inflates their
        token scores → higher quality_delta → lower utility. This steers the
        solver away from truncating requests that would be expensive to redo
        with little remaining benefit.

        Returns 1.0 (no penalty) for requests with unknown completion info.
        """
        if req.max_output_tokens <= 0 or req.num_computed_tokens <= 0:
            return 1.0
        num_output = max(0, req.num_computed_tokens - req.num_prompt_tokens)
        completion = min(1.0, num_output / req.max_output_tokens)
        # Quadratic ramp: 0% → 1.0×, 50% → 2.0×, 80% → 3.56×, 100% → 5.0×
        return 1.0 + completion * completion * 4.0

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[CompressionAction]:
        """Mode A: 质量感知的请求级驱逐排序。

        quality_delta = 1 + 2*completion + starvation

        - freed 主导排序（高效回收 KV 空间）
        - completion 提供 ≤3× 的次级保护
        - anti-starvation 保护被反复 preempt 的请求

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``delta_budget``：覆盖默认值。

        Returns
        -------
        list[CompressionAction]
            按 utility 降序排列的驱逐操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        pool_mgr = BidPoolManager(enabled=True)

        for req in candidates:
            if req.current_tokens <= 1:
                continue

            tokens_freed = req.current_tokens

            # Compute completion ratio: 0 (just started) → 1 (done)
            output_generated = max(0, req.num_computed_tokens - req.num_prompt_tokens)
            completion = 0.0
            if req.max_output_tokens > 0:
                completion = min(1.0, output_generated / req.max_output_tokens)

            # quality_delta = 1 + 0.5*completion + starvation
            # ─────────────────────────────────────────────────
            # Denominator always ≥ 1, so U ≤ freed.
            # Max protection ratio (completion 0→1) is 1.5×.
            # freed strongly dominates ordering — victim selection
            # closely tracks freed-first (like h2o-style) for
            # differently-sized requests, with completion providing
            # only a mild tiebreaker. Anti-starvation (+0.3 per
            # prior preemption) is the primary differentiator.
            quality_delta = 1.0 + 0.5 * completion

            # Anti-starvation: previously preempted requests get
            # additional denominator weight to reduce their utility.
            if req.num_preemptions > 0:
                quality_delta += req.num_preemptions * 0.3

            bid = CompressionBid(
                bid_id=make_bid_id(req.request_id, 0),
                request_id=req.request_id,
                algorithm_id="bidkv",
                tokens_freed=tokens_freed,
                quality_delta=quality_delta,
                compress_latency_ms=0.0,
                confidence=0.8,
                metadata={
                    "completion": round(completion, 4),
                    "num_preemptions": req.num_preemptions,
                    "mode": "A",
                },
            )
            pool_mgr.submit_bids(req.request_id, [bid])

        pool = pool_mgr.get_pool_snapshot()
        if not pool.bids:
            return []

        # Relaxed delta budget for Mode A: rank ALL candidates
        # (delta_budget only constrains how many the solver picks, but we
        # want a complete ordering for the priority cache)
        total_delta = sum(b.quality_delta for b in pool.bids)
        mode_a_budget = max(total_delta + 1.0, 100.0)

        acceptance = self._solver.solve(
            pool,
            needed_tokens,
            mode_a_budget,
            decision_reason="baseline_bidkv",
        )

        if acceptance.is_empty:
            return []

        bid_index = {b.bid_id: b for b in pool.bids}
        actions: list[CompressionAction] = []
        for bid_id in acceptance.accepted_bid_ids:
            bid = bid_index.get(bid_id)
            if bid is None:
                continue
            actions.append(
                CompressionAction(
                    request_id=bid.request_id,
                    action_type="evict",
                    target_tokens=bid.tokens_freed,
                    metadata={
                        "strategy": "bidkv",
                        "bid_id": bid.bid_id,
                        "quality_delta": bid.quality_delta,
                        "utility": bid.utility,
                    },
                )
            )

        return actions
