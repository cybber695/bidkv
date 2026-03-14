"""Global-NoBid baseline — 系统自动推断 utility，无用户 bid。

**关键归因 baseline**：对比 Global-NoBid → BidKV 可揭示
bid 接口（用户显式偏好）的增量价值。

设计理由：Global-NoBid 与 BidKV 使用相同的 H2OScoring 评分，
但 **系统自动推断 utility**（不暴露 bid 接口给用户）。
如果 BidKV > Global-NoBid，则证明用户显式 bid 比系统推断更有价值。

选择公式：
- U_sys = r / (δ_H2O + ε)，其中 δ_H2O 由 H2OScoring 估算
- 按 U_sys 贪心选择（与 GreedyBidSolver 相同算法）
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.protocol.bid import _UTILITY_EPSILON
from bidkv.scoring import H2OScoring


class GlobalNoBidStrategy(BaselineStrategy):
    """Global-NoBid：系统推断 utility + greedy 选择（无 bid 接口）。

    流程：
    1. 对每个候选请求，用 H2OScoring 评分
    2. 根据评分估算压缩的 quality_delta
    3. 计算系统推断 utility：U_sys = tokens_freed / (δ_H2O + ε)
    4. 按 U_sys 降序贪心选择

    Parameters
    ----------
    scoring:
        H2OScoring 实例。若为 None，使用默认配置创建。
    delta_budget:
        质量损失上限（Σδ ≤ delta_budget）。默认 0.15。
    compressible_ratio:
        每个请求可压缩的 token 比例。默认 0.6。
    """

    def __init__(
        self,
        *,
        scoring: H2OScoring | None = None,
        delta_budget: float = 0.15,
        compressible_ratio: float = 0.6,
    ) -> None:
        self._scoring = scoring or H2OScoring()
        self._delta_budget = delta_budget
        self._compressible_ratio = compressible_ratio

    @property
    def name(self) -> str:
        return "global-nobid"

    @property
    def scoring(self) -> H2OScoring:
        """当前使用的 H2OScoring 实例。"""
        return self._scoring

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """系统自动推断 utility 并贪心选择。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``scoring_states``：dict[str, H2OScoring]，
            每个请求独立的 H2OScoring 实例。
            可选 ``delta_budget``：覆盖默认 delta_budget。

        Returns
        -------
        list[CompressionAction]
            压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        scoring_states: dict[str, H2OScoring] = kwargs.get("scoring_states", {})
        delta_budget = kwargs.get("delta_budget", self._delta_budget)

        # 为每个候选生成系统推断的 bid
        system_bids: list[tuple[RequestState, int, float, float]] = []
        for req in candidates:
            if req.current_tokens <= 1:
                continue

            scorer = scoring_states.get(req.request_id, self._scoring)
            scores = scorer.score(req.token_ids) if req.token_ids else []

            # 计算可释放 token 数
            tokens_freed = max(1, int(req.current_tokens * self._compressible_ratio))

            # 估算 quality_delta（基于 H2O scoring）
            quality_delta = self._estimate_delta(scores, tokens_freed, req.current_tokens)

            # 系统推断 utility
            utility = tokens_freed / (quality_delta + _UTILITY_EPSILON)

            system_bids.append((req, tokens_freed, quality_delta, utility))

        # 按 utility 降序贪心选择
        system_bids.sort(key=lambda x: x[3], reverse=True)

        actions: list[CompressionAction] = []
        freed = 0
        total_delta = 0.0
        for req, tokens_to_free, delta, utility in system_bids:
            if freed >= needed_tokens:
                break
            if total_delta + delta > delta_budget:
                continue

            actual_free = min(tokens_to_free, needed_tokens - freed)
            # 按比例调整 delta
            adjusted_delta = delta * (actual_free / tokens_to_free) if tokens_to_free > 0 else 0.0

            actions.append(
                CompressionAction(
                    request_id=req.request_id,
                    action_type="compress",
                    target_tokens=actual_free,
                    metadata={
                        "strategy": "global-nobid",
                        "system_utility": utility,
                        "estimated_quality_delta": adjusted_delta,
                    },
                )
            )
            freed += actual_free
            total_delta += adjusted_delta

        return actions

    def _estimate_delta(
        self, scores: list[float], tokens_to_compress: int, total_tokens: int
    ) -> float:
        """根据 H2O scoring 估算压缩的 quality delta。

        被压缩的 token 是重要度最低的。delta 等于被压缩 token 的平均重要度。
        """
        if not scores:
            # 无 scoring 数据时，用比例启发式
            ratio = tokens_to_compress / max(1, total_tokens)
            return min(1.0, ratio * 0.5)

        sorted_scores = sorted(scores)
        n_compress = min(tokens_to_compress, len(sorted_scores))
        if n_compress == 0:
            return 0.0
        return min(1.0, sum(sorted_scores[:n_compress]) / n_compress)
