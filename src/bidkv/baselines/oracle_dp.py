"""Oracle DP baseline — 离线精确最优解（动态规划）。

设计理由：提供理论上界参考。在已知所有信息（所有候选 bid + trace）的情况下，
通过动态规划找到精确最优解。不可实时使用（需要离线 trace），
仅用于衡量各 baseline 与最优解的差距（competitive gap）。

**Strict definition**：same pressure event / candidate pool / trace / KV accounting。
Oracle 使用与其他 baseline 完全相同的候选池和 bid 集合。

算法：Grouped Knapsack DP
- 每个 request 为一组（group），组内有多个 compression level（item）
- 约束 1：每个 request 最多选择一个 compression level
- 约束 2：Σδ ≤ delta_budget
- 目标：最大化 Σ tokens_freed
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.protocol.bid import CompressionBid
from bidkv.scoring import H2OScoring


class OracleDPStrategy(BaselineStrategy):
    """Oracle DP：离线精确最优解。

    使用 Grouped Knapsack 动态规划在所有候选 bid 中找到
    释放 token 最多且满足质量约束的最优组合。

    Parameters
    ----------
    delta_budget:
        质量损失上限。默认 0.15。
    dp_resolution:
        DP 质量维度的离散化精度（步数）。越大精度越高、计算量越大。
        默认 1000。
    scoring:
        H2OScoring 实例（用于生成 bid）。若为 None 使用默认配置。
    compression_levels:
        生成 bid 时的压缩级别。默认 (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)。
    """

    def __init__(
        self,
        *,
        delta_budget: float = 0.15,
        dp_resolution: int = 1000,
        scoring: H2OScoring | None = None,
        compression_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    ) -> None:
        if dp_resolution < 1:
            raise ValueError(f"dp_resolution must be >= 1, got {dp_resolution}")
        if delta_budget <= 0:
            raise ValueError(f"delta_budget must be > 0, got {delta_budget}")
        self._delta_budget = delta_budget
        self._dp_resolution = dp_resolution
        self._scoring = scoring or H2OScoring()
        self._compression_levels = compression_levels

    @property
    def name(self) -> str:
        return "oracle-dp"

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """使用 DP 求解精确最优解。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``bids_by_request``：dict[str, list[CompressionBid]]。
            预生成的 bid（candidate-universe consistency）。
            可选 ``scoring_states``：dict[str, H2OScoring]。
            可选 ``delta_budget``：覆盖默认值。

        Returns
        -------
        list[CompressionAction]
            精确最优压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        delta_budget = kwargs.get("delta_budget", self._delta_budget)
        pre_bids: dict[str, list[CompressionBid]] | None = kwargs.get("bids_by_request")
        scoring_states: dict[str, H2OScoring] = kwargs.get("scoring_states", {})

        # 获取或生成每个请求的 bid
        bids_by_request: dict[str, list[CompressionBid]] = {}
        if pre_bids is not None:
            bids_by_request = dict(pre_bids)
        else:
            for req in candidates:
                if req.current_tokens <= 1 or not req.token_ids:
                    continue
                scorer = scoring_states.get(req.request_id, self._scoring)
                bids = scorer.generate_bids(
                    req.request_id,
                    req.token_ids,
                    self._compression_levels,
                )
                if bids:
                    bids_by_request[req.request_id] = bids

        if not bids_by_request:
            return []

        # 运行 Grouped Knapsack DP
        selected_bids = self._solve_grouped_knapsack(bids_by_request, needed_tokens, delta_budget)

        # 转换为 CompressionAction
        return [
            CompressionAction(
                request_id=bid.request_id,
                action_type="compress",
                target_tokens=bid.tokens_freed,
                metadata={
                    "strategy": "oracle-dp",
                    "bid_id": bid.bid_id,
                    "quality_delta": bid.quality_delta,
                    "utility": bid.utility,
                },
            )
            for bid in selected_bids
        ]

    def _solve_grouped_knapsack(
        self,
        bids_by_request: dict[str, list[CompressionBid]],
        needed_tokens: int,
        delta_budget: float,
    ) -> list[CompressionBid]:
        """Grouped Knapsack DP 求解。

        每个 request 是一个 group，组内有多个 bid（不同压缩级别）。
        约束：每个 group 最多选一个 bid，Σδ ≤ delta_budget。
        目标：最大化 Σ tokens_freed。

        Returns
        -------
        list[CompressionBid]
            被选中的 bid 列表（每个 request 最多一个）。
        """
        capacity = self._dp_resolution
        step_size = delta_budget / capacity  # 每个 DP 步代表的 delta

        # dp[j] = 使用 delta budget j 步时的最大 tokens_freed
        dp = [0] * (capacity + 1)
        # choice[j] = dp[j] 对应的 selected bid 列表
        choice: list[list[CompressionBid]] = [[] for _ in range(capacity + 1)]

        for _request_id, bids in bids_by_request.items():
            # 保存当前 DP 状态（"不选此 group" 的选项）
            old_dp = dp[:]
            old_choice = [c[:] for c in choice]

            for bid in bids:
                # 跳过 quality_delta 超过 delta_budget 的 bid
                if bid.quality_delta > delta_budget:
                    continue

                # 将 quality_delta 离散化到 DP 步数
                cost = max(1, round(bid.quality_delta / step_size)) if step_size > 0 else capacity
                cost = min(cost, capacity)
                value = bid.tokens_freed

                for j in range(capacity, cost - 1, -1):
                    candidate = old_dp[j - cost] + value
                    if candidate > dp[j]:
                        dp[j] = candidate
                        choice[j] = old_choice[j - cost] + [bid]

        # 找满足 needed_tokens 且 delta 最小的解
        best_j = -1
        for j in range(capacity + 1):
            if dp[j] >= needed_tokens:
                best_j = j
                break

        if best_j >= 0:
            return choice[best_j]

        # 无法满足 needed_tokens 时，返回最大释放量的解
        best_j = max(range(capacity + 1), key=lambda j: dp[j])
        return choice[best_j] if dp[best_j] > 0 else []
