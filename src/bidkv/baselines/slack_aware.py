"""Slack-Aware baseline — SLO 剩余时间感知调度。

设计理由：有调度信息（SLO deadline），但无 quality 信息（不使用 scoring）。
对比 Slack-Aware → BidKV 可揭示 quality-aware 信息的增量价值
（BidKV 额外知道压缩对质量的影响）。

选择公式：
- slack = deadline - now
- 按 slack 降序排列（deadline 远的先压缩 — 有更多时间余裕，可承受压缩）
- 没有 deadline 的请求优先被压缩（被视为无 SLO 保证）
"""

from __future__ import annotations

import time
from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState


class SlackAwareStrategy(BaselineStrategy):
    """Slack-Aware：根据 SLO 剩余时间选择压缩受害者。

    deadline 远的请求先压缩（它们有更多时间余裕，可承受质量下降）。
    没有 deadline（deadline_ms=None）的请求被视为最不紧迫，优先压缩。

    Parameters
    ----------
    compression_ratio:
        每个受害请求的压缩比例。默认 0.5。
    """

    def __init__(self, *, compression_ratio: float = 0.5) -> None:
        if not (0.0 < compression_ratio <= 1.0):
            raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")
        self._compression_ratio = compression_ratio

    @property
    def name(self) -> str:
        return "slack-aware"

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """按 SLO slack 降序选择受害者（deadline 远的先压缩）。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``now_ms``：当前时间戳（单调毫秒）。
            若未提供，使用 ``time.monotonic() * 1000``。

        Returns
        -------
        list[CompressionAction]
            压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        now_ms: float = kwargs.get("now_ms", time.monotonic() * 1000)

        # 计算每个请求的 slack
        # 无 deadline → 最大 slack（float('inf')）→ 优先被压缩
        def _slack(req: RequestState) -> float:
            if req.deadline_ms is None:
                return float("inf")
            return req.deadline_ms - now_ms

        # 按 slack 降序排列（最大 slack / 无 deadline 优先压缩）
        sorted_candidates = sorted(candidates, key=_slack, reverse=True)

        actions: list[CompressionAction] = []
        freed = 0
        for req in sorted_candidates:
            if freed >= needed_tokens:
                break
            if req.current_tokens <= 1:
                continue

            tokens_to_free = max(1, int(req.current_tokens * self._compression_ratio))
            tokens_to_free = min(tokens_to_free, req.current_tokens - 1, needed_tokens - freed)
            if tokens_to_free <= 0:
                continue

            slack = _slack(req)
            actions.append(
                CompressionAction(
                    request_id=req.request_id,
                    action_type="compress",
                    target_tokens=tokens_to_free,
                    metadata={
                        "strategy": "slack-aware",
                        "slack_ms": slack if slack != float("inf") else None,
                        "deadline_ms": req.deadline_ms,
                    },
                )
            )
            freed += tokens_to_free

        return actions
