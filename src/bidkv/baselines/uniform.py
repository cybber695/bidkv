"""Uniform baseline — 所有活跃请求均等压缩。

设计理由：隔离"差异化压缩"对结果的贡献。
Uniform 平等对待所有请求（每个释放相同 token 数），
不使用任何 scoring 或 priority 信息。
对比 Uniform → BidKV 可揭示差异化压缩的增量价值。

选择公式：∀req: compress(needed_tokens / N)
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState


class UniformStrategy(BaselineStrategy):
    """Uniform Compression：所有活跃请求均等压缩。

    每个请求释放 ``ceil(needed_tokens / N)`` 个 token，
    N = 候选请求数量。若某请求 token 不足则释放其全部。
    """

    @property
    def name(self) -> str:
        return "uniform"

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """均等压缩所有候选请求。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。

        Returns
        -------
        list[CompressionAction]
            压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        # 过滤掉 token <= 1 的请求（至少保留 1 个 token）
        compressible = [r for r in candidates if r.current_tokens > 1]
        if not compressible:
            return []

        n = len(compressible)
        per_request = max(1, -(-needed_tokens // n))  # ceil division

        actions: list[CompressionAction] = []
        freed = 0
        for req in compressible:
            if freed >= needed_tokens:
                break
            # 最多压缩到保留 1 个 token
            actual_free = min(per_request, req.current_tokens - 1, needed_tokens - freed)
            if actual_free <= 0:
                continue
            actions.append(
                CompressionAction(
                    request_id=req.request_id,
                    action_type="compress",
                    target_tokens=actual_free,
                    metadata={"strategy": "uniform", "per_request_target": per_request},
                )
            )
            freed += actual_free

        return actions
