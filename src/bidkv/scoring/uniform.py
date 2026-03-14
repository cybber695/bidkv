"""Uniform 评分策略 — 消融实验用基线。

所有 token 赋予相同的重要度分数，表示"无差别压缩"。
用于消融实验中验证评分策略的价值——如果 H2O/Attention 策略
无法显著优于 Uniform，则说明评分信号没有携带有效信息。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bidkv.protocol.bid import CompressionBid, make_bid_id


class UniformScoring:
    """Uniform scoring：所有 token 等权（baseline）。

    Parameters
    ----------
    uniform_score:
        所有 token 的固定分数。默认 0.5。
    algorithm_id:
        算法标识符。默认 "uniform"。
    """

    def __init__(
        self,
        *,
        uniform_score: float = 0.5,
        algorithm_id: str = "uniform",
    ) -> None:
        if not (0.0 <= uniform_score <= 1.0):
            raise ValueError(f"uniform_score must be in [0, 1], got {uniform_score}")
        self._uniform_score = uniform_score
        self._algorithm_id = algorithm_id

    @property
    def uniform_score(self) -> float:
        """固定分数值。"""
        return self._uniform_score

    def score(
        self,
        token_ids: Sequence[int],
        **context: Any,
    ) -> list[float]:
        """返回每个 token 的等权重要度分数。

        Parameters
        ----------
        token_ids:
            Token ID 序列。

        Returns
        -------
        list[float]
            所有值均为 ``uniform_score``。
        """
        return [self._uniform_score] * len(token_ids)

    def generate_bids(
        self,
        request_id: str,
        token_ids: Sequence[int],
        compression_levels: Sequence[float],
        **context: Any,
    ) -> list[CompressionBid]:
        """基于均匀评分生成 CompressionBid。

        由于所有 token 等权，quality_delta 与压缩比例成正比。
        """
        n = len(token_ids)
        if n == 0:
            return []

        bids = []
        for level_idx, level in enumerate(compression_levels):
            tokens_to_remove = max(1, int(n * level))
            tokens_freed = min(tokens_to_remove, n - 1)
            if tokens_freed <= 0:
                continue

            # Uniform scoring 下，quality_delta 正比于压缩比例
            quality_delta = min(1.0, self._uniform_score * (tokens_freed / n))

            bid = CompressionBid(
                bid_id=make_bid_id(request_id, level_idx),
                request_id=request_id,
                algorithm_id=self._algorithm_id,
                tokens_freed=tokens_freed,
                quality_delta=quality_delta,
                compress_latency_ms=0.1 * tokens_freed,
                confidence=1.0,  # 均匀分布无不确定性
                metadata={
                    "compression_level": level,
                    "uniform_score": self._uniform_score,
                    "scoring_method": "uniform",
                },
            )
            bids.append(bid)

        return bids
