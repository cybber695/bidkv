"""Random 评分策略 — 消融实验用基线。

为每个 token 随机赋予重要度分数。用于消融实验中验证：
有信息量的评分策略（H2O/Attention）vs 随机猜测的差距。

注意：使用固定 seed 时可复现。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from bidkv.protocol.bid import CompressionBid, make_bid_id


class RandomScoring:
    """Random scoring：随机分数（baseline）。

    Parameters
    ----------
    seed:
        随机种子。若为 None，则不设置 seed（不可复现）。
        默认 None。
    algorithm_id:
        算法标识符。默认 "random"。
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        algorithm_id: str = "random",
    ) -> None:
        self._seed = seed
        self._algorithm_id = algorithm_id
        self._rng = random.Random(seed)

    @property
    def seed(self) -> int | None:
        """随机种子。"""
        return self._seed

    def score(
        self,
        token_ids: Sequence[int],
        **context: Any,
    ) -> list[float]:
        """返回每个 token 的随机重要度分数。

        Parameters
        ----------
        token_ids:
            Token ID 序列。

        Returns
        -------
        list[float]
            每个值在 [0, 1] 内随机生成。
        """
        return [self._rng.random() for _ in token_ids]

    def generate_bids(
        self,
        request_id: str,
        token_ids: Sequence[int],
        compression_levels: Sequence[float],
        **context: Any,
    ) -> list[CompressionBid]:
        """基于随机评分生成 CompressionBid。"""
        n = len(token_ids)
        if n == 0:
            return []

        scores = self.score(token_ids, **context)
        bids = []

        for level_idx, level in enumerate(compression_levels):
            tokens_to_remove = max(1, int(n * level))
            tokens_freed = min(tokens_to_remove, n - 1)
            if tokens_freed <= 0:
                continue

            indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])
            removed_scores = [s for _, s in indexed_scores[:tokens_freed]]

            avg_removed_importance = sum(removed_scores) / len(removed_scores)
            quality_delta = min(1.0, avg_removed_importance)

            bid = CompressionBid(
                bid_id=make_bid_id(request_id, level_idx),
                request_id=request_id,
                algorithm_id=self._algorithm_id,
                tokens_freed=tokens_freed,
                quality_delta=quality_delta,
                compress_latency_ms=0.1 * tokens_freed,
                confidence=0.0,  # 随机评分完全无置信
                metadata={
                    "compression_level": level,
                    "seed": self._seed,
                    "scoring_method": "random",
                },
            )
            bids.append(bid)

        return bids
