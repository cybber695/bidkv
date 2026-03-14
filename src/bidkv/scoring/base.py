"""ScoringStrategy — Token 重要度评分策略 Protocol。

定义 BidKV scoring 层的核心抽象接口。所有评分策略必须满足此 Protocol，
生产部署、参考基线和消融实验均通过可插拔的 ScoringStrategy 实现区分。

评分三级分类
------------
- **Practical Scoring**：H2OScoring — 生产部署中实际使用的评分代理
- **Reference Scoring**：AttentionWeightScoring — 精度上界参考（需 output_attentions）
- **Auxiliary Scoring**：UniformScoring / RandomScoring — 消融实验用基线
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from bidkv.protocol.bid import CompressionBid


@runtime_checkable
class ScoringStrategy(Protocol):
    """Token 重要度评分策略。

    任何实现 ``score()`` 和 ``generate_bids()`` 方法签名的对象，
    都可作为 ScoringStrategy 使用（structural subtyping，无需继承）。
    """

    def score(
        self,
        token_ids: Sequence[int],
        **context: Any,
    ) -> list[float]:
        """返回每个 token 的重要度分数。

        Parameters
        ----------
        token_ids:
            Token ID 序列。
        **context:
            策略特定的上下文信息（例如 attention 权重、decode step 统计等）。

        Returns
        -------
        list[float]
            长度与 ``token_ids`` 相同的分数列表，值域 [0, 1]，越高越重要。
        """
        ...

    def generate_bids(
        self,
        request_id: str,
        token_ids: Sequence[int],
        compression_levels: Sequence[float],
        **context: Any,
    ) -> list[CompressionBid]:
        """基于评分，针对多个压缩级别生成 CompressionBid。

        Parameters
        ----------
        request_id:
            推理请求 ID。
        token_ids:
            Token ID 序列。
        compression_levels:
            压缩比例列表（0~1），例如 [0.2, 0.4, 0.6, 0.8]，
            表示分别尝试保留 80%、60%、40%、20% 的 token。
        **context:
            策略特定的上下文信息。

        Returns
        -------
        list[CompressionBid]
            按压缩级别生成的 bid 列表，字段符合 CompressionBid 三层体系。
        """
        ...
