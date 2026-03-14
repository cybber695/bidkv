"""FrameworkAdapter ABC — Minimum Viable Cross-Framework Abstraction.

5 层职责边界：
1. **KV stats 获取**：``get_kv_stats() → (used, max)``
2. **Pressure interception**：在框架原生 preemption / eviction **之前**获得压缩尝试机会
3. **Compression 执行**：``execute_compression()`` 委托框架原生 KV 操作
4. **Scoring 回调**：decode step 后更新 H2OScoring
5. **Lifecycle 管理**：``on_request_complete()`` 清理 bid
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bidkv.config import BidKVConfig
from bidkv.scoring.base import ScoringStrategy


class FrameworkAdapter(ABC):
    """框架适配器 — Minimum Viable Cross-Framework Abstraction。

    bidkv 与 LLM serving 框架（vLLM、SGLang 等）之间的桥梁。
    每个框架需实现本 ABC 的所有抽象方法。

    Parameters
    ----------
    config:
        BidKV 全局配置（feature gate + kill switch）。
    scoring:
        Token 重要度评分策略实例。
    """

    def __init__(self, config: BidKVConfig, scoring: ScoringStrategy) -> None:
        self._config = config
        self._scoring = scoring

    @property
    def config(self) -> BidKVConfig:
        """当前 BidKV 配置。"""
        return self._config

    @property
    def scoring(self) -> ScoringStrategy:
        """当前评分策略。"""
        return self._scoring

    @abstractmethod
    def install(self) -> None:
        """将 bidkv 注入到框架中（monkey-patch / subclass / plugin）。

        调用后，框架的调度路径中应包含 bidkv 压力检测与压缩逻辑。
        """

    @abstractmethod
    def get_kv_stats(self) -> tuple[int, int]:
        """返回 (used_tokens, max_tokens)。

        由 PressureDetector 轮询调用。
        """

    @abstractmethod
    def execute_compression(self, request_id: str, target_tokens: int) -> int:
        """在框架中执行 KV 压缩，返回实际释放 token 数。

        Parameters
        ----------
        request_id:
            目标请求 ID。
        target_tokens:
            期望释放的 token 数量。

        Returns
        -------
        int
            实际释放的 token 数量（>= 0）。
        """

    @abstractmethod
    def on_request_complete(self, request_id: str) -> None:
        """请求完成时清理 bid 和内部状态。"""
