"""VLLMAdapter — BidKV 在 vLLM 框架上的适配器。

vLLM 0.11+ 使用 v1 调度架构：
- ``Scheduler`` 在 ``schedule()`` 中通过 ``kv_cache_manager.allocate_slots()`` 分配 KV
- 分配失败时 preempt 最低优先级 running request
- BidKV 在 preemption 前注入压缩尝试，减少不必要的 preemption

核心职责：
1. KV stats 获取：从 ``KVCacheManager.block_pool`` 读取 usage
2. Pressure interception：在 vLLM preempt 路径前获得压缩尝试机会
3. Compression 执行：通过 block-level 操作释放 KV（标记 + 释放尾部 blocks）
4. Scoring 回调：decode step 后更新 H2OScoring
5. Lifecycle 管理：请求完成时清理 bid 和内部状态
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from bidkv.adapters.base import FrameworkAdapter
from bidkv.config import BidKVConfig
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
from bidkv.scoring.base import ScoringStrategy
from bidkv.solver import GreedyBidSolver, SolverConfig

if TYPE_CHECKING:
    from bidkv.protocol.bid import BidAcceptance

logger = logging.getLogger(__name__)

# 默认压缩级别（论文 §4 标准设置）
DEFAULT_COMPRESSION_LEVELS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


class VLLMAdapter(FrameworkAdapter):
    """BidKV 在 vLLM 框架上的适配器。

    通过 monkey-patch vLLM 的 Scheduler，在 preemption 路径之前注入
    BidKV 压缩尝试。如果压缩能释放足够空间，则避免 preemption。

    Parameters
    ----------
    config:
        BidKV 全局配置（feature gate + kill switch）。
    scoring:
        评分策略实例（通常为 H2OScoring）。
    scheduler:
        vLLM 的 Scheduler 实例（``vllm.v1.core.sched.scheduler.Scheduler``）。
        若为 None，需在 ``install()`` 前通过 ``set_scheduler()`` 设置。
    pressure_config:
        PressureDetector 配置。若为 None 使用默认值（threshold 与 BidKV 对齐）。
    solver_config:
        GreedyBidSolver 配置。若为 None 使用默认值。
    compression_levels:
        bid 生成使用的压缩级别。默认 (0.2, 0.4, 0.6, 0.8)。
    """

    def __init__(
        self,
        config: BidKVConfig,
        scoring: ScoringStrategy,
        *,
        scheduler: Any = None,
        pressure_config: PressureConfig | None = None,
        solver_config: SolverConfig | None = None,
        compression_levels: Sequence[float] | None = None,
    ) -> None:
        super().__init__(config, scoring)
        self._scheduler = scheduler

        # BidKV 核心组件
        p_cfg = pressure_config or PressureConfig(enabled=config.is_active)
        s_cfg = solver_config or SolverConfig(
            enabled=config.is_active,
            delta_budget=config.delta_budget,
        )
        self._pressure_detector = PressureDetector(p_cfg)
        self._pool_manager = BidPoolManager(
            enabled=config.is_active,
            kill_switch=config.kill_switch,
        )
        self._solver = GreedyBidSolver(s_cfg)
        self._compression_levels = tuple(compression_levels or DEFAULT_COMPRESSION_LEVELS)

        # 请求追踪
        # {request_id: list[int]} — 每个请求的 token ids
        self._request_tokens: dict[str, list[int]] = {}
        # 已安装标记
        self._installed: bool = False
        # 原始方法备份（用于 uninstall）
        self._original_methods: dict[str, Any] = {}

        # Metrics
        self._metrics = AdapterMetrics()

        logger.info(
            "VLLMAdapter created: enabled=%s, kill_switch=%s, compression_levels=%s",
            config.is_active,
            config.kill_switch,
            self._compression_levels,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pressure_detector(self) -> PressureDetector:
        """内部 PressureDetector 实例。"""
        return self._pressure_detector

    @property
    def pool_manager(self) -> BidPoolManager:
        """内部 BidPoolManager 实例。"""
        return self._pool_manager

    @property
    def solver(self) -> GreedyBidSolver:
        """内部 GreedyBidSolver 实例。"""
        return self._solver

    @property
    def metrics(self) -> AdapterMetrics:
        """适配器运行指标。"""
        return self._metrics

    @property
    def installed(self) -> bool:
        """是否已安装到 vLLM 框架。"""
        return self._installed

    # ------------------------------------------------------------------
    # FrameworkAdapter interface
    # ------------------------------------------------------------------

    def install(self) -> None:
        """将 bidkv 注入到 vLLM 调度路径。

        Monkey-patch vLLM Scheduler 的 schedule()、update_from_output()
        和 _free_request() 方法。

        Raises
        ------
        RuntimeError
            如果 scheduler 未设置。
        """
        if not self._config.is_active:
            logger.info("VLLMAdapter.install: BidKV not active, skipping injection")
            return

        if self._scheduler is None:
            raise RuntimeError(
                "VLLMAdapter.install: scheduler not set. Call set_scheduler() before install()."
            )

        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        install_scheduler_hook(self._scheduler, self)
        self._installed = True
        logger.info("VLLMAdapter: installed into vLLM scheduler")

    def uninstall(self) -> None:
        """移除 bidkv 注入，恢复 vLLM 原始行为。"""
        if not self._installed:
            return

        from bidkv.adapters.vllm.scheduler_hook import uninstall_scheduler_hook

        uninstall_scheduler_hook(self._scheduler, self)
        self._installed = False
        logger.info("VLLMAdapter: uninstalled from vLLM scheduler")

    def set_scheduler(self, scheduler: Any) -> None:
        """设置 vLLM Scheduler 实例。

        Parameters
        ----------
        scheduler:
            vLLM v1 ``Scheduler`` 实例。
        """
        self._scheduler = scheduler

    def get_kv_stats(self) -> tuple[int, int]:
        """从 vLLM KVCacheManager 获取 KV 使用统计。

        通过 ``block_pool.get_usage()`` 和 ``block_pool.get_num_free_blocks()``
        计算 token 级别的使用量。

        Returns
        -------
        tuple[int, int]
            (used_tokens, max_tokens)。以 block 粒度对齐到 token 数。
        """
        if self._scheduler is None:
            return (0, 0)

        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return (0, 0)

        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return (0, 0)

        block_size = getattr(kv_cache_manager, "block_size", None)
        if block_size is None or block_size <= 0:
            return (0, 0)

        num_free = block_pool.get_num_free_blocks()
        usage = block_pool.get_usage()

        # 估算总 block 数：free / (1 - usage)，避免除零
        total_blocks = (int(num_free / (1.0 - usage)) if num_free > 0 else 0) if usage < 1.0 else 0

        # 更准确：直接从 block_pool 获取（如果有 _num_blocks 属性）
        total_blocks_attr = getattr(block_pool, "_num_blocks", None)
        if total_blocks_attr is not None and total_blocks_attr > 0:
            total_blocks = total_blocks_attr

        total_tokens = total_blocks * block_size
        used_tokens = int(total_tokens * usage)

        return (used_tokens, total_tokens)

    def execute_compression(self, request_id: str, target_tokens: int) -> int:
        """在 vLLM 中执行 KV 压缩。

        vLLM 使用 block-level KV 管理，不原生支持 token-level 部分释放。
        压缩策略：

        1. 获取评分，标记低重要度 token
        2. 计算可以释放的完整 block 数
        3. 通过 coordinator 释放尾部 block

        注意：由于 block 对齐限制，实际释放量可能小于 target_tokens。

        Parameters
        ----------
        request_id:
            目标请求 ID。
        target_tokens:
            期望释放的 token 数量。

        Returns
        -------
        int
            实际释放的 token 数量 (block-aligned)。
        """
        if not self._config.is_active:
            return 0

        token_ids = self._request_tokens.get(request_id)
        if not token_ids:
            logger.debug("execute_compression: no tokens tracked for request %s", request_id)
            return 0

        # 获取 block_size 用于对齐
        block_size = self._get_block_size()
        if block_size <= 0:
            return 0

        # 获取评分
        scores = self._scoring.score(token_ids)
        if len(scores) != len(token_ids):
            logger.warning(
                "execute_compression: score length mismatch (scores=%d, tokens=%d)",
                len(scores),
                len(token_ids),
            )
            return 0

        # 计算需要释放的 block 数量（向下对齐到 block_size）
        blocks_to_free = target_tokens // block_size
        if blocks_to_free <= 0:
            return 0

        tokens_to_free = blocks_to_free * block_size

        # 确保至少保留一部分 token（不完全释放）
        if tokens_to_free >= len(token_ids):
            # 最多释放 n-1 个 block 的 token，保留至少一个 block
            blocks_to_free = max(0, len(token_ids) // block_size - 1)
            tokens_to_free = blocks_to_free * block_size

        if tokens_to_free <= 0:
            return 0

        # 实际释放逻辑：通过 vLLM 的 block 管理释放
        actual_freed = self._free_tail_blocks(request_id, blocks_to_free, block_size)

        if actual_freed > 0:
            # 更新内部追踪：缩减 token 列表
            remaining_count = max(1, len(token_ids) - actual_freed)
            self._request_tokens[request_id] = token_ids[:remaining_count]

        self._metrics.record_compression(request_id, actual_freed)
        logger.debug(
            "execute_compression: request=%s, target=%d, block_aligned=%d, actual_freed=%d",
            request_id,
            target_tokens,
            tokens_to_free,
            actual_freed,
        )
        return actual_freed

    def on_request_complete(self, request_id: str) -> None:
        """请求完成时清理 bid 和内部状态。"""
        self._pool_manager.remove_by_request(request_id)
        self._request_tokens.pop(request_id, None)
        self._metrics.record_request_complete(request_id)
        logger.debug("on_request_complete: request=%s", request_id)

    # ------------------------------------------------------------------
    # BidKV Pipeline — Pressure-triggered compression cycle
    # ------------------------------------------------------------------

    def try_compress(self) -> int:
        """执行一轮 BidKV 压缩周期（压力驱动）。

        在 vLLM scheduler 的 preemption 路径之前调用。
        流程：
        1. 更新 KV stats → PressureDetector
        2. 检查是否处于压力态
        3. 为所有追踪的请求生成/刷新 bids
        4. Solver 选择最优 bid 组合
        5. 执行压缩

        Returns
        -------
        int
            本轮实际释放的总 token 数。0 表示未触发或无需压缩。
        """
        if not self._config.is_active:
            return 0

        # Step 1: 更新 KV stats
        used, total = self.get_kv_stats()
        self._pressure_detector.update_stats(used, total)

        # Step 2: 检查压力
        if not self._pressure_detector.is_under_pressure():
            return 0

        self._metrics.record_pressure_event()

        # Step 3: 为追踪的请求刷新 bids
        self._refresh_bids()

        # Step 4: Solver 求解
        pool_snapshot = self._pool_manager.get_pool_snapshot()
        tokens_needed = self._pressure_detector.needed_tokens()
        acceptance = self._solver.solve(
            pool_snapshot,
            tokens_needed,
            decision_reason="vllm_kv_pressure",
        )

        if acceptance.is_empty:
            return 0

        # Step 5: 执行压缩
        total_freed = self._execute_acceptance(acceptance)
        return total_freed

    def try_compress_for_request(self, needed_blocks: int) -> int:
        """尝试压缩以为特定请求腾出 block 空间。

        在 allocate_slots 返回 None 时，preempt 之前调用。

        Parameters
        ----------
        needed_blocks:
            需要释放的 block 数量。

        Returns
        -------
        int
            实际释放的 block 数量（以 token 计）。
        """
        if not self._config.is_active:
            return 0

        block_size = self._get_block_size()
        if block_size <= 0:
            return 0

        needed_tokens = needed_blocks * block_size

        # 更新 KV stats
        used, total = self.get_kv_stats()
        self._pressure_detector.update_stats(used, total)
        self._metrics.record_pressure_event()

        # 刷新 bids
        self._refresh_bids()

        # Solver 求解
        pool_snapshot = self._pool_manager.get_pool_snapshot()
        acceptance = self._solver.solve(
            pool_snapshot,
            needed_tokens,
            decision_reason="vllm_allocate_slots_pressure",
        )

        if acceptance.is_empty:
            return 0

        total_freed = self._execute_acceptance(acceptance)
        return total_freed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_bids(self) -> None:
        """为所有追踪的请求重新生成 bids。"""
        for request_id, token_ids in self._request_tokens.items():
            if not token_ids:
                continue
            bids = self._scoring.generate_bids(
                request_id=request_id,
                token_ids=token_ids,
                compression_levels=self._compression_levels,
            )
            self._pool_manager.submit_bids(request_id, bids)

    def _execute_acceptance(self, acceptance: BidAcceptance) -> int:
        """执行 Solver 接受的 bid 组合。"""
        total_freed = 0
        for bid_id in acceptance.accepted_bid_ids:
            bid = self._pool_manager.get_bid(bid_id)
            if bid is None:
                continue
            freed = self.execute_compression(bid.request_id, bid.tokens_freed)
            total_freed += freed
        return total_freed

    def _get_block_size(self) -> int:
        """获取 vLLM 的 block size。"""
        if self._scheduler is None:
            return 0
        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return 0
        block_size = getattr(kv_cache_manager, "block_size", None)
        return block_size if block_size and block_size > 0 else 0

    def _free_tail_blocks(self, request_id: str, num_blocks: int, block_size: int) -> int:
        """通过 vLLM 的 KV cache coordinator 释放请求的尾部 block。

        vLLM v1 的 KVCacheManager 使用 coordinator 管理 block 分配。
        释放尾部 block 是最安全的操作，因为不会破坏前缀缓存。

        Parameters
        ----------
        request_id:
            请求 ID。
        num_blocks:
            要释放的 block 数量。
        block_size:
            每个 block 的 token 数。

        Returns
        -------
        int
            实际释放的 token 数量。
        """
        if self._scheduler is None or num_blocks <= 0:
            return 0

        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return 0

        coordinator = getattr(kv_cache_manager, "coordinator", None)
        if coordinator is None:
            return 0

        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return 0

        # 获取该请求当前的 block 列表
        try:
            blocks = coordinator.get_blocks(request_id)
        except (KeyError, AttributeError):
            logger.debug("_free_tail_blocks: request %s not found in coordinator", request_id)
            return 0

        if not blocks:
            return 0

        # blocks 是 tuple[list[KVCacheBlock], ...] — 每个 kv_cache_group 一个列表
        # 我们只释放最后 num_blocks 个 block（尾部释放最安全）
        freed_count = 0
        for group_blocks in blocks:
            if not group_blocks:
                continue
            n_to_free = min(num_blocks - freed_count, len(group_blocks) - 1)
            if n_to_free <= 0:
                continue
            # 取尾部 block 释放
            tail_blocks = group_blocks[-n_to_free:]
            block_pool.free_blocks(tail_blocks)
            # 从追踪中移除这些 block
            del group_blocks[-n_to_free:]
            freed_count += n_to_free

        return freed_count * block_size

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def track_request(self, request_id: str, token_ids: list[int]) -> None:
        """开始追踪一个请求的 token。

        Parameters
        ----------
        request_id:
            请求 ID。
        token_ids:
            请求的 token ID 列表。
        """
        self._request_tokens[request_id] = list(token_ids)
        logger.debug(
            "track_request: request=%s, tokens=%d",
            request_id,
            len(token_ids),
        )

    def get_tracked_requests(self) -> list[str]:
        """返回当前追踪的所有请求 ID。"""
        return list(self._request_tokens.keys())

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self) -> None:
        """激活 kill switch，立即停止所有 BidKV 操作。"""
        self._config = BidKVConfig(
            enabled=self._config.enabled,
            kill_switch=True,
            delta_budget=self._config.delta_budget,
            max_bids_per_solve=self._config.max_bids_per_solve,
        )
        self._pool_manager.activate_kill_switch()
        self._solver.update_config(
            SolverConfig(
                enabled=self._solver._config.enabled,
                kill_switch=True,
                delta_budget=self._solver._config.delta_budget,
            )
        )
        self._pressure_detector.set_enabled(False)
        self._metrics.record_kill_switch()
        logger.warning("VLLMAdapter: KILL SWITCH activated")

    def deactivate_kill_switch(self) -> None:
        """解除 kill switch，恢复 BidKV 操作。"""
        self._config = BidKVConfig(
            enabled=self._config.enabled,
            kill_switch=False,
            delta_budget=self._config.delta_budget,
            max_bids_per_solve=self._config.max_bids_per_solve,
        )
        self._pool_manager.enable()
        self._solver.update_config(
            SolverConfig(
                enabled=True,
                kill_switch=False,
                delta_budget=self._config.delta_budget,
            )
        )
        self._pressure_detector.set_enabled(True)
        logger.info("VLLMAdapter: kill switch deactivated, BidKV resumed")

    # ------------------------------------------------------------------
    # H2O decode step callback
    # ------------------------------------------------------------------

    def on_decode_step(self, request_id: str, attention_pattern: Sequence[float]) -> None:
        """decode step 完成后的回调，更新 H2O scoring。

        由 h2o_hook 在每个 decode step 后调用。

        Parameters
        ----------
        request_id:
            请求 ID。
        attention_pattern:
            当前 decode step 中 query token 对所有 KV token 的注意力权重。
        """
        if not self._config.is_active:
            return
        if hasattr(self._scoring, "update_from_decode_step"):
            self._scoring.update_from_decode_step(attention_pattern)
        self._metrics.record_decode_step(request_id)


class AdapterMetrics:
    """vLLM adapter 运行指标。"""

    def __init__(self) -> None:
        self.total_compressions: int = 0
        self.total_tokens_freed: int = 0
        self.total_pressure_events: int = 0
        self.total_requests_completed: int = 0
        self.total_decode_steps: int = 0
        self.kill_switch_activations: int = 0
        self.preemptions_avoided: int = 0

    def record_compression(self, request_id: str, tokens_freed: int) -> None:
        if tokens_freed > 0:
            self.total_compressions += 1
            self.total_tokens_freed += tokens_freed

    def record_pressure_event(self) -> None:
        self.total_pressure_events += 1

    def record_request_complete(self, request_id: str) -> None:
        self.total_requests_completed += 1

    def record_decode_step(self, request_id: str) -> None:
        self.total_decode_steps += 1

    def record_kill_switch(self) -> None:
        self.kill_switch_activations += 1

    def record_preemption_avoided(self) -> None:
        self.preemptions_avoided += 1

    def as_dict(self) -> dict[str, int]:
        """返回所有指标的字典形式。"""
        return {
            "total_compressions": self.total_compressions,
            "total_tokens_freed": self.total_tokens_freed,
            "total_pressure_events": self.total_pressure_events,
            "total_requests_completed": self.total_requests_completed,
            "total_decode_steps": self.total_decode_steps,
            "kill_switch_activations": self.kill_switch_activations,
            "preemptions_avoided": self.preemptions_avoided,
        }
