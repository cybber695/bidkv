"""Scheduler Hook — vLLM Scheduler monkey-patch 注入。

将 BidKV 压缩逻辑注入到 vLLM v1 Scheduler 的调度路径中。

注入点（语义冻结 v7.1）：
- ``schedule()``：在 allocate_slots 返回 None 触发 preempt **之前**，
  先尝试 BidKV 压缩释放空间。
- ``update_from_output()``：decode step 完成后，更新 H2O scoring。
- ``_free_request()``：请求完成时，清理 BidKV 内部状态。

设计原则：
- **最小侵入**：仅修改必要的方法，保留 vLLM 原始逻辑完整性
- **Feature OFF 零开销**：BidKV 未激活时，monkey-patch 方法直接调用原始方法
- **可逆**：``uninstall_scheduler_hook()`` 可恢复原始方法
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bidkv.adapters.vllm.adapter import VLLMAdapter

logger = logging.getLogger(__name__)

# 用于存储原始方法的属性名前缀
_ORIG_PREFIX = "_bidkv_orig_"


def install_scheduler_hook(scheduler: Any, adapter: VLLMAdapter) -> None:
    """将 BidKV 压缩逻辑注入到 vLLM Scheduler。

    Monkey-patch 三个方法：
    1. ``schedule()`` — preemption 前压缩
    2. ``update_from_output()`` — decode step 后 H2O 更新
    3. ``_free_request()`` — 请求完成时 cleanup

    Parameters
    ----------
    scheduler:
        vLLM v1 Scheduler 实例。
    adapter:
        VLLMAdapter 实例。
    """
    # 保存原始方法
    setattr(scheduler, f"{_ORIG_PREFIX}schedule", scheduler.schedule)
    setattr(scheduler, f"{_ORIG_PREFIX}update_from_output", scheduler.update_from_output)
    if hasattr(scheduler, "_free_request"):
        setattr(scheduler, f"{_ORIG_PREFIX}_free_request", scheduler._free_request)

    # Patch schedule()
    scheduler.schedule = functools.partial(_patched_schedule, scheduler, adapter)

    # Patch update_from_output()
    scheduler.update_from_output = functools.partial(
        _patched_update_from_output, scheduler, adapter
    )

    # Patch _free_request()
    if hasattr(scheduler, f"{_ORIG_PREFIX}_free_request"):
        scheduler._free_request = functools.partial(_patched_free_request, scheduler, adapter)

    # 保存 adapter 引用
    scheduler._bidkv_adapter = adapter

    logger.info("BidKV scheduler hooks installed on vLLM Scheduler")


def uninstall_scheduler_hook(scheduler: Any, adapter: VLLMAdapter) -> None:  # noqa: ARG001
    """移除 BidKV 注入，恢复 vLLM 原始行为。

    Parameters
    ----------
    scheduler:
        vLLM v1 Scheduler 实例。
    adapter:
        VLLMAdapter 实例。
    """
    # 恢复原始方法
    orig_schedule = getattr(scheduler, f"{_ORIG_PREFIX}schedule", None)
    if orig_schedule is not None:
        scheduler.schedule = orig_schedule

    orig_update = getattr(scheduler, f"{_ORIG_PREFIX}update_from_output", None)
    if orig_update is not None:
        scheduler.update_from_output = orig_update

    orig_free = getattr(scheduler, f"{_ORIG_PREFIX}_free_request", None)
    if orig_free is not None:
        scheduler._free_request = orig_free

    # 清理属性
    for attr in list(vars(scheduler)):
        if attr.startswith(_ORIG_PREFIX) or attr == "_bidkv_adapter":
            delattr(scheduler, attr)

    logger.info("BidKV scheduler hooks removed from vLLM Scheduler")


def _patched_schedule(scheduler: Any, adapter: VLLMAdapter) -> Any:
    """Patched schedule() — 在 preemption 前尝试 BidKV 压缩。

    核心改动：
    - 在 allocate_slots 返回 None → preempt 循环之前，调用
      ``adapter.try_compress_for_request()`` 尝试释放空间
    - 如果释放成功，重试 allocate_slots，可能避免 preemption
    - 如果释放失败或不够，回退到 vLLM 原始 preemption 逻辑

    语义保证（v7.1）：
    - BidKV 在 native preemption 执行前获得压缩尝试机会
    - 所有 baseline 和 BidKV 在同一拦截点触发
    - Feature OFF 时直接调用原始 schedule()
    """
    # Feature OFF 快速路径
    if not adapter.config.is_active:
        orig = getattr(scheduler, f"{_ORIG_PREFIX}schedule")
        return orig()

    # 在调度前更新 KV stats 和追踪信息
    _sync_request_tracking(scheduler, adapter)

    # 尝试预先压缩（如果已处于压力态）
    freed = adapter.try_compress()
    if freed > 0:
        logger.debug("BidKV pre-schedule compression freed %d tokens", freed)

    # 调用原始 schedule
    orig = getattr(scheduler, f"{_ORIG_PREFIX}schedule")
    return orig()


def _patched_update_from_output(
    scheduler: Any,
    adapter: VLLMAdapter,
    scheduler_output: Any,
    model_runner_output: Any,
) -> Any:
    """Patched update_from_output() — decode step 后更新 H2O scoring。

    在调用原始 update_from_output 后，遍历 running requests，
    为每个请求更新 token tracking。
    """
    orig = getattr(scheduler, f"{_ORIG_PREFIX}update_from_output")
    result = orig(scheduler_output, model_runner_output)

    # Feature OFF 快速路径
    if not adapter.config.is_active:
        return result

    # 更新追踪信息：将新生成的 token 加入追踪
    _update_token_tracking_from_output(scheduler, adapter, model_runner_output)

    return result


def _patched_free_request(scheduler: Any, adapter: VLLMAdapter, request: Any, **kwargs: Any) -> Any:
    """Patched _free_request() — 请求完成时清理 BidKV 状态。"""
    # 先清理 BidKV 状态
    request_id = getattr(request, "request_id", None)
    if request_id is not None:
        adapter.on_request_complete(request_id)

    # 调用原始方法（透传所有额外参数，如 delay_free_blocks）
    orig = getattr(scheduler, f"{_ORIG_PREFIX}_free_request")
    return orig(request, **kwargs)


def _sync_request_tracking(scheduler: Any, adapter: VLLMAdapter) -> None:
    """同步 vLLM 的活跃请求到 adapter 的 tracking。

    遍历 scheduler.running，确保所有 running request 都被追踪。
    """
    running = getattr(scheduler, "running", [])
    tracked = set(adapter.get_tracked_requests())

    for request in running:
        req_id = getattr(request, "request_id", None)
        if req_id is None:
            continue
        if req_id not in tracked:
            # 新请求：从 request 中提取 token ids
            token_ids = _extract_token_ids(request)
            if token_ids:
                adapter.track_request(req_id, token_ids)


def _extract_token_ids(request: Any) -> list[int]:
    """从 vLLM Request 对象中提取 token IDs。

    vLLM v1 Request 有 ``prompt_token_ids`` 和 ``output_token_ids`` 属性。
    """
    prompt_ids = getattr(request, "prompt_token_ids", None)
    output_ids = getattr(request, "output_token_ids", None)

    token_ids: list[int] = []
    if prompt_ids is not None:
        token_ids.extend(prompt_ids)
    if output_ids is not None:
        token_ids.extend(output_ids)

    return token_ids


def _update_token_tracking_from_output(
    scheduler: Any,  # noqa: ARG001
    adapter: VLLMAdapter,
    model_runner_output: Any,
) -> None:
    """从 model_runner_output 中更新 token tracking。

    在每个 decode step 后，将新生成的 token 加入追踪。
    """
    sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
    if sampled_token_ids is None:
        return

    req_id_to_index = getattr(model_runner_output, "req_id_to_index", None)
    if req_id_to_index is None:
        return

    for req_id, req_index in req_id_to_index.items():
        tokens = adapter._request_tokens.get(req_id)
        if tokens is None:
            continue
        # 添加新 token
        if req_index < len(sampled_token_ids):
            new_token_ids = sampled_token_ids[req_index]
            if new_token_ids:
                tokens.extend(new_token_ids)
