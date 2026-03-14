"""bidkv.adapters.vllm — vLLM framework adapter.

vLLM 0.11+ (v1 架构) 使用 KVCacheManager + BlockPool 进行分页 KV 管理。
本模块提供 BidKV 在 vLLM 上的完整适配层。

核心注入点：
- Scheduler.schedule() 的 preemption 路径（allocate_slots 返回 None 时）
- Scheduler.update_from_output() 的 decode step 回调（H2O scoring 更新）
- Scheduler._free_request() 的请求完成回调（lifecycle cleanup）
"""

from __future__ import annotations

from bidkv.adapters.vllm.adapter import VLLMAdapter

__all__ = [
    "VLLMAdapter",
]
