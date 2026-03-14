"""bidkv.adapters.sglang — SGLang framework adapter.

SGLang 使用 RadixAttention（树状 KV 管理），架构上与 vLLM（扁平分页）截然不同。
本模块提供 BidKV 在 SGLang 上的完整适配层。

关键差异 vs vLLM：
- KV 管理：树状 RadixAttention（前缀共享）vs 扁平 BlockTable
- 驱逐策略：radix tree 节点 LRU vs seq_group 整体 preempt
- 调度入口：``get_next_batch_to_run()`` vs ``_schedule()``
- KV 内存池：``TokenToKVPool`` + ``ReqToTokenPool`` vs ``BlockAllocator``
"""

from __future__ import annotations

from bidkv.adapters.sglang.adapter import SGLangAdapter

__all__ = [
    "SGLangAdapter",
]
