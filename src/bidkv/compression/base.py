"""CompressionExecutor Protocol — 压缩执行抽象。

定义 FrameworkAdapter 必须实现的压缩执行接口。
bidkv 核心模块（Solver、Pool）不执行实际压缩，
由各框架（sagellm、vllm、sglang 等）通过 adapter 实现此接口。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CompressionExecutor(Protocol):
    """压缩执行器协议 — 由 FrameworkAdapter 实现。

    执行实际的 KV 压缩操作，返回实际释放的 token 数。
    bidkv 核心模块通过此接口与框架解耦。

    实现者须知
    ----------
    本接口为结构化子类型（structural subtyping），无需继承此类，
    只需实现 ``execute()`` 方法签名即可。
    """

    def execute(self, request_id: str, target_tokens: int) -> int:
        """执行压缩，返回实际释放的 token 数。

        Parameters
        ----------
        request_id:
            目标请求 ID，标识需要压缩的 KV cache。
        target_tokens:
            期望释放的 token 数量。

        Returns
        -------
        int
            实际释放的 token 数量（>= 0）。
            可能小于 target_tokens（块对齐、最小保留等约束）。
        """
        ...
