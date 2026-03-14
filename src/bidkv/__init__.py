"""bidkv — CompressionBid protocol layer for KV cache scheduling primitives.

零外部依赖的独立 Python 包，定义 BidKV 协议层核心类型。
"""

from bidkv._version import __version__
from bidkv.config import BidKVConfig
from bidkv.protocol import (
    FEATURE_GATE_ID,
    BidAcceptance,
    BidCapacityError,
    BidExecutionError,
    BidExpiredError,
    BidPool,
    CompressionBid,
    CompressionBidError,
    CompressionBidProvider,
    compute_utility,
    make_bid_id,
)

__all__ = [
    "__version__",
    "FEATURE_GATE_ID",
    "BidAcceptance",
    "BidCapacityError",
    "BidExecutionError",
    "BidExpiredError",
    "BidKVConfig",
    "BidPool",
    "CompressionBid",
    "CompressionBidError",
    "CompressionBidProvider",
    "compute_utility",
    "make_bid_id",
]
