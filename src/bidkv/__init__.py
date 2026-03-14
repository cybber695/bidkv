"""bidkv — CompressionBid framework-adaptable library for KV cache scheduling primitives.

零外部依赖的独立 Python 包，提供：
- Protocol 层：核心类型（CompressionBid, BidPool, BidAcceptance）
- Scoring 层：token 重要度评分策略
- Core 层：BidPoolManager, GreedyBidSolver, PressureDetector, CompressionExecutor
- Baselines 层：7 个 baseline 策略 + Oracle DP 上界
"""

from bidkv._version import __version__
from bidkv.adapters import FrameworkAdapter
from bidkv.baselines import (
    BaselineContext,
    BaselineRegistry,
    BaselineStrategy,
    BidKVStrategy,
    CompressionAction,
    GlobalNoBidStrategy,
    H2OStyleStrategy,
    OracleDPStrategy,
    PreemptEvictStrategy,
    RequestState,
    SlackAwareStrategy,
    StaticRandomStrategy,
    UniformStrategy,
)
from bidkv.compression import CompressionExecutor
from bidkv.config import BidKVConfig
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
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
from bidkv.scoring import (
    AttentionWeightScoring,
    H2OScoring,
    RandomScoring,
    ScoringStrategy,
    UniformScoring,
)
from bidkv.solver import ExecutionResult, GreedyBidSolver, SolverConfig

__all__ = [
    "__version__",
    "FEATURE_GATE_ID",
    # Protocol
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
    # Core
    "BidPoolManager",
    "CompressionExecutor",
    "ExecutionResult",
    "GreedyBidSolver",
    "PressureConfig",
    "PressureDetector",
    "SolverConfig",
    # Scoring
    "AttentionWeightScoring",
    "H2OScoring",
    "RandomScoring",
    "ScoringStrategy",
    "UniformScoring",
    # Baselines
    "BaselineContext",
    "BaselineRegistry",
    "BaselineStrategy",
    "BidKVStrategy",
    "CompressionAction",
    "GlobalNoBidStrategy",
    "H2OStyleStrategy",
    "OracleDPStrategy",
    "PreemptEvictStrategy",
    "RequestState",
    "SlackAwareStrategy",
    "StaticRandomStrategy",
    "UniformStrategy",
    # Adapters
    "FrameworkAdapter",
]
