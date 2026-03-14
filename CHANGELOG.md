# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Scoring Strategy Layer** (#043): Token 重要度评分策略
  - `ScoringStrategy` Protocol（`score()` + `generate_bids()`）
  - `H2OScoring`: Heavy Hitter Oracle — 基于累积注意力的 practical scoring（CPU, 无 GPU 依赖）
  - `AttentionWeightScoring`: Full attention weight aggregate — reference scoring
  - `UniformScoring`: 等权基线（消融实验用）
  - `RandomScoring`: 随机基线（消融实验用）
  - `generate_bids()` 按压缩级别生成合法 CompressionBid（三层体系完整）
  - H2O vs AttentionWeight Spearman rank correlation ≥ 0.7 验证通过
  - H2OScoring 精度边界声明文档（稀疏度 >90%、>32K token、多轮对话退化条件）
  - 60 个 scoring 单元测试全部通过

## [0.1.0] - 2026-03-14

### Added

- 从 `sagellm-protocol` 提取 CompressionBid 协议层为独立包 (issue #040)
- `bidkv.protocol.bid`: CompressionBid, BidPool, BidAcceptance 核心数据结构
- `bidkv.protocol.errors`: CompressionBidError 异常层次结构
- `bidkv.protocol.provider`: CompressionBidProvider Protocol 接口
- `bidkv.config`: BidKVConfig (feature gate + kill switch, 默认 OFF)
- CompressionBid 字段三层体系标注 (Layer 1/2/3)
- `compute_utility()` 标注为 "operational ranking signal, not ground-truth"
- 零外部依赖（仅 Python stdlib）
- 77 个单元测试全部通过
