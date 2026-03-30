# BidKV 系统架构分析报告
> **文档版本**: v1.1 (2026-03-27)  
> **项目状态**: Phase C 实施中 (Wave 1-3 已完成)  
> **代码规模**: ~12,000 LOC Python, 447+ tests

---

## 目录

1. [分析目标与结论摘要](#1-分析目标与结论摘要)
2. [全局架构总览](#2-全局架构总览)
3. [公开 API 与包边界](#3-公开-api-与包边界)
4. [协议与类型系统分析](#4-协议与类型系统分析)
5. [配置治理：默认 OFF 与 Kill Switch](#5-配置治理默认-off-与-kill-switch)
6. [Core Pipeline 分析](#6-core-pipeline-分析)
7. [Baseline 体系分析](#7-baseline-体系分析)
8. [Framework Adapter 层分析](#8-framework-adapter-层分析)
9. [实验框架分析](#9-实验框架分析)
10. [关键数据流与控制流](#10-关键数据流与控制流)
11. [设计模式识别](#11-设计模式识别)
12. [零外部依赖约束对架构的影响](#12-零外部依赖约束对架构的影响)
13. [可测试性、可维护性、可扩展性评估](#13-可测试性可维护性可扩展性评估)
14. [面向研究与实现的协同机制](#14-面向研究与实现的协同机制)
15. [潜在限制与风险点](#15-潜在限制与风险点)
16. [不违反冻结约束的改进建议](#16-不违反冻结约束的改进建议)
17. [综合评价](#17-综合评价)
18. [一句话判断](#18-一句话判断)

---
## 1. 分析目标与结论摘要

本报告面向“研究方案实施 + 工程实现落地”两个目标，对 BidKV 项目的整体代码结构进行系统性分析。

结论先行：

1. BidKV 的架构是一个典型的“协议层先行、核心算法内聚、框架适配外置、实验系统并行演化”的研究型工程结构。
2. 它没有把自己做成某个 serving framework 的插件集合，而是先定义了一套独立的中间抽象：CompressionBid、BidPool、BidAcceptance、ScoringStrategy、FrameworkAdapter、BaselineStrategy。
3. 项目最重要的工程选择不是某个具体算法，而是“零外部依赖 + 默认关闭 + kill switch + 冻结实验配置”这四个制度化约束，它们决定了代码组织方式、依赖边界和扩展姿势。
4. 从研究实现协同看，这个仓库已经不是单纯的算法原型，而是一个同时服务于论文实验、跨框架移植、基线对比和结果复现实验的平台型代码库。
5. 从可扩展性看，新增 baseline、新 scoring、新 adapter 的入口都已经被显式建模；从可维护性看，最大的复杂度集中在两类位置：框架 hook 边界、实验运行与真实引擎交互边界。

### 项目规模与质量指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 代码规模 | ~12,000 LOC | 纯 Python 实现，零外部依赖 |
| 测试覆盖 | 447+ tests | 覆盖 protocol、scoring、baseline、adapter、experiment |
| 模块数量 | 8 个核心模块 | protocol, scoring, pool, pressure, solver, compression, baselines, adapters |
| 框架支持 | 2 个 | vLLM 0.17.1 (v1) + SGLang (RadixAttention) |
| 策略数量 | 7 个 | 6 baseline + BidKV 完整系统 |

当前工程重心是"把论文中的关键机制对象化并验证其边界"，代码质量已达到可发表研究系统标准。

## 2. 全局架构总览

### 2.1 分层结构

BidKV 采用六层架构设计，从公开 API 到实验复现层，每层职责明确：

| 层级 | 模块 | 职责 |
|------|------|------|
| 1. 公开 API 层 | `src/bidkv/__init__.py` | 统一对外接口 |
| 2. 协议与类型层 | `protocol/`, `config.py` | CompressionBid / BidPool / BidAcceptance |
| 3. 决策核心层 | `scoring/`, `pool/`, `pressure/`, `solver/`, `compression/` | 核心算法链路 |
| 4. 策略与基线层 | `baselines/` | 7 个策略 + 注册表 |
| 5. 框架适配层 | `adapters/vllm/`, `adapters/sglang/` | 跨框架移植接口 |
| 6. 实验编排层 | `experiments/`, `scripts/`, `results/`, `paper/` | 论文实验系统 |

**架构数据流示意**：

```text
┌─────────────────────────────────────────────────┐
│ Layer 1: Public API                             │
│ src/bidkv/__init__.py                          │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│ Layer 2: Protocol & Type System                │
│ protocol/, config.py                           │
│ CompressionBid / BidPool / BidAcceptance       │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│ Layer 3: Core Decision Pipeline                │
│ scoring/ → pool/ → pressure/ → solver/         │
│ → compression/                                 │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│ Layer 4: Strategy Layer                        │
│ baselines/                                     │
│ 7 strategies + registry                        │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│ Layer 5: Framework Adapter Layer               │
│ adapters/vllm/, adapters/sglang/               │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│ Layer 6: Experiment & Reproducibility Layer    │
│ experiments/, scripts/, results/, paper/       │
└─────────────────────────────────────────────────┘
```

### 2.2 架构核心思想

这个项目的核心不是“做压缩”，而是“把压缩调度问题重新表述为 bid-based scheduling primitive”。

也就是说，它先定义问题，再允许不同框架、不同 scoring、不同 baseline 在同一问题接口上比较。这个思路使得：

1. 研究问题和工程实现之间有稳定边界。
2. 论文中的概念可以一一映射到代码对象。
3. 框架移植不需要重写算法，只需要重写 adapter。
4. baseline 比较可以共享候选池和指标口径。

## 3. 公开 API 与包边界

顶层导出文件 [src/bidkv/__init__.py](src/bidkv/__init__.py) 扮演了"架构目录"的角色，而不只是普通的聚合导入文件。

### 3.1 API 分组设计

它向外暴露了 6 组对象，形成清晰的功能层次：

| 分组 | 导出对象 | 职责 |
|------|----------|------|
| **Protocol** | `CompressionBid`, `BidPool`, `BidAcceptance`, 异常类, utility 函数 | 核心协议与类型 |
| **Core** | `BidPoolManager`, `CompressionExecutor`, `GreedyBidSolver`, `PressureDetector` | 决策引擎核心组件 |
| **Scoring** | `ScoringStrategy`, `H2OScoring`, `AttentionWeightScoring`, etc. | Token 重要度评分策略 |
| **Baselines** | 7 个策略类, `BaselineRegistry`, `RequestState`, `CompressionAction` | 策略体系与注册表 |
| **Adapters** | `FrameworkAdapter`, `BaseAdapterMetrics`, `VLLMAdapter`, `SGLangAdapter` | 跨框架适配接口 |
| **Experiments** | `ExperimentMetrics`, `BaseExperimentRunner` | 实验编排与指标采集 |

### 3.2 设计意图

这说明该包的对外定位**不是某个单点功能库，而是一个完整的调度原语工具包**。

这类 API 设计有两个重要效果：

1. **依赖简化**：上层用户可以只依赖 `bidkv` 包名，而不需要关心内部目录细节。
2. **概念统一**：仓库内部的研究模块和实验模块可以共用同一组稳定对象，不必各自复制概念定义。

从软件工程角度看，这种设计避免了"研究原型中概念分散"的常见问题，为跨框架移植和实验复现提供了稳定的抽象基础。

## 4. 协议与类型系统分析

### 4.1 CompressionBid：全系统的语义原子

`CompressionBid` 的设计非常关键。它**不是简单的数据类**，而是整个系统的**"研究问题编码"**。

#### 字段分层设计

其字段被明确分为三层，各有不同的消费者和语义：

| 层级 | 字段 | 消费者 | 语义 |
|------|------|---------|------|
| **Layer 1** | `tokens_freed`, `quality_delta` | Solver | 优化算法核心输入，与框架解耦 |
| **Layer 2** | `request_id`, `compress_latency_ms` | BidPool | 缓存、过滤、路由，不污染求解逻辑 |
| **Layer 3** | `confidence`, `metadata` | 日志/分析 | 扩展空间，不侵入主算法 |

#### 设计价值

这种分层设计的价值在于：

1. **算法解耦**：Solver 只消费 Layer 1，因此优化算法与框架细节解耦。
2. **职责清晰**：BidPool 主要关心 Layer 2，因此缓存、过滤、路由不会污染核心求解逻辑。
3. **可扩展性**：Layer 3 为日志、分析、未来扩展保留空间，不会过早侵入主算法。

**本质语义**：把"质量损失换空间释放"的交换关系固化为统一对象，这是 BidKV 系统最核心的抽象。

### 4.2 BidPool：候选宇宙快照，而非在线数据库

`BidPool` 的语义是某时刻所有活跃 bid 的**不可变快照**（immutable snapshot）。这里的关键词是 **snapshot**。

#### 设计意图

这意味着：

1. **稳定输入**：Solver 面对的是一个稳定输入，而不是会在求解期间变化的动态集合。
2. **一致性保证**：candidate-universe consistency 可以通过同一份快照自然保证。
3. **可复现性**：实验公平性和算法可复现性更容易建立。

这是一个**明显服务于研究评估**的设计，而不仅仅是工程便利。从软件工程视角看，它把"流式更新"与"批量决策"的边界显式化，避免了在线系统中常见的"决策时观察不一致"问题。

### 4.3 BidAcceptance：决策结果的统一表示

`BidAcceptance` **不是直接执行结果**，而是**"决策层输出"**。这让系统形成了清晰的两阶段链路：

1. **Decide 阶段**：哪些 bid 被接受（solver 输出 `BidAcceptance`）
2. **Execute 阶段**：这些 bid 在框架中如何真正释放 KV（adapter 执行）

#### 研究价值

这种抽象分离对分析 **estimated vs actual** 的偏差很重要，也是研究系统中非常必要的抽象分离。它使得：

- Solver 可以独立验证决策正确性
- Adapter 可以独立验证执行正确性
- 实验系统可以对比预期与实际的释放量差异

### 4.4 Protocol 与 ABC 混用的类型策略

仓库同时使用了 Python `Protocol` 和 `ABC`，这是一种**成熟的类型设计策略**：

| 类型机制 | 使用场景 | 示例 | 设计意图 |
|---------|---------|------|---------|
| **Protocol** | 结构化鸭子类型接口 | `ScoringStrategy`, `BaselineContext`, `CompressionBidProvider` | 对外部实现者友好，降低接入成本 |
| **ABC** | 强制继承和模板方法 | `BaselineStrategy`, `FrameworkAdapter` | 对框架主骨架严格约束，保持行为一致性 |

#### 设计合理性

这种混用是合理的：

1. **降低接入门槛**：对外部实现者友好的接口用 Protocol，不强制继承。
2. **保持框架约束**：对框架主骨架要求严格的位置用 ABC，确保关键行为不漏实现。
3. **类型安全**：既保留了静态类型检查能力，又不过度约束实现方式。

这比"一律用继承"或"一律用 duck typing"都更成熟，体现了对 Python 类型系统的深刻理解和工程权衡。

## 5. 配置治理：默认 OFF 与 Kill Switch

`BidKVConfig` 是项目**最关键的治理对象之一**。

### 5.1 三条核心制度

| 制度 | 配置字段 | 默认值 | 治理意图 |
|------|---------|--------|---------|
| **默认关闭** | `enabled` | `False` | BidKV 可以在生产框架中以"零侵入默认关闭"方式存在 |
| **Kill Switch** | `kill_switch` | `False` | 可全局绕过所有 bid 逻辑，即使 adapter 已安装 |
| **求解预算** | `delta_budget`, `max_bids_per_solve` | 有限值 | 控制单次决策的计算开销和质量损失上界 |

### 5.2 工程价值

这一设计的工程含义非常明确：

1. **零侵入**：BidKV 可以在生产框架中以"零侵入默认关闭"方式存在。
2. **风险优先**：风险控制优先于功能启用，特别适合实验性系统。
3. **紧急退出**：即使 adapter 已安装，只要 kill switch 打开，整个链路立即退化为 no-op。

### 5.3 研究系统走向生产的安全治理

这**不是单纯的配置项堆积**，而是**实验系统走向真实系统时必须具备的安全治理手段**。它体现了以下认知：

- 研究原型进入生产框架时，必须有"功能完全关闭"的能力
- 即使代码已部署，也要能通过配置即刻回退到无 BidKV 状态
- 预算控制是防止单次求解失控的最后防线

从软件工程角度，这是"feature gate + kill switch + circuit breaker"三重保护模式的体现。

## 6. Core Pipeline 分析

### 6.1 Scoring 层：score-only 契约

`ScoringStrategy` 被定义成**非常瘦的接口**：只要求实现 `score(token_ids, **context) -> list[float]`。

#### 设计决策价值

这是一个重要设计决策。它意味着：

| 职责边界 | 说明 | 效果 |
|---------|------|------|
| **只负责评分** | scoring 只负责估计 token importance | 实现非常可替换 |
| **bid 生成统一** | bid 生成统一由 `build_bids` 负责 | 避免各 scorer 重复实现 |
| **解耦其他组件** | scorer 不需要知道 solver、pool 或 adapter 的存在 | 依赖关系清晰 |

从架构上看，**scoring 层承担的是"感知"职责，而不是"决策"职责**。这种克制使得 token 重要度估计算法可以独立演化，而不影响调度主逻辑。

### 6.2 多种 scoring 的研究角色划分

从文档和目录结构看，scoring 被分成三类角色：

| 类型 | 实现 | 研究角色 |
|------|------|----------|
| **Practical** | `H2OScoring` | 可部署近似代理，decode-step 轻量级评分 |
| **Reference** | `AttentionWeightScoring` | 更强但可能更重的参考上界 |
| **Auxiliary** | `UniformScoring`, `RandomScoring` | 消融和对照组 |

这**不是纯工程分类，而是实验语义分类**。其意义是：

1. H2O 对应论文中可实际部署的 surrogate scoring
2. Attention 对应更接近 ground truth 但开销更大的评分方式
3. Uniform/Random 对应消融实验的基线

因此 **scoring 目录既是算法实现区，也是实验变量区**。

### 6.3 BidPoolManager：在线缓存与快照桥梁

`BidPoolManager` 的职责边界很清晰：

| 方法 | 职责 | 调用方 |
|------|------|--------|
| `submit_bids` | 由外部生成并提交 bid | adapter 或 baseline |
| `get_pool_snapshot` | 将在线缓存转成不可变决策输入 | solver |
| `remove/invalidate` | 生命周期清理 | adapter |
| `is_active` | 受 feature gate 与 kill switch 统一控制 | 全局 |

#### 设计克制

它**不自己生成 bid，也不自己做决策**。这种克制是正确的。

`BidPoolManager` 的关键价值不在复杂逻辑，而在**"把流式更新转成稳定求解输入"**。它是 online system 和 batch decision 之间的桥梁组件。

### 6.4 PressureDetector：唯一 KV 压力入口

`PressureDetector` 有两个很强的架构信号：

| 设计约束 | 内容 | 目的 |
|---------|------|------|
| **禁止平滑** | 明确禁止 rolling window 和指数平滑 | 保持触发语义清晰 |
| **单一真相来源** | 明确声明自己是 KV 状态唯一来源 | 避免多个组件各自计算 needed_tokens |

#### 可解释性保护

这两个约束本质上是在**保护实验可解释性**：

- 如果引入平滑，pressure trigger 的语义就会变得模糊
- 如果多个组件各自计算 needed_tokens，系统就会失去单一真相来源

#### 当前实现：瞬时值决策

当前实现采用瞬时值决策：

1. occupancy 超过阈值触发
2. 高优先级 pending 存在且 free tokens 低于阈值触发
3. `needed_tokens` 返回当前安全线缺口

这使得 **pressure → solve 的链路可以被直接审计和复现**。

### 6.5 GreedyBidSolver：研究论文到代码的一一映射

`GreedyBidSolver` 是**论文 Algorithm 1 的核心实现**。其求解逻辑非常直接：

#### 算法流程

1. 按 $U = \frac{r}{\delta + \varepsilon}$ 排序（utility-ratio）
2. 每个 request 最多选一个 bid
3. 总 quality delta 不超过预算
4. 达到 `tokens_needed` 后提前停止

#### 优点与边界

| 维度 | 优点 | 边界/限制 |
|------|------|----------|
| **透明性** | 算法含义非常透明，与论文叙述一致 | — |
| **可核对性** | 易于审稿人和研究者核对 | — |
| **复杂度** | 工程行为可预测，$O(n \log n)$ 排序主导 | — |
| **全局最优** | — | 不是全局最优求解器，而是 utility-ratio greedy |
| **代理依赖** | — | 依赖 `quality_delta` 的可信度 |
| **组合精度** | — | 每请求最多一档 bid 的约束降低了搜索空间 |

#### 研究权衡

对研究实现而言，这种折中是**合理的**，因为它换来了：

- 可解释性：每次决策可以逐步审计
- 稳定性：不依赖复杂的全局优化器
- 可复现性：确定性算法，相同输入产生相同输出

### 6.6 执行层与实际释放量校验

solver 中的 `execute_accepted` 明确区分 `estimated_freed` 和 `actual_freed`。**这一点很重要**。

#### 现实承认

它说明系统承认一个现实：

1. **决策层**：估计释放多少 token
2. **框架层**：未必真的按预期释放成功

#### 工程价值

因此项目**没有把"求解正确"误当成"执行成功"**。这对真实 serving 系统尤为关键，因为：

- KV 释放可能因共享前缀而实际释放量更少
- 框架内部状态可能导致部分 token 无法立即释放
- 执行超时或失败会导致零释放

这种区分使得系统可以：
- 诊断 solver 估计偏差
- 评估框架适配器的执行质量
- 为下一轮决策提供更准确的反馈

## 7. Baseline 体系分析

### 7.1 Baseline 抽象的建模质量较高

BaselineStrategy 统一使用 select_victims(candidates, needed_tokens, **kwargs) -> list[CompressionAction]。

它的优点有三点：

1. 所有 baseline 共用同一输入候选池，便于公平对照。
2. 输出统一为 CompressionAction，执行层可以复用。
3. kwargs 允许保留实验扩展能力，而不破坏基类接口。

这使 baseline 不是“散落的脚本”，而是一个真正的策略子系统。

### 7.2 7 个 baseline 形成的是归因链，而不是平铺列表

从 baseline-specs 文档和 registry 默认注册顺序看，这 7 个策略承担的是不同层次的归因任务：

1. Preempt-Evict：框架默认下界
2. Static-Random：无信息随机对照
3. H2O-Style：有 token-level scoring，但不走 bid
4. Uniform：无差异化压缩
5. Global-NoBid：系统自动推断 utility，不暴露用户 bid
6. Slack-Aware：只用 deadline/slack
7. BidKV：完整路径

这里最重要的并不是策略数目，而是它们构成了一条有语义的研究归因链。

这说明 baselines/ 目录不是附属实验代码，而是论文论证结构的程序化表达。

### 7.3 Registry 模式的作用

BaselineRegistry 通过 register/get/create_default_registry 管理策略实例。

它的作用不是复杂工厂，而是：

1. 消除字符串到类的硬编码分散
2. 为 CLI、实验运行器、plugin 环境变量路由提供统一入口
3. 把“实验配置中的策略名”映射到“实际策略对象”

这让实验系统与算法实现之间的耦合控制在一个干净接口上。

### 7.4 BidKVStrategy 是主流程的 baseline 适配器

BidKVStrategy 很有代表性。它不是重新发明一套逻辑，而是把 scoring → build_bids → pool → solver 这条主链包装成 baseline 接口。

这表明项目刻意避免“研究主算法”和“baseline 调度接口”分叉演化。

这是一个成熟信号：主算法复用核心流水线，而不是在 baseline 层再复制一遍实现。

## 8. Framework Adapter 层分析

### 8.1 FrameworkAdapter ABC 定义了跨框架最小公分母

FrameworkAdapter 被定义为 5 层职责边界：

1. KV stats 获取
2. Pressure interception
3. Compression 执行
4. Scoring 回调
5. Lifecycle 管理

这 5 个点恰好是把任意 serving framework 接入 BidKV 所必需的最小能力集合。

换句话说，BidKV 并不要求框架暴露完整内部实现，只要求框架能在这 5 个接口点被观察和操作。

这就是“framework-portable”真正落在代码里的地方。

### 8.2 vLLMAdapter：在受限框架中做最小侵入接入

vLLMAdapter 的关键特征是：

1. 通过 plugin.py 以 vllm.general_plugins 入口注入
2. 通过 monkey-patch Scheduler.__init__ 自动安装 hook
3. 通过 scheduler 路径在 preemption 之前尝试压缩
4. 在当前实现里，execute_compression 路由到 tail truncation 路径
5. 在实验设计层面保留了基于策略名的 baseline 路由能力

vLLM 适配的难点不在 BidKV 逻辑本身，而在“vLLM 并没有天然提供为 BidKV 设计的扩展面”。

因此它采用的是典型的研究型工程策略：

1. 尽量不改框架源代码
2. 优先在调度边界打补丁
3. 把复杂性局部化在 adapter/hook 层

这带来的代价是：

1. 实现更脆弱，依赖框架内部对象结构
2. 需要同步请求状态和 KV state 的一致性
3. Mode A 与 Mode B 的执行语义更依赖引擎内部不变量

### 8.3 SGLangAdapter：利用原生细粒度 KV 结构获得更自然接入

SGLangAdapter 的总体结构与 vLLMAdapter 一致，但其可操作空间更好：

1. SGLang 的 RadixAttention 天然带有树状前缀结构
2. 可以做 token-level 或 node-level 释放
3. 共享前缀可以通过 ref count 类语义进行保护
4. shared_positions 这类显式追踪结构使“不可压缩共享前缀”可编程表达

radix_hook.py 展示了这一点：

1. 可以从 token position 定位 KV slot
2. 可以判断 slot 是否共享
3. 可以跳过共享前缀，只释放私有位置

因此 SGLang 更接近 BidKV 机制的理想宿主。

### 8.4 两类 adapter 的共同点与差异

共同点：

1. 都复用 PressureDetector、BidPoolManager、GreedyBidSolver
2. 都保存 request token 跟踪信息
3. 都通过 experiment_strategy 接口支持 baseline 实验路由
4. 都具有 metrics 采集能力

差异点：

1. vLLM 更偏 block-level / scheduler patching
2. SGLang 更偏 token-level / radix structure exploitation
3. vLLM 的正确性风险更多来自内部状态同步
4. SGLang 的正确性风险更多来自共享前缀保护和池映射一致性

研究意义上，这两个 adapter 共同构成了“可移植性”主张的工程证据。

## 9. 实验框架分析

### 9.1 experiments/ 不是辅助目录，而是第二主系统

这个项目有一个容易被低估的事实：experiments/ 不是简单脚本集合，而是与 src/bidkv/ 并行的第二套系统。

它承担的是：

1. 冻结实验配置
2. 运行矩阵规划
3. trace 复用
4. 结果落盘
5. 审计与报告
6. 论文图表输入生成

这意味着仓库本质上是“双核结构”：

1. 核心算法系统
2. 论文实验系统

### 9.2 共享实验抽象：BaseExperimentRunner

BaseExperimentRunner 提供了统一的 ExperimentConfig 和 plan 机制。

它体现出项目对实验可重复性的要求：

1. run 的笛卡尔积是显式可计算的
2. framework、strategy、workload、concurrency、run_id 都有明确枚举
3. 输出结果对象化，而不是依赖临时脚本打印

这是标准研究工程化的特征。

### 9.3 vLLM 与 SGLang 配置的冻结机制

vLLM 和 SGLang 配置文件有一个共同特点：不是“默认建议值”，而是“冻结后的实验制度编码”。

例如：

1. strategy 集合固定
2. workload 集合固定
3. per-workload request rates 固定
4. 配置提供 get_rates_for_workload 这类显式查询接口

这里最重要的不是代码技巧，而是研究治理：代码文件本身就是 freeze artifact。

### 9.4 指标模式：ExperimentMetrics

ExperimentMetrics 的字段覆盖了论文主指标：

1. SLO attainment
2. p99 TTFT
3. throughput
4. compression coverage
5. 可选质量指标
6. adapter 内部指标

其设计说明：

1. 论文外显指标与系统内部指标被统一存储
2. 系统行为既从用户侧看，也从机制侧看
3. 指标 schema 是论文叙事和工程调试的交汇点

这对于后续做 figure/table 生成非常重要。

## 10. 关键数据流与控制流

### 10.1 一次 compression 决策的完整生命周期

可以把主数据流写成：

```text
Request state / token ids
	↓
ScoringStrategy.score()
	↓
build_bids()
	↓
BidPoolManager.submit_bids()
	↓
BidPool snapshot
	↓
PressureDetector.needed_tokens()
	↓
GreedyBidSolver.solve()
	↓
BidAcceptance
	↓
Adapter.execute_compression()
	↓
ExecutionResult(actual vs estimated)
	↓
Metrics / audit / results
```

这条数据流最大的优点是阶段清晰，几乎每一步都有独立对象承载。

### 10.2 Pressure-trigger 控制流

控制流上，典型触发顺序是：

1. adapter 获取当前 KV stats
2. PressureDetector.update_stats
3. PressureDetector.is_under_pressure 判断是否触发
4. 若触发，使用 detector.needed_tokens 作为 solve 目标
5. 读取 pool snapshot
6. solver 返回 acceptance
7. adapter 执行 compression

这种控制流体现了单向责任传递：

1. detector 决定是否需要行动
2. solver 决定行动方案
3. adapter 决定如何在 конкрет框架上落地

三者职责没有混合。

### 10.3 Mode A / Mode B 的架构差异

从仓库说明与 adapter 代码可见，vLLM 有双轨执行语义：

1. Mode A：recompute fallback 语义，核心思想是在 preemption/recompute 语义下避免不必要的代价
2. Mode B：tail truncation 语义，尝试直接减少请求尾部 KV footprint

从架构角度，两者差异在于：

1. Mode A 更接近“调度策略替换”，对底层 KV 结构侵入更小
2. Mode B 更接近“内核级执行语义扩展”，需要更强的一致性维护
3. Mode A 的研究意义是先证明 selection 机制有效
4. Mode B 的研究意义是进一步逼近 token-level 真实释放

因此双轨并不是重复实现，而是研究路径分层：先证明调度价值，再推进执行精度。

## 11. 设计模式识别

该项目至少稳定使用了以下设计模式：

1. Strategy Pattern：ScoringStrategy、BaselineStrategy
2. Registry Pattern：BaselineRegistry
3. Adapter Pattern：FrameworkAdapter、VLLMAdapter、SGLangAdapter
4. Template Method 风格：BaseExperimentRunner
5. Snapshot Pattern：BidPool
6. Facade 风格：顶层 __init__.py 对外导出统一 API
7. Plugin / Hook Pattern：vLLM plugin.py、scheduler hook、SGLang radix hook

需要强调的是，这些模式不是为了“面向对象好看”，而是直接服务于实验可控性和跨框架移植性。

## 12. 零外部依赖约束对架构的影响

pyproject.toml 明确声明 dependencies = []。

这条约束对架构产生了深远影响：

1. 核心逻辑必须只建立在 Python stdlib 和自有协议之上
2. 所有框架依赖必须延迟到 adapter 内 runtime import
3. scoring、solver、pool、pressure 这些核心模块不能假设 numpy/torch 存在
4. 类型与数据结构设计必须尽量轻量

这种约束的优点：

1. 核心包易测试
2. 安装和移植简单
3. 跨框架耦合降低

代价：

1. 某些高性能数值处理不能直接借助外部库
2. adapter 层要承担更多“与重框架接触”的复杂性

这是一个很典型的“把复杂性推迟到边缘层”的架构选择。

## 13. 可测试性、可维护性、可扩展性评估

### 13.1 可测试性

优点：

1. protocol、scoring、baseline、adapter 都有独立测试入口
2. 核心模块低依赖，适合单元测试
3. Protocol 与 ABC 的接口清晰，便于 fake object 测试
4. 实验配置和值对象天然适合验证

限制：

1. 真实 adapter 行为仍依赖框架内部对象结构，集成测试成本高
2. hook 路径比纯函数逻辑更难完全覆盖

总体评价：可测试性较强，尤其是核心算法层。

### 13.2 可维护性

优点：

1. 模块边界清晰
2. 每个目录承担相对单一职责
3. 配置冻结和文档约束使演化方向可控

主要维护风险：

1. adapter 对框架内部 API 演化敏感
2. 实验系统与论文语义高度绑定，重构需要非常谨慎
3. 双平台、双模式会增加实现分叉压力

总体评价：中高，可维护性建立在“严格遵守边界与冻结规则”之上。

### 13.3 可扩展性

新增 baseline：较容易

1. 实现 BaselineStrategy
2. 注册到 BaselineRegistry
3. 加入实验配置与测试

新增 scoring：容易

1. 实现 score-only 接口
2. 接入 build_bids 链路
3. 补充实验和对比

新增 framework adapter：中等偏难

1. 实现 FrameworkAdapter 5 个职责点
2. 找到框架的 pressure interception 边界
3. 找到真实 KV release 或 fallback 语义
4. 建立 request/token 生命周期跟踪

总体评价：算法扩展容易，框架扩展昂贵但路径清晰。

## 14. 面向研究与实现的协同机制

BidKV 的一个突出优点是，它把研究活动直接编码进了软件结构中。

这种协同主要体现在：

1. 论文概念有直接代码对象映射
2. baseline 不是论文附录，而是正式子系统
3. freeze rules 通过配置和文档进入工程流程
4. result、paper、scripts、experiments 与 src 同仓管理
5. metrics schema 同时服务机制验证和论文汇报

这种结构非常适合研究型系统项目，因为它避免了常见问题：

1. 论文中的术语和代码对象脱节
2. baseline 只能靠临时脚本复现
3. 实验配置散落在 shell 命令中
4. 框架适配逻辑和算法逻辑互相污染

从这个角度看，BidKV 已经具备“可发表研究系统仓库”的组织特征。

## 15. 潜在限制与风险点

以下是当前架构中最值得关注的限制点：

1. vLLM adapter 的稳定性高度依赖内部实现细节，版本升级风险大。
2. Mode A 与 Mode B 的叙事、执行、指标解释必须严格区分，否则容易出现研究语义混淆。
3. quality_delta 是 surrogate signal，若评分代理失真，solver 的 utility 排序会被系统性偏置。
4. 每 request 最多选一个 bid 的约束虽然简化实现，但会限制更细粒度组合优化。
5. SGLang 的共享前缀保护逻辑正确性非常关键，一旦误判，可能影响缓存一致性或复用效率。
6. experiments 与 results 的冻结制度虽然保障论文严谨性，但也提高了后期重构成本。

这些风险并不意味着架构有问题，而是说明该架构已进入“真实研究系统”而非“玩具原型”的复杂度阶段。

## 16. 不违反冻结约束的改进建议

以下建议只涉及工程质量提升，不改变冻结实验方向、策略列表或 figure 语义。

### 16.1 为 adapter 增加更强的不变量检查

建议在 vLLM 和 SGLang adapter 中进一步集中化以下断言：

1. request token 数与框架内部状态的一致性
2. compression 前后 request 生命周期合法性
3. actual_freed 与内部池状态变化的一致性

这会显著降低 hook 类 bug 的排查成本。

### 16.2 把实验路由与运行时路由的边界再显式化

目前 experiment_strategy_name、registry、plugin 环境变量路由已经存在。可以进一步把“研究实验路由”和“生产接入路由”做更明确的对象化封装，减少 adapter 初始化参数的语义负担。

### 16.3 强化 snapshot 审计

可以为每次 pressure event 记录标准化的审计对象：

1. 当前候选池摘要
2. detector stats
3. acceptance 原因
4. execution 结果

这样对复现实验和诊断异常 run 都更有帮助。

### 16.4 为新增 adapter 准备模板层

可以抽出一个更接近脚手架的 adapter mixin 或 checklist，专门服务未来接入新框架，例如 TensorRT-LLM。这样能减少重复踩坑，但不会改变现有抽象。

## 17. 综合评价

从系统架构角度看，BidKV 的最大优点不是“代码写得多复杂”，而是它在研究系统中做对了三件最难的事：

1. 把论文问题建模成稳定、可复用的协议对象
2. 把框架差异隔离在 adapter，而不是污染核心算法
3. 把实验治理、冻结配置、baseline 归因和结果输出纳入同一个工程体系

如果把它放在“研究原型”到“系统论文实现”这一光谱上，BidKV 明显已经位于偏后段：

1. 核心机制抽象完整
2. 跨框架路径真实存在
3. baseline 体系系统化
4. 实验与论文产物联动紧密

它当前最需要持续投入的，不是再发明新的核心抽象，而是继续提升 adapter 边界的稳健性、执行路径的一致性验证，以及实验审计链路的可诊断性。

## 18. 一句话判断

BidKV 不是“给某个框架加一个压缩技巧”，而是把 KV 压缩调度上升为一个跨框架、可比较、可复现实验的系统原语；其代码结构已经很好地服务了这个目标。
