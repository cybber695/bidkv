# BidKV 实验思路总结

## 一、核心问题

**KV cache 空间不足时，应该驱逐哪个请求？**

vLLM 默认按 LIFO（后进先出）驱逐，不感知请求的质量偏好。BidKV 提出用 utility ratio 来指导决策：

$$U = \frac{r}{\delta + \varepsilon}$$

驱逐 $U$ 最高的请求 = 每单位质量损失能释放最多 KV 空间的请求。

---

## 二、七策略归因链

```
preempt-evict  → 零信息基线（纯 LIFO）
   ↑ 加入随机压缩
static-random  → 随机选择基线
   ↑ 加入均等压缩
uniform        → 所有请求均等压缩
   ↑ 加入 token-level scoring
h2o-style      → 按 attention 找低重要度 token 压缩
   ↑ 加入 SLO deadline 感知
slack-aware    → 按剩余时间松弛度决策
   ↑ 加入 utility 量化 + 贪心 solver
global-nobid   → 系统自动推断 utility，无 bid 协议层
   ↑ 加入显式 bid 接口
bidkv          → 完整 bid pipeline（本文方法）
```

每一步差距 = 对应信息/机制的增量价值。

---

## 三、执行链（所有策略统一）

```
KV 压力触发（scheduler_hook.py monkey-patch）
    ↓
strategy.select_victims(candidates, needed_tokens)
    ↓  ← 7 种策略的唯一差异点
list[CompressionAction]
    ↓
_execute_baseline_actions()  ← 忽略 action_type，统一执行
    ↓
execute_compression(request_id, target_tokens)
    ↓
_execute_tail_truncation() → vLLM 原生 preempt + recompute
```

`action_type`（`"evict"` / `"compress"`）字段只是语义标签，底层执行对所有策略完全相同，保证实验的单一变量原则。

---

## 四、实验矩阵

| 维度 | vLLM（主平台） | SGLang（可移植性验证） |
|------|----------------|----------------------|
| 策略数 | 7 | 3（preempt-evict / slack-aware / bidkv） |
| Workload | 2（mixed / long_context） | 2 |
| Rate | 3（各自校准冻结） | 3 |
| Repeat | 3 | 3 |
| **总 runs** | **126** | **54** |

冻结 rates：`mixed=(2.0, 3.8, 5.7) req/s`，`long_context=(0.35, 0.5, 0.7) req/s`，seed=42 frozen trace。

---

## 五、公平性保证

| 机制 | 实现 |
|------|------|
| 相同输入 | 所有策略共用同一冻结 trace（SHA-256 验证） |
| 相同候选池 | 同一压力事件下，所有策略看到相同的 `candidates` 快照 |
| 相同执行 | `_execute_baseline_actions()` 对所有策略走同一底层代码 |
| 单一变量 | 策略是唯一变量，rate/trace/模型/硬件全部冻结 |

---

## 六、当前进度

| 阶段 | 状态 |
|------|------|
| Phase 0-1 方案冻结 | ✅ v2.3-frozen |
| Phase 2-3 Pilot + Trace 冻结 | ✅ #055 完成 |
| SGLang smoke test | ✅ #052 完成 |
| **全量 126+54 runs** | ⏳ 待执行（最紧迫） |
| Phase 7 论文图表生成 | ⏳ 依赖全量数据 |

---

## 七、补充问题解答

### Q1：相较之前版本，本次代码主要变化是什么？

本次只做了一处修改：在 `adapters/vllm/adapter.py` 和 `adapters/sglang/adapter.py` 的
`_execute_baseline_actions()` 方法中，**添加了明确注释**，说明 `action_type` 字段被有意忽略的原因
（单一变量实验设计）。代码逻辑本身没有任何改变。

---

### Q2：Global-NoBid 和 BidKV 除了 bid 接口外是否没有区别？

**是的，在当前实验中两者几乎等价。**

| 步骤 | Global-NoBid | BidKV |
|------|-------------|-------|
| scoring | `H2OScoring.score()` | `H2OScoring.score()` |
| delta 计算 | `_estimate_delta()`（自己实现） | `build_bids()`（`bid_builder.py` 统一实现） |
| utility 公式 | $U = r/(\delta + \varepsilon)$（手写） | $U = r/(\delta + \varepsilon)$（`CompressionBid.utility`） |
| 贪心约束 | 手写（约束 A + B） | `GreedyBidSolver`（相同约束） |
| bid 协议层 | **无** | `CompressionBid` 对象 + `BidPoolManager` |

两者输入相同、算法相同，差距只在协议抽象层，**数值结果在实验中会非常接近**。

---

### Q3：BidKV 用的是 H2O 而非用户主动提供 bid？

**正确。** 当前实验中 bid 完全由系统自动生成：

```
H2OScoring.score(token_ids)   ← 用 attention 权重估算 token 重要度
    ↓
build_bids()                  ← 按 compression_levels 生成 CompressionBid 对象
    ↓
BidPoolManager.submit_bids()  ← 汇入 bid 池
```

`CompressionBid` 的 `quality_delta` 字段由"被移除 token 的平均重要度分数"自动计算，不是用户填写的。
这是实验阶段的 **proxy bid** 设计——用 H2O scoring 代理应用层偏好。

论文的架构贡献是 bid **接口**本身（应用层可以注入真实偏好），而非实验中的 bid 信号来源。
在真实部署中，不同业务（实时对话 vs 后台任务）可以提交差异化的 `quality_delta`，使 BidKV 真正区别于 Global-NoBid。

---

### Q4：选出 actions 后是驱逐还是压缩？

**在当前 Mode A 下，底层统一调用 `_execute_tail_truncation()`，但效果因 `target_tokens` 不同而分化：**

| 策略 | `target_tokens` 的值 | 实际效果 |
|------|---------------------|---------|
| preempt-evict | `req.current_tokens`（全量） | 释放整个请求的全部 KV → 等价于完整驱逐 |
| 其余 6 种 | `current_tokens × ratio`（部分） | 只截断尾部若干 blocks，请求继续运行 |

`action_type` 字段（`"evict"` / `"compress"`）在执行层被有意忽略，差异完全由 `target_tokens` 的大小体现，
确保底层执行路径统一，不引入混淆变量。
