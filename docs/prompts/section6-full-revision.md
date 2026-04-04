# §6 Evaluation 章节大改指令

> 本提示词面向资深系统会议论文写手，负责 §6 Evaluation 及 §8 Conclusion 的全面修改。
> 生成日期：2026-04-04
> 数据来源：`results/vllm_v8_full_validation/`（mixed 63 runs）、`results/vllm_v8_long_context/`（LC 36+ runs）
> 数据状态：v8-frozen，所有数值从实验 JSON 原始数据计算，mean of 3 runs

---

## 第一部分：强制修改（实验设定 & 数据客观事实）

以下内容涉及实验配置和实测数据的客观准确性，**必须严格按照修改**，不可偏离。

### 1.1 策略列表：7 → 5

当前论文表格中有 7 个策略，正文 Baselines 段列了 5 个但包含 PE-SJF 而缺少 Slack-Aware。**统一修改为以下 5 个策略**：

| # | 代码名 | 论文显示名 | 保留理由 |
|---|--------|-----------|---------|
| 1 | preempt-evict | PE (default) | vLLM 原生 baseline |
| 2 | static-random | Static-Random | 随机 victim selection |
| 3 | largest-first | Largest-First | 容量贪心 victim selection |
| 4 | slack-aware | Slack-Aware | SLO-deadline aware |
| 5 | bidkv | BidKV | 完整系统 |

**移除 PE-SJF 和 Uniform**。所有表格、正文、图中涉及这两个策略的行/描述全部删除。

### 1.2 指标体系：5 列 → 4 列

当前表格使用 5 列指标（Goodput 500ms, SLO 300ms, TTFT P95, TPOT P95, Norm Lat）。**修改为 4 列**：

| 列 | 指标名 | 单位 | 方向 |
|---|--------|------|------|
| 1 | Throughput | req/s | ↑ |
| 2 | SLO attainment (300ms) | % | ↑ |
| 3 | TTFT P95 | ms | ↓ |
| 4 | TPOT P95 | ms | ↓ |

**移除 Goodput(500ms) 和 Normalized Latency**。正文 Metrics 段中对应描述也需同步删除。

### 1.3 表格数据（必须使用以下精确数值）

#### Table 1 — `paper/tables/table1_main.tex`

Mixed workload, rate=3.8 req/s, mean of 3 runs。Caption 中更新列名和策略数。

| Strategy | Throughput | SLO (300ms) | TTFT P95 | TPOT P95 |
|:---|---:|---:|---:|---:|
| PE (default) | 3.48 | 68.4 | 6769 | 97.0 |
| Static-Random | **3.57** | <u>84.1</u> | 1216 | **90.9** |
| Largest-First | 3.28 | 79.9 | <u>677</u> | 112.9 |
| Slack-Aware | <u>3.47</u> | 68.2 | 5471 | <u>93.9</u> |
| **BidKV** | 3.42 | **83.2** | **631** | 107.2 |

#### Table 2 — `paper/tables/table2_rate_full.tex`

Mixed workload, 3 rates, mean of 3 runs。

**Rate = 2.0：**

| Strategy | Throughput | SLO | TTFT P95 | TPOT P95 |
|:---|---:|---:|---:|---:|
| PE (default) | 1.96 | 92.0 | 683 | 78.1 |
| Static-Random | 1.96 | <u>96.5</u> | <u>266</u> | <u>72.8</u> |
| Largest-First | 1.96 | 95.5 | 291 | 74.5 |
| Slack-Aware | 1.96 | 91.8 | 963 | 81.5 |
| **BidKV** | 1.96 | **97.0** | **258** | **72.4** |

**Rate = 3.8：** （同 Table 1）

**Rate = 5.7：**

| Strategy | Throughput | SLO | TTFT P95 | TPOT P95 |
|:---|---:|---:|---:|---:|
| PE (default) | 3.49 | 56.2 | 8699 | 119.9 |
| Static-Random | **4.09** | 80.4 | 1792 | **94.4** |
| Largest-First | 3.53 | 77.7 | <u>793</u> | 113.2 |
| Slack-Aware | <u>3.72</u> | 57.2 | 5867 | <u>104.7</u> |
| **BidKV** | 3.60 | **81.0** | **797** | 110.0 |

#### Table 3 — `paper/tables/table3_long_context.tex`

Long-context workload, cross-rate average (rates 0.35/0.5/0.7), SLO threshold = **2000ms**（非 300ms）。**Slack-Aware 数据缺失（0 runs），仅 4 行**。需在 caption 或脚注说明。删除红色 placeholder 文字。

| Strategy | Throughput | SLO (2000ms) | TTFT P95 | TPOT P95 |
|:---|---:|---:|---:|---:|
| PE (default) | 0.374 | 21.2 | 80135 | 181.4 |
| Static-Random | <u>0.447</u> | <u>75.0</u> | **14569** | <u>162.5</u> |
| Largest-First | 0.435 | **76.6** | 16262 | 211.3 |
| **BidKV** | **0.443** | 75.6 | <u>15492</u> | 179.6 |

### 1.4 正文中引用的具体数字

以下正文中硬编码的数据点**必须更新为真实值**（如果当前值与下方不同）：

- rate=3.8 BidKV TTFT P95 = 631ms
- rate=3.8 BidKV SLO = 83.2%
- rate=3.8 Static-Random TTFT = 1216ms, SLO = 84.1%, Throughput = 3.57
- rate=3.8 Largest-First TTFT = 677ms, SLO = 79.9%
- rate=3.8 PE SLO = 68.4%
- rate=3.8 Slack-Aware SLO = 68.2%
- rate=2.0 BidKV SLO = 97.0%, TTFT = 258ms, TPOT = 72.4ms
- rate=5.7 BidKV SLO = 81.0%, TTFT = 797ms
- rate=5.7 Static-Random Throughput = 4.09
- cross-rate BidKV SLO = 87.1%, TTFT P95 = 562ms
- cross-rate Static-Random Throughput = 3.21, BidKV Throughput = 2.99（差 ~7%）

所有引用 PE-SJF 或 Uniform 数字的句子必须删除或改写为引用现有 5 策略的数据。

### 1.5 Baselines 段的策略机制描述（客观事实部分）

以下机制描述是代码实现的客观事实，**必须准确**：

- **PE**：FCFS admission, LIFO preemption, 无 proactive reclamation
- **Static-Random**：SJF admission, proactive reclamation (KV>90%), random victim, SRPT (KV>80%)
- **Largest-First**：SJF admission, proactive reclamation (KV>90%), capacity-greedy victim（驱逐占用最多 KV blocks 的请求）, SRPT (KV>80%)
- **Slack-Aware**：EDF admission（按到达序, cached priority）, proactive reclamation (KV>90%), SLO-deadline aware victim, SRPT (KV>80%)
- **BidKV**：SJF admission, 95% KV 压力门控 running reorder, U = r/(δ+ε) utility-ratio victim selection, **显式禁用 SRPT 和 proactive preempt**

### 1.6 Conclusion 段同步修改

- "five-strategy evaluation" 保留，但策略组合修改为：PE, Static-Random, Largest-First, Slack-Aware, BidKV
- 引用的核心数字更新为：cross-rate SLO 87.1%, TTFT P95 562ms

### 1.7 文件清单

必须修改的文件：
1. `paper/bidkv_sc2026.tex`：§6 (~L628-L860) + §8 Conclusion
2. `paper/tables/table1_main.tex`
3. `paper/tables/table2_rate_full.tex`
4. `paper/tables/table3_long_context.tex`

---

## 第二部分：建议修改（叙事、解读、写法）

以下内容属于论文叙事和主观解读。作为资深系统会议论文写手，请根据自身判断选择性采纳。这些是基于数据特征给出的写作方向建议，不做强制要求。

### 2.1 Metrics 段的解释文字

**建议**：新增 Throughput 的解释（标准 LLM serving 指标，vLLM/Orca/SGLang 均使用），简化 SLO 定义段（删掉提及 Goodput 500ms 的冗余句子）。TPOT P95 可引用 Sarathi-Serve (OSDI'24)。P95 vs P99 的理由段写得不错，建议保留。

### 2.2 §6.2 Key Findings 的叙事结构

当前写法合理，但基于 5 策略数据，**建议考虑以下叙事要点**：

- BidKV 在 rate=3.8 的核心优势是 **TTFT P95 #1**（631ms），SLO 是 #2（83.2%，比 Static-Random 低 0.9pp）。可以强调 BidKV 在 TTFT 这个最直接的用户体验指标上排名第一
- Slack-Aware 是一个值得点评的对比：尽管有 SLO-deadline 信号 + proactive preempt，其 SLO (68.2%) 与无 proactive 的 PE (68.4%) 几乎相同，说明错误的 victim selection 信号抵消了 proactive 的收益
- Static-Random 的 throughput 优势来自 SRPT aggressive preemption，代价是 TTFT 翻倍。这个 tradeoff 对 latency-sensitive 场景不利

### 2.3 Attribution Chain

由于移除了 PE-SJF，原有的四步归因链断裂。**建议**调整为：

- PE → Static-Random：加入 SJF + proactive + random victim → SLO +15.7pp，TTFT 6769→1216ms
- Static-Random → Largest-First：random → capacity-greedy victim → SLO -4.2pp 但 TTFT 1216→677ms（更精准的 victim 降低了尾部延迟，虽然 SLO 因 victim 选择过于保守略有下降）
- Largest-First → BidKV：capacity-greedy → utility-ratio → SLO +3.3pp, TTFT 677→631ms（coordinated cost-aware selection 的增量价值）

但如果你认为有更好的归因结构，请自行调整。

### 2.4 Rate Sensitivity 三 Regime 叙事

**建议**的三段组织：

- **Low (2.0)**：BidKV 全 4 指标 #1，但差距小。可一笔带过
- **Medium (3.8)**：主要分化区间，核心段落。BidKV TTFT #1 + SLO #2；SRPT tradeoff（Static-Random throughput 高但 TTFT 1.9×）；Slack-Aware 失效现象
- **High (5.7)**：BidKV SLO #1 (81.0%)，TTFT #1 (797ms)。Static-Random throughput 最高 (4.09, +13.6%) 但 TTFT 2.2×。高压下 BidKV 的 SLO 优势扩大

cross-rate 段：BidKV SLO #1 (87.1%, +0.1pp over Static-Random), TTFT #1 (562ms, 48% below Static-Random 1091ms)。Throughput #3 (2.99 vs 3.21, -6.8%)。**建议**明确点出"~7% throughput 代价换取显著更好的 admission responsiveness"这个 tradeoff。

### 2.5 Long-Context 讨论

Table 3 是新增内容。**建议**简短讨论（半段到一段）：
- PE 在 LC 下严重退化（SLO 21.2%，TTFT 80s）
- BidKV 保持竞争力（Throughput #1, SLO #3, TTFT #2），但不再全面领先
- Largest-First 的简单"大优先释放"在极端 KV 压力下有其优势（SLO #1, 76.6%）
- **建议**不过度渲染 LC 结果，因为 Slack-Aware 数据缺失使对比不完整

### 2.6 Reclamation Event Analysis

**建议**调整为 5 策略的驱逐行为描述。可以考虑新增 Slack-Aware 的观察：其 mixed rate=3.8 下 proactive evictions=27, freed=16k tokens，但 SLO 仅 68.2%——频率低且释放量不足，说明 deadline-aware victim 信号在选择释放效率高的请求上的局限性。

### 2.7 论文叙事的总体方针

**建议**（非强制）：

- 核心定位：BidKV 是 **admission-responsiveness-first** 策略，优化目标是 SLO + TTFT
- 必须显式声明 throughput/TPOT 非 #1 的 tradeoff，不回避
- Slack-Aware 的"失效"可以作为论文的一个有价值的 negative result：纯 SLO-deadline 信号不等于好的 victim selection
- LC 结果作为 completeness，不作为核心 claim

---

## 不可修改的约束（贯穿全文）

- ❌ 不修改 §1-§5 的内容方向
- ❌ 不修改 utility ratio 公式 U = r/(δ+ε) 或 δ = 1 + 0.5c + 0.3P
- ❌ 不声称 BidKV "全面领先"
- ❌ 不隐瞒 throughput/TPOT 的非 #1 表现
- ❌ 不隐瞒 Slack-Aware LC 数据缺失
- ✅ Best **bold**, second-best underline
- ✅ Mean of 3 runs
- ✅ 保持 `\input{tables/...}` 引用路径不变
