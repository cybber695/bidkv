# Decision Prompt: Introduction Preliminary Results Figure

## 任务

请分析以下问题，并给出具体的设计方案建议（包括图的内容、数据、caption 草稿）。

---

## 背景：论文定位

**论文题目**：BidKV: A Bid-Based Scheduling Abstraction for Active KV-Cache Reclamation（SC 2026 投稿）

**核心问题**：GPU 内存有限，在线 LLM 服务中当 KV cache 占用超过阈值时，serving engine 必须从正在运行的请求中"回收"（reclaim）KV 状态。关键问题是**选谁做受害者（victim selection）**。vLLM 默认用 LIFO（最后进入 running 的先被驱逐），不考虑每个请求的回收代价差异。

**BidKV 的方案**：每个请求提交一个 bid（包含可回收 KV tokens 数 + 估算的 disruption cost），调度器按 utility = reclaimable_tokens / (disruption_estimate + ε) 排序，选 utility 最高的受害者，通过 vLLM/SGLang 原生 preempt+recompute 机制执行。

**实验设置**：
- 硬件：NVIDIA RTX A6000 48GB，Llama-3.1-8B-Instruct (bf16)
- KV budget：600 blocks × 16 tokens = 9,600 token slots（故意压缩以制造 KV 压力）
- workload：ShareGPT 混合长度，Poisson 到达，1000 requests/run，3 rates: {2.0, 3.8, 5.7} req/s
- 5 个策略对比：PE（vLLM 默认 LIFO）、PE-SJF（+SJF 准入排序）、Static-Random、Largest-First、BidKV

**主要实验结论（rate=3.8，cross-rate average）**：

| 策略 | Throughput (req/s) | SLO(300ms) % | TTFT P95 (ms) | TPOT P95 (ms) |
|------|--------------------|-------------|----------------|---------------|
| PE（LIFO）    | 2.98 | 72.2 | 6,769 | 97.0  |
| PE-SJF        | 3.06 | 79.6 | 666   | 153.5 |
| Static-Random | 3.57 | 84.1 | 1,216 | 90.9  |
| Largest-First | 3.51 | 79.9 | 677   | 99.4  |
| **BidKV**     | 3.42 | 83.2 | **631** | 107.2 |

BidKV cross-rate average vs. PE default: SLO +14.1 pp, TTFT P95 −90% (5,241→544 ms)，代价是 throughput −7.1%。

---

## 现有论文结构

```
§1 Introduction（当前无图）
  ¶1: LLM serving KV pressure context
  ¶2: victim selection problem + "Because reclamation costs are heterogeneous..."
  ¶3: existing systems 不足（LIFO/FCFS 无 cost signal）
  ¶4: BidKV 方案介绍
  ¶5: contributions

§2 Background and Motivation
  §2.1 KV Cache Memory Model
  §2.2 The Victim-Selection Problem
       → 数学形式化（min-cost covering problem）
       → Empirical evidence 段落（当前有图 fig:motivation）

§6 Evaluation
  §6.2 Main Comparison（rate=3.8结果，文字归因链）
  §6.3 Rate Sensitivity（Table 2: 7×3×4矩阵）
  §6.4 Long-context workload（Table 3）
  §6.5 Reclamation Event Analysis（fig:preempt_analysis）
```

---

## 现有 §2 的 Empirical Evidence 图（fig:motivation）

当前在 §2.2 中放了一张图，数据完全来自已有实验（rate=3.8，250个 KV 压力事件）：

**内容**：CDF of per-event reclamation-opportunity spread = max(R) / min(R)，
其中 $R_i = B_i \cdot (1 - p_i)$，$B_i$ = 当前持有 KV tokens，$p_i = g_i/G_i$ = 事后完成率（$G_i$ 为实际最终输出长度）。

**结论**：
- 99.6% 的 250 个压力事件中，最高/最低 opportunity 比值 ≥ 10×
- 中位数 128×，94.8% 超过 20×
- 结论：real workload 下 victim 异质性极强，仅凭到达顺序无法区分好坏受害者

**注意**：$p_i = g_i/G_i$ 使用 post-hoc 完成率（推断时不知道 $G_i$），因此这是一个 analysis-only 的 proxy，**方法无关**，不引用任何调度公式。

---

## 审稿人质疑

> "Because reclamation costs ...那一段写的很好，但是少了实验证据佐证"，要求在 **Introduction 右上角放个 preliminary results 作证现有工作的不足**。

---

## 已排除的方案及原因

### 方案 X1：在 §1 末尾加文字引用 §6.5
- **为什么不够**：审稿人明确要求 Introduction 放图，加文字引用不满足位置要求。

### 方案 X2：Panel (b) = PE vs BidKV TTFT CDF（率=3.8）
- **为什么有问题**：§6 Table 2 + §6.2 归因链已经包含完全相同的信息（BidKV 631ms vs PE 6769ms）。在 §1 提前展示 BidKV 的结果会导致 §6 重复，且在 Introduction 放出 solution 的结果会破坏叙事顺序（"问题→方案→证明"应分开）。

### 方案 X3：Panel (b) = LIFO 与"最优受害者"的对比
- **为什么不行**：$R$ 的定义中"最优受害者"= argmax $R_i$ 依赖 $G_i$（实际最终输出长度），这是事后 oracle 信息。任何在线策略都会输给事后 oracle，这是不公平对比，reviewer 会质疑。

---

## 核心决策问题

**已确认的约束**：

1. **位置**：图必须在 §1（Introduction），承接"Because reclamation costs are heterogeneous..."段落
2. **内容**：证明"现有工作的不足"（existing policies fail），而非只证明"问题存在"
3. **无新实验**：所有数据必须来自已有结果（`results/vllm_v8_full_validation/`，rate=3.8，triple runs 已冻结）
4. **无 oracle**：不使用运行时不可知的信息（如 $G_i$）作为"最优"基准
5. **最小与 §6 重叠**：§6 Table 2 已有完整 7 策略 × 3 rates 对比，§1 图不应重复说同一件事

**当前倾向方案（需要验证）**：

双 panel 图放 §1 末尾（fig:intro_evidence），将现有 §2 的单图升级为：

- **Panel (a)**：保留现有 $\max R / \min R$ CDF → 证明"问题结构存在" (heterogeneity is pervasive)
- **Panel (b)**：PE (LIFO 默认) 在 rate=3.8 的 TTFT **CDF**，只画 PE 一条曲线，标注 P95 竖线和 SLO(300ms) 竖线，不展示 BidKV -> 直接证明"现有工作的不足"（6,769ms P95，68.4% SLO 达标率），不涉及 BidKV solution

§2.2 的 "Empirical evidence" 段落：删除当前图，改为 "Figure~\ref{fig:intro_evidence} (shown in Section~\ref{sec:intro}) establishes this heterogeneity..."

---

## 请分析以下问题

1. **Panel (b) 是否会与 §6 形成不可接受的重复？** 考虑 §1 的 Panel (b) 只展示 PE 一条曲线（不含其他基线和 BidKV），而 §6 Table 2 是 5 策略 × 3 rates 的完整矩阵。两者的叙事功能是否足够区分（"问题示例"vs "系统性评估"）？

2. **Panel (b) 展示 PE 的 TTFT CDF 是否足够有说服力？** PE 的 P95=6,769ms 是一个"崩溃"级别的数字，但审稿人可能认为这是"问题的 consequence"而非"现有工作的不足"——因为 PE 已经是 vLLM 的 default，审稿人可能知道 LIFO 在极端压力下表现差并不意外。是否需要加入一条"理论下界"或"简单改进"的曲线来形成对比？

3. **是否有更好的 Panel (b) 方案**，既能证明"existing scheduling policies fail"，又不需要 oracle 信息，不提前泄露 BidKV 结果，不与 §6 重复？例如：
   - 展示"不同完成率请求被 LIFO 选中的频率分布"（证明 LIFO 偏向高完成率受害者）
   - 展示"LIFO 每次驱逐事件中选中的受害者 $B_i$ 和 $p_i$（已生成 tokens / 当时 live tokens）的散点图"
   - 其他

4. **图的位置**：审稿人说"Introduction 右上角"，在双栏 ACM 格式下这意味着 `\begin{figure}[t]` floats 到第一页右栏顶部。双 panel 图在列宽（8.5cm）内能否清晰显示？是否应该考虑 `\begin{figure*}` 跨栏（会放在第 2 页顶部，位置不理想）？

5. **§2.2 的 Empirical Evidence 段落要如何处理**？如果图移到 §1，§2.2 现有的数学推导（$R_i = B_i(1-p_i)$，CDF 解读，250 个事件的统计数字）是继续保留还是删减？保留数字但移除图，§2.2 是否显得"图文不符"？

6. **整体叙事连贯性**：如果 §1 放 preliminary results，§2.2 仍有 empirical evidence 段落，§6.5 有 reclamation event analysis，三者在叙事上如何形成递进而非重复？请给出每个位置的叙事定位建议。

---

## 附：相关代码/数据路径（供 Gemini 了解实施可行性）

- 实验结果：`results/vllm_v8_full_validation/preempt-evict__mixed__rate3.8__r{0,1,2}.json`
- 每个 JSON 中 `request_results` 列表，每项含 `ttft_ms`（float，毫秒）、`error`（空字符串=成功）
- 现有 motivation 图生成脚本：`scripts/generate_paper_figures.py`
- 现有 motivation 图文件：`paper/figures/fig1_motivation_1panel.pdf`
- 论文主文件：`paper/bidkv_sc2026.tex`

---

## 期望输出

请给出：
1. 问题 1-6 的分析
2. 你认为最优的 Panel (b) 方案（含具体展示内容、x/y 轴定义、标注要点）
3. Panel (b) 的 caption 草稿（英文，50-80 字）
4. §2.2 Empirical Evidence 段落的建议处理方式
5. 如果双 panel 方案有重大缺陷，请提出替代方案
