# §6.5 Reclamation Event Analysis 修订指令

> 生成日期：2026-04-06
> 面向：论文写作 Agent
> 任务范围：仅修订 §6.5 正文（`\subsection{Reclamation Event Analysis}` 到下一个 `\subsection` 之前）
> 图和 caption 已由代码自动更新，**不需要改动**

---

## 背景：数据来源

### 原始数据

- 目录：`results/vllm_fig3_mixed_rate38/`
- 策略 × 运行：5 strategies × 3 runs = 15 runs（全部有效，1000/1000 成功）
- 工作负载：mixed，rate=3.8 req/s
- 聚合结果：`results/vllm_fig3_mixed_rate38/analysis/summary.json`

### 图表

- 论文引用图：`paper/figures/fig5_compress_coverage.pdf`（已更新）
- 图的结构：**单图双 Y 轴，每策略两柱**
  - 左柱（实心，策略配色）= 总回收次数（含 native LIFO + proactive，左 Y 轴）
  - 右柱（斜线纹理，半透明，同色）= 总释放 token 数（×1000，右 Y 轴）
- 新图与旧图的关键区别：旧图仅统计 proactive eviction（PE/PE-SJF 均为 0），新图统计所有回收路径（PE/PE-SJF 均有非零值）

---

## 关键数据（3 runs 均值，mixed rate=3.8）

| Strategy | Reclamation Count | Tokens Freed | Freed / Event |
|---|---:|---:|---:|
| PE | 247 | 196k | 794 |
| PE-SJF | 336 | 366k | 1,087 |
| Static-Random | 329 | 88k | 269 |
| Largest-First | 296 | 325k | 1,100 |
| BidKV | 323 | 174k | 539 |

**Freed/Event = Tokens Freed ÷ Reclamation Count**，衡量每次回收的平均释放量，是本节核心叙事指标。

---

## 修订重点

### 1. PE 和 PE-SJF 描述——必须纠正

旧文本称两者"triggers zero proactive evictions"。**新数据显示两者均有非零 reclamation count**（PE=247，PE-SJF=336），这些来自 vLLM native LIFO `_preempt_request` 路径——即 vLLM 自身在 KV 压力时触发的回收，之前未被计入统计。需根据新数据重新描述其回收行为。

### 2. 补充 Static-Random 的 bullet——旧文本缺失

旧文本的 itemize 未包含 Static-Random。需补充：其回收频率（329）与 BidKV 相当，但每次仅释放 88k tokens（Freed/Event=269），是 5 策略中最低。说明随机选 victim 经常选到小请求，需要更多次回收才能缓解同等 KV 压力。

### 3. Freed/Event 是核心叙事——需要在每个 bullet 中体现

各策略 Freed/Event 的对比揭示了 victim selection 机制的本质差异：

- **PE & PE-SJF**：LIFO 机制选择最近进入 running 的请求，通常是 decode 阶段刚开始的短完成度请求，释放量中等（PE=794，PE-SJF=1,087）
- **Static-Random**：随机选，命中短请求概率高，Freed/Event 最低（269）
- **Largest-First**：选 KV 占用最大的请求，Freed/Event 最高（1,100），但这些往往是接近完成的长请求，recompute 代价最高
- **BidKV**：$\delta = 1 + 0.5c + 0.3P$（$c$=completion ratio，$P$=prior preemptions）对高完成度请求施加惩罚，Freed/Event 中等（539），在释放量与 recompute 代价之间取得平衡

### 4. 与主指标（TTFT P95）的因果链——需要显式构建

叙事链：**victim selection → Freed/Event → recompute overhead → prefill bandwidth → TTFT**

- Largest-First 的高 Freed/Event 看似高效，但被选中的近完成请求 recompute 代价高，占用 prefill 带宽，升高 TTFT
- BidKV 的中等 Freed/Event 避免了此类代价高昂的 recompute，prefill 带宽保留给 waiting queue，TTFT P95 在 5 策略中最低

---

## 保持不变的内容

- `\begin{figure}` … `\end{figure}` 环境（图 + caption + label）
- `\subsection{Framework Portability}` 及之后所有内容
- 论文术语规范：reclamation（不用 eviction/compression），disruption estimate $\delta$，$U = r/(\delta + \varepsilon)$

---

## 定位修改范围

```
% 从此行开始修改：
\subsection{Reclamation Event Analysis}\label{sec:eval:mechanism}

% 到此行停止（不含此行及之后）：
\subsection{Framework Portability}\label{sec:eval:sglang}
```
