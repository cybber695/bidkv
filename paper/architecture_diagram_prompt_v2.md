# BidKV Architecture Diagram Prompt

## 总体要求

为 SC 2026 论文绘制 BidKV 系统架构图（单栏宽度，约 3.3 inch）。

核心叙事：**BidKV 在后台预计算一张 victim 排名表，当 KV 压力到达门槛时，把排名投射回引擎的运行队列，让引擎自己的 LIFO 驱逐机制自动选中最优 victim。**

图必须同时传达两件事：
1. **三层架构分层**（LLM Framework / Runtime Adapter / BidKV Core）— 这是论文 §4 的核心设计
2. **两个过程**（后台排名 + 门控驱逐）在三层之间的数据流动

---

## 系统工作流程（画图前必读）

架构图描述的是 BidKV 在一个 LLM 推理引擎（如 vLLM）中的**运行时行为**。整个系统由**两个异步过程**协作完成，它们通过一张**缓存排名表**（Cached Priority Map）解耦：

### 过程 A：后台排名（Background Ranking，每 ~3 秒一次）

这个过程在后台周期性运行，负责"算出谁应该被驱逐"：

1. **采集状态**：Runtime Adapter（中间层）从引擎的 Running Batch 中提取每个正在运行请求的三项信息 —— 完成进度 `c`（已生成 token 数 / 预估总 token 数）、被驱逐次数 `P`、当前占用的 KV token 数 `r`。

2. **计算代价**：这些状态被传递到 BidKV Core（底层）。Bid Generation Layer 为每个请求计算一个**扰动估计（disruption estimate）**：$\delta = 1 + 0.5c + 0.3P$。含义是：一个快完成的请求（c 高）或已经被驱逐过多次的请求（P 高），驱逐它的代价更大。

3. **排序**：Utility-Ranked Selection Layer 用 $U = r / (\delta + \varepsilon)$ 给所有请求排序。$U$ 高意味着"释放多（r大）且代价低（δ小）"—— 这样的请求是最优的驱逐目标。排序结果返回到中间层的 Cached Priority Map 中缓存。

**关键**：这个过程**不触发任何驱逐**。它只产出一张排好序的表，等待过程 B 来使用。

### 过程 B：门控驱逐（Gated Reclamation，每个调度 tick）

这个过程在每次调度时运行，负责"在需要时执行驱逐"：

1. **Pressure Probe（压力探测）**：Adapter 从引擎的 KV Block Pool 读取当前使用率。

2. **Pressure Gate（门控判断）**：如果 KV 使用率 < 95%，什么都不做，引擎继续用默认的 LIFO 顺序。**大部分时间系统处于这种状态** —— BidKV 零开销。

3. **Ranking Lookup（排名查询）**：Gate 通过后，Adapter 从 Cached Priority Map 中读取上一轮过程 A 预计算好的 victim 排名列表。这一步的数据流是 **Pressure Gate → Cached Priority Map**，发生在 Layer 2 内部。

4. **Priority Projection（优先级投射）**：Adapter 将读取到的排名"投射"到引擎的 Running Batch 上 —— 具体做法是按排名重新排列请求顺序，让 $U$ 最高的请求（最优 victim）被推到队列末尾（tail 位置）。这一步的数据流是 **Layer 2 → Layer 1**，从 Cached Priority Map 向上指向 Running Batch。

5. **Native Eviction（原生驱逐）**：引擎自身的 LIFO 驱逐机制从队列末尾弹出请求（这是 vLLM 的原生行为），该请求的 KV blocks 被全部释放。

6. **Feedback Re-entry（反馈重入）**：被驱逐的请求回到 Waiting Queue，其 $P$ 加 1 —— 下一轮排名中它的 $\delta$ 会增大，$U$ 会降低，从而不容易再次被选为 victim（**反饥饿机制**）。

### 两个过程的耦合方式

过程 A 和过程 B **不直接通信**。它们唯一的共享状态是 Cached Priority Map：
- 过程 A 每 ~3 秒**写入**一次新排名
- 过程 B 在 KV ≥ 95% 时**读取**缓存的排名

这种设计使得排名计算（可能涉及所有 running 请求的遍历和评分）与实际调度决策在时间上解耦，不增加关键路径延迟。

---

## 整体布局：三层 + 两个过程

采用 **三个水平层带（horizontal bands）** 从上到下排列，代表三个架构层。两个过程（A 蓝色、B 红色）的箭头在三层之间穿行。

```
┌═══════════════════════════════════════════════════════════┐
│  Layer 1: LLM Serving Engine (e.g., vLLM)    浅灰色背景   │
│                                                           │
│   ┌─────────────┐  ┌───────────────────┐  ┌───────────┐  │
│   │ Waiting     │  │ Running Batch     │  │ KV Block  │  │
│   │ Queue       │  │ R5  R6  R7  [R4]▮│  │ Pool      │  │
│   │ R1  R2  R3  │  │              tail │  │ ████░ 95% │  │
│   └─────────────┘  └───────────────────┘  └───────────┘  │
│         ▲ B4              ▲ A1  ▼ B2          │ B1       │
├═════════╪══════════════════╪════╪══════════════╪═════════╤┤
│  Layer 2: Runtime Adapter Layer        蓝色虚线边框      ││
│                                                          ││
│    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  ││
│    │ State    │◄───│ Pressure │◄───┤                  │  ││
│    │ Collector│    │ Gate     │    │ Cached Priority  │  ││
│    │          │    │ KV≥95%? │    │ Map       ⛁      │  ││
│    └────┬─────┘    └──┬──────┘    └────────▲─────────┘  ││
│         │ A2       yes│ no→(LIFO)          │ A3         ││
│         ▼             ▼ B2                 │            ││
├═════════╪═════════════╪════════════════════╪════════════╤┤
│  Layer 3: BidKV Core                    橙色边框        ││
│                                                         ││
│    ┌─────────────────┐    ┌─────────────────────────┐   ││
│    │ Bid Generation  │───→│ Utility-Ranked Selection│   ││
│    │ Layer            │    │ Layer                    │   ││
│    │ δ = 1+0.5c+0.3P │    │ Sort by U=r/(δ+ε)       │   ││
│    └─────────────────┘    └─────────────────────────┘   ││
│                                                         ││
└═════════════════════════════════════════════════════════╤┘
```

**关键设计意图**：
- 三个层带的背景色/边框必须 **视觉分明**，让读者一眼看到三层架构
- 每个层带左侧有清晰的层标签（Layer 1 / Layer 2 / Layer 3）
- 两组箭头（蓝/红）**垂直穿越层边界**，体现跨层数据流动

---

## Layer 1：LLM Serving Engine（浅灰色背景带）

最上方层带，标签 "**LLM Serving Engine (e.g., vLLM)**"。

包含三个并排组件（从左到右）：

- **Waiting Queue**（左）：3 个请求色块横排（R1 R2 R3），每个是一个**小圆角矩形**，用不同柔和色调（淡绿、淡蓝、淡紫）区分，内写编号。色块间留小间距，像排队一样。
- **Running Batch**（中）：4 个请求色块横排（R5 R6 R7 R4）。前三个用同系列淡色，**R4 用红色填充 + 白色文字**，视觉醒目。R4 正下方画一个小 ↓ 箭头 + "tail" 标注。整个框底部可画一条半透明**进度条**（每个请求占不同宽度），暗示请求各有不同完成度。
- **KV Block Pool**（右）：画成**竖向柱状仪表盘（gauge）** —— 竖长条，下部用渐变灰色填充到约 95% 高度，顶部留一小段空白标注 "free"。在 95% 位置画一条**红色虚线横线** + 标注 "95%"。仪表盘视觉暗示"快满了"，比横条更直观。

这三个组件是 **纯 LLM 引擎原生的**，BidKV 不拥有它们。用统一的浅灰色背景给它们归属感。

---

## Layer 2：Runtime Adapter Layer（浅蓝色 #E8F0FE + 蓝色虚线边框）

中间层带，标签 "**Runtime Adapter Layer**"，整体用蓝色虚线边框包围。

包含三个模块（从左到右），模块之间有明确的内部逻辑连线：

### State Collector（左侧圆角矩形）
框内显示采集的三个字段，用**图标 + 标签**卡片式排列，每行一个：
- 📊 `c` — completion progress（小半填充圆环图标，暗示"进度"）
- 🔄 `P` — preemption count（小循环箭头图标，暗示"被驱逐过"）
- 🧱 `r` — KV tokens held（小方块堆叠图标，暗示"占用量"）

三行之间用细灰线分隔，像一张**采集卡片**。整个框用**蓝色虚线边框**（它属于过程 A 的左侧通道）。

### Pressure Gate（中央圆角矩形，不用菱形）
框内分上下两区：
- **上区**：标题 "Pressure Gate"，小号灰色字体
- **下区**：判断条件 "KV ≥ 95%?"，**加粗大字**

框内还可放一个**小信号灯图标**（🔴 红灯亮起），直觉传达"压力报警"。

框的右侧出口处画一个**小闸门/栏杆符号** ┃╱（竖线 + 斜线，表示闸门抬起），暗示"只有压力到了才打开通路"。这个小符号是整张图中**条件触发**的核心视觉锚点。

框下方画两条出口：
- **左下**：灰色短箭头 + "no → LIFO"（灰色淡化文字，表示默认行为）
- **右侧**：红色箭头穿过闸门 → 指向 Cached Priority Map（B2）

**不要画菱形**。用普通圆角矩形即可。整个框用**红色实线边框**（它属于过程 B 的右侧通道），与左侧 State Collector 的蓝色虚线形成对比。

### Cached Priority Map（右侧圆角矩形 + 圆柱图标）
框内画一个**迷你排行榜**，每行是 rank → request：
```
  rank  request
  ──────────────
   0    R4  ← victim    🔴
   1    R7
   2    R6
   3    R5  ← safest    🟢
```

R4 行用**淡红底色**高亮，R5 行用**淡绿底色**，中间行白底。像一个小 leaderboard。

框顶部放一个小**圆柱数据库图标 ⛁**，暗示缓存数据结构。框的左上角标注 "✍ A writes"（蓝色小字），右上角标注 "👁 B reads"（红色小字），强化它作为两个过程唯一耦合点的角色。

整体用**浅黄底 (#FFF9C4) + 加粗边框 + 微阴影**，在三个 L2 模块中**视觉最突出**。这是两个过程的 **唯一耦合点**。

### Layer 2 内部连线

除了跨层的 A/B 箭头外，Layer 2 内部有一条关键连线：

- **B2 (Ranking Lookup)**：Pressure Gate → Cached Priority Map，红色短线，标注 "Read ranking"。这条线是过程 B 在 Layer 2 内部的唯一水平流动，表示 Gate 通过后去查表。

### 条件触发区域（Conditional Region）

**关键视觉**：用一个**红色虚线圆角矩形**把 B2（Ranking Lookup）和 B3（Priority Projection）整体圈起来，形成一个"条件触发区域"。这个虚线框：
- 从 Pressure Gate 的右侧出口开始，包围 Cached Priority Map 和向上到 Running Batch 的 B3 箭头
- 左上角标注 **"⚡ only when KV ≥ 95%"**（红色小字 + 闪电图标）
- 用**浅红色半透明填充**（如 rgba(219,68,55,0.05)），让这块区域与周围形成微弱色差

这样读者一眼就能看出：B2 和 B3 不是无条件执行的，它们被 Pressure Gate 守护着。Gate 之前（B1）是每 tick 必定执行的探测；Gate 之后的一切都只在压力超阈值时触发。

**重要**：State Collector 和 Pressure Gate 之间 **没有连线**。它们分属过程 A 和过程 B，互不直接通信。它们的唯一间接关系是都通过 Cached Priority Map 耦合（A 写入 → B 读取）。

Adapter 的职责：**向下采集、向上投射**。它不做计算，只做转发和门控。

---

## Layer 3：BidKV Core（橙色实线边框带）

最下方层带，标签 "**BidKV Core**"，整体用橙色实线边框包围。

包含两个子模块（从左到右）：

- **Bid Generation Layer**（左）：框内画一个**公式卡片** —— 浅橙底圆角矩形内写 $\delta = 1 + 0.5c + 0.3P$，用 serif 数学字体。标题 "Bid Generation" 小号字体在框顶。
- **Utility-Ranked Selection Layer**（右）：同样公式卡片风格，写 $U = r/(\delta+\varepsilon)$。下方可画一个**迷你排序图示**：3-4 个高度递减的小竖条从左到右排列（红→黄→绿），暗示排序结果。标题 "Utility-Ranked Selection" 在框顶。

两个子模块之间画一条**橙色粗箭头**，从 Bid Generation → Selection，箭头上标注 "bids"。

BidKV Core 的职责：**只管算**。接收 RequestState → 输出 ranked ordering。整个层带用橙色实线边框，与上方蓝色边框形成暖/冷色对比。

---

## 过程 A：Background Ranking（蓝色虚线箭头）

每 ~3 秒执行一次，穿越三层。箭头走图的 **左侧**。

在 A1 箭头起始位置旁放一个小**时钟图标 🕐** + "~3 s"，直观表示这是周期性后台任务。

| 步骤 | 跨层方向 | 起点 → 终点 | 标注 |
|------|---------|------------|------|
| **A1** | Layer 1 → Layer 2 | Running Batch → State Collector | "Collect state" |
| **A2** | Layer 2 → Layer 3 | State Collector → Bid Generation Layer | "Request states" |
| **A3** | Layer 3 → Layer 2 | Selection Layer → Cached Priority Map | "$U$-ranked list" |

注意 A3 是从 Layer 3 **向上** 回到 Layer 2 的 Cached Priority Map（不回到 Layer 1）。A3 箭头终点处可画一个小**写入图标 ✍**，强调这一步是把排名缓存起来。

---

## 过程 B：Gated Reclamation（红色实线箭头）

每个 scheduling tick 执行，穿越 Layer 1 和 Layer 2。箭头走图的 **右侧**。

在 B1 箭头起始位置旁放一个小**闪电图标 ⚡** + "per tick"，与过程 A 的 🕐 形成对比 —— A 是慢周期后台，B 是快速实时。

| 步骤 | 名称 | 跨层方向 | 起点 → 终点 | 标注 |
|------|------|---------|------------|------|
| **B1** | Pressure Probe | Layer 1 → Layer 2 | KV Block Pool → Pressure Gate | "Read usage" |
| **B2** | Ranking Lookup | Layer 2 内部 | Pressure Gate → Cached Priority Map | "Read ranking" |
| **B3** | Priority Projection | Layer 2 → Layer 1 | Cached Priority Map → Running Batch | "Reorder queue" |
| **B4** | Native Eviction | Layer 1 内部 | Running Batch tail → R4 弹出 | "Evict tail" |
| **B5** | Feedback Re-entry | Layer 1 内部 | R4 → Waiting Queue | "Requeue ($P$++)" |

B2→B3 的关键视觉：B2 是 Layer 2 内部的短水平箭头（Pressure Gate → Cached Priority Map），B3 是从 Cached Priority Map 向上穿越层界指向 Running Batch 的竖向箭头。两者合起来构成"查表 + 投射"的完整链路。

B4→B5 在 Layer 1 内部形成一条**弧线**：R4 从 Running Batch 尾部弹出，画一条弯曲箭头飞向左侧 Waiting Queue。弧线中间画一个小 **"×" 释放标记**（表示 KV blocks 被释放），终点处标注 "P++"。这条弧线让读者直观看到被驱逐请求的"旅程"。

---

## 视觉规范

### 颜色

| 元素 | 颜色 |
|------|------|
| 上栏背景 | 浅蓝色 (#E8F0FE) |
| 下栏背景 | 浅红色 (#FDE8E8) |
| Running Batch / Waiting Queue / KV Pool | 浅灰色 (#F0F0F0) |
| Runtime Adapter 边框 | 蓝色虚线 |
| BidKV Core 边框 | 橙色实线 |
| 循环 A 箭头 + 编号 | 蓝色 (#2F63B7) 虚线 |
| 循环 B 箭头 + 编号 | 红色 (#D5623B, RGB 213,98,59) 实线 |
| Victim 请求 (R4) | 红色填充 (#D5623B) |
| Cached Priority Map | 白色圆柱 + 加粗边框 |

### 编号样式

- A1, A2, A3：图中蓝色箭头旁的标签，caption 中用纯文本 **A1**–**A3** 引用
- B1, B2, B3, B4：图中红色箭头旁的标签，caption 中用纯文本 **B1**–**B4** 引用
- 编号紧贴箭头起点，不要放在箭头中间

### 图例（右下角，紧凑一行）

🕐 `── ── ──` Background ranking (~3 s)　　⚡ `────────` Per-tick gated reclamation

### 简洁原则

- **不写**任何函数名或代码标识符
- **不写** δ 的展开公式（仅在 BidKV Core 框内写 $U = r/(\delta+\varepsilon)$）
- 每支箭头旁 **最多 3 个英文单词**
- 请求方块内只写 R1–R7 编号，不标注 token 数
- Running Batch 在上下栏各画一次，但应让读者理解是同一个队列

---

## 论文术语对齐

| 正确术语 | 禁用术语 |
|----------|----------|
| disruption estimate ($\delta$) | quality delta |
| reclamation utility ($U$) | compression ratio |
| reclaim / preempt | compress |
| Scheduling Bid | compression bid |
| Bid Generation | scoring (as proper noun) |
| Utility-Ranked Selection | solver |
| Runtime Adapter | hook layer |

---

## Caption

```latex
\caption{\bidkv scheduling overview.  Two asynchronous processes govern
victim selection.  \emph{Background ranking}~(blue, every
${\sim}3$\,s): the runtime adapter collects per-request lifecycle
state---completion progress~$c$, preemption count~$P$, and KV tokens
held~$r$---from the running batch~(A1), passes it to the
BidKV core which computes a disruption estimate
$\qualdelta = 1 + 0.5c + 0.3P$ and ranks all candidates by
reclamation utility $U = r/(\qualdelta+\varepsilon)$~(A2);
the resulting ordering is written to the cached priority
map~(A3).  \emph{Gated reclamation}~(red, every scheduling
tick): the adapter reads KV utilization~(B1); when usage
$\geq 95\%$, it looks up the cached ranking~(B2) and projects
it onto the running queue, placing the highest-$U$ candidate at the
tail~(B3).  The framework's native LIFO eviction removes the
tail request, frees its KV blocks, and requeues it with
$P{+}{+}$~(B4), raising~$\qualdelta$ in future ranking cycles
(anti-starvation).  Below the pressure threshold, no reorder occurs
and default LIFO governs eviction.}
```

**Caption 标签与图中箭头的对应关系**：

| Caption 标签 | 对应箭头 | 步骤 |
|-------------|---------|------|
| A1 | A1 | Collect state: Running Batch → State Collector |
| A2 | A2 + L3 内部 | Compute δ, rank by U |
| A3 | A3 | Cache: Selection Layer → Cached Priority Map |
| B1 | B1 | Pressure Probe: KV Block Pool → Pressure Gate |
| B2 | B2 | Ranking Lookup: Gate → Cached Priority Map |
| B3 | B3 | Priority Projection: Map → Running Batch |
| B4 | B4 | Native Eviction + Requeue P++ |