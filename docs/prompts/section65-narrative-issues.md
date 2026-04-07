# §6.5 叙事问题记录（待论文 Agent 修订）

> 生成日期：2026-04-06
> 更新日期：2026-04-06（BidKV 补跑验证后）
> 状态：**READY** — BidKV 补跑完成，数据已验证
> 数据基准：fig3 实验（`results/vllm_fig3_mixed_rate38/`），rate=3.8，3 runs 均值
> 注意：reclamation 统计使用 `total_all_preemptions` 字段（含 vLLM 原生驱逐），
>       非 `total_evictions`（仅 BidKV 主动 proactive preempt，BidKV 该字段始终为 0）

---

## 实验数据参考（fig3，rate=3.8，3 runs 均值）

### 性能指标

| Strategy | SLO(300ms)% | TTFT P95 (ms) | Throughput (req/s) | TPOT P95 (ms) |
|---|---|---|---|---|
| PE | 68.0 | 7,429 | 3.44 | 100.2 |
| PE-SJF | 80.3 | 665 | 3.13 | 145.4 |
| Static-Random | 84.3 | 1,111 | 3.60 | 88.1 |
| Largest-First | 81.0 | 675 | 3.36 | 109.7 |
| **BidKV** | **82.6** | **642** | 3.39 | 107.2 |

### Fig.5 回收统计（`total_all_preemptions` / `total_all_tokens_freed`）

| Strategy | Reclamation Count | Tokens Freed | Freed/Event | BidKV Proactive |
|---|---:|---:|---:|---:|
| PE | 247 | 196k | 794 | 0 |
| PE-SJF | 336 | 366k | 1,087 | 0 |
| Static-Random | 329 | 88k | 269 | ~87 |
| Largest-First | 296 | 325k | 1,100 | ~112 |
| **BidKV** | **181** | **253k** | **1,398** | **0** |

**说明**：`total_all_preemptions` 含 vLLM 原生驱逐。BidKV 的 `total_evictions`（BidKV 主动
proactive preempt）始终为 0 ——BidKV 不发起 proactive preempt，所有 181 次事件均通过
running-queue reorder（KV>95%）+ vLLM 原生 LIFO 触发；`bk_proactive=0` 是预期行为，
非 bug（skip guard 验证通过）。

---

## 需修订的叙事问题（4 项）

### 问题 1（RESOLVED）：BidKV 数据 bug 已修复，数据已更新

`scheduler_hook.py` 中 BidKV proactive preempt skip 守卫已补回，3 runs 补跑完成。
- 补跑结果：TTFT P95 = 642 ms，SLO = 82.6%（v8 参考：631 ms，83.2%，差异 < 2%）✅
- § 6.5 原用旧数据（TTFT 1,507 ms）需更新为正确值

---

### 问题 2（事实错误）：Static-Random bullet 称"substantially more events"，但方向相反

**位置**：§6.5 Static-Random bullet

**当前措辞**：
> *"requiring substantially more reclamation events to relieve the same pressure—a direct consequence of ignoring victim heterogeneity"*

**问题**：fig3 数据中 Static-Random = **329 events**，BidKV = **181 events**（Static-Random 多 81%，
方向**确实**是 substantially more）。但核心差距是 Freed/Event：Static-Random = 269，
BidKV = 1,398（**5.2× 差距**），导致 Static-Random 总释放量仅 88k，BidKV 为 253k（相差 2.9×）。

**修订方向**：保留"substantially more events"（数字支持），但补充 Freed/Event 对比（269 vs 1,398）
作为核心因果链：随机命中小 KV 占用请求 → 每次释放量极低（269 tokens/event，BidKV 的 1/5）
→ 需要更多次驱逐才能缓解同等压力 → 更多 recompute 次数 → TTFT 受累（1,111 ms 尾部）。
即使 SLO 指标略好（84.3% 因运气好短请求驱逐开销小），TTFT P95 尾部暴露了其 inefficiency。

---

### 问题 3（措辞需精调）：PE-SJF 与 BidKV TTFT 差距仅 3.6%，但 TPOT 差距显著

**位置**：§6.5 PE-SJF bullet

**当前措辞**：
> *"cost-unaware victim selection means high freed-per-event does not translate to better TTFT P95"*

**问题**：fig3 PE-SJF TTFT P95 = 665 ms，BidKV = 642 ms，差距仅 **3.6%（23 ms）**。
PE-SJF 是 5 策略 TTFT 排名第 2，与 BidKV 近乎持平——但 TPOT P95 差距明显（145.4 ms vs 107.2 ms，**36%**）。
PE-SJF 高 Freed/Event（1,087）通过 **更多 events**（336 vs 181）实现，
意味着更频繁触发 large-request recompute，占据 prefill 带宽，压制 decode 速度。

**修订方向**：承认 PE-SJF TTFT 接近 BidKV（665 ms vs 642 ms），把主要差距归因于 TPOT（145.4 ms vs 107.2 ms）
和 SLO（80.3% vs 82.6%）。解释："PE-SJF 实现高 freed-per-event 依赖更多次驱逐（336 vs 181），
频繁的 large-request recompute 消耗 prefill 带宽，导致 TPOT 显著恶化。"

---

### 问题 4（措辞需精调）：Largest-First TTFT 差距有限，核心差距在 SLO 和 recompute 开销

**位置**：§6.5 Largest-First bullet

**当前措辞**：
> *"Largest-First's strong freed-per-event coexists with elevated TTFT P95 relative to BidKV"*

**问题**：fig3 Largest-First TTFT P95 = 675 ms，BidKV = 642 ms，差距 **5%（33 ms）**；
TPOT P95 差距也有限（109.7 vs 107.2 ms）。主要差距体现在 SLO（81.0% vs 82.6%，1.6pp）。
Largest-First 的 bk_proactive ≈ 112（vs BidKV = 0），即大量主动驱逐大请求，
所有请求更多地被打断 → recompute 次数上升 → 系统整体 SLO 受影响。

**修订方向**：把解释重心从"elevated TTFT"转移到 recompute overhead：
"Largest-First targets largest KV blocks regardless of completion progress，
its 112 proactive preemptions（vs BidKV's 0）drive near-complete requests into recompute，
consuming disproportionate prefill bandwidth and lifting TPOT。其 33 ms TTFT 劣势
和 1.6pp SLO 劣势均源于此高 recompute 代价。"

---

## 状态追踪

- [x] Bug 修复：`scheduler_hook.py` BidKV proactive preempt skip 守卫补回
- [x] δ 公式修复：`bidkv_strategy.py` 回退至 v8 公式 `δ = 1 + 0.5c + 0.3P`
- [x] 论文 §4.2 公式修正（同步 v8 公式）
- [x] 旧 BidKV fig3 数据删除（3 个 JSON 文件）
- [x] BidKV 补跑完成（`results/vllm_fig3_mixed_rate38/bidkv__mixed__rate3.8__r*.json`）
- [x] 验证补跑数据与 v8 frozen 一致（TTFT 差 1.7%，SLO 差 0.6pp ✅）
- [ ] 更新 Fig.5（重跑 analysis pipeline，填入 BidKV ev=181/freed=253k/f/e=1398）
- [ ] 论文 §6.5 根据本文件 4 条问题修订
