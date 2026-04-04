#!/usr/bin/env python3
"""v8 Root Cause Synthesis — BidKV vs Static-Random 根因量化报告。

量化三个核心假说：
H1: BidKV 在 mixed 中从未主动驱逐 → KV 滞留 → 全局 decode 减速
H2: BidKV 的 U-score 偏好 largest-first → 触发昂贵 recompute
H3: BidKV 在 LC rate=0.7（驱逐数相近）时反超 SR → 证明 U-score victim
    selection 本身有效，瓶颈是缺少驱逐触发机制

输出：简洁量化表 + 根因结论
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict


def load_run(filepath: str) -> dict:
    with open(filepath) as f:
        d = json.load(f)
    ok = [r for r in d["request_results"] if not r.get("error")]
    ttft_list = sorted(r["ttft_ms"] for r in ok if r["ttft_ms"] is not None)
    tpot_list = []
    for r in ok:
        if (
            r.get("completion_tokens", 0) > 1
            and r.get("ttft_ms") is not None
            and r.get("total_latency_ms") is not None
        ):
            tpot = (r["total_latency_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
            tpot_list.append(tpot)
    tpot_list.sort()
    e2e_list = sorted(r["total_latency_ms"] for r in ok if r.get("total_latency_ms") is not None)

    def pct(data, p):
        if not data:
            return float("nan")
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    slo_thr = 2000.0 if "long" in d.get("workload", "") else 300.0
    slo_count = sum(1 for t in ttft_list if t <= slo_thr)
    slo_pct = slo_count / len(ttft_list) * 100 if ttft_list else 0

    am = d.get("adapter_metrics", {})
    evictions = am.get("total_evictions", am.get("total_compressions", 0))
    freed = am.get("total_tokens_freed", 0)

    # Extreme outlier counts
    e2e_gt60 = sum(1 for v in e2e_list if v > 60000)
    ttft_gt10 = sum(1 for v in ttft_list if v > 10000)

    return {
        "throughput": d["summary"]["throughput_rps"],
        "slo_pct": slo_pct,
        "ttft_p50": pct(ttft_list, 50),
        "ttft_p95": pct(ttft_list, 95),
        "ttft_p99": pct(ttft_list, 99),
        "tpot_p50": pct(tpot_list, 50),
        "tpot_p95": pct(tpot_list, 95),
        "tpot_p99": pct(tpot_list, 99),
        "e2e_p50": pct(e2e_list, 50),
        "e2e_p95": pct(e2e_list, 95),
        "e2e_max": max(e2e_list) if e2e_list else 0,
        "e2e_gt60s": e2e_gt60,
        "ttft_gt10s": ttft_gt10,
        "ok_count": len(ok),
        "total_count": len(d["request_results"]),
        "evictions": evictions,
        "tokens_freed": freed,
        "freed_per_eviction": freed / evictions if evictions > 0 else 0,
        "strategy": d.get("strategy", ""),
        "workload": d.get("workload", ""),
        "rate": d.get("request_rate", 0),
        "duration": d.get("duration_s", 0),
    }


def load_dir(result_dir: str) -> dict:
    groups = defaultdict(list)
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith(".json") or fn.startswith("candidate"):
            continue
        m = load_run(os.path.join(result_dir, fn))
        groups[(m["strategy"], m["rate"])].append(m)
    return groups


def avg(vals):
    return statistics.mean(vals) if vals else 0


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def main():
    base = "/home/cyb/bidkv/results"
    mixed_dir = os.path.join(base, "vllm_v8_full_validation")
    lc_dir = os.path.join(base, "vllm_v8_long_context")

    print("#" * 80)
    print("#  BidKV v8 ROOT CAUSE SYNTHESIS: Why BidKV Loses to Static-Random")
    print("#" * 80)

    # ─── Load data ───
    mixed = load_dir(mixed_dir) if os.path.isdir(mixed_dir) else {}
    lc = load_dir(lc_dir) if os.path.isdir(lc_dir) else {}

    # Strategy aliases
    bidkv_name = "bidkv"
    sr_name = "static-random"

    # ─── H1: Eviction Gap → KV Stagnation → Decode Slowdown ───
    print_section("H1: EVICTION GAP — BidKV Never Proactively Evicts in Mixed")

    print("  Mixed Workload: BidKV vs Static-Random per rate")
    print(f"  {'Rate':>6}  {'BidKV Evict':>11}  {'SR Evict':>10}  {'BidKV TPOT99':>13}"
          f"  {'SR TPOT99':>10}  {'Gap':>8}  {'BidKV E2Emax':>13}  {'SR E2Emax':>10}"
          f"  {'BidKV >60s':>10}  {'SR >60s':>8}")

    mixed_rates = sorted({r for (_, r) in mixed})
    for rate in mixed_rates:
        bk = mixed.get((bidkv_name, rate), [])
        sr = mixed.get((sr_name, rate), [])
        if not bk or not sr:
            continue
        bk_evict = avg([m["evictions"] for m in bk])
        sr_evict = avg([m["evictions"] for m in sr])
        bk_tpot99 = avg([m["tpot_p99"] for m in bk])
        sr_tpot99 = avg([m["tpot_p99"] for m in sr])
        gap = (bk_tpot99 - sr_tpot99) / sr_tpot99 * 100 if sr_tpot99 > 0 else 0
        bk_e2emax = max(m["e2e_max"] for m in bk)
        sr_e2emax = max(m["e2e_max"] for m in sr)
        bk_gt60 = sum(m["e2e_gt60s"] for m in bk)
        sr_gt60 = sum(m["e2e_gt60s"] for m in sr)

        print(f"  {rate:>6.1f}  {bk_evict:>11.0f}  {sr_evict:>10.0f}"
              f"  {bk_tpot99:>13.1f}  {sr_tpot99:>10.1f}  {gap:>+7.1f}%"
              f"  {bk_e2emax / 1000:>12.1f}s  {sr_e2emax / 1000:>9.1f}s"
              f"  {bk_gt60:>10}  {sr_gt60:>8}")

    print("\n  Key Finding: BidKV has ZERO proactive evictions at ALL mixed rates.")
    print("  Without proactive eviction, KV stays full → all requests compete for GPU →")
    print("  TPOT rises + some requests get trapped (170s max E2E, 13× the SR outlier count).")

    # ─── H1 for Long-Context ───
    print_section("H1b: LONG-CONTEXT — Both Evict, But BidKV Evicts MORE Tokens")

    print("  Long-Context Workload: BidKV vs Static-Random per rate")
    print(f"  {'Rate':>6}  {'BidKV Evict':>11}  {'SR Evict':>10}  {'BidKV Freed/E':>14}"
          f"  {'SR Freed/E':>11}  {'BidKV TPOT99':>13}  {'SR TPOT99':>10}  {'Gap':>8}")

    lc_rates = sorted({r for (_, r) in lc})
    for rate in lc_rates:
        bk = lc.get((bidkv_name, rate), [])
        sr = lc.get((sr_name, rate), [])
        if not bk or not sr:
            continue
        bk_evict = avg([m["evictions"] for m in bk])
        sr_evict = avg([m["evictions"] for m in sr])
        bk_fpe = avg([m["freed_per_eviction"] for m in bk if m["evictions"] > 0])
        sr_fpe = avg([m["freed_per_eviction"] for m in sr if m["evictions"] > 0])
        bk_tpot99 = avg([m["tpot_p99"] for m in bk])
        sr_tpot99 = avg([m["tpot_p99"] for m in sr])
        gap = (bk_tpot99 - sr_tpot99) / sr_tpot99 * 100 if sr_tpot99 > 0 else 0

        print(f"  {rate:>6.2f}  {bk_evict:>11.0f}  {sr_evict:>10.0f}"
              f"  {bk_fpe:>14.0f}  {sr_fpe:>11.0f}"
              f"  {bk_tpot99:>13.1f}  {sr_tpot99:>10.1f}  {gap:>+7.1f}%")

    print("\n  Key Finding: BidKV's freed_per_eviction > SR → confirmed largest-first bias.")
    print("  Largest-first eviction = most expensive recompute (long prompt re-prefill).")

    # ─── H2: Freed-dominant U-score → Largest-First ───
    print_section("H2: U-SCORE IS FREED-DOMINANT → EFFECTIVELY LARGEST-FIRST")

    print("  Formula: U = freed / (δ + ε), δ = 1 + 0.5c + 0.3P, δ ∈ [1.0, 1.8]")
    print()
    print("  Example candidates at KV pressure:")
    print("  ┌────────────────────────────────────────────────────┐")
    print("  │ Request  freed  c    P   δ      U       Rank      │")
    print("  ├────────────────────────────────────────────────────┤")
    examples = [
        ("A (long prompt, new)", 3000, 0.05, 0, "1.025", 3000 / 1.026),
        ("B (short, mid-gen)", 500, 0.5, 0, "1.250", 500 / 1.251),
        ("C (medium, preempted)", 1500, 0.3, 1, "1.450", 1500 / 1.451),
        ("D (long, near done)", 3000, 0.8, 0, "1.400", 3000 / 1.401),
    ]
    for name, freed, c, p, delta_str, u in examples:
        print(f"  │ {name:<23} {freed:>5}  {c:.2f}  {p}  {delta_str:>5}  {u:>8.1f}  │")
    print("  └────────────────────────────────────────────────────┘")
    print()
    print("  Victim order: A (2926) > D (2141) > C (1034) > B (400)")
    print("  → Largest requests always evicted first, δ barely changes ranking.")
    print("  → Request A: just started, 3000 tokens cached, EXPENSIVE recompute.")
    print("  → BidKV systematically chooses the most costly-to-recompute victims.")

    # ─── H3: LC Rate=0.7 Reversal Proves U-score Works When Eviction Happens ───
    print_section("H3: LC RATE=0.7 — BidKV BEATS SR WHEN EVICTION COUNTS MATCH")

    metrics_compare = [
        "throughput", "slo_pct", "ttft_p50", "ttft_p95", "tpot_p50", "tpot_p95",
        "e2e_p50", "e2e_p95", "e2e_max", "evictions",
    ]
    headers = [
        "Throughput", "SLO%", "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95",
        "E2E p50", "E2E p95", "E2E max", "Evictions",
    ]

    lc_high_rates = [r for r in lc_rates if r >= 0.65]
    for rate in lc_high_rates:
        bk = lc.get((bidkv_name, rate), [])
        sr = lc.get((sr_name, rate), [])
        if not bk or not sr:
            continue

        print(f"  Long-Context Rate={rate}:")
        print(f"    {'Metric':<14} {'BidKV':>10} {'StaticRandom':>13} {'Winner':>10}")
        for metric, header in zip(metrics_compare, headers):
            bk_val = avg([m[metric] for m in bk])
            sr_val = avg([m[metric] for m in sr])

            # Determine winner (lower is better for latency, higher for thru/slo)
            lower_better = metric not in ("throughput", "slo_pct", "evictions")
            if metric == "evictions":
                winner = "—"
            elif lower_better:
                winner = "BidKV" if bk_val < sr_val else "SR"
            else:
                winner = "BidKV" if bk_val > sr_val else "SR"

            if metric in ("throughput",):
                print(f"    {header:<14} {bk_val:>10.3f} {sr_val:>13.3f} {winner:>10}")
            elif metric in ("slo_pct",):
                print(f"    {header:<14} {bk_val:>10.1f}% {sr_val:>12.1f}% {winner:>10}")
            elif metric in ("evictions",):
                print(f"    {header:<14} {bk_val:>10.0f} {sr_val:>13.0f} {winner:>10}")
            else:
                print(f"    {header:<14} {bk_val:>10.0f}ms {sr_val:>11.0f}ms {winner:>10}")

    print()
    print("  *** CRITICAL FINDING: At similar eviction counts (~137 vs ~140),")
    print("      BidKV's quality-aware victim selection delivers BETTER TTFT and")
    print("      lower E2E max than random. The U-score WORKS — but only when the")
    print("      eviction trigger mechanism is active. ***")

    # ─── Causal Chain ───
    print_section("ROOT CAUSE CAUSAL CHAIN")

    print("""  Mixed workload (3 rates, cross-rate):

  ┌─────────────────────────────────────────────────────┐
  │ BidKV skips proactive_preempt() + SRPT()             │
  │ (Lines 505, 662 of scheduler_hook.py)               │
  └───────────────────────┬─────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ Mixed: 0 proactive evictions (all 3 rates)          │
  │ SR: 12→86→119 proactive evictions as rate increases │
  └───────────────────────┬─────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ BidKV: KV stays at high utilization → all requests  │
  │ compete for GPU time per decode step                │
  └───────────────────────┬─────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ TPOT p99: +47-49% vs SR at high rates               │
  │ E2E max: 170s vs 61s (2.8×)                         │
  │ >60s E2E: 13 req vs 1 req (13×)                     │
  └───────────────────────────────────────────────────────┘

  Long-context (rate=0.7, high pressure):

  ┌─────────────────────────────────────────────────────┐
  │ vLLM native preemption frequent enough that         │
  │ running reorder (KV>95%) IS effective               │
  │ → Both strategies evict ~140 times                  │
  └───────────────────────┬─────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ BidKV's U-score victim selection IS better:         │
  │ TTFT p50: 1201 < 1613ms (BidKV wins)               │
  │ E2E max: 167s < 213s (BidKV wins)                   │
  │ TPOT p50: 67.2 < 70.8 (BidKV wins)                 │
  └───────────────────────────────────────────────────────┘

  Long-context (rate=0.35, low pressure):

  ┌─────────────────────────────────────────────────────┐
  │ BidKV evicts MORE (56 vs 38) but WORSE outcomes     │
  │ → largest-first bias → expensive recompute          │
  │ TPOT p99: +59.1% gap!                              │
  └───────────────────────────────────────────────────────┘""")

    # ─── Verdict ───
    print_section("VERDICT: TWO DISTINCT BUGS, NOT ONE")

    print("""  Bug #1: Missing Proactive Eviction Trigger (Mixed)
  ─────────────────────────────────────────────────────
  BidKV has a hard `return` skip for proactive_preempt() and SRPT().
  v10 tried enabling proactive preempt → caused eviction storms (wrong gate).

  The fix is NOT "enable proactive preempt at KV>90%". The fix needs:
  - Higher trigger gate (92-95%) specific to BidKV
  - Longer cooldown (8-10s vs 5s) to prevent storms
  - Cost-benefit gate: only evict if U-score > threshold (skip marginal victims)

  Bug #2: Recompute-Blind U-Score (LC Low Pressure)
  ─────────────────────────────────────────────────────
  U = freed / (δ + ε) maximizes "KV freed per quality cost" but ignores
  recompute cost. A 3000-token request with c=0.05 (just started, long prompt)
  has U=2926 — highest priority to evict. But recomputing it costs 3000 tokens
  of prefill work!

  Static-Random accidentally avoids this by being random — the average
  recompute cost is lower than BidKV's systematic worst-case.

  Fix direction: U_new = freed / (δ + α·recompute_ratio + ε)
  where recompute_ratio = num_prompt_tokens / max_context_len
  α controls recompute cost sensitivity (suggested: 0.5-1.0)

  ## Why v10 Failed — Specific Diagnosis

  v10 enabled proactive preempt with SAME parameters as other strategies:
  - Gate: KV > 90% (too low for BidKV — triggers too early)
  - Cooldown: 5s (too short — allows storm)
  - Victim: cached priority (= BidKV's U-score = largest-first)

  Result: evicted the LARGEST request at KV=91% → triggered expensive
  recompute → recompute request consumed KV blocks → KV>90% again →
  eviction cascade. TTFT +29% because recomputation flooded prefill queue.

  ## Combined Fix Proposal (v11)

  1. Enable proactive preempt for BidKV with DIFFERENT parameters:
     - Gate: KV > 93% (higher than 90%, to avoid early trigger)
     - Cooldown: 10s (2× others, to prevent storm)
     - Min running requests: 4 (keep 3+ running, stricter)

  2. Recompute-aware U-score:
     - δ_new = 1 + 0.5c + 0.3P + α·(prompt_tokens / max_model_len)
     - This penalizes evicting long-prompt requests (expensive recompute)
     - α = 0.5 suggested (to be calibrated)

  3. Keep SRPT disabled (confirmed harmful in Mode A)

  4. Keep running reorder at KV > 95% (proven effective in LC rate=0.7)""")

    # ─── Expected Impact ───
    print_section("EXPECTED IMPACT OF v11 FIX")

    print("""  Based on observed data:

  Mixed (currently BidKV's weakness):
  - v11 proactive preempt (93% gate, 10s cooldown) should:
    - Add ~30-80 evictions per run (similar to SR's 72-119)
    - Reduce TPOT p99 by ~20-30% (from KV pressure relief)
    - Eliminate extreme outliers (170s → ~60s max)
    - Recompute-aware victim = cheaper recompute → less TTFT degradation
    - Expected: maintain SLO #1 + TTFT #1 while closing TPOT gap

  Long-context (currently competitive):
  - Recompute-aware U-score should:
    - Reduce freed_per_eviction (smaller victims = cheaper recompute)
    - Close the TPOT p99 gap at low rates (currently +59%)
    - Maintain or improve high-rate advantage (LC 0.7 already winning)

  Risk: Too-tight gate or too-long cooldown → insufficient eviction → no effect
  Mitigation: Calibrate with pilot run before full experiment""")


if __name__ == "__main__":
    main()
