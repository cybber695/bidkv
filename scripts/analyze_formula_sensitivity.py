"""Analyze whether better formula parameters could help BidKV.

Key question: BidKV's throughput gap is ~7% cross-rate. Is this fundamental
to the no-proactive-preempt design, or could formula tuning help?

Analysis:
1. Compare victim quality across rates (what does native preemption evict?)
2. Model the effect of different δ parameters on victim ordering
3. Check if stale priority cache contributes to suboptimal ordering
"""
from __future__ import annotations

import json
import os
import statistics


def load_requests(filepath: str) -> list[dict]:
    d = json.load(open(filepath))
    ok = [r for r in d['request_results'] if not r.get('error')]
    return ok, d


def analyze_victim_quality(result_dir: str, rate: float) -> None:
    """Analyze request-level patterns to understand victim selection quality."""
    bidkv_files = []
    competitor_files = {}

    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        d = json.load(open(os.path.join(result_dir, fn)))
        if abs(d.get('request_rate', 0) - rate) > 0.01:
            continue
        strat = d['strategy']
        path = os.path.join(result_dir, fn)
        if strat == 'bidkv':
            bidkv_files.append(path)
        else:
            competitor_files.setdefault(strat, []).append(path)

    if not bidkv_files:
        return

    print(f"\n{'='*80}")
    print(f"  VICTIM QUALITY ANALYSIS — Rate={rate}")
    print(f"{'='*80}")

    # For each strategy, compute the decode efficiency distribution
    # (ms per output token) — this shows how effectively each strategy
    # converts compute into output.
    for strat_name, files in [('bidkv', bidkv_files)] + list(competitor_files.items()):
        all_decode_eff = []
        all_ttft = []
        all_completion_tokens = []

        for fp in files:
            ok, d = load_requests(fp)
            for r in ok:
                ct = r.get('completion_tokens', 0)
                ttft = r.get('ttft_ms')
                total = r.get('total_latency_ms')
                if ct > 1 and ttft is not None and total is not None:
                    decode_time = total - ttft
                    decode_eff = decode_time / (ct - 1)
                    all_decode_eff.append(decode_eff)
                    all_ttft.append(ttft)
                    all_completion_tokens.append(ct)

        if not all_decode_eff:
            continue

        all_decode_eff.sort()
        all_ttft.sort()
        all_completion_tokens.sort()

        n = len(all_decode_eff)
        p50_de = all_decode_eff[n // 2]
        p95_de = all_decode_eff[int(n * 0.95)]
        p50_t = all_ttft[n // 2]
        p95_t = all_ttft[int(n * 0.95)]
        avg_ct = statistics.mean(all_completion_tokens)

        print(f"\n  {strat_name:<22} (n={n})")
        print(f"    Decode eff:  p50={p50_de:.1f}  p95={p95_de:.1f} ms/tok")
        print(f"    TTFT:        p50={p50_t:.1f}  p95={p95_t:.1f} ms")
        print(f"    Completion:  avg={avg_ct:.0f} tokens")

    # Compute the theoretical effect of different δ parameters
    print(f"\n--- δ Parameter Sensitivity ---")
    print(f"  Given BidKV's U = freed / (δ + ε), how does δ affect victim ordering?")
    print()

    # Simulate victim selection with different δ formulas
    # at representative KV pressure points
    test_victims = [
        # (label, freed_tokens, completion_ratio, num_preemptions)
        ("new-short",    100, 0.05, 0),
        ("new-long",     400, 0.02, 0),
        ("mid-short",    100, 0.50, 0),
        ("mid-long",     300, 0.30, 0),
        ("near-done",    200, 0.85, 0),
        ("preempted-1x", 150, 0.10, 1),
        ("preempted-2x", 150, 0.10, 2),
    ]

    formulas = {
        "v8 (0.5c+0.3P)": lambda c, p: 1.0 + 0.5 * c + 0.3 * p,
        "weak-c (0.3c+0.3P)": lambda c, p: 1.0 + 0.3 * c + 0.3 * p,
        "strong-c (0.8c+0.3P)": lambda c, p: 1.0 + 0.8 * c + 0.3 * p,
        "strong-P (0.5c+0.5P)": lambda c, p: 1.0 + 0.5 * c + 0.5 * p,
        "flat (0.2c+0.2P)": lambda c, p: 1.5 + 0.2 * c + 0.2 * p,
    }

    eps = 1e-6
    for fname, formula in formulas.items():
        print(f"  Formula: {fname}")
        scored = []
        for label, freed, c, p in test_victims:
            delta = formula(c, p)
            u = freed / (delta + eps)
            scored.append((u, label, freed, c, delta))
        scored.sort(reverse=True)
        for rank, (u, label, freed, c, delta) in enumerate(scored, 1):
            marker = " ← best victim" if rank == 1 else ""
            print(f"    #{rank} {label:<16} freed={freed:3d} c={c:.2f} "
                  f"δ={delta:.2f} U={u:.0f}{marker}")
        print()


def analyze_throughput_decomposition(result_dir: str) -> None:
    """Decompose throughput into admission wait + decode + overhead."""
    print(f"\n{'='*80}")
    print(f"  THROUGHPUT DECOMPOSITION — Cross-Rate")
    print(f"{'='*80}")

    files_by_strat = {}
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        d = json.load(open(os.path.join(result_dir, fn)))
        strat = d['strategy']
        files_by_strat.setdefault(strat, []).append(os.path.join(result_dir, fn))

    for strat in sorted(files_by_strat.keys()):
        all_wait = []
        all_decode = []
        all_overhead = []

        for fp in files_by_strat[strat]:
            ok, d = load_requests(fp)
            for r in ok:
                ttft = r.get('ttft_ms')
                total = r.get('total_latency_ms')
                ct = r.get('completion_tokens', 0)
                if ttft is not None and total is not None and ct > 0:
                    # Wait time ≈ TTFT (includes both queuing + prefill)
                    wait = ttft
                    # Decode time = total - TTFT
                    decode = total - ttft
                    # Total per-request time
                    all_wait.append(wait)
                    all_decode.append(decode)

        if not all_wait:
            continue

        all_wait.sort()
        all_decode.sort()
        n = len(all_wait)

        print(f"\n  {strat:<22} (n={n})")
        print(f"    Wait (≈TTFT):  p50={all_wait[n//2]:.0f}  "
              f"p95={all_wait[int(n*0.95)]:.0f}  "
              f"avg={statistics.mean(all_wait):.0f} ms")
        print(f"    Decode time:   p50={all_decode[n//2]:.0f}  "
              f"p95={all_decode[int(n*0.95)]:.0f}  "
              f"avg={statistics.mean(all_decode):.0f} ms")
        print(f"    Total:         p50={all_wait[n//2]+all_decode[n//2]:.0f}  "
              f"p95={all_wait[int(n*0.95)]+all_decode[int(n*0.95)]:.0f}  "
              f"avg={statistics.mean(all_wait)+statistics.mean(all_decode):.0f} ms")


if __name__ == '__main__':
    result_dir = '/home/cyb/bidkv/results/vllm_v8_full_validation'

    for rate in [2.0, 3.8, 5.7]:
        analyze_victim_quality(result_dir, rate)

    analyze_throughput_decomposition(result_dir)
