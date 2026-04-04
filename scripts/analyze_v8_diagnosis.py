"""Diagnose BidKV vs competitors: completion time, decode efficiency, queue dynamics."""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict


def load_requests(filepath: str) -> list[dict]:
    d = json.load(open(filepath))
    return [r for r in d['request_results'] if not r.get('error')]


def analyze_rate(result_dir: str, rate: float, workload: str) -> None:
    slo = 2000.0 if 'long' in workload else 300.0
    strategies = {}
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        d = json.load(open(os.path.join(result_dir, fn)))
        if abs(d.get('request_rate', 0) - rate) > 0.01:
            continue
        strat = d['strategy']
        if strat not in strategies:
            strategies[strat] = []
        strategies[strat].extend(
            r for r in d['request_results'] if not r.get('error')
        )

    print(f"\n{'='*80}")
    print(f"  DETAILED ANALYSIS: {workload} rate={rate}")
    print(f"{'='*80}")

    # 1. Arrival vs completion order analysis
    # How much does each strategy reorder requests?
    for strat in ['bidkv', 'static-random', 'uniform', 'h2o-style',
                  'preempt-evict-sjf', 'largest-first']:
        reqs = strategies.get(strat, [])
        if not reqs:
            continue
        # Sort by submit_time, track finish order
        by_submit = sorted(reqs, key=lambda r: r['submit_time'])
        by_finish = sorted(reqs, key=lambda r: r['finish_time'])

        # Kendall tau-like: how many pairs are out of order
        finish_rank = {r['request_id']: i for i, r in enumerate(by_finish)}
        inversions = 0
        n = len(by_submit)
        for i in range(min(n, 500)):  # sample first 500
            for j in range(i + 1, min(n, 500)):
                ri = finish_rank.get(by_submit[i]['request_id'], i)
                rj = finish_rank.get(by_submit[j]['request_id'], j)
                if ri > rj:
                    inversions += 1
        total_pairs = min(n, 500) * (min(n, 500) - 1) // 2
        reorder_pct = inversions / max(1, total_pairs) * 100
        print(f"  {strat:<22} reorder: {reorder_pct:.1f}% inversions "
              f"(n={n}, sampled={min(n,500)})")

    # 2. E2E latency vs completion tokens scatter
    print(f"\n--- E2E latency by completion token buckets ---")
    for strat in ['bidkv', 'static-random', 'uniform', 'h2o-style']:
        reqs = strategies.get(strat, [])
        if not reqs:
            continue
        buckets = defaultdict(list)
        for r in reqs:
            ct = r.get('completion_tokens', 0)
            lat = r.get('total_latency_ms', 0)
            if ct <= 0 or lat <= 0:
                continue
            if ct < 30:
                buckets['<30'].append(lat)
            elif ct < 100:
                buckets['30-100'].append(lat)
            elif ct < 300:
                buckets['100-300'].append(lat)
            else:
                buckets['>=300'].append(lat)

        parts = []
        for label in ['<30', '30-100', '100-300', '>=300']:
            data = buckets.get(label, [])
            if data:
                parts.append(f"{label}:n={len(data)},p50={statistics.median(data):.0f}")
        print(f"  {strat:<22} {', '.join(parts)}")

    # 3. TTFT vs queue position (submit order)
    print(f"\n--- TTFT by arrival position (first 200 vs last 200) ---")
    for strat in ['bidkv', 'static-random', 'uniform', 'h2o-style']:
        reqs = strategies.get(strat, [])
        if not reqs:
            continue
        by_submit = sorted(reqs, key=lambda r: r['submit_time'])
        early = [r['ttft_ms'] for r in by_submit[:200] if r.get('ttft_ms')]
        late = [r['ttft_ms'] for r in by_submit[-200:] if r.get('ttft_ms')]
        if early and late:
            print(f"  {strat:<22} early200 p50={statistics.median(early):.0f} "
                  f"p95={sorted(early)[int(len(early)*0.95)]:.0f} | "
                  f"late200 p50={statistics.median(late):.0f} "
                  f"p95={sorted(late)[int(len(late)*0.95)]:.0f}")

    # 4. Throughput timeline: completions per 30s window
    print(f"\n--- Completion rate over time (completions per 30s window) ---")
    for strat in ['bidkv', 'static-random', 'uniform', 'h2o-style']:
        reqs = strategies.get(strat, [])
        if not reqs:
            continue
        by_finish = sorted(reqs, key=lambda r: r['finish_time'])
        if not by_finish:
            continue
        t0 = by_finish[0]['finish_time']
        windows = defaultdict(int)
        for r in by_finish:
            w = int((r['finish_time'] - t0) / 30)
            windows[w] += 1
        max_w = max(windows.keys()) if windows else 0
        rates_list = [windows.get(w, 0) / 30.0 for w in range(max_w + 1)]
        if rates_list:
            print(f"  {strat:<22} avg={statistics.mean(rates_list):.2f} req/s, "
                  f"min={min(rates_list):.2f}, max={max(rates_list):.2f}, "
                  f"windows={len(rates_list)}")

    # 5. Decode efficiency: total_latency / completion_tokens
    print(f"\n--- Decode efficiency: ms per output token (total) ---")
    for strat in ['bidkv', 'static-random', 'uniform', 'h2o-style']:
        reqs = strategies.get(strat, [])
        if not reqs:
            continue
        efficiencies = []
        for r in reqs:
            ct = r.get('completion_tokens', 0)
            lat = r.get('total_latency_ms', 0)
            if ct > 0 and lat > 0:
                efficiencies.append(lat / ct)
        if efficiencies:
            efficiencies.sort()
            n = len(efficiencies)
            print(f"  {strat:<22} p50={efficiencies[n//2]:.1f} "
                  f"p95={efficiencies[int(n*0.95)]:.1f} "
                  f"p99={efficiencies[int(n*0.99)]:.1f} ms/token")


if __name__ == '__main__':
    mixed_dir = '/home/cyb/bidkv/results/vllm_v8_full_validation'
    lc_dir = '/home/cyb/bidkv/results/vllm_v8_long_context'

    # Focus on high-pressure rates where BidKV loses
    if os.path.isdir(mixed_dir):
        analyze_rate(mixed_dir, 3.8, 'mixed')
        analyze_rate(mixed_dir, 5.7, 'mixed')
    if os.path.isdir(lc_dir):
        analyze_rate(lc_dir, 0.5, 'long_context')
        analyze_rate(lc_dir, 0.7, 'long_context')
