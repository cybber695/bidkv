"""Check statistical significance of mixed workload gaps.

Analyzes per-run variance to determine if BidKV's SLO/TTFT gaps are
within experimental noise or genuinely significant.
"""
from __future__ import annotations

import json
import os
import statistics


def load_run(filepath):
    d = json.load(open(filepath))
    ok = [r for r in d['request_results'] if not r.get('error')]
    ttft_list = sorted(r['ttft_ms'] for r in ok if r['ttft_ms'] is not None)

    def percentile(data, p):
        if not data:
            return float('nan')
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    slo_count = sum(1 for t in ttft_list if t <= 300.0)
    slo_pct = slo_count / len(ttft_list) * 100 if ttft_list else 0

    return {
        'slo_pct': slo_pct,
        'ttft_p95': percentile(ttft_list, 95),
        'strategy': d.get('strategy', ''),
        'rate': d.get('request_rate', 0),
        'run_index': d.get('run_index', 0),
    }


def main():
    base = '/home/cyb/bidkv/results/vllm_v8_full_validation'

    # Focus on the gap rates
    for rate in [3.8, 5.7]:
        print(f"\n{'='*60}")
        print(f"  Rate = {rate}")
        print(f"{'='*60}")

        # Collect per-run data for key strategies
        strategies = ['bidkv', 'static-random', 'uniform', 'preempt-evict-sjf']
        strat_data = {}

        for fn in sorted(os.listdir(base)):
            if not fn.endswith('.json') or fn.startswith('candidate'):
                continue
            filepath = os.path.join(base, fn)
            m = load_run(filepath)
            if m['rate'] != rate:
                continue
            if m['strategy'] not in strategies:
                continue
            strat_data.setdefault(m['strategy'], []).append(m)

        print(f"\n  SLO(300ms) per-run values:")
        for s in strategies:
            runs = strat_data.get(s, [])
            vals = [r['slo_pct'] for r in runs]
            if len(vals) >= 2:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals)
                print(f"    {s:<22} runs={vals}  mean={mean:.1f}  std={std:.2f}")
            elif vals:
                print(f"    {s:<22} runs={vals}  mean={vals[0]:.1f}  (single run)")

        print(f"\n  TTFT p95 per-run values:")
        for s in strategies:
            runs = strat_data.get(s, [])
            vals = [r['ttft_p95'] for r in runs]
            if len(vals) >= 2:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals)
                print(f"    {s:<22} runs={vals}  mean={mean:.0f}  std={std:.0f}")
            elif vals:
                print(f"    {s:<22} runs={vals}  mean={vals[0]:.0f}  (single run)")

        # Overlap analysis
        print(f"\n  Overlap analysis (BidKV vs best):")
        bidkv_slo = [r['slo_pct'] for r in strat_data.get('bidkv', [])]
        bidkv_ttft = [r['ttft_p95'] for r in strat_data.get('bidkv', [])]

        for s in ['static-random', 'uniform', 'preempt-evict-sjf']:
            comp_slo = [r['slo_pct'] for r in strat_data.get(s, [])]
            comp_ttft = [r['ttft_p95'] for r in strat_data.get(s, [])]

            if bidkv_slo and comp_slo:
                overlap_slo = max(min(bidkv_slo), min(comp_slo)) <= min(max(bidkv_slo), max(comp_slo))
                print(f"    BidKV SLO [{min(bidkv_slo):.1f},{max(bidkv_slo):.1f}] vs {s} [{min(comp_slo):.1f},{max(comp_slo):.1f}] overlap={'YES' if overlap_slo else 'NO'}")
            if bidkv_ttft and comp_ttft:
                overlap_ttft = max(min(bidkv_ttft), min(comp_ttft)) <= min(max(bidkv_ttft), max(comp_ttft))
                print(f"    BidKV TTFT [{min(bidkv_ttft):.0f},{max(bidkv_ttft):.0f}] vs {s} [{min(comp_ttft):.0f},{max(comp_ttft):.0f}] overlap={'YES' if overlap_ttft else 'NO'}")


if __name__ == '__main__':
    main()
