"""Compare v14 BidKV (no guard) vs v8 baselines in long_context.

Run after v14 LC experiment completes:
  conda run -n sagellm python scripts/v14_lc_compare.py
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict


def load_run(filepath):
    d = json.load(open(filepath))
    ok = [r for r in d['request_results'] if not r.get('error')]
    ttft_list = sorted(r['ttft_ms'] for r in ok if r['ttft_ms'] is not None)

    tpot_list = []
    for r in ok:
        if (r.get('completion_tokens', 0) > 1
                and r.get('ttft_ms') is not None
                and r.get('total_latency_ms') is not None):
            tpot = (r['total_latency_ms'] - r['ttft_ms']) / (r['completion_tokens'] - 1)
            tpot_list.append(tpot)
    tpot_list.sort()

    def percentile(data, p):
        if not data:
            return float('nan')
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    slo_count = sum(1 for t in ttft_list if t <= 2000.0)
    slo_pct = slo_count / len(ttft_list) * 100 if ttft_list else 0

    am = d.get('adapter_metrics', {})
    evictions = am.get('total_evictions', am.get('total_compressions', 0))

    return {
        'throughput': d['summary']['throughput_rps'],
        'slo_pct': slo_pct,
        'ttft_p50': percentile(ttft_list, 50),
        'ttft_p95': percentile(ttft_list, 95),
        'ttft_p99': percentile(ttft_list, 99),
        'tpot_p50': percentile(tpot_list, 50),
        'tpot_p95': percentile(tpot_list, 95),
        'tpot_p99': percentile(tpot_list, 99),
        'ok_count': len(ok),
        'total_count': len(d['request_results']),
        'evictions': evictions,
        'tokens_freed': am.get('total_tokens_freed', 0),
        'strategy': d.get('strategy', ''),
        'workload': d.get('workload', ''),
        'rate': d.get('request_rate', 0),
    }


def main():
    base = '/home/cyb/bidkv/results'
    v14_dir = os.path.join(base, 'vllm_v14_long_context')
    v8_dir = os.path.join(base, 'vllm_v8_long_context')

    # Check data availability
    v14_files = [f for f in os.listdir(v14_dir) if f.endswith('.json') and f.startswith('bidkv')]
    if not v14_files:
        print("No v14 LC results yet!")
        return

    print(f"v14 BidKV LC files: {len(v14_files)}")
    rates = [0.35, 0.5, 0.7]

    # Load v14 BidKV data
    v14_data: dict[float, list] = defaultdict(list)
    for fn in v14_files:
        m = load_run(os.path.join(v14_dir, fn))
        v14_data[m['rate']].append(m)

    # Load v8 baselines + v8 BidKV
    v8_data: dict[tuple[str, float], list] = defaultdict(list)
    for fn in sorted(os.listdir(v8_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        m = load_run(os.path.join(v8_dir, fn))
        v8_data[(m['strategy'], m['rate'])].append(m)

    metrics = ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']
    higher_better = {'throughput': True, 'slo_pct': True, 'ttft_p95': False, 'tpot_p95': False}

    print("\n" + "=" * 90)
    print("  v14 BidKV (no guard) vs v8 baselines — LONG_CONTEXT (SLO=2000ms)")
    print("=" * 90)

    for rate in rates:
        v14_runs = v14_data.get(rate, [])
        if not v14_runs:
            print(f"\n  Rate={rate}: no v14 data yet")
            continue

        print(f"\n--- Rate = {rate} ---")

        # Collect all strategies for this rate
        strat_avg = {}

        # v14 BidKV
        strat_avg['bidkv-v14'] = {m: statistics.mean([r[m] for r in v14_runs]) for m in metrics}
        strat_avg['bidkv-v14']['evictions'] = statistics.mean([r['evictions'] for r in v14_runs])
        strat_avg['bidkv-v14']['n_runs'] = len(v14_runs)

        # v8 BidKV (with guard)
        v8_bidkv = v8_data.get(('bidkv', rate), [])
        if v8_bidkv:
            strat_avg['bidkv-v8'] = {m: statistics.mean([r[m] for r in v8_bidkv]) for m in metrics}
            strat_avg['bidkv-v8']['evictions'] = statistics.mean([r['evictions'] for r in v8_bidkv])
            strat_avg['bidkv-v8']['n_runs'] = len(v8_bidkv)

        # v8 baselines
        for s in ['h2o-style', 'static-random', 'preempt-evict', 'preempt-evict-sjf', 'uniform']:
            runs = v8_data.get((s, rate), [])
            if runs:
                strat_avg[s] = {m: statistics.mean([r[m] for r in runs]) for m in metrics}
                strat_avg[s]['evictions'] = statistics.mean([r['evictions'] for r in runs])
                strat_avg[s]['n_runs'] = len(runs)

        # Rank
        for m in metrics:
            vals = {s: v[m] for s, v in strat_avg.items() if s != 'bidkv-v8' and v[m] == v[m]}
            sorted_strats = sorted(vals.items(), key=lambda x: x[1], reverse=higher_better[m])
            for i, (s, _) in enumerate(sorted_strats):
                strat_avg[s][f'{m}_rank'] = i + 1

        # Print
        hdr = f"{'Strategy':<20} {'Thru':>7} {'SLO%':>7} {'TTFT95':>8} {'TPOT95':>8} | {'Thru#':>5} {'SLO#':>5} {'TTFT#':>5} {'TPOT#':>5} | {'N':>3} {'Evict':>6}"
        print(hdr)
        print('-' * len(hdr))

        for s in ['bidkv-v14', 'bidkv-v8', 'h2o-style', 'static-random', 'preempt-evict', 'preempt-evict-sjf', 'uniform']:
            if s not in strat_avg:
                continue
            v = strat_avg[s]
            ranks = ''.join([f"{v.get(f'{m}_rank', '-'):>6}" for m in metrics]) if s != 'bidkv-v8' else '   (v8 reference)'
            marker = " <<<" if s == 'bidkv-v14' else " (OLD)" if s == 'bidkv-v8' else ""
            ev = v.get('evictions', 0)
            n = v.get('n_runs', 0)
            print(f"{s:<20} {v['throughput']:>7.3f} {v['slo_pct']:>7.1f} {v['ttft_p95']:>8.0f} {v['tpot_p95']:>8.1f} | {ranks} | {n:>3} {ev:>6.0f}{marker}")

        # Delta analysis
        if 'bidkv-v14' in strat_avg and 'bidkv-v8' in strat_avg:
            print(f"\n  v14 vs v8 delta:")
            for m in metrics:
                v14_val = strat_avg['bidkv-v14'][m]
                v8_val = strat_avg['bidkv-v8'][m]
                delta = v14_val - v8_val
                better_dir = '+' if higher_better[m] else '-'
                improved = (higher_better[m] and delta > 0) or (not higher_better[m] and delta < 0)
                status = "✅ improved" if improved else "❌ regressed" if abs(delta) > 0.1 else "→ unchanged"
                print(f"    {m}: v14={v14_val:.1f} v8={v8_val:.1f} delta={delta:+.1f} {status}")

    # Cross-rate summary
    print(f"\n{'='*60}")
    print("  Cross-Rate Average (v14 BidKV vs competitors)")
    print(f"{'='*60}")

    cross_v14 = defaultdict(list)
    cross_v8 = defaultdict(list)
    for rate in rates:
        for r in v14_data.get(rate, []):
            for m in metrics:
                cross_v14[m].append(r[m])
        for s in ['h2o-style', 'static-random', 'preempt-evict', 'preempt-evict-sjf', 'uniform']:
            for r in v8_data.get((s, rate), []):
                cross_v8.setdefault(s, defaultdict(list))
                for m in metrics:
                    cross_v8[s][m].append(r[m])

    if cross_v14:
        print(f"\n  BidKV-v14 cross-rate: ", end='')
        for m in metrics:
            val = statistics.mean(cross_v14[m]) if cross_v14[m] else float('nan')
            print(f"{m}={val:.1f}  ", end='')
        print()

        for s in ['h2o-style', 'static-random', 'preempt-evict', 'preempt-evict-sjf', 'uniform']:
            if s in cross_v8:
                print(f"  {s:<22} ", end='')
                for m in metrics:
                    val = statistics.mean(cross_v8[s][m]) if cross_v8[s][m] else float('nan')
                    print(f"{m}={val:.1f}  ", end='')
                print()


if __name__ == '__main__':
    main()
