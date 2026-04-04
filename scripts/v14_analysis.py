"""v14 baseline comprehensive analysis.

Analyzes v8 mixed (63 runs) + v8 long_context (available runs).
Computes per-rate and cross-rate rankings for all 4 main metrics.
Identifies where BidKV is NOT #1 in SLO and TTFT p95.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict


def load_run(filepath: str) -> dict:
    """Load a single experiment result file, return standardized metrics."""
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

    workload = d.get('workload', '')
    slo_threshold = 2000.0 if 'long' in workload else 300.0
    slo_count = sum(1 for t in ttft_list if t <= slo_threshold)
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
        'workload': workload,
        'rate': d.get('request_rate', 0),
    }


def load_all_runs(result_dir: str) -> dict:
    """Load all runs, grouped by (strategy, rate)."""
    groups = defaultdict(list)
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        filepath = os.path.join(result_dir, fn)
        try:
            m = load_run(filepath)
            groups[(m['strategy'], m['rate'])].append(m)
        except Exception as e:
            print(f"  WARN: skipping {fn}: {e}")
    return groups


def avg_metric(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r[key] == r[key]]  # skip nan
    return statistics.mean(vals) if vals else float('nan')


def rank_strategies(strat_vals: dict[str, float], higher_is_better: bool) -> dict[str, int]:
    """Rank strategies. Returns {strategy: rank} (1-based, 1=best)."""
    items = sorted(strat_vals.items(), key=lambda x: x[1], reverse=higher_is_better)
    return {s: i + 1 for i, (s, _) in enumerate(items)}


def analyze_workload(result_dir: str, workload_name: str):
    """Full analysis for one workload."""
    print(f"\n{'='*80}")
    print(f"  WORKLOAD: {workload_name}")
    print(f"  Data dir: {result_dir}")
    print(f"{'='*80}\n")

    groups = load_all_runs(result_dir)
    if not groups:
        print("  No data found!")
        return {}

    # Get unique rates and strategies
    rates = sorted(set(r for _, r in groups.keys()))
    strategies = sorted(set(s for s, _ in groups.keys()))

    print(f"Strategies ({len(strategies)}): {strategies}")
    print(f"Rates: {rates}")
    print(f"Total runs: {sum(len(v) for v in groups.values())}")
    print()

    # Per-rate analysis
    metrics = ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']
    higher_better = {'throughput': True, 'slo_pct': True, 'ttft_p95': False, 'tpot_p95': False}
    metric_fmt = {'throughput': '.2f', 'slo_pct': '.1f', 'ttft_p95': '.0f', 'tpot_p95': '.1f'}
    metric_unit = {'throughput': 'req/s', 'slo_pct': '%', 'ttft_p95': 'ms', 'tpot_p95': 'ms'}

    all_rate_data = {}

    for rate in rates:
        print(f"--- Rate = {rate} ---")
        rate_data = {}
        for s in strategies:
            runs = groups.get((s, rate), [])
            if not runs:
                continue
            rate_data[s] = {m: avg_metric(runs, m) for m in metrics}
            rate_data[s]['n_runs'] = len(runs)
            rate_data[s]['ok_avg'] = avg_metric(runs, 'ok_count')
            rate_data[s]['evictions'] = avg_metric(runs, 'evictions')

        if not rate_data:
            print("  No data\n")
            continue

        # Rank for each metric
        rankings = {}
        for m in metrics:
            strat_vals = {s: v[m] for s, v in rate_data.items() if v[m] == v[m]}
            rankings[m] = rank_strategies(strat_vals, higher_better[m])

        # Print table
        hdr = f"{'Strategy':<22} {'Thru':>7} {'SLO%':>7} {'TTFT95':>8} {'TPOT95':>8} | {'Thru#':>5} {'SLO#':>5} {'TTFT#':>5} {'TPOT#':>5} | {'Runs':>4} {'Ok':>5} {'Evict':>6}"
        print(hdr)
        print('-' * len(hdr))
        for s in sorted(rate_data.keys()):
            v = rate_data[s]
            r = {m: rankings[m].get(s, '-') for m in metrics}
            marker = " <<<" if s == 'bidkv' else ""
            print(f"{s:<22} {v['throughput']:>7.2f} {v['slo_pct']:>7.1f} {v['ttft_p95']:>8.0f} {v['tpot_p95']:>8.1f} | "
                  f"{r['throughput']:>5} {r['slo_pct']:>5} {r['ttft_p95']:>5} {r['tpot_p95']:>5} | "
                  f"{v['n_runs']:>4} {v['ok_avg']:>5.0f} {v['evictions']:>6.0f}{marker}")
        print()

        all_rate_data[rate] = rate_data

    # Cross-rate average
    print(f"\n--- Cross-Rate Average ---")
    cross_avg = {}
    for s in strategies:
        vals = defaultdict(list)
        for rate in rates:
            if s in all_rate_data.get(rate, {}):
                for m in metrics:
                    v = all_rate_data[rate][s][m]
                    if v == v:  # not nan
                        vals[m].append(v)
        if any(vals.values()):
            cross_avg[s] = {m: statistics.mean(vals[m]) if vals[m] else float('nan') for m in metrics}

    if cross_avg:
        rankings = {}
        for m in metrics:
            strat_vals = {s: v[m] for s, v in cross_avg.items() if v[m] == v[m]}
            rankings[m] = rank_strategies(strat_vals, higher_better[m])

        hdr = f"{'Strategy':<22} {'Thru':>7} {'SLO%':>7} {'TTFT95':>8} {'TPOT95':>8} | {'Thru#':>5} {'SLO#':>5} {'TTFT#':>5} {'TPOT#':>5} | RankSum Wins"
        print(hdr)
        print('-' * len(hdr))
        for s in sorted(cross_avg.keys()):
            v = cross_avg[s]
            r = {m: rankings[m].get(s, 99) for m in metrics}
            rank_sum = sum(r.values())
            wins = sum(1 for m in metrics if r[m] == 1)
            marker = " <<<" if s == 'bidkv' else ""
            print(f"{s:<22} {v['throughput']:>7.2f} {v['slo_pct']:>7.1f} {v['ttft_p95']:>8.0f} {v['tpot_p95']:>8.1f} | "
                  f"{r['throughput']:>5} {r['slo_pct']:>5} {r['ttft_p95']:>5} {r['tpot_p95']:>5} | {rank_sum:>7} {wins:>4}{marker}")
        print()

    # BidKV gap analysis
    print(f"\n--- BidKV Gap Analysis (SLO + TTFT focus) ---")
    if 'bidkv' in cross_avg:
        for rate in rates:
            if rate not in all_rate_data or 'bidkv' not in all_rate_data[rate]:
                continue
            bv = all_rate_data[rate]['bidkv']
            for m in ['slo_pct', 'ttft_p95']:
                best_s, best_v = None, None
                for s, v in all_rate_data[rate].items():
                    if s == 'bidkv':
                        continue
                    val = v[m]
                    if val != val:
                        continue
                    if best_v is None:
                        best_s, best_v = s, val
                    elif higher_better[m] and val > best_v:
                        best_s, best_v = s, val
                    elif not higher_better[m] and val < best_v:
                        best_s, best_v = s, val

                bidkv_val = bv[m]
                if best_v is not None:
                    is_better = (higher_better[m] and bidkv_val >= best_v) or \
                                (not higher_better[m] and bidkv_val <= best_v)
                    status = "✅ #1" if is_better else f"❌ behind {best_s}"
                    delta = bidkv_val - best_v
                    print(f"  rate={rate} {m}: BidKV={bidkv_val:{metric_fmt[m]}} vs best={best_v:{metric_fmt[m]}} ({best_s}) "
                          f"delta={delta:+{metric_fmt[m]}} {status}")

    return all_rate_data


def main():
    base = '/home/cyb/bidkv/results'

    print("=" * 80)
    print("  v14 BASELINE COMPREHENSIVE ANALYSIS")
    print("  v14 code = v8 committed code (all v11 changes reverted)")
    print("=" * 80)

    # Mixed analysis
    mixed_data = analyze_workload(
        os.path.join(base, 'vllm_v8_full_validation'),
        'MIXED (SLO threshold=300ms)'
    )

    # Long context analysis
    lc_data = analyze_workload(
        os.path.join(base, 'vllm_v8_long_context'),
        'LONG_CONTEXT (SLO threshold=2000ms)'
    )

    # Summary
    print("\n" + "=" * 80)
    print("  OVERALL SUMMARY: Where BidKV needs improvement")
    print("=" * 80)
    print()
    print("Target: SLO #1 + TTFT p95 #1 across ALL rates in BOTH workloads")
    print("Current gaps identified above — focus optimization on these gaps.")


if __name__ == '__main__':
    main()
