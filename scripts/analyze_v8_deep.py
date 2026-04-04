"""Deep analysis of v8 mixed + long_context data for BidKV optimization insights.

Focuses on:
1. Per-rate BidKV weakness diagnosis (where exactly does BidKV lose?)
2. Request-level TTFT/TPOT distribution comparison
3. Eviction pattern analysis (who gets evicted, when, cost)
4. High-pressure behavior (rate=5.7 mixed, rate=0.7 LC)
5. Comparison with top competitors at each rate
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict


def load_run(filepath: str) -> dict:
    d = json.load(open(filepath))
    ok = [r for r in d['request_results'] if not r.get('error')]
    fail = [r for r in d['request_results'] if r.get('error')]

    ttft_list = sorted([r['ttft_ms'] for r in ok if r['ttft_ms'] is not None])
    tpot_list = []
    for r in ok:
        if (r.get('completion_tokens', 0) > 1
                and r.get('ttft_ms') is not None
                and r.get('total_latency_ms') is not None):
            tpot = (r['total_latency_ms'] - r['ttft_ms']) / (r['completion_tokens'] - 1)
            tpot_list.append(tpot)
    tpot_list.sort()

    def pct(data: list, p: int) -> float:
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
        'ttft_p50': pct(ttft_list, 50),
        'ttft_p95': pct(ttft_list, 95),
        'ttft_p99': pct(ttft_list, 99),
        'tpot_p50': pct(tpot_list, 50),
        'tpot_p95': pct(tpot_list, 95),
        'tpot_p99': pct(tpot_list, 99),
        'ok_count': len(ok),
        'fail_count': len(fail),
        'total_count': len(d['request_results']),
        'evictions': evictions,
        'tokens_freed': am.get('total_tokens_freed', 0),
        'pressure_events': am.get('total_pressure_events', 0),
        'decode_steps': am.get('total_decode_steps', 0),
        'strategy': d.get('strategy', ''),
        'workload': workload,
        'rate': d.get('request_rate', 0),
        'duration': d.get('duration_s', 0),
        'ttft_list': ttft_list,
        'tpot_list': tpot_list,
        'request_results': ok,
    }


def load_all(result_dir: str) -> dict:
    groups = defaultdict(list)
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        m = load_run(os.path.join(result_dir, fn))
        groups[(m['strategy'], m['rate'])].append(m)
    return groups


def analyze_workload(result_dir: str, workload_name: str) -> None:
    groups = load_all(result_dir)
    rates = sorted({r for _, r in groups.keys()})
    strategies = sorted({s for s, _ in groups.keys()})

    print(f"\n{'='*80}")
    print(f"  {workload_name} WORKLOAD DEEP ANALYSIS")
    print(f"{'='*80}")

    slo_threshold = 2000.0 if 'long' in workload_name else 300.0

    # 1. Per-rate detailed comparison
    for rate in rates:
        print(f"\n--- Rate = {rate} ---")
        print(f"{'Strategy':<22} {'Thru':>6} {'SLO%':>7} {'TTFT50':>8} {'TTFT95':>8} "
              f"{'TTFT99':>8} {'TPOT50':>8} {'TPOT95':>8} {'TPOT99':>8} "
              f"{'Evict':>6} {'Freed':>8} {'Fail':>5}")
        for strat in strategies:
            runs = groups.get((strat, rate), [])
            if not runs:
                continue
            avg = lambda key: statistics.mean([r[key] for r in runs])
            print(f"{strat:<22} {avg('throughput'):>6.2f} {avg('slo_pct'):>6.1f}% "
                  f"{avg('ttft_p50'):>8.0f} {avg('ttft_p95'):>8.0f} "
                  f"{avg('ttft_p99'):>8.0f} {avg('tpot_p50'):>8.1f} "
                  f"{avg('tpot_p95'):>8.1f} {avg('tpot_p99'):>8.1f} "
                  f"{avg('evictions'):>6.0f} {avg('tokens_freed'):>8.0f} "
                  f"{avg('fail_count'):>5.0f}")

    # 2. Cross-rate ranking
    print(f"\n--- Cross-Rate Average Ranking ---")
    strat_avg = defaultdict(lambda: defaultdict(list))
    for (strat, rate), runs in groups.items():
        for r in runs:
            for k in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']:
                strat_avg[strat][k].append(r[k])

    cross = {}
    for strat, metrics in strat_avg.items():
        cross[strat] = {k: statistics.mean(v) for k, v in metrics.items()}

    # Rank each metric
    for metric in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']:
        reverse = metric in ('throughput', 'slo_pct')  # higher is better
        ranked = sorted(cross.items(), key=lambda x: x[1][metric], reverse=reverse)
        for rank, (strat, _) in enumerate(ranked, 1):
            cross[strat][f'{metric}_rank'] = rank

    print(f"{'Strategy':<22} {'Thru':>6} {'R':>2} {'SLO%':>7} {'R':>2} "
          f"{'TTFT95':>8} {'R':>2} {'TPOT95':>8} {'R':>2} {'Sum':>4}")
    ranked_strats = sorted(cross.items(),
                           key=lambda x: sum(x[1].get(f'{m}_rank', 99)
                                             for m in ['throughput', 'slo_pct',
                                                        'ttft_p95', 'tpot_p95']))
    for strat, m in ranked_strats:
        rsum = sum(m.get(f'{k}_rank', 99)
                   for k in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95'])
        print(f"{strat:<22} {m['throughput']:>6.2f} #{m['throughput_rank']:<1} "
              f"{m['slo_pct']:>6.1f}% #{m['slo_pct_rank']:<1} "
              f"{m['ttft_p95']:>8.0f} #{m['ttft_p95_rank']:<1} "
              f"{m['tpot_p95']:>8.1f} #{m['tpot_p95_rank']:<1} "
              f"{rsum:>4}")

    # 3. BidKV vs top competitor gap analysis at highest rate
    high_rate = max(rates)
    print(f"\n--- BidKV Gap Analysis at Rate={high_rate} (highest pressure) ---")
    bidkv_runs = groups.get(('bidkv', high_rate), [])
    if bidkv_runs:
        bidkv_avg = {k: statistics.mean([r[k] for r in bidkv_runs])
                     for k in ['throughput', 'slo_pct', 'ttft_p50', 'ttft_p95',
                                'tpot_p50', 'tpot_p95', 'evictions', 'tokens_freed',
                                'fail_count', 'pressure_events']}
        print(f"  BidKV: Thru={bidkv_avg['throughput']:.2f}, SLO={bidkv_avg['slo_pct']:.1f}%, "
              f"TTFT95={bidkv_avg['ttft_p95']:.0f}, TPOT95={bidkv_avg['tpot_p95']:.1f}")
        print(f"  Evictions={bidkv_avg['evictions']:.0f}, Freed={bidkv_avg['tokens_freed']:.0f}, "
              f"Pressure={bidkv_avg['pressure_events']:.0f}, Fails={bidkv_avg['fail_count']:.0f}")

        for strat in strategies:
            if strat == 'bidkv':
                continue
            runs = groups.get((strat, high_rate), [])
            if not runs:
                continue
            other = {k: statistics.mean([r[k] for r in runs])
                     for k in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95',
                                'evictions', 'tokens_freed', 'fail_count']}
            thru_d = (other['throughput'] - bidkv_avg['throughput']) / bidkv_avg['throughput'] * 100
            slo_d = other['slo_pct'] - bidkv_avg['slo_pct']
            ttft_d = other['ttft_p95'] - bidkv_avg['ttft_p95']
            tpot_d = other['tpot_p95'] - bidkv_avg['tpot_p95']
            print(f"  vs {strat:<20}: Thru {thru_d:>+6.1f}%, SLO {slo_d:>+5.1f}pp, "
                  f"TTFT95 {ttft_d:>+8.0f}ms, TPOT95 {tpot_d:>+6.1f}ms")

    # 4. TTFT distribution analysis at highest rate
    print(f"\n--- TTFT Distribution at Rate={high_rate} ---")
    for strat in ['bidkv', 'static-random', 'uniform', 'h2o-style', 'largest-first']:
        runs = groups.get((strat, high_rate), [])
        if not runs:
            continue
        all_ttft = []
        for r in runs:
            all_ttft.extend(r['ttft_list'])
        if not all_ttft:
            continue
        all_ttft.sort()
        n = len(all_ttft)
        buckets = {
            f'<{int(slo_threshold)}ms': sum(1 for t in all_ttft if t <= slo_threshold),
            f'<1s': sum(1 for t in all_ttft if t <= 1000),
            '<5s': sum(1 for t in all_ttft if t <= 5000),
            '>=5s': sum(1 for t in all_ttft if t > 5000),
        }
        print(f"  {strat:<22} (n={n}): ", end='')
        for label, cnt in buckets.items():
            print(f"{label}={cnt}({cnt/n*100:.1f}%) ", end='')
        print()

    # 5. TPOT distribution analysis at highest rate
    print(f"\n--- TPOT Distribution at Rate={high_rate} ---")
    for strat in ['bidkv', 'static-random', 'uniform', 'h2o-style', 'largest-first']:
        runs = groups.get((strat, high_rate), [])
        if not runs:
            continue
        all_tpot = []
        for r in runs:
            all_tpot.extend(r['tpot_list'])
        if not all_tpot:
            continue
        all_tpot.sort()
        n = len(all_tpot)
        buckets = {
            '<50ms': sum(1 for t in all_tpot if t <= 50),
            '<100ms': sum(1 for t in all_tpot if t <= 100),
            '<200ms': sum(1 for t in all_tpot if t <= 200),
            '>=200ms': sum(1 for t in all_tpot if t >= 200),
        }
        print(f"  {strat:<22} (n={n}): ", end='')
        for label, cnt in buckets.items():
            print(f"{label}={cnt}({cnt/n*100:.1f}%) ", end='')
        print()

    # 6. Request completion tokens distribution vs TPOT
    print(f"\n--- Completion Tokens vs TPOT at Rate={high_rate} (BidKV) ---")
    bidkv_runs = groups.get(('bidkv', high_rate), [])
    if bidkv_runs:
        short_tpot = []  # completion < 50
        med_tpot = []     # 50-200
        long_tpot = []    # > 200
        for r in bidkv_runs:
            for req in r['request_results']:
                ct = req.get('completion_tokens', 0)
                if ct <= 1 or req.get('ttft_ms') is None or req.get('total_latency_ms') is None:
                    continue
                tpot = (req['total_latency_ms'] - req['ttft_ms']) / (ct - 1)
                if ct < 50:
                    short_tpot.append(tpot)
                elif ct <= 200:
                    med_tpot.append(tpot)
                else:
                    long_tpot.append(tpot)
        for label, data in [('short(<50)', short_tpot), ('med(50-200)', med_tpot),
                            ('long(>200)', long_tpot)]:
            if data:
                data.sort()
                print(f"  {label:<15} n={len(data):>4}, p50={data[len(data)//2]:.1f}ms, "
                      f"p95={data[int(len(data)*0.95)]:.1f}ms, "
                      f"p99={data[int(len(data)*0.99)]:.1f}ms")

    # 7. Requests with TTFT > 5s analysis
    print(f"\n--- Slow Requests (TTFT>5s) at Rate={high_rate} ---")
    for strat in strategies:
        runs = groups.get((strat, high_rate), [])
        if not runs:
            continue
        slow_count = 0
        total = 0
        for r in runs:
            for req in r['request_results']:
                if req.get('error'):
                    continue
                total += 1
                if req.get('ttft_ms', 0) and req['ttft_ms'] > 5000:
                    slow_count += 1
        print(f"  {strat:<22}: {slow_count}/{total} ({slow_count/max(1,total)*100:.1f}%) "
              f"requests with TTFT>5s")


if __name__ == '__main__':
    mixed_dir = '/home/cyb/bidkv/results/vllm_v8_full_validation'
    lc_dir = '/home/cyb/bidkv/results/vllm_v8_long_context'

    if os.path.isdir(mixed_dir):
        analyze_workload(mixed_dir, 'MIXED')

    if os.path.isdir(lc_dir):
        analyze_workload(lc_dir, 'LONG_CONTEXT')
