"""Comprehensive v8 analysis for paper evaluation.

Objective assessment of BidKV vs all baselines across mixed + long_context.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict


def load_run(filepath: str) -> dict:
    d = json.load(open(filepath))
    ok = [r for r in d['request_results'] if not r.get('error')]
    ttft_list = sorted([r['ttft_ms'] for r in ok if r['ttft_ms'] is not None])
    tpot_list = []
    for r in ok:
        if (r.get('completion_tokens', 0) > 1
                and r.get('ttft_ms') is not None
                and r.get('total_latency_ms') is not None):
            tpot = (r['total_latency_ms'] - r['ttft_ms']) / (r['completion_tokens'] - 1)
            tpot_list.append(tpot)
    tpot_list.sort()

    def pct(data, p):
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
        'fail_count': len(d['request_results']) - len(ok),
        'total_count': len(d['request_results']),
        'evictions': evictions,
        'tokens_freed': am.get('total_tokens_freed', 0),
        'strategy': d.get('strategy', ''),
        'workload': d.get('workload', ''),
        'rate': d.get('request_rate', 0),
    }


def load_dir(result_dir: str) -> dict:
    groups = defaultdict(list)
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        m = load_run(os.path.join(result_dir, fn))
        groups[(m['strategy'], m['rate'])].append(m)
    return groups


DISPLAY = {
    'bidkv': 'BidKV',
    'static-random': 'Static-Random',
    'uniform': 'Uniform',
    'h2o-style': 'Largest-First',
    'slack-aware': 'Slack-Aware',
    'preempt-evict': 'PE (Default)',
    'preempt-evict-sjf': 'PE-SJF',
}

METRICS_4 = ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']
HIGHER_BETTER = {'throughput': True, 'slo_pct': True, 'ttft_p95': False, 'tpot_p95': False}
METRIC_FMT = {'throughput': '.2f', 'slo_pct': '.1f', 'ttft_p95': '.0f', 'tpot_p95': '.1f'}
METRIC_UNIT = {'throughput': 'req/s', 'slo_pct': '%', 'ttft_p95': 'ms', 'tpot_p95': 'ms'}


def avg_runs(runs: list[dict], key: str) -> float:
    return statistics.mean([r[key] for r in runs])


def std_runs(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs]
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def analyze_workload(groups: dict, workload_name: str, slo_label: str) -> dict:
    """Full analysis for one workload. Returns cross-rate summary."""
    # Get all strategies and rates
    strategies = sorted(set(s for s, r in groups.keys()))
    rates = sorted(set(r for s, r in groups.keys()))

    print(f"\n{'#'*90}")
    print(f"#  WORKLOAD: {workload_name.upper()}  —  SLO threshold: {slo_label}")
    print(f"#  Strategies: {len(strategies)}  |  Rates: {rates}  |  Runs/combo: 3")
    print(f"{'#'*90}")

    # ================================================================
    # Section 1: Per-Rate Detailed Tables
    # ================================================================
    rate_rankings = defaultdict(lambda: defaultdict(dict))  # rate -> metric -> strat -> rank

    for rate in rates:
        print(f"\n{'='*85}")
        print(f"  Rate = {rate} req/s")
        print(f"{'='*85}")

        # Collect per-strategy averages
        strat_data = {}
        for strat in strategies:
            key = (strat, rate)
            if key not in groups:
                continue
            runs = groups[key]
            strat_data[strat] = {
                m: avg_runs(runs, m) for m in METRICS_4
            }
            strat_data[strat]['_std'] = {
                m: std_runs(runs, m) for m in METRICS_4
            }
            strat_data[strat]['_n'] = len(runs)
            strat_data[strat]['ok_pct'] = avg_runs(runs, 'ok_count') / avg_runs(runs, 'total_count') * 100
            strat_data[strat]['evictions'] = avg_runs(runs, 'evictions')
            strat_data[strat]['tokens_freed'] = avg_runs(runs, 'tokens_freed')

        if not strat_data:
            continue

        # Rank per metric
        for m in METRICS_4:
            reverse = HIGHER_BETTER[m]
            ranked = sorted(strat_data.keys(),
                            key=lambda s: strat_data[s][m],
                            reverse=reverse)
            for rank, s in enumerate(ranked, 1):
                strat_data[s][f'{m}_rank'] = rank
                rate_rankings[rate][m][s] = rank

        # Print table
        print(f"\n  {'Strategy':<16} {'Thru':>7} {'SLO%':>7} {'TTFT95':>8} {'TPOT95':>8} "
              f"{'RankSum':>8} {'#1s':>4} {'Evict':>6} {'Freed':>8} {'OK%':>6}")
        print(f"  {'-'*84}")

        for strat in sorted(strat_data.keys(),
                            key=lambda s: sum(strat_data[s].get(f'{m}_rank', 99) for m in METRICS_4)):
            d = strat_data[strat]
            rank_sum = sum(d.get(f'{m}_rank', 99) for m in METRICS_4)
            wins = sum(1 for m in METRICS_4 if d.get(f'{m}_rank') == 1)
            marker = " ◄" if strat == 'bidkv' else ""
            print(f"  {DISPLAY.get(strat, strat):<16} "
                  f"{d['throughput']:>7.2f} "
                  f"{d['slo_pct']:>6.1f}% "
                  f"{d['ttft_p95']:>8.0f} "
                  f"{d['tpot_p95']:>8.1f} "
                  f"{rank_sum:>8d} "
                  f"{wins:>4d} "
                  f"{d['evictions']:>6.0f} "
                  f"{d['tokens_freed']:>8.0f} "
                  f"{d['ok_pct']:>5.1f}%{marker}")

        # Print ranks explicitly
        print(f"\n  Ranks:  ", end="")
        for strat in sorted(strat_data.keys(),
                            key=lambda s: sum(strat_data[s].get(f'{m}_rank', 99) for m in METRICS_4)):
            ranks = "/".join(str(strat_data[strat].get(f'{m}_rank', '?')) for m in METRICS_4)
            print(f"  {DISPLAY.get(strat, strat)[:8]}({ranks})", end="")
        print()

        # BidKV specific analysis
        if 'bidkv' in strat_data:
            bd = strat_data['bidkv']
            print(f"\n  BidKV rank: throughput=#{bd.get('throughput_rank','?')}, "
                  f"SLO=#{bd.get('slo_pct_rank','?')}, "
                  f"TTFT95=#{bd.get('ttft_p95_rank','?')}, "
                  f"TPOT95=#{bd.get('tpot_p95_rank','?')}")

            # Gap to #1 for each metric
            for m in METRICS_4:
                if bd.get(f'{m}_rank') != 1:
                    best_strat = [s for s in strat_data if strat_data[s].get(f'{m}_rank') == 1][0]
                    best_val = strat_data[best_strat][m]
                    gap = bd[m] - best_val
                    gap_pct = gap / max(abs(best_val), 0.01) * 100
                    print(f"    {m}: gap to #{1} ({DISPLAY.get(best_strat, best_strat)}) = "
                          f"{gap:+.1f} ({gap_pct:+.1f}%)")

    # ================================================================
    # Section 2: Cross-Rate Average
    # ================================================================
    print(f"\n{'='*85}")
    print(f"  CROSS-RATE AVERAGE  ({workload_name})")
    print(f"{'='*85}")

    cross_data = {}
    for strat in strategies:
        all_vals = defaultdict(list)
        for rate in rates:
            key = (strat, rate)
            if key not in groups:
                continue
            for run in groups[key]:
                for m in METRICS_4 + ['evictions', 'tokens_freed', 'ok_count', 'total_count']:
                    all_vals[m].append(run[m])
        if not all_vals['throughput']:
            continue
        cross_data[strat] = {m: statistics.mean(all_vals[m]) for m in METRICS_4}
        cross_data[strat]['evictions'] = statistics.mean(all_vals['evictions'])
        cross_data[strat]['tokens_freed'] = statistics.mean(all_vals['tokens_freed'])
        cross_data[strat]['n_runs'] = len(all_vals['throughput'])
        # Standard deviations
        for m in METRICS_4:
            cross_data[strat][f'{m}_std'] = statistics.stdev(all_vals[m]) if len(all_vals[m]) >= 2 else 0.0

    # Rank
    for m in METRICS_4:
        reverse = HIGHER_BETTER[m]
        ranked = sorted(cross_data.keys(), key=lambda s: cross_data[s][m], reverse=reverse)
        for rank, s in enumerate(ranked, 1):
            cross_data[s][f'{m}_rank'] = rank

    print(f"\n  {'Strategy':<16} {'Thru':>7} {'SLO%':>7} {'TTFT95':>8} {'TPOT95':>8} "
          f"{'RankSum':>8} {'#1s':>4} {'n':>4}")
    print(f"  {'-'*68}")

    for strat in sorted(cross_data.keys(),
                        key=lambda s: sum(cross_data[s].get(f'{m}_rank', 99) for m in METRICS_4)):
        d = cross_data[strat]
        rank_sum = sum(d.get(f'{m}_rank', 99) for m in METRICS_4)
        wins = sum(1 for m in METRICS_4 if d.get(f'{m}_rank') == 1)
        ranks_str = "/".join(str(d.get(f'{m}_rank', '?')) for m in METRICS_4)
        marker = " ◄" if strat == 'bidkv' else ""
        print(f"  {DISPLAY.get(strat, strat):<16} "
              f"{d['throughput']:>7.2f} "
              f"{d['slo_pct']:>6.1f}% "
              f"{d['ttft_p95']:>8.0f} "
              f"{d['tpot_p95']:>8.1f} "
              f"{rank_sum:>8d} "
              f"{wins:>4d} "
              f"{d['n_runs']:>4d} "
              f" ({ranks_str}){marker}")

    # ================================================================
    # Section 3: Stability Analysis (standard deviations)
    # ================================================================
    print(f"\n{'='*85}")
    print(f"  STABILITY (Cross-Rate Std Dev)  ({workload_name})")
    print(f"{'='*85}")
    print(f"\n  {'Strategy':<16} {'Thru±':>9} {'SLO±':>8} {'TTFT95±':>10} {'TPOT95±':>10}")
    print(f"  {'-'*55}")
    for strat in sorted(cross_data.keys()):
        d = cross_data[strat]
        print(f"  {DISPLAY.get(strat, strat):<16} "
              f"{d['throughput_std']:>9.3f} "
              f"{d['slo_pct_std']:>7.2f}% "
              f"{d['ttft_p95_std']:>10.0f} "
              f"{d['tpot_p95_std']:>10.1f}")

    # ================================================================
    # Section 4: BidKV Dominance / Reversal Check
    # ================================================================
    if 'bidkv' in cross_data:
        print(f"\n{'='*85}")
        print(f"  BIDKV PAIRWISE COMPARISON  ({workload_name})")
        print(f"{'='*85}")
        bd = cross_data['bidkv']
        for other in sorted(cross_data.keys()):
            if other == 'bidkv':
                continue
            od = cross_data[other]
            wins_b = 0
            wins_o = 0
            details = []
            for m in METRICS_4:
                diff = bd[m] - od[m]
                hb = HIGHER_BETTER[m]
                if (hb and diff > 0) or (not hb and diff < 0):
                    wins_b += 1
                    details.append(f"{m}:BidKV+")
                elif (hb and diff < 0) or (not hb and diff > 0):
                    wins_o += 1
                    details.append(f"{m}:{DISPLAY.get(other, other)[:6]}+")
                else:
                    details.append(f"{m}:tie")
            verdict = "BidKV wins" if wins_b > wins_o else \
                      f"{DISPLAY.get(other, other)} wins" if wins_o > wins_b else "TIE"
            print(f"  vs {DISPLAY.get(other, other):<16}: {wins_b}-{wins_o} ({verdict})  "
                  f"[{', '.join(details)}]")

    return cross_data


def print_combined_summary(mixed_data: dict, lc_data: dict) -> None:
    """Combined paper-readiness assessment."""
    print(f"\n{'#'*90}")
    print(f"#  COMBINED ASSESSMENT — Paper Readiness")
    print(f"{'#'*90}")

    # Strategies present in both
    both = set(mixed_data.keys()) & set(lc_data.keys())
    print(f"\n  Strategies in both workloads: {sorted(both)}")

    if 'bidkv' not in both:
        print("  ERROR: bidkv not in both workloads!")
        return

    # Combined ranking (average rank across both workloads)
    print(f"\n  {'Strategy':<16} {'Mixed RS':>9} {'LC RS':>7} {'Comb RS':>8} {'Mixed #1':>9} {'LC #1':>6}")
    print(f"  {'-'*55}")

    combined = {}
    for strat in sorted(both):
        m_rs = sum(mixed_data[strat].get(f'{m}_rank', 99) for m in METRICS_4)
        l_rs = sum(lc_data[strat].get(f'{m}_rank', 99) for m in METRICS_4)
        m_wins = sum(1 for m in METRICS_4 if mixed_data[strat].get(f'{m}_rank') == 1)
        l_wins = sum(1 for m in METRICS_4 if lc_data[strat].get(f'{m}_rank') == 1)
        combined[strat] = m_rs + l_rs

    for strat in sorted(combined.keys(), key=lambda s: combined[s]):
        m_rs = sum(mixed_data[strat].get(f'{m}_rank', 99) for m in METRICS_4)
        l_rs = sum(lc_data[strat].get(f'{m}_rank', 99) for m in METRICS_4)
        m_wins = sum(1 for m in METRICS_4 if mixed_data[strat].get(f'{m}_rank') == 1)
        l_wins = sum(1 for m in METRICS_4 if lc_data[strat].get(f'{m}_rank') == 1)
        marker = " ◄" if strat == 'bidkv' else ""
        print(f"  {DISPLAY.get(strat, strat):<16} {m_rs:>9} {l_rs:>7} {combined[strat]:>8} "
              f"{m_wins:>9} {l_wins:>6}{marker}")

    # BidKV specific verdict
    bd_mixed = mixed_data['bidkv']
    bd_lc = lc_data['bidkv']

    print(f"\n  {'='*60}")
    print(f"  BidKV Cross-Rate Ranks:")
    print(f"    Mixed:        Thru=#{bd_mixed.get('throughput_rank')}, "
          f"SLO=#{bd_mixed.get('slo_pct_rank')}, "
          f"TTFT95=#{bd_mixed.get('ttft_p95_rank')}, "
          f"TPOT95=#{bd_mixed.get('tpot_p95_rank')}")
    print(f"    Long-Context: Thru=#{bd_lc.get('throughput_rank')}, "
          f"SLO=#{bd_lc.get('slo_pct_rank')}, "
          f"TTFT95=#{bd_lc.get('ttft_p95_rank')}, "
          f"TPOT95=#{bd_lc.get('tpot_p95_rank')}")

    # Weakness analysis
    print(f"\n  BidKV Weaknesses (metrics where rank > 3):")
    for wl_name, data in [('Mixed', bd_mixed), ('LC', bd_lc)]:
        for m in METRICS_4:
            rank = data.get(f'{m}_rank', 99)
            if rank > 3:
                print(f"    {wl_name} {m}: rank #{rank}")

    # Key competitor comparison
    print(f"\n  Key Competitor: Static-Random")
    if 'static-random' in mixed_data and 'static-random' in lc_data:
        sr_m = mixed_data['static-random']
        sr_l = lc_data['static-random']
        print(f"    Mixed:  BidKV thru={bd_mixed['throughput']:.2f} vs SR={sr_m['throughput']:.2f} "
              f"({(bd_mixed['throughput']/sr_m['throughput']-1)*100:+.1f}%)")
        print(f"            BidKV SLO={bd_mixed['slo_pct']:.1f}% vs SR={sr_m['slo_pct']:.1f}% "
              f"({bd_mixed['slo_pct']-sr_m['slo_pct']:+.1f}pp)")
        print(f"            BidKV TTFT95={bd_mixed['ttft_p95']:.0f} vs SR={sr_m['ttft_p95']:.0f} "
              f"({(bd_mixed['ttft_p95']/sr_m['ttft_p95']-1)*100:+.1f}%)")
        print(f"            BidKV TPOT95={bd_mixed['tpot_p95']:.1f} vs SR={sr_m['tpot_p95']:.1f} "
              f"({(bd_mixed['tpot_p95']/sr_m['tpot_p95']-1)*100:+.1f}%)")
        print(f"    LC:     BidKV thru={bd_lc['throughput']:.2f} vs SR={sr_l['throughput']:.2f} "
              f"({(bd_lc['throughput']/sr_l['throughput']-1)*100:+.1f}%)")
        print(f"            BidKV SLO={bd_lc['slo_pct']:.1f}% vs SR={sr_l['slo_pct']:.1f}% "
              f"({bd_lc['slo_pct']-sr_l['slo_pct']:+.1f}pp)")
        print(f"            BidKV TTFT95={bd_lc['ttft_p95']:.0f} vs SR={sr_l['ttft_p95']:.0f} "
              f"({(bd_lc['ttft_p95']/sr_l['ttft_p95']-1)*100:+.1f}%)")
        print(f"            BidKV TPOT95={bd_lc['tpot_p95']:.1f} vs SR={sr_l['tpot_p95']:.1f} "
              f"({(bd_lc['tpot_p95']/sr_l['tpot_p95']-1)*100:+.1f}%)")


def print_per_rate_win_matrix(mixed_groups: dict, lc_groups: dict) -> None:
    """Show BidKV's rank at every individual rate."""
    print(f"\n{'='*85}")
    print(f"  BIDKV PER-RATE WIN MATRIX")
    print(f"{'='*85}")

    for label, groups, rates in [
        ("Mixed", mixed_groups, [2.0, 3.8, 5.7]),
        ("Long-Context", lc_groups, [0.35, 0.5, 0.7]),
    ]:
        strategies = sorted(set(s for s, r in groups.keys()))
        print(f"\n  {label}:")
        print(f"  {'Rate':>6}  {'Thru':>6} {'SLO':>5} {'TTFT95':>7} {'TPOT95':>7} {'RankSum':>8} {'#1':>3} {'Top3':>5}")

        for rate in rates:
            strat_vals = {}
            for strat in strategies:
                key = (strat, rate)
                if key not in groups:
                    continue
                runs = groups[key]
                strat_vals[strat] = {m: avg_runs(runs, m) for m in METRICS_4}

            if 'bidkv' not in strat_vals:
                continue

            ranks = {}
            for m in METRICS_4:
                reverse = HIGHER_BETTER[m]
                ranked = sorted(strat_vals.keys(),
                                key=lambda s: strat_vals[s][m],
                                reverse=reverse)
                for r, s in enumerate(ranked, 1):
                    ranks.setdefault(s, {})[m] = r

            br = ranks.get('bidkv', {})
            rs = sum(br.get(m, 99) for m in METRICS_4)
            n1 = sum(1 for m in METRICS_4 if br.get(m) == 1)
            t3 = sum(1 for m in METRICS_4 if br.get(m, 99) <= 3)
            print(f"  {rate:>6.2f}  "
                  f"#{br.get('throughput','?'):<4} "
                  f"#{br.get('slo_pct','?'):<4} "
                  f"#{br.get('ttft_p95','?'):<6} "
                  f"#{br.get('tpot_p95','?'):<6} "
                  f"{rs:>8} "
                  f"{n1:>3} "
                  f"{t3:>4}/4")


def print_scenario_check(mixed_data: dict, lc_data: dict) -> None:
    """Check Scenario A/B switching rules."""
    print(f"\n{'='*85}")
    print(f"  SCENARIO A/B CHECK (RULE SCENARIO-SWITCH)")
    print(f"{'='*85}")
    print(f"\n  Rule: BidKV SLO Δ_avg ≥ 10pp over PE baseline + no reversal")

    if 'bidkv' not in mixed_data or 'preempt-evict' not in mixed_data:
        print("  SKIP: missing data")
        return

    for wl, data, label in [('Mixed', mixed_data, '300ms'), ('LC', lc_data, '2000ms')]:
        if 'bidkv' not in data or 'preempt-evict' not in data:
            continue
        delta = data['bidkv']['slo_pct'] - data['preempt-evict']['slo_pct']
        print(f"\n  {wl} (SLO {label}):")
        print(f"    BidKV SLO = {data['bidkv']['slo_pct']:.1f}%")
        print(f"    PE SLO    = {data['preempt-evict']['slo_pct']:.1f}%")
        print(f"    Δ_avg     = {delta:+.1f}pp  {'≥ 10pp ✓ PASS' if delta >= 10 else '< 10pp ✗ FAIL'}")


if __name__ == '__main__':
    mixed_groups = load_dir('/home/cyb/bidkv/results/vllm_v8_full_validation')
    lc_groups = load_dir('/home/cyb/bidkv/results/vllm_v8_long_context')

    mixed_cross = analyze_workload(mixed_groups, "Mixed (1000 reqs)", "TTFT ≤ 300ms")
    lc_cross = analyze_workload(lc_groups, "Long-Context (500 reqs)", "TTFT ≤ 2000ms")

    print_per_rate_win_matrix(mixed_groups, lc_groups)
    print_combined_summary(mixed_cross, lc_cross)
    print_scenario_check(mixed_cross, lc_cross)
