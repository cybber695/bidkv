"""Deep behavioral diagnosis: BidKV vs Static-Random.

Multi-dimensional comparison to find the TRUE bottleneck.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict


def load_all_requests(result_dir: str, strategy: str, rate: float) -> list[dict]:
    """Load all request-level data for a strategy×rate across all runs."""
    all_reqs = []
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        d = json.load(open(os.path.join(result_dir, fn)))
        if d.get('strategy') != strategy or abs(d.get('request_rate', 0) - rate) > 0.01:
            continue
        run_idx = d.get('run_index', 0)
        for r in d['request_results']:
            r['_run'] = run_idx
        all_reqs.extend(d['request_results'])
    return all_reqs


def load_adapter_metrics(result_dir: str, strategy: str, rate: float) -> list[dict]:
    """Load adapter_metrics for a strategy×rate across all runs."""
    mets = []
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        d = json.load(open(os.path.join(result_dir, fn)))
        if d.get('strategy') != strategy or abs(d.get('request_rate', 0) - rate) > 0.01:
            continue
        am = d.get('adapter_metrics', {})
        am['_duration'] = d.get('duration_s', 0)
        am['_summary'] = d.get('summary', {})
        mets.append(am)
    return mets


def pct(data: list, p: float) -> float:
    if not data:
        return float('nan')
    data_s = sorted(data)
    idx = int(len(data_s) * p / 100)
    return data_s[min(idx, len(data_s) - 1)]


def analyze_latency_outliers(reqs: list[dict], label: str) -> None:
    """Analyze extreme latency outliers."""
    ok = [r for r in reqs if not r.get('error')]
    if not ok:
        return

    # Total latency distribution
    lats = sorted([r['total_latency_ms'] for r in ok if r.get('total_latency_ms')])
    ttfts = sorted([r['ttft_ms'] for r in ok if r.get('ttft_ms') is not None])

    print(f"\n  [{label}] Latency Distribution (n={len(ok)})")
    print(f"    E2E:  p50={pct(lats,50):.0f}  p90={pct(lats,90):.0f}  "
          f"p95={pct(lats,95):.0f}  p99={pct(lats,99):.0f}  "
          f"max={max(lats):.0f} ms")
    print(f"    TTFT: p50={pct(ttfts,50):.0f}  p90={pct(ttfts,90):.0f}  "
          f"p95={pct(ttfts,95):.0f}  p99={pct(ttfts,99):.0f}  "
          f"max={max(ttfts):.0f} ms")

    # Count extreme outliers
    thresholds = [10000, 20000, 30000, 60000]
    for t in thresholds:
        e2e_cnt = sum(1 for l in lats if l > t)
        ttft_cnt = sum(1 for t2 in ttfts if t2 > t)
        if e2e_cnt > 0 or ttft_cnt > 0:
            print(f"    >{t/1000:.0f}s:  E2E={e2e_cnt} ({e2e_cnt/len(lats)*100:.1f}%)  "
                  f"TTFT={ttft_cnt} ({ttft_cnt/len(ttfts)*100:.1f}%)")

    # Characterize the worst 1% of requests by E2E
    worst_n = max(1, len(ok) // 100)
    worst_by_e2e = sorted(ok, key=lambda r: r.get('total_latency_ms', 0), reverse=True)[:worst_n]

    print(f"\n    Worst {worst_n} requests (by E2E):")
    ct_list = [r.get('completion_tokens', 0) for r in worst_by_e2e]
    ttft_list = [r.get('ttft_ms', 0) for r in worst_by_e2e]
    e2e_list = [r.get('total_latency_ms', 0) for r in worst_by_e2e]
    decode_list = [e - t for e, t in zip(e2e_list, ttft_list)]

    print(f"      Completion tokens: min={min(ct_list)} avg={statistics.mean(ct_list):.0f} max={max(ct_list)}")
    print(f"      TTFT:    min={min(ttft_list):.0f} avg={statistics.mean(ttft_list):.0f} max={max(ttft_list):.0f}")
    print(f"      E2E:     min={min(e2e_list):.0f} avg={statistics.mean(e2e_list):.0f} max={max(e2e_list):.0f}")
    print(f"      Decode:  min={min(decode_list):.0f} avg={statistics.mean(decode_list):.0f} max={max(decode_list):.0f}")

    # Worst 1% by TTFT
    worst_by_ttft = sorted(ok, key=lambda r: r.get('ttft_ms', 0) or 0, reverse=True)[:worst_n]
    ct_list2 = [r.get('completion_tokens', 0) for r in worst_by_ttft]
    ttft_list2 = [r.get('ttft_ms', 0) for r in worst_by_ttft]

    print(f"\n    Worst {worst_n} requests (by TTFT):")
    print(f"      Completion tokens: min={min(ct_list2)} avg={statistics.mean(ct_list2):.0f} max={max(ct_list2)}")
    print(f"      TTFT:    min={min(ttft_list2):.0f} avg={statistics.mean(ttft_list2):.0f} max={max(ttft_list2):.0f}")


def analyze_completion_token_distribution(reqs: list[dict], label: str) -> None:
    """Analyze completion tokens and decode efficiency by buckets."""
    ok = [r for r in reqs if not r.get('error')]
    if not ok:
        return

    # Bucket by completion tokens
    buckets = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 9999)]
    print(f"\n  [{label}] Decode Efficiency by Completion Length")
    print(f"    {'Bucket':<12} {'Count':>6} {'%':>6} {'TPOT_p50':>9} {'TPOT_p95':>9} "
          f"{'TTFT_p50':>9} {'TTFT_p95':>9} {'E2E_p95':>9}")

    for lo, hi in buckets:
        bucket_reqs = [r for r in ok
                       if lo <= r.get('completion_tokens', 0) < hi
                       and r.get('completion_tokens', 0) > 1
                       and r.get('ttft_ms') is not None
                       and r.get('total_latency_ms') is not None]
        if not bucket_reqs:
            continue

        tpots = sorted([(r['total_latency_ms'] - r['ttft_ms']) / (r['completion_tokens'] - 1)
                        for r in bucket_reqs])
        ttfts = sorted([r['ttft_ms'] for r in bucket_reqs])
        e2es = sorted([r['total_latency_ms'] for r in bucket_reqs])

        label_str = f"[{lo},{hi})" if hi < 9999 else f"[{lo},∞)"
        print(f"    {label_str:<12} {len(bucket_reqs):>6} "
              f"{len(bucket_reqs)/len(ok)*100:>5.1f}% "
              f"{pct(tpots,50):>9.1f} {pct(tpots,95):>9.1f} "
              f"{pct(ttfts,50):>9.0f} {pct(ttfts,95):>9.0f} "
              f"{pct(e2es,95):>9.0f}")


def analyze_time_phases(reqs: list[dict], label: str) -> None:
    """Decompose request lifecycle into wait (TTFT) + decode phases."""
    ok = [r for r in reqs if not r.get('error') and r.get('ttft_ms') is not None
          and r.get('total_latency_ms') is not None and r.get('completion_tokens', 0) > 0]
    if not ok:
        return

    waits = [r['ttft_ms'] for r in ok]
    decodes = [r['total_latency_ms'] - r['ttft_ms'] for r in ok]
    ratios = [w / max(d, 1) for w, d in zip(waits, decodes)]

    print(f"\n  [{label}] Phase Decomposition (n={len(ok)})")
    print(f"    Wait (TTFT):  avg={statistics.mean(waits):.0f}  "
          f"p50={pct(waits,50):.0f}  p95={pct(waits,95):.0f}  p99={pct(waits,99):.0f}")
    print(f"    Decode:       avg={statistics.mean(decodes):.0f}  "
          f"p50={pct(decodes,50):.0f}  p95={pct(decodes,95):.0f}  p99={pct(decodes,99):.0f}")
    print(f"    Wait/Decode:  avg={statistics.mean(ratios):.3f}  "
          f"p50={pct(ratios,50):.3f}  p99={pct(ratios,99):.3f}")


def analyze_arrival_position_effect(reqs: list[dict], label: str) -> None:
    """TTFT by arrival position (early vs late arrivals)."""
    ok = [r for r in reqs if not r.get('error') and r.get('ttft_ms') is not None
          and r.get('submit_time') is not None]
    if not ok:
        return

    # Sort by submit_time within each run
    runs = defaultdict(list)
    for r in ok:
        runs[r.get('_run', 0)].append(r)

    quartile_ttft = defaultdict(list)
    quartile_e2e = defaultdict(list)
    for run_reqs in runs.values():
        run_reqs.sort(key=lambda r: r['submit_time'])
        n = len(run_reqs)
        for i, r in enumerate(run_reqs):
            q = min(3, int(i / n * 4))  # 0,1,2,3
            quartile_ttft[q].append(r['ttft_ms'])
            if r.get('total_latency_ms') is not None:
                quartile_e2e[q].append(r['total_latency_ms'])

    print(f"\n  [{label}] TTFT by Arrival Quartile")
    print(f"    {'Q':>3} {'Count':>6} {'TTFT_p50':>9} {'TTFT_p95':>9} {'TTFT_p99':>9} "
          f"{'E2E_p50':>9} {'E2E_p95':>9}")
    for q in range(4):
        if q not in quartile_ttft:
            continue
        tt = sorted(quartile_ttft[q])
        ee = sorted(quartile_e2e.get(q, []))
        print(f"    Q{q}  {len(tt):>6} "
              f"{pct(tt,50):>9.0f} {pct(tt,95):>9.0f} {pct(tt,99):>9.0f} "
              f"{pct(ee,50):>9.0f} {pct(ee,95):>9.0f}")


def analyze_tpot_per_request(reqs_a: list[dict], reqs_b: list[dict],
                             label_a: str, label_b: str) -> None:
    """Side-by-side TPOT distribution comparison."""
    def compute_tpot(reqs):
        tpots = []
        for r in reqs:
            if (not r.get('error') and r.get('completion_tokens', 0) > 1
                    and r.get('ttft_ms') is not None
                    and r.get('total_latency_ms') is not None):
                tpot = (r['total_latency_ms'] - r['ttft_ms']) / (r['completion_tokens'] - 1)
                tpots.append(tpot)
        return sorted(tpots)

    tpots_a = compute_tpot(reqs_a)
    tpots_b = compute_tpot(reqs_b)

    print(f"\n  TPOT Distribution Comparison:")
    print(f"    {'Percentile':>12} {label_a:>14} {label_b:>14} {'Delta':>10} {'Pct':>8}")
    for p in [50, 75, 90, 95, 99]:
        va = pct(tpots_a, p)
        vb = pct(tpots_b, p)
        d = va - vb
        dp = d / max(abs(vb), 0.01) * 100
        print(f"    p{p:<11} {va:>14.1f} {vb:>14.1f} {d:>+10.1f} {dp:>+7.1f}%")


def analyze_concurrent_pressure(reqs: list[dict], label: str, duration_s: float) -> None:
    """Simulate concurrent load to understand queuing behavior."""
    ok = [r for r in reqs if r.get('submit_time') is not None and r.get('finish_time') is not None
          and not r.get('error')]
    if not ok:
        return

    # Time-window concurrency
    t_min = min(r['submit_time'] for r in ok)
    t_max = max(r['finish_time'] for r in ok)

    window = 5.0  # 5 second windows
    slots = int((t_max - t_min) / window) + 1

    concurrent = [0] * slots
    waiting_at = [0] * slots

    for r in ok:
        s_slot = int((r['submit_time'] - t_min) / window)
        f_slot = int((r['finish_time'] - t_min) / window)
        ftt_slot = int(((r.get('first_token_time') or r['submit_time']) - t_min) / window)

        for slot in range(s_slot, min(f_slot + 1, slots)):
            concurrent[slot] += 1
        for slot in range(s_slot, min(ftt_slot + 1, slots)):
            waiting_at[slot] += 1

    # Only look at active period (middle 80%)
    active_start = int(slots * 0.1)
    active_end = int(slots * 0.9)
    active_conc = concurrent[active_start:active_end]
    active_wait = waiting_at[active_start:active_end]

    if active_conc:
        print(f"\n  [{label}] Concurrency ({window:.0f}s windows, middle 80%)")
        print(f"    Active:  avg={statistics.mean(active_conc):.1f}  "
              f"max={max(active_conc)}")
        if active_wait:
            print(f"    Waiting: avg={statistics.mean(active_wait):.1f}  "
                  f"max={max(active_wait)}")


def analyze_slow_requests_characteristics(reqs_a: list[dict], reqs_b: list[dict],
                                          label_a: str, label_b: str,
                                          top_n: int = 30) -> None:
    """Deep dive into the slowest requests of each strategy."""
    def get_ok(reqs):
        return [r for r in reqs if not r.get('error')
                and r.get('ttft_ms') is not None
                and r.get('total_latency_ms') is not None
                and r.get('completion_tokens', 0) > 0]

    ok_a = get_ok(reqs_a)
    ok_b = get_ok(reqs_b)

    # Sort by E2E descending
    ok_a.sort(key=lambda r: r['total_latency_ms'], reverse=True)
    ok_b.sort(key=lambda r: r['total_latency_ms'], reverse=True)

    print(f"\n  Slowest {top_n} Requests Comparison")
    for label, slow in [(label_a, ok_a[:top_n]), (label_b, ok_b[:top_n])]:
        e2es = [r['total_latency_ms'] for r in slow]
        ttfts = [r['ttft_ms'] for r in slow]
        cts = [r['completion_tokens'] for r in slow]
        decodes = [e - t for e, t in zip(e2es, ttfts)]
        tpots = [(e - t) / max(c - 1, 1) for e, t, c in zip(e2es, ttfts, cts)]

        print(f"\n    [{label}] Top {top_n} slowest:")
        print(f"      E2E range:    {min(e2es):.0f} – {max(e2es):.0f} ms")
        print(f"      TTFT range:   {min(ttfts):.0f} – {max(ttfts):.0f} ms")
        print(f"      Decode range: {min(decodes):.0f} – {max(decodes):.0f} ms")
        print(f"      Tokens range: {min(cts)} – {max(cts)}")
        print(f"      TPOT range:   {min(tpots):.1f} – {max(tpots):.1f} ms/tok")

        # What fraction are TTFT-dominated vs Decode-dominated?
        ttft_dom = sum(1 for t, d in zip(ttfts, decodes) if t > d)
        print(f"      TTFT-dominated: {ttft_dom}/{len(slow)} "
              f"({ttft_dom/len(slow)*100:.0f}%)")

        # Token length distribution of slow requests
        short = sum(1 for c in cts if c < 50)
        med = sum(1 for c in cts if 50 <= c < 200)
        long_ = sum(1 for c in cts if c >= 200)
        print(f"      Token dist: short(<50)={short} med(50-200)={med} long(≥200)={long_}")


def analyze_completion_rate_over_time(reqs: list[dict], label: str) -> None:
    """Track how completion rate changes over the experiment duration."""
    ok = [r for r in reqs if not r.get('error') and r.get('finish_time') is not None]
    if not ok:
        return

    # Group by run
    runs = defaultdict(list)
    for r in ok:
        runs[r.get('_run', 0)].append(r)

    print(f"\n  [{label}] Completion Rate Over Time (30s windows)")

    for run_id in sorted(runs.keys())[:1]:  # Just first run for clarity
        run_reqs = sorted(runs[run_id], key=lambda r: r['finish_time'])
        if not run_reqs:
            continue
        t0 = min(r.get('submit_time', r['finish_time']) for r in run_reqs)
        t_max = max(r['finish_time'] for r in run_reqs)

        window = 30.0
        n_windows = int((t_max - t0) / window) + 1

        completions = [0] * n_windows
        slow_ttft = [0] * n_windows  # TTFT > 1000ms
        for r in run_reqs:
            w = min(int((r['finish_time'] - t0) / window), n_windows - 1)
            completions[w] += 1
            if r.get('ttft_ms', 0) and r['ttft_ms'] > 1000:
                slow_ttft[w] += 1

        print(f"    Run {run_id}:")
        print(f"    {'Window':>8} {'Compl':>7} {'SlowTTFT':>9}")
        for w in range(min(n_windows, 20)):  # First 20 windows
            t = w * window
            print(f"    {t:>6.0f}s  {completions[w]:>7} {slow_ttft[w]:>9}")


def run_comparison(result_dir: str, rate: float, workload: str) -> None:
    """Run full comparison between BidKV and Static-Random at a given rate."""
    slo_label = "2000ms" if 'long' in workload else "300ms"

    print(f"\n{'#'*90}")
    print(f"#  DEEP BEHAVIORAL DIAGNOSIS: BidKV vs Static-Random")
    print(f"#  Workload={workload}  Rate={rate}  SLO={slo_label}")
    print(f"{'#'*90}")

    reqs_b = load_all_requests(result_dir, 'bidkv', rate)
    reqs_r = load_all_requests(result_dir, 'static-random', rate)
    mets_b = load_adapter_metrics(result_dir, 'bidkv', rate)
    mets_r = load_adapter_metrics(result_dir, 'static-random', rate)

    if not reqs_b or not reqs_r:
        print(f"  Missing data for rate={rate}")
        return

    # Adapter metrics comparison
    print(f"\n{'='*80}")
    print(f"  ADAPTER METRICS")
    print(f"{'='*80}")
    for label, mets in [("BidKV", mets_b), ("StaticRandom", mets_r)]:
        if not mets:
            continue
        avg_ev = statistics.mean([m.get('total_evictions', m.get('total_compressions', 0)) for m in mets])
        avg_freed = statistics.mean([m.get('total_tokens_freed', 0) for m in mets])
        avg_pressure = statistics.mean([m.get('total_pressure_events', 0) for m in mets])
        avg_complete = statistics.mean([m.get('total_requests_completed', 0) for m in mets])
        avg_steps = statistics.mean([m.get('total_decode_steps', 0) for m in mets])
        avg_dur = statistics.mean([m.get('_duration', 0) for m in mets])
        avg_thru = statistics.mean([m.get('_summary', {}).get('throughput_rps', 0) for m in mets])
        print(f"\n  [{label}] (avg over {len(mets)} runs)")
        print(f"    Evictions:       {avg_ev:.1f}")
        print(f"    Tokens freed:    {avg_freed:.0f}")
        print(f"    Pressure events: {avg_pressure:.1f}")
        print(f"    Completed:       {avg_complete:.0f}")
        print(f"    Decode steps:    {avg_steps:.0f}")
        print(f"    Duration:        {avg_dur:.0f}s")
        print(f"    Throughput:      {avg_thru:.2f} req/s")
        if avg_ev > 0:
            print(f"    Freed/eviction:  {avg_freed/avg_ev:.0f} tokens")

    # Section 1: Latency outliers
    print(f"\n{'='*80}")
    print(f"  LATENCY OUTLIER ANALYSIS")
    print(f"{'='*80}")
    analyze_latency_outliers(reqs_b, "BidKV")
    analyze_latency_outliers(reqs_r, "Static-Random")

    # Section 2: TPOT distribution
    print(f"\n{'='*80}")
    print(f"  TPOT DISTRIBUTION")
    print(f"{'='*80}")
    analyze_tpot_per_request(reqs_b, reqs_r, "BidKV", "Static-Random")

    # Section 3: Completion token effects
    print(f"\n{'='*80}")
    print(f"  DECODE EFFICIENCY BY COMPLETION LENGTH")
    print(f"{'='*80}")
    analyze_completion_token_distribution(reqs_b, "BidKV")
    analyze_completion_token_distribution(reqs_r, "Static-Random")

    # Section 4: Phase decomposition
    print(f"\n{'='*80}")
    print(f"  PHASE DECOMPOSITION")
    print(f"{'='*80}")
    analyze_time_phases(reqs_b, "BidKV")
    analyze_time_phases(reqs_r, "Static-Random")

    # Section 5: Arrival position
    print(f"\n{'='*80}")
    print(f"  TTFT BY ARRIVAL POSITION")
    print(f"{'='*80}")
    analyze_arrival_position_effect(reqs_b, "BidKV")
    analyze_arrival_position_effect(reqs_r, "Static-Random")

    # Section 6: Concurrency
    print(f"\n{'='*80}")
    print(f"  CONCURRENT LOAD")
    print(f"{'='*80}")
    for mets in mets_b[:1]:
        dur = mets.get('_duration', 300)
        analyze_concurrent_pressure(reqs_b[:len(reqs_b)//len(mets_b)], "BidKV", dur)
    for mets in mets_r[:1]:
        dur = mets.get('_duration', 300)
        analyze_concurrent_pressure(reqs_r[:len(reqs_r)//len(mets_r)], "Static-Random", dur)

    # Section 7: Slowest request deep dive
    print(f"\n{'='*80}")
    print(f"  SLOWEST REQUEST DEEP DIVE")
    print(f"{'='*80}")
    analyze_slow_requests_characteristics(reqs_b, reqs_r, "BidKV", "Static-Random", top_n=30)

    # Section 8: Completion rate over time
    print(f"\n{'='*80}")
    print(f"  COMPLETION RATE TIMELINE")
    print(f"{'='*80}")
    analyze_completion_rate_over_time(reqs_b, "BidKV")
    analyze_completion_rate_over_time(reqs_r, "Static-Random")


if __name__ == '__main__':
    mixed_dir = '/home/cyb/bidkv/results/vllm_v8_full_validation'
    lc_dir = '/home/cyb/bidkv/results/vllm_v8_long_context'

    # Mixed: all 3 rates
    for rate in [2.0, 3.8, 5.7]:
        run_comparison(mixed_dir, rate, 'mixed')

    # Long-context: all 3 rates
    for rate in [0.35, 0.5, 0.7]:
        run_comparison(lc_dir, rate, 'long_context')
