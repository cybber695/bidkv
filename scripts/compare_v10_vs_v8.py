"""Compare v10 optimization results vs v8 baseline."""
from __future__ import annotations

import json
import os


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
        'evictions': evictions,
        'tokens_freed': am.get('total_tokens_freed', 0),
        'pressure_events': am.get('total_pressure_events', 0),
        'strategy': d.get('strategy', ''),
        'rate': d.get('request_rate', 0),
    }


def show_comparison(v10_file: str, v8_dir: str, rate: float) -> None:
    v10 = load_run(v10_file)

    # Load v8 baseline (3 runs average)
    v8_runs = []
    for fn in os.listdir(v8_dir):
        if fn.startswith('bidkv__mixed__rate') and fn.endswith('.json'):
            d = json.load(open(os.path.join(v8_dir, fn)))
            if abs(d.get('request_rate', 0) - rate) < 0.01:
                v8_runs.append(load_run(os.path.join(v8_dir, fn)))

    if not v8_runs:
        print(f"  No v8 baseline data for rate={rate}")
        return

    import statistics
    v8_avg = {k: statistics.mean([r[k] for r in v8_runs])
              for k in ['throughput', 'slo_pct', 'ttft_p50', 'ttft_p95', 'ttft_p99',
                        'tpot_p50', 'tpot_p95', 'tpot_p99', 'evictions', 'tokens_freed',
                        'pressure_events', 'ok_count', 'fail_count']}

    # Also load competitor baselines for context
    competitors = {}
    for fn in os.listdir(v8_dir):
        if not fn.endswith('.json') or fn.startswith('candidate') or fn.startswith('bidkv'):
            continue
        d = json.load(open(os.path.join(v8_dir, fn)))
        if abs(d.get('request_rate', 0) - rate) < 0.01:
            strat = d['strategy']
            if strat not in competitors:
                competitors[strat] = []
            competitors[strat].append(load_run(os.path.join(v8_dir, fn)))

    comp_avg = {}
    for strat, runs in competitors.items():
        comp_avg[strat] = {k: statistics.mean([r[k] for r in runs])
                           for k in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']}

    print(f"\n{'='*80}")
    print(f"  v10 vs v8 COMPARISON — Rate={rate}")
    print(f"{'='*80}")
    print(f"\n{'Metric':<14} {'v10':>10} {'v8(avg3)':>10} {'Delta':>10} {'pct':>8}")
    print(f"{'-'*52}")
    for metric, higher_better in [
        ('throughput', True), ('slo_pct', True),
        ('ttft_p50', False), ('ttft_p95', False), ('ttft_p99', False),
        ('tpot_p50', False), ('tpot_p95', False), ('tpot_p99', False),
        ('evictions', None), ('tokens_freed', None),
        ('pressure_events', None),
    ]:
        v10_val = v10[metric]
        v8_val = v8_avg[metric]
        delta = v10_val - v8_val
        pct = delta / max(abs(v8_val), 0.01) * 100

        # Format: highlight improvements in green direction
        if higher_better is True:
            sign = '+' if delta > 0 else ''
            good = delta > 0
        elif higher_better is False:
            sign = '' if delta > 0 else ''
            good = delta < 0
        else:
            sign = '+' if delta > 0 else ''
            good = None

        fmt = f"  {metric:<14} {v10_val:>10.1f} {v8_val:>10.1f} {sign}{delta:>+9.1f} {pct:>+7.1f}%"
        if good is True:
            fmt += " ✓"
        elif good is False:
            fmt += " ✗"
        print(fmt)

    print(f"\n  Failures: v10={v10['fail_count']}, v8_avg={v8_avg['fail_count']:.0f}")

    # Context: where does v10 rank vs all strategies?
    print(f"\n--- v10 BidKV vs v8 Competitors at rate={rate} ---")
    all_strats = {'v10-bidkv': {k: v10[k] for k in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']},
                  'v8-bidkv': {k: v8_avg[k] for k in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']}}
    all_strats.update(comp_avg)

    print(f"  {'Strategy':<22} {'Thru':>6} {'SLO%':>7} {'TTFT95':>8} {'TPOT95':>8}")
    for metric in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95']:
        reverse = metric in ('throughput', 'slo_pct')
        ranked = sorted(all_strats.items(), key=lambda x: x[1][metric], reverse=reverse)
        for rank, (strat, _) in enumerate(ranked, 1):
            all_strats[strat][f'{metric}_rank'] = rank

    for strat in sorted(all_strats.keys()):
        m = all_strats[strat]
        print(f"  {strat:<22} {m['throughput']:>6.2f} {m['slo_pct']:>6.1f}% "
              f"{m['ttft_p95']:>8.0f} {m['tpot_p95']:>8.1f} "
              f"(R: {m.get('throughput_rank','?')}/{m.get('slo_pct_rank','?')}/"
              f"{m.get('ttft_p95_rank','?')}/{m.get('tpot_p95_rank','?')})")


if __name__ == '__main__':
    v8_dir = '/home/cyb/bidkv/results/vllm_v8_full_validation'
    v10_dir = '/home/cyb/bidkv/results/vllm_v10_test'

    for rate in [3.8, 5.7]:
        fn = f'bidkv__mixed__rate{rate}__r0.json'
        path = os.path.join(v10_dir, fn)
        if os.path.exists(path):
            show_comparison(path, v8_dir, rate)
