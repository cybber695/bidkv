#!/usr/bin/env python3
"""Generate paper figures from vllm_v8_full_validation results.

Produces:
  - paper/figures/fig3_rate_sensitivity.{pdf,png}
  - paper/figures/fig5_compress_coverage.{pdf,png}

Reads from results/vllm_v8_full_validation/ (mixed, 5 strategies × 3 rates × 3 runs).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "vllm_v8_full_validation"
FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

STRATEGIES = [
    "preempt-evict",
    "preempt-evict-sjf",
    "static-random",
    "h2o-style",
    "bidkv",
]
STRATEGY_DISPLAY = {
    "preempt-evict": "PE",
    "preempt-evict-sjf": "PE-SJF",
    "static-random": "Static-Random",
    "h2o-style": "Largest-First",
    "bidkv": "BidKV",
}
RATES = [2.0, 3.8, 5.7]
SLO_TTFT_MS = 300.0

COLORS = {
    "preempt-evict": "#7f7f7f",
    "preempt-evict-sjf": "#aec7e8",
    "static-random": "#1f77b4",
    "h2o-style": "#ff7f0e",
    "bidkv": "#d62728",
}
MARKERS = {
    "preempt-evict": "s",
    "preempt-evict-sjf": "^",
    "static-random": "v",
    "h2o-style": "D",
    "bidkv": "o",
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(len(s) * p / 100)
    return s[min(k, len(s) - 1)]


def load_run(filepath: Path) -> dict:
    with open(filepath) as f:
        d = json.load(f)
    rr = d.get("request_results", [])
    am = d.get("adapter_metrics", {})
    ok = [r for r in rr if not r.get("error")]
    ttfts = sorted(r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None)
    tpots = []
    for r in ok:
        ct = r.get("completion_tokens", 0)
        ttft = r.get("ttft_ms")
        tot = r.get("total_latency_ms")
        if ct > 1 and ttft is not None and tot is not None and tot > ttft:
            tpots.append((tot - ttft) / (ct - 1))
    tpots.sort()
    return {
        "strategy": d.get("strategy", ""),
        "rate": d.get("request_rate", 0),
        "throughput": d["summary"]["throughput_rps"],
        "ttft_p95": percentile(ttfts, 95),
        "tpot_p95": percentile(tpots, 95),
        "slo_pct": sum(1 for t in ttfts if t <= SLO_TTFT_MS) / len(ttfts) * 100 if ttfts else 0,
        "evictions": am.get("total_evictions", am.get("total_compressions", 0)),
        "tokens_freed": am.get("total_tokens_freed", 0),
    }


def load_all() -> dict[tuple[str, float], list[dict]]:
    from collections import defaultdict
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("candidate"):
            continue
        row = load_run(f)
        if row["strategy"] in STRATEGIES:
            groups[(row["strategy"], row["rate"])].append(row)
    return groups


def avg(runs: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in runs)


def save_fig(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    print(f"  Saved {stem}.{{pdf,png}}")


def generate_fig3(groups: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.8, "lines.markersize": 7,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for strat in STRATEGIES:
        rd = []
        for rate in RATES:
            runs = groups.get((strat, rate), [])
            if runs:
                rd.append((rate, avg(runs, "throughput"), avg(runs, "ttft_p95")))
        if not rd:
            continue
        rs = [d[0] for d in rd]
        thpt = [d[1] for d in rd]
        ttft = [d[2] for d in rd]
        kw = dict(color=COLORS[strat], marker=MARKERS[strat],
                  label=STRATEGY_DISPLAY[strat],
                  linewidth=2.5 if strat == "bidkv" else 1.8,
                  zorder=10 if strat == "bidkv" else 5,
                  markeredgecolor="white", markeredgewidth=0.5)
        ax1.plot(rs, thpt, **kw)
        ax2.plot(rs, ttft, **kw)

    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("Throughput (req/s)")
    ax1.set_xticks(RATES)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_title("(a) Throughput vs. Request Rate", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.set_xlabel("Request Rate (req/s)")
    ax2.set_ylabel("TTFT P95 (ms)")
    ax2.set_yscale("log")
    ax2.set_xticks(RATES)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_title("(b) TTFT P95 vs. Request Rate", fontsize=10)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout(w_pad=3)
    save_fig(fig, "fig3_rate_sensitivity")
    plt.close(fig)


def generate_fig5(groups: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    rate = 3.8
    strats_ok, labels, evicts, freed = [], [], [], []
    for strat in STRATEGIES:
        runs = groups.get((strat, rate), [])
        if not runs:
            continue
        strats_ok.append(strat)
        labels.append(STRATEGY_DISPLAY[strat])
        evicts.append(avg(runs, "evictions"))
        freed.append(avg(runs, "tokens_freed") / 1000)

    n = len(strats_ok)
    x = list(range(n))
    bw = 0.35
    bar_c = [COLORS[s] for s in strats_ok]

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax2r = ax1.twinx()

    ax1.bar([i - bw / 2 for i in x], evicts, bw,
            color=bar_c, alpha=0.85, edgecolor="black", linewidth=0.5,
            label="Proactive Evictions")
    ax2r.bar([i + bw / 2 for i in x], freed, bw,
             color=bar_c, alpha=0.4, edgecolor="black", linewidth=0.5,
             hatch="//", label="Tokens Freed (×1000)")

    ax1.set_xlabel("Strategy")
    ax1.set_ylabel("Proactive Eviction Count")
    ax2r.set_ylabel("Tokens Freed (×1000)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")

    me = max(evicts) if evicts and max(evicts) > 0 else 1
    mf = max(freed) if freed and max(freed) > 0 else 1
    for i, (ev, fr) in enumerate(zip(evicts, freed)):
        if ev > 0:
            ax1.text(i - bw / 2, ev + me * 0.02, f"{ev:.0f}",
                     ha="center", va="bottom", fontsize=8)
        if fr > 0:
            ax2r.text(i + bw / 2, fr + mf * 0.02, f"{fr:.0f}k",
                      ha="center", va="bottom", fontsize=8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.2, linestyle="--")

    fig.tight_layout()
    save_fig(fig, "fig5_compress_coverage")
    plt.close(fig)


def main() -> None:
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: {RESULTS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_all()
    print(f"Loaded {sum(len(v) for v in groups.values())} runs "
          f"across {len(groups)} (strategy, rate) groups.\n")

    for strat in STRATEGIES:
        for rate in RATES:
            runs = groups.get((strat, rate), [])
            if runs:
                print(f"  {STRATEGY_DISPLAY[strat]:<15} rate={rate}: "
                      f"Thru={avg(runs, 'throughput'):.2f}, "
                      f"SLO={avg(runs, 'slo_pct'):.1f}%, "
                      f"TTFT={avg(runs, 'ttft_p95'):.0f}, "
                      f"TPOT={avg(runs, 'tpot_p95'):.1f}, "
                      f"Evict={avg(runs, 'evictions'):.0f}, "
                      f"Freed={avg(runs, 'tokens_freed'):.0f}")

    print()
    generate_fig3(groups)
    generate_fig5(groups)
    print("\nDone.")


if __name__ == "__main__":
    main()
