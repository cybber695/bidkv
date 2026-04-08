#!/usr/bin/env python3
"""Generate paper figures from vllm_v8_full_validation results.

Produces:
  - paper/figures/fig1_intro_evidence_panel_b.{pdf,png}  (scatter: LIFO blind selection)
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
FIG3_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "vllm_fig3_mixed_rate38"
FIG3_RATE57_DIR = Path(__file__).resolve().parent.parent / "results" / "vllm_fig3_mixed_rate57"
FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

STRATEGIES = [
    "preempt-evict",
    "preempt-evict-sjf",
    "static-random",
    "largest-first",
    "bidkv",
]
STRATEGY_DISPLAY = {
    "preempt-evict": "PE",
    "preempt-evict-sjf": "PE-SJF",
    "static-random": "Static-Random",
    "largest-first": "Largest-First",
    "h2o-style": "Largest-First",
    "bidkv": "BidKV",
}
RATES = [2.0, 3.8, 5.7]
SLO_TTFT_MS = 300.0

COLORS = {
    "preempt-evict": "#7f7f7f",
    "preempt-evict-sjf": "#aec7e8",
    "static-random": "#1f77b4",
    "largest-first": "#ff7f0e",
    "h2o-style": "#ff7f0e",
    "bidkv": "#d62728",
}
MARKERS = {
    "preempt-evict": "s",
    "preempt-evict-sjf": "^",
    "static-random": "v",
    "largest-first": "D",
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
    """Load all runs, using vllm_fig3_mixed_rate57 data for rate=5.7 (overrides v8)."""
    from collections import defaultdict
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    # Load v8 baseline for rate=2.0 and 3.8
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("candidate"):
            continue
        row = load_run(f)
        strat = row["strategy"]
        # map legacy h2o-style to largest-first
        if strat == "h2o-style":
            strat = "largest-first"
            row["strategy"] = "largest-first"
        if strat in STRATEGIES and row["rate"] != 5.7:
            groups[(strat, row["rate"])].append(row)
    # Override rate=5.7 with updated data from vllm_fig3_mixed_rate57
    if FIG3_RATE57_DIR.is_dir():
        for f in sorted(FIG3_RATE57_DIR.glob("*.json")):
            if f.name.startswith("candidate"):
                continue
            row = load_run(f)
            strat = row["strategy"]
            if strat == "h2o-style":
                strat = "largest-first"
                row["strategy"] = "largest-first"
            if strat in STRATEGIES and row["rate"] == 5.7:
                groups[(strat, row["rate"])].append(row)
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


def load_fig3_run(filepath: Path) -> dict:
    """Load a single run from vllm_fig3_mixed_rate38 using all-path fields."""
    with open(filepath) as f:
        d = json.load(f)
    am = d.get("adapter_metrics", {})
    return {
        "strategy": d.get("strategy", ""),
        "rate": d.get("request_rate", 0),
        "all_preemptions": am.get("total_all_preemptions", 0),
        "all_tokens_freed": am.get("total_all_tokens_freed", 0),
    }


def load_fig3_all() -> dict[tuple[str, float], list[dict]]:
    from collections import defaultdict
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for f in sorted(FIG3_RESULTS_DIR.glob("*.json")):
        if f.name.startswith("candidate"):
            continue
        row = load_fig3_run(f)
        strat = row["strategy"]
        # map legacy h2o-style to largest-first
        if strat == "h2o-style":
            strat = "largest-first"
            row["strategy"] = "largest-first"
        if strat in STRATEGIES:
            groups[(strat, row["rate"])].append(row)
    return groups


def load_fig5_all() -> dict[tuple[str, float], list[dict]]:
    """Load rate=5.7 reclamation data for fig5 from vllm_fig3_mixed_rate57."""
    from collections import defaultdict
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for f in sorted(FIG3_RATE57_DIR.glob("*.json")):
        if f.name.startswith("candidate"):
            continue
        row = load_fig3_run(f)
        strat = row["strategy"]
        if strat == "h2o-style":
            strat = "largest-first"
            row["strategy"] = "largest-first"
        if strat in STRATEGIES:
            groups[(strat, row["rate"])].append(row)
    return groups


def generate_fig5(groups: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig5_groups = load_fig3_all()
    rate = 3.8
    strats_ok, labels, evicts, freed = [], [], [], []
    for strat in STRATEGIES:
        runs = fig5_groups.get((strat, rate), [])
        if not runs:
            continue
        strats_ok.append(strat)
        labels.append(STRATEGY_DISPLAY[strat])
        evicts.append(sum(r["all_preemptions"] for r in runs) / len(runs))
        freed.append(sum(r["all_tokens_freed"] for r in runs) / len(runs) / 1000)

    n = len(strats_ok)
    x = list(range(n))
    bw = 0.35
    bar_c = [COLORS[s] for s in strats_ok]

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax2r = ax1.twinx()

    ax1.bar([i - bw / 2 for i in x], evicts, bw,
            color=bar_c, alpha=0.85, edgecolor="black", linewidth=0.5,
            label="Reclamation Count (All Paths)")
    ax2r.bar([i + bw / 2 for i in x], freed, bw,
             color=bar_c, alpha=0.4, edgecolor="black", linewidth=0.5,
             hatch="//", label="Tokens Freed (×1000)")

    ax1.set_xlabel("Strategy")
    ax1.set_ylabel("Reclamation Count")
    ax2r.set_ylabel("Tokens Freed (×1000)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")

    me = max(evicts) if evicts and max(evicts) > 0 else 1
    mf = max(freed) if freed and max(freed) > 0 else 1
    ax1.set_ylim(0, me * 1.18)
    ax2r.set_ylim(0, mf * 1.18)
    for i, (ev, fr) in enumerate(zip(evicts, freed)):
        if ev > 0:
            ax1.text(i - bw / 2, ev + me * 0.02, f"{ev:.0f}",
                     ha="center", va="bottom", fontsize=8)
        if fr > 0:
            ax2r.text(i + bw / 2, fr + mf * 0.02, f"{fr:.0f}k",
                      ha="center", va="bottom", fontsize=8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2,
               loc="lower left", bbox_to_anchor=(0, 1.02),
               ncol=2, fontsize=8, borderaxespad=0)
    ax1.grid(True, axis="y", alpha=0.2, linestyle="--")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig5_compress_coverage")
    plt.close(fig)


def generate_fig1_panel_b_scatter() -> None:
    """Generate panel (b) for Figure 1: victim-selection CDF profiles.

    For each of 150 KV-pressure snapshots, record the chosen victim's:
      - per-snapshot KV-footprint rank (0 = smallest candidate, 1 = largest)
      - completion ratio c = progress in [0, 1]  (0 = just started, 1 = done)

    Two stacked CDFs make the contrast overlap-free:
      Top:    distribution of KV rank recovered  — higher is better
      Bottom: distribution of recompute cost c   — lower (left) is better

    Expected result:
      KV CDFs:   LIFO gradual (random KV), LF step at 1.0, BidKV near-1
      Cost CDFs: LIFO spike at c≈0, LF roughly uniform, BidKV left-shifted vs LF

    Output: paper/figures/fig1_intro_evidence_panel_b.{pdf,png}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pe_path = RESULTS_DIR / "preempt-evict__mixed__rate3.8__r0.json"
    if not pe_path.exists():
        print(f"  SKIP fig1_panel_b: {pe_path} not found", file=sys.stderr)
        return

    with open(pe_path) as f:
        d = json.load(f)

    ok = [
        r for r in d["request_results"]
        if not r.get("error")
        and r.get("first_token_time")
        and r.get("finish_time")
        and r["finish_time"] > r["first_token_time"]
    ]

    t0 = min(r["submit_time"] for r in ok)
    t1 = max(r["finish_time"] for r in ok)

    def _c_kv(r: dict, t: float) -> tuple[float, float]:
        progress = (t - r["first_token_time"]) / max(
            r["finish_time"] - r["first_token_time"], 1e-6
        )
        progress = min(max(progress, 0.001), 0.999)
        gen = float(r["completion_tokens"]) * progress
        est_prompt = min(r["ttft_ms"], 1000.0) * 0.3 + 50.0
        return progress, est_prompt + gen  # (c, Y)

    N_SAMPLES = 150
    MIN_CONC = 6
    EPSILON = 0.01

    lifo_ky: list[float] = []
    lifo_c: list[float] = []
    lf_ky: list[float] = []
    lf_c: list[float] = []
    bk_ky: list[float] = []
    bk_c: list[float] = []

    for i in range(N_SAMPLES):
        t = t0 + (t1 - t0) * (0.1 + 0.8 * i / N_SAMPLES)
        snap = [r for r in ok if r["first_token_time"] <= t <= r["finish_time"]]
        if len(snap) < MIN_CONC:
            continue

        n = len(snap)
        pairs = [_c_kv(r, t) for r in snap]  # (c, Y)

        # Per-snapshot rank on Y = KV footprint (0 = smallest, 1 = largest)
        sy = sorted(range(n), key=lambda j: pairs[j][1])
        rank_y = [0.0] * n
        for rk, idx in enumerate(sy):
            rank_y[idx] = rk / max(n - 1, 1)

        # LIFO: most recently started decoding (max first_token_time)
        li = max(range(n), key=lambda j: snap[j]["first_token_time"])
        lifo_ky.append(rank_y[li])
        lifo_c.append(pairs[li][0])

        # Largest-first: max KV footprint (rank_y always 1.0 by definition)
        lfi = max(range(n), key=lambda j: pairs[j][1])
        lf_ky.append(rank_y[lfi])
        lf_c.append(pairs[lfi][0])

        # BidKV: max utility U = Y / (1 + 0.5*c + ε)
        us = [
            pairs[j][1] / (1.0 + 0.5 * pairs[j][0] + EPSILON)
            for j in range(n)
        ]
        bi = max(range(n), key=lambda j: us[j])
        bk_ky.append(rank_y[bi])
        bk_c.append(pairs[bi][0])

    n_snaps = len(lifo_ky)

    def _cdf(vals: list[float]) -> tuple[list[float], list[float]]:
        sv = sorted(vals)
        probs = [(i + 1) / len(sv) for i in range(len(sv))]
        return sv, probs

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 3.0), sharex=False)
    colors = {"LIFO": "#d62728", "Largest-first": "#1f77b4", "BidKV": "#2ca02c"}

    # ── Top: CDF of KV-footprint rank ─────────────────────────────────────────
    for label, vals in [("LIFO", lifo_ky), ("Largest-first", lf_ky), ("BidKV", bk_ky)]:
        xs, ys = _cdf(vals)
        ax1.plot([0.0] + xs + [1.0], [0.0] + ys + [1.0],
                 color=colors[label], lw=1.5, label=label, drawstyle="steps-post")
    ax1.set_xlim(-0.02, 1.04)
    ax1.set_ylim(-0.03, 1.08)
    ax1.set_xlabel("KV-Footprint Rank of Selected Victim  (1 = largest)", labelpad=2)
    ax1.set_ylabel("CDF", labelpad=2)
    ax1.text(0.01, 0.97, "more KV freed →", transform=ax1.transAxes,
             ha="left", va="top", fontsize=5.5, color="#555555", style="italic")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=6.5,
               handlelength=1.4, borderpad=0.4, labelspacing=0.25)
    ax1.grid(True, linestyle=":", alpha=0.35)

    # ── Bottom: CDF of completion ratio c ────────────────────────────────────
    for label, vals in [("LIFO", lifo_c), ("Largest-first", lf_c), ("BidKV", bk_c)]:
        xs, ys = _cdf(vals)
        ax2.plot([0.0] + xs + [1.0], [0.0] + ys + [1.0],
                 color=colors[label], lw=1.5, label=label, drawstyle="steps-post")
    ax2.set_xlim(-0.02, 1.04)
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_xlabel("Completion Ratio $c$ of Selected Victim  (0 = newest)", labelpad=2)
    ax2.set_ylabel("CDF", labelpad=2)
    ax2.text(0.99, 0.03, "← less recompute cost", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=5.5, color="#555555", style="italic")
    ax2.grid(True, linestyle=":", alpha=0.35)

    fig.suptitle("(b) Victim-Selection Profiles  ($n$=" + str(n_snaps) + " snapshots)",
                 fontsize=8, y=1.01)
    fig.tight_layout(pad=0.5, h_pad=0.8)
    save_fig(fig, "fig1_intro_evidence_panel_b")
    plt.close(fig)

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else float("nan")

    print(
        f"  Panel (b): {n_snaps} snapshots\n"
        f"    KV rank mean  — LIFO:{_mean(lifo_ky):.3f}  LF:{_mean(lf_ky):.3f}  BidKV:{_mean(bk_ky):.3f}\n"
        f"    c mean        — LIFO:{_mean(lifo_c):.3f}  LF:{_mean(lf_c):.3f}  BidKV:{_mean(bk_c):.3f}"
    )


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
    generate_fig1_panel_b_scatter()
    print("\nDone.")


if __name__ == "__main__":
    main()
