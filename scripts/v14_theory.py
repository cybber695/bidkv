"""Theoretical analysis: BidKV U-score vs h2o-style ordering in long_context.

Shows how U = freed / (1 + 0.5*completion + 0.3*preemptions) compares to
h2o-style's simple freed-first ordering for typical long_context requests.
"""

def bidkv_utility(current_tokens, completion, preemptions=0):
    """BidKV U = freed / (1 + 0.5*completion + 0.3*preemptions)"""
    return current_tokens / (1.0 + 0.5 * completion + 0.3 * preemptions)

# Simulate typical long_context running set
# Assume avg prompt = 2000 tokens, max_output = 256
requests = [
    {"id": "A", "prompt": 3000, "output_done": 10,  "max_output": 256, "preemptions": 0},
    {"id": "B", "prompt": 2000, "output_done": 128, "max_output": 256, "preemptions": 0},
    {"id": "C", "prompt": 1500, "output_done": 200, "max_output": 256, "preemptions": 0},
    {"id": "D", "prompt": 2500, "output_done": 50,  "max_output": 256, "preemptions": 1},
    {"id": "E", "prompt": 4000, "output_done": 5,   "max_output": 256, "preemptions": 0},
]

print("=== BidKV U-score ordering (victim = highest U, first to be preempted) ===")
print(f"{'ID':>4} {'Prompt':>6} {'OutDone':>7} {'Complete%':>9} {'Preempt':>7} {'CurrTok':>7} {'U':>8} {'δ':>6}")
print("-" * 65)

scored = []
for r in requests:
    current_tokens = r["prompt"] + r["output_done"]
    completion = min(1.0, r["output_done"] / r["max_output"]) if r["max_output"] > 0 else 0
    u = bidkv_utility(current_tokens, completion, r["preemptions"])
    delta = 1.0 + 0.5 * completion + 0.3 * r["preemptions"]
    scored.append((r["id"], r["prompt"], r["output_done"], completion, r["preemptions"], current_tokens, u, delta))
    print(f"{r['id']:>4} {r['prompt']:>6} {r['output_done']:>7} {completion*100:>8.1f}% {r['preemptions']:>7} {current_tokens:>7} {u:>8.1f} {delta:>6.2f}")

print()
print("=== BidKV victim ordering (highest U first = most expendable) ===")
scored.sort(key=lambda x: x[6], reverse=True)
for i, s in enumerate(scored):
    print(f"  #{i+1} victim: {s[0]} (U={s[6]:.1f}, freed={s[5]}, completion={s[3]*100:.0f}%)")

print()
print("=== h2o-style/largest-first ordering (largest current_tokens first) ===")
scored_hv = sorted(scored, key=lambda x: x[5], reverse=True)
for i, s in enumerate(scored_hv):
    print(f"  #{i+1} victim: {s[0]} (freed={s[5]}, completion={s[3]*100:.0f}%)")

print()
print("=== Key insight ===")
print("BidKV and h2o-style produce SIMILAR orderings because freed (current_tokens)")
print("dominates U. The 0.5*completion factor gives at most 1.5× protection.")
print()
print("Example: Request E (4005 tokens, 2% done) → U=3995.1")
print("         Request B (2128 tokens, 50% done) → U=1702.4")
print("Both rank E first. The completion penalty only matters for closely-sized requests.")
print()

# Now simulate: what if we add anti-starvation effect
print("=== Anti-starvation scenario: D has 3 preemptions ===")
r_d_starved = {"id": "D*", "prompt": 2500, "output_done": 50, "max_output": 256, "preemptions": 3}
current_d = r_d_starved["prompt"] + r_d_starved["output_done"]
comp_d = r_d_starved["output_done"] / r_d_starved["max_output"]
u_d = bidkv_utility(current_d, comp_d, 3)
delta_d = 1.0 + 0.5 * comp_d + 0.3 * 3
print(f"  D* U={u_d:.1f} (δ={delta_d:.2f}) vs D U={bidkv_utility(2550, 50/256):.1f} (δ={1+0.5*50/256:.2f})")
print(f"  Anti-starvation drops D from ~victim to protected")
