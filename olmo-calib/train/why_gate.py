"""Stop rule for the Brief 2 wave (why_tests/PREDICTIONS.md), applied after the 12 runs of predictions 1 and 3.

Reads results_ckpt/small/wy_{plain,w5,simpo,simporef}_s{1,2,3}/summary.json (+ pairs.json) and prints per-condition
means and seed ranges of MMLU T* and learned. Decision, exactly as pre-registered:
  REFUTED (exit 3)      3a fails (SimPO T* not >= 0.1 below plain, or learned > 3 points lower), or the 3b ratio
                        (T*(simporef) - T*(simpo)) / (T*(plain) - T*(simpo)) <= 0.25
  CONFIRMED (exit 0)    3a holds and ratio >= 0.5
  INCONCLUSIVE (exit 0) 3a holds and 0.25 < ratio < 0.5
Writes the decision and numbers to /workspace/why_gate.json.

Usage: python why_gate.py /workspace/results_ckpt/small
"""
import json
import os
import statistics as st
import sys

R = sys.argv[1]
conds = ("plain", "w5", "simpo", "simporef")
vals = {}
for c in conds:
    t, l = [], []
    for s in (1, 2, 3):
        d = os.path.join(R, f"wy_{c}_s{s}")
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        sm = json.load(open(os.path.join(d, "summary.json")))
        if sm.get("letters_mmlu_t_star") is None or sm.get("pairs_acc_ref_disagreed") is None:
            continue
        t.append(sm["letters_mmlu_t_star"])
        l.append(100 * sm["pairs_acc_ref_disagreed"])
    vals[c] = {"t_star": t, "learned": l}
    if t:
        print(f"{c:<9} n {len(t)}  T* mean {st.mean(t):.3f} (range {min(t):.2f}-{max(t):.2f})  "
              f"learned mean {st.mean(l):.1f} (range {min(l):.1f}-{max(l):.1f})")
missing = [c for c in ("plain", "simpo", "simporef") if len(vals[c]["t_star"]) < 2]
if missing:
    decision, reason = "REFUTED", f"cannot evaluate: fewer than 2 finished seeds for {missing} (treated as stop)"
else:
    m = {c: (st.mean(vals[c]["t_star"]), st.mean(vals[c]["learned"])) for c in conds if vals[c]["t_star"]}
    gap = m["plain"][0] - m["simpo"][0]
    a_ok = gap >= 0.1 and m["simpo"][1] >= m["plain"][1] - 3
    ratio = (m["simporef"][0] - m["simpo"][0]) / gap if gap > 0 else float("nan")
    if not a_ok:
        decision, reason = "REFUTED", f"3a fails: plain - simpo T* gap {gap:+.3f}, learned simpo {m['simpo'][1]:.1f} vs plain {m['plain'][1]:.1f}"
    elif ratio <= 0.25:
        decision, reason = "REFUTED", f"3b ratio {ratio:.2f} <= 0.25"
    elif ratio >= 0.5:
        decision, reason = "CONFIRMED", f"3a holds (gap {gap:+.3f}); 3b ratio {ratio:.2f} >= 0.5"
    else:
        decision, reason = "INCONCLUSIVE", f"3a holds (gap {gap:+.3f}); 3b ratio {ratio:.2f} between 0.25 and 0.5"
print(f"DECISION: {decision} — {reason}")
json.dump({"decision": decision, "reason": reason, "values": vals}, open("/workspace/why_gate.json", "w"), indent=1)
sys.exit(3 if decision == "REFUTED" else 0)
