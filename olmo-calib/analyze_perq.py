"""Per-question view of direct answering vs reasoning (cot_test.py per-question files), SFT vs DPO.

Per question and model: direct confidence (letter probability, chat format, no reasoning) and
self-consistency (share of 8 sampled reasoning chains agreeing on the majority answer).

Usage: python analyze_perq.py results_cot
"""
import json
import sys
from collections import defaultdict

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "results_cot"
L = "ABCD"


def load(name):
    direct = {}
    for l in open(f"{d}/{name}.questions.jsonl"):
        r = json.loads(l)
        if r["mode"] == "direct":
            direct[r["qi"]] = r
    sc = {json.loads(l)["qi"]: json.loads(l) for l in open(f"{d}/{name}.sc8.questions.jsonl")}
    out = {}
    for qi, s in sc.items():
        p = direct[qi]["probs"]
        votes = s["votes"]
        valid = [v for v in votes if v]
        counts = {x: votes.count(x) for x in L}
        maj = max(L, key=lambda x: counts[x])
        out[qi] = {
            "src": s["src"], "subject": s["subject"], "answer": s["answer"], "question": s["question"],
            "choices": s["choices"], "chains": s["chains"],
            "d_pred": L[int(np.argmax(p))], "d_conf": max(p), "d_p_correct": p[L.index(s["answer"])],
            "sc_pred": maj, "sc_conf": counts[maj] / len(votes),
            "sc_conf_valid": counts[maj] / len(valid) if valid else 0.0, "no_answer": votes.count(None),
            "votes": votes,
        }
        out[qi]["d_ok"] = out[qi]["d_pred"] == s["answer"]
        out[qi]["sc_ok"] = maj == s["answer"]
    return out


S, D = load("Olmo-3-7B-Instruct-SFT"), load("Olmo-3-7B-Instruct-DPO")
qs = sorted(S)


def excess(m, key_conf, key_ok, src):
    ids = [q for q in qs if m[q]["src"] == src]
    return np.mean([m[q][key_conf] for q in ids]) - np.mean([m[q][key_ok] for q in ids])


print("== 1. DPO's excess overconfidence (conf - acc, DPO minus SFT) ==")
for src in ("mmlu", "arc"):
    for label, kc, ko in [("direct", "d_conf", "d_ok"), ("reasoning, missing answers as votes", "sc_conf", "sc_ok"),
                          ("reasoning, missing answers dropped", "sc_conf_valid", "sc_ok")]:
        print(f"  {src:<5}{label:<38}{excess(D, kc, ko, src) - excess(S, kc, ko, src):+.3f}")
print(f"  missing answers: SFT {np.mean([S[q]['no_answer'] for q in qs]) / 8:.1%}, "
      f"DPO {np.mean([D[q]['no_answer'] for q in qs]) / 8:.1%}")

print("\n== 2. Where DPO's direct confidence rises (MMLU+ARC), by SFT's direct confidence ==")
print(f"{'SFT conf':<12}{'n':>5}{'SFT conf':>9}{'DPO conf':>9}{'SFT acc':>8}{'DPO acc':>8}{'same answer':>12}")
for lo, hi in [(0, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)]:
    ids = [q for q in qs if lo <= S[q]["d_conf"] < hi]
    if ids:
        print(f"[{lo:.2f},{min(hi, 1):.2f}){len(ids):>5}{np.mean([S[q]['d_conf'] for q in ids]):>9.3f}"
              f"{np.mean([D[q]['d_conf'] for q in ids]):>9.3f}{np.mean([S[q]['d_ok'] for q in ids]):>8.3f}"
              f"{np.mean([D[q]['d_ok'] for q in ids]):>8.3f}{np.mean([S[q]['d_pred'] == D[q]['d_pred'] for q in ids]):>12.1%}")

print("\n== 3. Confidently wrong when answering directly (conf >= 0.9, wrong) ==")
for name, m in (("SFT", S), ("DPO", D)):
    bad = [q for q in qs if m[q]["d_conf"] >= 0.9 and not m[q]["d_ok"]]
    if not bad:
        continue
    print(f"  {name}: {len(bad)} questions ({len(bad) / len(qs):.1%}); reasoning fixes {np.mean([m[q]['sc_ok'] for q in bad]):.0%} of them; "
          f"mean agreement across chains {np.mean([m[q]['sc_conf'] for q in bad]):.2f}; "
          f"all 8 chains agree on the same wrong answer in {np.mean([m[q]['sc_conf'] == 1 and not m[q]['sc_ok'] for q in bad]):.0%}")

print("\n== 4. Does direct confidence track reasoning-based certainty? (per-question correlation) ==")
for name, m in (("SFT", S), ("DPO", D)):
    a = np.array([m[q]["d_conf"] for q in qs])
    b = np.array([m[q]["sc_conf"] for q in qs])
    same = np.mean([m[q]["d_pred"] == m[q]["sc_pred"] for q in qs])
    print(f"  {name}: corr(direct conf, chain agreement) = {np.corrcoef(a, b)[0, 1]:+.3f}; "
          f"direct answer = reasoning majority on {same:.0%} of questions")

print("\n== 5. Per subject (MMLU, >= 12 questions): DPO minus SFT ==")
by = defaultdict(list)
for q in qs:
    if S[q]["src"] == "mmlu":
        by[S[q]["subject"]].append(q)
rows = []
for sj, ids in by.items():
    if len(ids) < 12:
        continue
    rows.append((sj, len(ids),
                 np.mean([D[q]["d_conf"] - S[q]["d_conf"] for q in ids]), np.mean([D[q]["d_ok"] - S[q]["d_ok"] for q in ids]),
                 np.mean([D[q]["sc_conf"] - S[q]["sc_conf"] for q in ids]), np.mean([D[q]["sc_ok"] - S[q]["sc_ok"] for q in ids])))
print(f"{'subject':<36}{'n':>4}{'direct conf':>12}{'direct acc':>11}{'chain conf':>11}{'chain acc':>10}")
for r in sorted(rows, key=lambda r: -(r[2] - r[3]))[:8] + [None] + sorted(rows, key=lambda r: r[2] - r[3])[:4]:
    if r is None:
        print("  ...")
        continue
    print(f"{r[0][:35]:<36}{r[1]:>4}{r[2]:>+12.3f}{r[3]:>+11.3f}{r[4]:>+11.3f}{r[5]:>+10.3f}")

print("\n== 6. Examples: DPO confidently wrong directly (>= 0.95), SFT unsure (< 0.6) ==")
ex = [q for q in qs if D[q]["d_conf"] >= 0.95 and not D[q]["d_ok"] and S[q]["d_conf"] < 0.6]
print(f"  {len(ex)} such questions")
for q in ex[:5]:
    r = D[q]
    print(f"\n  [{r['src']}/{r['subject']}] {r['question'][:200]!r}")
    print("   " + "  ".join(f"{x}. {c[:40]}" for x, c in zip(L, r["choices"])))
    print(f"   correct {r['answer']} | SFT direct {S[q]['d_pred']} ({S[q]['d_conf']:.2f}) | DPO direct {r['d_pred']} ({r['d_conf']:.2f})"
          f" | SFT chains {''.join(v or '-' for v in S[q]['votes'])} | DPO chains {''.join(v or '-' for v in r['votes'])}")
