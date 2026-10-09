"""Where inside DPO pairs does the push toward overconfidence sit? (reads token_attribution.py output)

A pair's push = mean_t d_t(chosen) − mean_t d_t(rejected), so chosen token t contributes d_t/L_c
and rejected token t contributes −d_t/L_r. Contributions are summed over all pairs and split by
position in the response, by the SFT model's uncertainty at that token (k = exp(entropy)), and by
token type.

Usage: python analyze_tokens.py results_tokens/instruct_tokens.jsonl [results_attr/instruct_pairs.jsonl]
"""
import json
import math
import re
import sys
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

recs = [json.loads(l) for l in open(sys.argv[1])]
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
print(f"{len(recs)} pairs")

# 1. Does the finite-difference push agree with the earlier gradient attribution?
if len(sys.argv) > 2:
    old = {r["idx"]: r for r in (json.loads(l) for l in open(sys.argv[2]))}
    pa = np.array([r["push_acc"] for r in old.values()])
    pc = np.array([r["push_conf"] for r in old.values()])
    slope = np.cov(pa, pc)[0, 1] / pa.var(ddof=1)  # as in analyze_attr.py
    for r in old.values():
        r["push_conf_resid"] = r["push_conf"] - slope * r["push_acc"]
    both = [(r["push"], old[r["idx"]]["push_conf"], old[r["idx"]]["push_conf_resid"]) for r in recs if r["idx"] in old]
    a = np.array(both)
    print(f"corr(token push, gradient push_conf) = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.3f}, "
          f"with push_conf_resid = {np.corrcoef(a[:, 0], a[:, 2])[0, 1]:+.3f}  (n = {len(a)})")

pushes = np.array([r["push"] for r in recs])
print(f"mean push {pushes.mean():+.4f}, share of pairs > 0: {(pushes > 0).mean():.0%}")


def kind(t):
    s = tok.decode([t])
    if t in tok.all_special_ids:
        return "special/end"
    if re.fullmatch(r"\s*\(?[A-Ea-e][\)\.:]?\s*", s) and s.strip():
        return "option letter"
    if re.fullmatch(r"\s*\d+[\.,]?\d*\s*", s) and s.strip():
        return "number"
    if not s.strip():
        return "whitespace/newline"
    if re.fullmatch(r"\s*[^\w\s]+\s*", s):
        return "punctuation/markdown"
    return "word"


K_EDGES = [1.1, 1.5, 2.5, 4, 8, 16]
K_NAMES = ["<1.1", "1.1-1.5", "1.5-2.5", "2.5-4", "4-8", "8-16", ">=16"]
POS = [(0, 5, "tokens 0-4"), (5, 20, "tokens 5-19"), (20, 100, "tokens 20-99"), (100, 10 ** 9, "tokens 100+")]

by = {"position": defaultdict(float), "SFT uncertainty k": defaultdict(float), "token type": defaultdict(float)}
count = {k: defaultdict(int) for k in by}
by_side = defaultdict(float)
contribs, strings = [], defaultdict(float)
kind_cache = {}
for r in recs:
    for side, sign in (("chosen", 1), ("rejected", -1)):
        s = r[side]
        L = len(s["d"])
        for t, (tid, d, ent) in enumerate(zip(s["tokens"], s["d"], s["ent"])):
            c = sign * d / L
            by_side[side] += c
            contribs.append(c)
            pos = next(name for lo, hi, name in POS if lo <= t < hi)
            kb = K_NAMES[sum(math.exp(ent) >= e for e in K_EDGES)]
            if tid not in kind_cache:
                kind_cache[tid] = kind(tid)
            for key, g in (("position", pos), ("SFT uncertainty k", kb), ("token type", kind_cache[tid])):
                by[key][g] += c
                count[key][g] += 1
            strings[(side, tok.decode([tid]))] += c

total = sum(by_side.values())
print(f"\ntotal push {total:+.3f}: chosen side {by_side['chosen']:+.3f}, rejected side {by_side['rejected']:+.3f}")
n_tok = sum(count["position"].values())
for key, groups in by.items():
    print(f"\n{key:<24}{'share of push':>14}{'share of tokens':>17}{'push per token (x1e3)':>23}")
    order = K_NAMES if key == "SFT uncertainty k" else [n for _, _, n in POS] if key == "position" else sorted(groups, key=lambda g: -groups[g])
    for g in order:
        if g in groups:
            print(f"{g:<24}{groups[g] / total:>14.1%}{count[key][g] / n_tok:>17.1%}{1e3 * groups[g] / count[key][g]:>23.3f}")

c = np.sort(np.abs(np.array(contribs)))[::-1]
print(f"\nconcentration: top 1% of tokens carry {c[: len(c) // 100].sum() / c.sum():.0%} of |contribution|, "
      f"top 10% carry {c[: len(c) // 10].sum() / c.sum():.0%}")
for side in ("chosen", "rejected"):
    items = sorted(((v, s) for (sd, s), v in strings.items() if sd == side), reverse=True)
    print(f"\n{side}: token strings with the largest total push toward confidence")
    print("   " + "  ".join(f"{s!r}:{v:+.2f}" for v, s in items[:15]))
    print(f"{side}: largest push against")
    print("   " + "  ".join(f"{s!r}:{v:+.2f}" for v, s in items[-10:]))
