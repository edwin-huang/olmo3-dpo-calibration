"""When a response states its answer mid-text, does the push concentrate where the answer is?

Finds answer statements after the first 5 tokens ("the answer is", "Answer:", "Final answer",
"\\boxed{...}", "correct option is", "**(B)**" ...) plus the ~12 characters after them (the
answer itself), maps them to tokens, and compares push per token at those tokens with the other
body tokens (position >= 5) of the same response. Also reports the SFT model's entropy there.

Usage: python answer_locus.py results_tokens/instruct_tokens.jsonl
"""
import json
import math
import re
import sys

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
STATE = re.compile(r"(final answer|the answer is|answer\s*(is)?\s*:|correct (answer|option|choice) is|"
                   r"\\boxed\{|the (best|correct|right) (answer|option|choice)|\*\*\(?[A-E]\)?[\.:]?\*\*|"
                   r"\b(option|choice)\s+\(?[A-E]\)?\b)", re.I)

recs = [json.loads(l) for l in open(sys.argv[1])]
stats = {s: {"ans": [], "other": [], "ent_ans": [], "ent_other": [], "n_resp": 0} for s in ("chosen", "rejected")}
signed = {s: {"ans": 0.0, "other": 0.0} for s in ("chosen", "rejected")}
total = sum(r["push"] for r in recs)
for r in recs:
    for side, sign in (("chosen", 1), ("rejected", -1)):
        s = r[side]
        pieces = [tok.decode([t]) for t in s["tokens"]]
        starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
        text = "".join(pieces)
        L = len(s["d"])
        hit = np.zeros(L, dtype=bool)
        for m in STATE.finditer(text):
            lo, hi = m.start(), m.end() + 12
            hit |= (starts + np.array([len(p) for p in pieces]) > lo) & (starts < hi)
        hit[:5] = False  # body only: the opening is analysed separately
        body = np.arange(L) >= 5
        if not hit.any():
            continue
        st = stats[side]
        st["n_resp"] += 1
        c = sign * np.array(s["d"]) / L
        ent = np.array(s["ent"])
        st["ans"] += c[hit].tolist()
        st["other"] += c[body & ~hit].tolist()
        st["ent_ans"] += ent[hit].tolist()
        st["ent_other"] += ent[body & ~hit].tolist()
        signed[side]["ans"] += c[hit].sum()
        signed[side]["other"] += c[body & ~hit].sum()

print(f"total push {total:+.2f} over {len(recs)} pairs\n")
for side in ("chosen", "rejected"):
    st = stats[side]
    if not st["n_resp"]:
        continue
    a, o = np.array(st["ans"]), np.array(st["other"])
    print(f"{side}: {st['n_resp']} responses state an answer after token 5 "
          f"({len(a)} answer tokens vs {len(o)} other body tokens)")
    print(f"   mean |push| per token: answer {1e3 * np.abs(a).mean():.3f}e-3, other {1e3 * np.abs(o).mean():.3f}e-3"
          f"  -> x{np.abs(a).mean() / np.abs(o).mean():.1f}")
    print(f"   signed push: answer tokens {signed[side]['ans'] / total:+.1%} of total, "
          f"other body tokens of these responses {signed[side]['other'] / total:+.1%}")
    print(f"   SFT uncertainty k = exp(entropy): answer {math.exp(np.mean(st['ent_ans'])):.2f}, "
          f"other {math.exp(np.mean(st['ent_other'])):.2f}")
