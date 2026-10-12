"""Data-only follow-ups to "overconfidence comes from pairs the model already agrees with" (no GPU).

1. Share of q25_all pairs where Instruct-SFT already prefers the chosen response (mean per-token log-prob margin > 0,
   from q25's reference cache), by preference type (delta / GPT-judged / multi-turn), by D4 flag (answer-first or
   degenerate rejected), and by option-listing prompt.
2. Across every small test (results_ckpt/small/*/summary.json with a matching dpo_data/<set>.parquet, plain DPO at
   LR 8e-6 only): per-set agreement share, mean margin, median chosen/rejected length -> correlation with log T* and a
   two-feature least-squares fit.

Usage: python agreement_analysis.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt /workspace/results_ckpt/small
"""
import json
import math
import os
import random
import re
import statistics as st
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

D, CACHE, SMALL = sys.argv[1:4]
OPT = re.compile(r"(?m)^\s*(?:\(?[A-Ea-e][\).:]|\(?[1-5][\).])\s+\S")
q25 = pq.read_table(f"{D}/q25_all.parquet").to_pylist()
c = torch.load(CACHE, map_location="cpu")
lc, lr = c["chosen_logps"].float().tolist(), c["rejected_logps"].float().tolist()
margin = [a - b for a, b in zip(lc, lr)]
meta = pq.read_table(f"{D}/instruct_meta.parquet").to_pylist()
perm = list(range(len(meta)))
random.Random(0).shuffle(perm)  # make_dpo_data.py's shuffle -> q25_all = perm[:65000]
info = [meta[i] for i in perm[:65000]]


def user(r):
    u = [m["content"] or "" for m in r["chosen"] if m["role"] == "user"]
    return u[-1] if u else ""


print("== 1. agreement share in q25_all (Instruct-SFT margin > 0)")
groups = {
    "all": range(65000),
    "delta_learning": [i for i in range(65000) if info[i]["preference_type"] == "delta_learning"],
    "llm_judged": [i for i in range(65000) if info[i]["preference_type"] == "llm_judged"],
    "multiturn": [i for i in range(65000) if info[i]["preference_type"].startswith("multiturn")],
    "D4-flagged": [i for i in range(65000) if info[i]["flag"]],
    "not flagged": [i for i in range(65000) if not info[i]["flag"]],
    "option-listing prompt": [i for i in range(65000) if len(OPT.findall(user(q25[i]))) >= 3],
}
for g, ix in groups.items():
    m = [margin[i] for i in ix]
    print(f"  {g:<22} n {len(ix):>6}  agree {sum(x > 0 for x in m) / len(m):5.1%}  mean margin {st.mean(m):+.3f}  "
          f"median {st.median(m):+.3f}")

print("== 2. per-set features vs outcome (plain DPO, LR 8e-6)")
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
key = lambda r: (r["chosen"][-1]["content"][:200], r["rejected"][-1]["content"][:200])
idx = {key(r): i for i, r in enumerate(q25)}
rows = []
for name in sorted(os.listdir(SMALL)):
    s_path = f"{SMALL}/{name}/summary.json"
    if not os.path.exists(s_path) or name.endswith(("_rep",)):
        continue
    first = open(f"{SMALL}/{name}/train_first_step.txt").read() if os.path.exists(f"{SMALL}/{name}/train_first_step.txt") else ""
    if "LR: 8e-07" not in first and "LR:" in first and "e-07" not in first:
        pass
    data = {"pilot8a": "s8_a", "pilot8b": "s8_b", "pilot8c": "s8_c", "pilot8d": "s8_d"}.get(name, name)
    if name in ("pilot16a", "pilotlo8a", "neg8a", "neglo8a", "pos8a", "rev8a", "swap8a") or not os.path.exists(f"{D}/{data}.parquet"):
        continue
    sset = pq.read_table(f"{D}/{data}.parquet").to_pylist()
    ix = [idx[key(r)] for r in sset if key(r) in idx]
    if len(ix) < 0.9 * len(sset):
        continue
    smp = random.Random(0).sample(sset, min(800, len(sset)))
    n_c = st.median(len(tok(r["chosen"][-1]["content"] or "", add_special_tokens=False).input_ids) for r in smp)
    n_r = st.median(len(tok(r["rejected"][-1]["content"] or "", add_special_tokens=False).input_ids) for r in smp)
    t = json.load(open(s_path))["letters_mmlu_t_star"]
    m = [margin[i] for i in ix]
    rows.append((name, sum(x > 0 for x in m) / len(m), st.mean(m), n_c, n_r, t))
    print(f"  {name:<10} agree {rows[-1][1]:5.1%}  margin {rows[-1][2]:+.3f}  median tokens chosen {n_c:>4.0f} rejected {n_r:>4.0f}  T* {t:.2f}")
y = np.log([r[5] for r in rows])
feats = {"agree": np.array([r[1] for r in rows]), "margin": np.array([r[2] for r in rows]),
         "log_len_chosen": np.log([r[3] for r in rows]), "log_len_rejected": np.log([r[4] for r in rows])}
for k, x in feats.items():
    print(f"  corr(log T*, {k}) = {np.corrcoef(x, y)[0, 1]:+.2f}")
for a, b in (("agree", "log_len_chosen"), ("margin", "log_len_chosen"), ("agree", "margin")):
    X = np.c_[feats[a], feats[b], np.ones(len(y))]
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  log T* ~ {coef[0]:+.2f} {a} {coef[1]:+.2f} {b} {coef[2]:+.2f}   R2 {r2:.2f}  (n {len(y)})")
