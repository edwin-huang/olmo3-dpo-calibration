"""Robustness of the rejected-likelihood result (rejlo8m T* 2.08 vs rejhi8m 1.52).

  rejlo8r / rejhi8r   replicate of rejlo8m / rejhi8m: same construction (rejected-length deciles, bottom/top 30% by
                      rejected log-prob under Instruct-SFT), different random draw within the bins
  rejlo8c / rejhi8c   ALSO matched on the chosen response's log-prob: bins = rejected-length tercile x chosen-log-prob
                      quintile; within each bin the bottom/top 40% by rejected log-prob, equal counts per bin. Separates
                      "the rejected response is unlikely" from "the pair is hard / SFT already disagrees".

Usage: python make_small_sets5.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
"""
import random
import statistics as st
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

D, CACHE = sys.argv[1], sys.argv[2]
rows = pq.read_table(f"{D}/q25_all.parquet").to_pylist()
cache = torch.load(CACHE, map_location="cpu")
lp_r, lp_c = cache["rejected_logps"].float().tolist(), cache["chosen_logps"].float().tolist()
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
n_rej = [len(tok(r["rejected"][-1]["content"] or "", add_special_tokens=False).input_ids) for r in rows]
N = len(rows)


def quantile_bins(ix, key, k):
    s = sorted(ix, key=key)
    return [s[j * len(s) // k:(j + 1) * len(s) // k] for j in range(k)]


rng = random.Random(51)  # different seed from make_small_sets4.py (41)
lo, hi = [], []
for b in quantile_bins(range(N), lambda i: n_rej[i], 10):
    b = sorted(b, key=lambda i: lp_r[i])
    cut = int(0.3 * len(b))
    lo += rng.sample(b[:cut], 820)
    hi += rng.sample(b[-cut:], 820)
sets = {"rejlo8r": lo[:8192], "rejhi8r": hi[:8192]}

lo, hi = [], []
cells = [c for b in quantile_bins(range(N), lambda i: n_rej[i], 3) for c in quantile_bins(b, lambda i: lp_c[i], 5)]
per = 8192 // len(cells) + 1
for c in cells:
    c = sorted(c, key=lambda i: lp_r[i])
    cut = int(0.4 * len(c))
    lo += rng.sample(c[:cut], min(per, cut))
    hi += rng.sample(c[-cut:], min(per, cut))
rng.shuffle(lo); rng.shuffle(hi)
sets.update({"rejlo8c": lo[:8192], "rejhi8c": hi[:8192]})

for name, ix in sets.items():
    marg = st.mean(lp_c[i] - lp_r[i] for i in ix)
    print(f"{name}: n {len(ix)}  rejected logp {st.mean(lp_r[i] for i in ix):.3f}  chosen logp {st.mean(lp_c[i] for i in ix):.3f}  "
          f"SFT margin (chosen - rejected) {marg:+.3f}  median rejected tokens {st.median(n_rej[i] for i in ix):.0f}")
    pq.write_table(pa.Table.from_pylist([{"chosen": rows[i]["chosen"], "rejected": rows[i]["rejected"]} for i in ix]),
                   f"{D}/{name}.parquet")
