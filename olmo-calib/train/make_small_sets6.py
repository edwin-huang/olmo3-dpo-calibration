"""Margin-matched rejected-likelihood split: rejlo8g / rejhi8g.

Bins = rejected-length tercile x SFT-margin quintile (margin = chosen minus rejected mean per-token log-prob under
Instruct-SFT); within each bin the bottom/top 40% by rejected log-prob, equal counts per bin. Holds "does SFT already
agree with the preference" fixed while varying how unlikely the rejected response is (the chosen response's log-prob
then moves with it). The squeezing account predicts rejlo8g still sharpens more.

Usage: python make_small_sets6.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
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
margin = [c - r for c, r in zip(lp_c, lp_r)]
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
n_rej = [len(tok(r["rejected"][-1]["content"] or "", add_special_tokens=False).input_ids) for r in rows]


def quantile_bins(ix, key, k):
    s = sorted(ix, key=key)
    return [s[j * len(s) // k:(j + 1) * len(s) // k] for j in range(k)]


rng = random.Random(61)
cells = [c for b in quantile_bins(range(len(rows)), lambda i: n_rej[i], 3) for c in quantile_bins(b, lambda i: margin[i], 5)]
per = 8192 // len(cells) + 1
lo, hi = [], []
for c in cells:
    c = sorted(c, key=lambda i: lp_r[i])
    cut = int(0.4 * len(c))
    lo += rng.sample(c[:cut], min(per, cut))
    hi += rng.sample(c[-cut:], min(per, cut))
rng.shuffle(lo); rng.shuffle(hi)
for name, ix in (("rejlo8g", lo[:8192]), ("rejhi8g", hi[:8192])):
    print(f"{name}: n {len(ix)}  rejected logp {st.mean(lp_r[i] for i in ix):.3f}  chosen logp {st.mean(lp_c[i] for i in ix):.3f}  "
          f"SFT margin {st.mean(margin[i] for i in ix):+.3f}  median rejected tokens {st.median(n_rej[i] for i in ix):.0f}")
    pq.write_table(pa.Table.from_pylist([{"chosen": rows[i]["chosen"], "rejected": rows[i]["rejected"]} for i in ix]),
                   f"{D}/{name}.parquet")
