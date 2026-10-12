"""Margin split at matched rejected likelihood: marglo8 / marghi8.

The margin-matched split (rejlo8g / rejhi8g: rejected log-prob −1.69 vs −0.61 at margin +0.30 vs +0.07) gave the same
T* (2.37 / 2.37), while every split where SFT's margin differed in sign did not. So the driver looks like SFT's own
preference margin (chosen minus rejected mean per-token log-prob), not how unlikely the rejected response is. This
holds the rejected response's likelihood (and length) fixed and varies the margin: bins = rejected-length tercile x
rejected-log-prob quintile, within each bin the bottom/top 40% by margin, equal counts per bin.
Prediction: marghi8 (SFT already agrees) sharpens; marglo8 (SFT prefers the rejected response) does not.

Usage: python make_small_sets7.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
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


rng = random.Random(71)
cells = [c for b in quantile_bins(range(len(rows)), lambda i: n_rej[i], 3) for c in quantile_bins(b, lambda i: lp_r[i], 5)]
per = 8192 // len(cells) + 1
lo, hi = [], []
for c in cells:
    c = sorted(c, key=lambda i: margin[i])
    cut = int(0.4 * len(c))
    lo += rng.sample(c[:cut], min(per, cut))
    hi += rng.sample(c[-cut:], min(per, cut))
rng.shuffle(lo); rng.shuffle(hi)
for name, ix in (("marglo8", lo[:8192]), ("marghi8", hi[:8192])):
    print(f"{name}: n {len(ix)}  rejected logp {st.mean(lp_r[i] for i in ix):.3f}  chosen logp {st.mean(lp_c[i] for i in ix):.3f}  "
          f"SFT margin {st.mean(margin[i] for i in ix):+.3f}  share margin>0 {sum(margin[i] > 0 for i in ix) / len(ix):.0%}  "
          f"median rejected tokens {st.median(n_rej[i] for i in ix):.0f}")
    pq.write_table(pa.Table.from_pylist([{"chosen": rows[i]["chosen"], "rejected": rows[i]["rejected"]} for i in ix]),
                   f"{D}/{name}.parquet")
