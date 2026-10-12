"""Length-matched rejected-likelihood split (after reverse DPO showed that WHAT is pushed down sets the direction).

From q25_all: bin pairs by rejected-response token length (deciles); within each bin take the 30% of pairs whose
rejected response is LEAST likely under Instruct-SFT (mean per-token log-prob, q25's reference cache) and the 30% MOST
likely; sample the same number per bin into rejlo8m / rejhi8m (8,192 each), so both have the same rejected-length
distribution. The squeezing account predicts rejlo8m sharpens more.

Usage: python make_small_sets4.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
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
n_cho = [len(tok(r["chosen"][-1]["content"] or "", add_special_tokens=False).input_ids) for r in rows]
order = sorted(range(len(rows)), key=lambda i: n_rej[i])
bins = [order[k * len(order) // 10:(k + 1) * len(order) // 10] for k in range(10)]
rng = random.Random(41)
lo, hi = [], []
for b in bins:
    b = sorted(b, key=lambda i: lp_r[i])
    cut = int(0.3 * len(b))
    lo += rng.sample(b[:cut], 820)
    hi += rng.sample(b[-cut:], 820)
for name, ix in (("rejlo8m", lo[:8192]), ("rejhi8m", hi[:8192])):
    print(f"{name}: n {len(ix)}  rejected logp {st.mean(lp_r[i] for i in ix):.3f}  chosen logp {st.mean(lp_c[i] for i in ix):.3f}  "
          f"median rejected tokens {st.median(n_rej[i] for i in ix):.0f}  median chosen tokens {st.median(n_cho[i] for i in ix):.0f}")
    pq.write_table(pa.Table.from_pylist([{"chosen": rows[i]["chosen"], "rejected": rows[i]["rejected"]} for i in ix]),
                   f"{D}/{name}.parquet")
