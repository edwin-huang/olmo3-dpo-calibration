"""Small training sets for the root-cause tests, all drawn from q25_all (65K random Instruct DPO pairs).

  s16_a     16,384 random pairs                     pilot at LR 4e-6 (same cumulative LR as q25 at 1e-6)
  s8_a      the first 8,192 of s16_a                pilot at LR 8e-6
  s8_b      8,192 other random pairs (disjoint)     data-subset noise: s8_a vs s8_b
  rejlo8    8,192 random from the quartile of pairs whose rejected response is LEAST likely under Instruct-SFT
  rejhi8    8,192 random from the quartile whose rejected response is MOST likely
            (mean per-token log-prob from q25's reference cache; the squeezing effect predicts rejlo8 sharpens more)
Prints each set's make-up (lengths, preference types, flag share, mean log-probs) since the likelihood split is
confounded with length and source.

Usage: python make_small_sets.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
"""
import collections
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
assert len(cache["rejected_logps"]) == len(rows) == 65000
lp_c, lp_r = cache["chosen_logps"].float().tolist(), cache["rejected_logps"].float().tolist()

# q25_all rows -> instruct_full rows (for preference type and D4 flag), as in build_rand10.py
full = pq.read_table(f"{D}/instruct_full.parquet", columns=["rejected"]).num_rows
meta = pq.read_table(f"{D}/instruct_meta.parquet").to_pylist()
perm = list(range(full))
random.Random(0).shuffle(perm)
info = [meta[i] for i in perm[:65000]]

tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
ntok = lambda m: len(tok(m[-1]["content"], add_special_tokens=False).input_ids)

rng = random.Random(11)
idx = list(range(65000))
rng.shuffle(idx)
order_r = sorted(range(65000), key=lambda i: lp_r[i])
q = 65000 // 4
sets = {
    "s16_a": idx[:16384],
    "s8_a": idx[:8192],
    "s8_b": idx[16384:16384 + 8192],
    "rejlo8": random.Random(12).sample(order_r[:q], 8192),
    "rejhi8": random.Random(13).sample(order_r[-q:], 8192),
}
print(f"{'set':<8}{'n':>7}{'lp chosen':>11}{'lp rej':>9}{'tok chosen':>12}{'tok rej':>9}{'flagged':>9}  types")
for name, ix in sets.items():
    sample = ix[:1500]
    types = collections.Counter(info[i]["preference_type"][:5] for i in ix)
    print(f"{name:<8}{len(ix):>7}{st.mean(lp_c[i] for i in ix):>11.3f}{st.mean(lp_r[i] for i in ix):>9.3f}"
          f"{st.median(ntok(rows[i]['chosen']) for i in sample):>12.0f}{st.median(ntok(rows[i]['rejected']) for i in sample):>9.0f}"
          f"{sum(info[i]['flag'] for i in ix) / len(ix):>9.1%}  {dict(types)}")
    pq.write_table(pa.Table.from_pylist([{"chosen": rows[i]["chosen"], "rejected": rows[i]["rejected"]} for i in ix]),
                   f"{D}/{name}.parquet")
