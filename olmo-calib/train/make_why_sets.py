"""Brief 2 data sets (run on the box, CPU only).

  why_hi8 / why_lo8   8,192 pairs each from q25_all, high vs low Instruct-SFT agreement, matched on preference source
                      and response length: strata = preference type x chosen-length decile x rejected-length 20-quantile
                      (token counts of the final responses); within each stratum the pairs are sorted by SFT margin and
                      the same number is drawn from the top 35% (hi) and the bottom 35% (lo), in proportion to stratum
                      size, so both sets keep q25_all's source mix and length distribution. Pairs with a response over
                      4,096 tokens are left out first (about 2%: the long tail, mostly repetitive rejected responses).
  why_pool.parquet    16,384 single-turn Instruct pairs that are in neither q25_all nor the held-out set, for the
                      Selective-DPO-style difficulty scoring (score_pool.py); their instruct_full row ids are kept.

Prints agreement, mean margin, source mix and median lengths of each set so the matching can be checked.

Usage: python make_why_sets.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
"""
import collections
import json
import random
import statistics as st
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

D, CACHE = sys.argv[1], sys.argv[2]
Q25 = 65000
rows = pq.read_table(f"{D}/q25_all.parquet").to_pylist()
c = torch.load(CACHE, map_location="cpu")
margin = (c["chosen_logps"] - c["rejected_logps"]).float().tolist()
meta = pq.read_table(f"{D}/instruct_meta.parquet").to_pylist()
full_n = len(meta)
perm = list(range(full_n))
random.Random(0).shuffle(perm)  # make_dpo_data.py's shuffle: q25_all = perm[:65000]
ptype = [meta[perm[i]]["preference_type"] for i in range(Q25)]
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
ntok = lambda msgs: len(tok(msgs[-1]["content"] or "", add_special_tokens=False).input_ids)
n_c = [ntok(r["chosen"]) for r in rows]
n_r = [ntok(r["rejected"]) for r in rows]


def quantile_bin(vals, k):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    q = [0] * len(vals)
    for rank, i in enumerate(order):
        q[i] = rank * k // len(vals)
    return q


qc, qr = quantile_bin(n_c, 10), quantile_bin(n_r, 20)  # fine bins so the long tail is matched too
strata = collections.defaultdict(list)
eligible = [i for i in range(Q25) if n_c[i] <= 4096 and n_r[i] <= 4096]  # drop the long tail (mostly degenerate loops)
print(f"eligible (both responses <= 4096 tokens): {len(eligible)} of {Q25}")
for i in eligible:
    strata[(ptype[i], qc[i], qr[i])].append(i)
rng = random.Random(2026)
hi, lo = [], []
for key, ix in sorted(strata.items()):
    ix = sorted(ix, key=lambda i: margin[i])
    cut = max(1, int(0.35 * len(ix)))
    k = round(8192 * 1.02 * len(ix) / len(eligible))  # 2% spare so the trim below lands exactly on 8,192
    k = min(k, cut)
    lo += rng.sample(ix[:cut], k)
    hi += rng.sample(ix[-cut:], k)
for name in ("hi", "lo"):  # trim or top up to exactly 8,192 from the matching tail of random strata
    s = hi if name == "hi" else lo
    rng.shuffle(s)
    del s[8192:]
assert len(hi) == len(lo) == 8192, (len(hi), len(lo))


def describe(name, ix):
    types = collections.Counter(ptype[i] for i in ix)
    print(f"{name}: n {len(ix)}  agreed {sum(margin[i] > 0 for i in ix) / len(ix):.3f}  mean margin {st.mean(margin[i] for i in ix):+.3f}  "
          f"median tokens chosen {st.median(n_c[i] for i in ix):.0f} rejected {st.median(n_r[i] for i in ix):.0f}  "
          f"mean tokens chosen {st.mean(n_c[i] for i in ix):.0f} rejected {st.mean(n_r[i] for i in ix):.0f}  "
          f"p90/p99 rejected {sorted(n_r[i] for i in ix)[int(.9 * len(ix))]}/{sorted(n_r[i] for i in ix)[int(.99 * len(ix))]}  "
          f"p90 chosen {sorted(n_c[i] for i in ix)[int(.9 * len(ix))]}  "
          f"types {dict((t, round(v / len(ix), 3)) for t, v in types.most_common())}")


describe("q25_all", range(Q25))
for name, ix in (("why_hi8", hi), ("why_lo8", lo)):
    describe(name, ix)
    pq.write_table(pa.Table.from_pylist([{"chosen": rows[i]["chosen"], "rejected": rows[i]["rejected"]} for i in ix]),
                   f"{D}/{name}.parquet")

# pool for the difficulty scoring: single-turn pairs outside q25_all and outside the held-out set
import os
if os.path.exists(f"{D}/why_pool.parquet"):
    print("why_pool.parquet exists; not rebuilt"); sys.exit()
full = pq.read_table(f"{D}/instruct_full.parquet", columns=["chosen", "rejected"])
import hashlib
text = lambda msgs: json.dumps([[m["role"], m["content"] or ""] for m in msgs], ensure_ascii=False)
pkey = lambda r: hashlib.sha1((text(r["chosen"]) + "\x00" + text(r["rejected"])).encode()).hexdigest()
q25_keys = {pkey(r) for r in rows}  # instruct_full has exact duplicate pairs at different rows (4 hit the first pool)
held = {r["chosen"][0]["content"] for r in pq.read_table(f"{D}/heldout_pairs.parquet").to_pylist()}
held |= {json.loads(l)["prompt"] for l in open(f"{D}/judge_prompts.jsonl")}
cand = []
for j in perm[Q25:]:
    r = full.slice(j, 1).to_pylist()[0]
    if len(r["chosen"]) != 2 or len(r["rejected"]) != 2 or r["chosen"][0]["content"] in held or pkey(r) in q25_keys:
        continue
    cand.append({"row": j, "chosen": r["chosen"], "rejected": r["rejected"]})
    if len(cand) == 16384:
        break
pq.write_table(pa.Table.from_pylist(cand), f"{D}/why_pool.parquet")
print(f"why_pool: {len(cand)} single-turn pairs outside q25_all and the held-out set")
