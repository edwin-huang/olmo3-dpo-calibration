"""Random-swap control for D4: q25_all with a RANDOM 6,275 pairs swapped out, built exactly like q25_d4 otherwise.

q25_d4 = q25_all minus its 6,275 flagged pairs (answer-first / degenerate rejected) plus the next unflagged pairs of the
same preference types from the shuffled pool, reshuffled. Here the removed pairs are drawn at random with the same
per-type counts, and the replacements are the next pairs of those types regardless of flag. The T* gap between this
and q25_all measures how much a 10% data change moves results by itself (the noise D4's +0.17 must exceed).

Usage: python build_rand10.py /workspace/dpo_data
"""
import collections
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

D = sys.argv[1]
full = pq.read_table(f"{D}/instruct_full.parquet").to_pylist()
meta = pq.read_table(f"{D}/instruct_meta.parquet").to_pylist()
perm = list(range(len(full)))
random.Random(0).shuffle(perm)  # make_dpo_data.py: rng = Random(0); rng.shuffle(rows) is the first use of rng
q25 = perm[:65000]
ref = pq.read_table(f"{D}/q25_all.parquet").slice(0, 50).to_pylist()
assert all(full[i]["chosen"] == r["chosen"] and full[i]["rejected"] == r["rejected"] for i, r in zip(q25, ref)), \
    "could not reproduce q25_all's order"

flagged = collections.Counter(meta[i]["preference_type"] for i in q25 if meta[i]["flag"])
print("flagged pairs in q25_all by type:", dict(flagged), "total", sum(flagged.values()))
rng = random.Random(1)
remove = set()
for t, k in flagged.items():
    remove |= set(rng.sample([i for i in q25 if meta[i]["preference_type"] == t], k))
add = []
for t, k in flagged.items():
    add += [i for i in perm[65000:] if meta[i]["preference_type"] == t][:k]
keep = [i for i in q25 if i not in remove] + add
rng.shuffle(keep)
print(f"removed {len(remove)} (of which flagged {sum(meta[i]['flag'] for i in remove)}), "
      f"added {len(add)} (of which flagged {sum(meta[i]['flag'] for i in add)}), total {len(keep)}")
pq.write_table(pa.Table.from_pylist([{"chosen": full[i]["chosen"], "rejected": full[i]["rejected"]} for i in keep]),
               f"{D}/q25_rand10.parquet")
