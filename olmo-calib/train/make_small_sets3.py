"""Third wave of small sets, after pilot8a (T* 2.39) and pilot8b (T* 1.95) showed large subset-to-subset variance.

  s8_c, s8_d   two more random 8,192-pair subsets of q25_all, disjoint from s16_a and s8_b (spread of T* across draws)
  nomc8a       s8_a without its option-listing prompts (same data otherwise, so subset noise mostly cancels vs pilot8a)
Prints each set's make-up.

Usage: python make_small_sets3.py /workspace/dpo_data
"""
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

D = sys.argv[1]
OPT = re.compile(r"(?m)^\s*(?:\(?[A-Ea-e][\).:]|\(?[1-5][\).])\s+\S")
key = lambda r: (r["chosen"][-1]["content"][:200], r["rejected"][-1]["content"][:200])
used = {key(r) for n in ("s16_a", "s8_b") for r in pq.read_table(f"{D}/{n}.parquet").to_pylist()}
pool = [r for r in pq.read_table(f"{D}/q25_all.parquet").to_pylist() if key(r) not in used]
random.Random(31).shuffle(pool)
s8a = pq.read_table(f"{D}/s8_a.parquet").to_pylist()


def is_mc(r):
    u = [m["content"] or "" for m in r["chosen"] if m["role"] == "user"]
    return len(OPT.findall(u[-1] if u else "")) >= 3


sets = {"s8_c": pool[:8192], "s8_d": pool[8192:16384], "nomc8a": [r for r in s8a if not is_mc(r)]}
for name, rows in sets.items():
    print(f"{name:<7} n {len(rows):>5}  option-listing prompts {sum(map(is_mc, rows)) / len(rows):5.1%}")
    pq.write_table(pa.Table.from_pylist([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]),
                   f"{D}/{name}.parquet")
