"""Second wave of small sets (8,192 pairs each), to test what in DPO's update drives the sharpening.

  rev8a    s8_a with chosen and rejected swapped in every pair     (does learning the preference matter, or just moving?)
  swap8a   s8_a with a random 50% of pairs swapped                  (no consistent preference signal)
  nomc8    8,192 pairs from q25_all whose prompt lists no options (A-D / 1-5); disjoint from s8_a
  short8   8,192 pairs whose chosen response is <= 40 tokens       (answer-commitment-heavy)
  long8    8,192 pairs whose chosen response is >= 300 tokens
Prints each set's make-up.

Usage: python make_small_sets2.py /workspace/dpo_data
"""
import collections
import random
import re
import statistics as st
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

D = sys.argv[1]
OPT = re.compile(r"(?m)^\s*(?:\(?[A-Ea-e][\).:]|\(?[1-5][\).])\s+\S")
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
s8a = pq.read_table(f"{D}/s8_a.parquet").to_pylist()
q25 = pq.read_table(f"{D}/q25_all.parquet").to_pylist()
key = lambda r: (r["chosen"][-1]["content"][:200], r["rejected"][-1]["content"][:200])
in_s8a = {key(r) for r in s8a}
pool = [r for r in q25 if key(r) not in in_s8a]
random.Random(21).shuffle(pool)


def user_text(r):
    u = [m["content"] or "" for m in r["chosen"] if m["role"] == "user"]
    return u[-1] if u else ""


def n_chosen(r):
    return len(tok(r["chosen"][-1]["content"] or "", add_special_tokens=False).input_ids)


rng = random.Random(22)
sets = {
    "rev8a": [{"chosen": r["rejected"], "rejected": r["chosen"]} for r in s8a],
    "swap8a": [({"chosen": r["rejected"], "rejected": r["chosen"]} if rng.random() < 0.5 else r) for r in s8a],
}
nomc, short, long_ = [], [], []
for r in pool:
    if len(nomc) < 8192 and len(OPT.findall(user_text(r))) < 3:
        nomc.append(r)
    if len(short) < 8192 or len(long_) < 8192:
        n = n_chosen(r)
        if n <= 40 and len(short) < 8192:
            short.append(r)
        elif n >= 300 and len(long_) < 8192:
            long_.append(r)
    if len(nomc) >= 8192 and len(short) >= 8192 and len(long_) >= 8192:
        break
sets.update({"nomc8": nomc, "short8": short, "long8": long_})
for name, rows in sets.items():
    sample = rows[:1000]
    mc = sum(len(OPT.findall(user_text(r))) >= 3 for r in rows) / len(rows)
    print(f"{name:<7} n {len(rows):>5}  option-listing prompts {mc:5.1%}  median chosen tokens "
          f"{st.median(n_chosen(r) for r in sample):>5.0f}  median rejected tokens "
          f"{st.median(len(tok(r['rejected'][-1]['content'] or '', add_special_tokens=False).input_ids) for r in sample):>5.0f}")
    pq.write_table(pa.Table.from_pylist([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]),
                   f"{D}/{name}.parquet")
