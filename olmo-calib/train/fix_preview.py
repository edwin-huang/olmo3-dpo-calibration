"""Offline preview of agreement-based data filters for Instruct DPO (no training).

For each candidate filter over q25_all (Instruct-SFT margins from q25's reference cache): share of pairs kept,
agreement share of the kept pairs, mix of preference types, and the log T* predicted by the cross-test fit over the 17
small plain-DPO tests (log T* ~ +0.91 agree + 0.06 log median chosen length - 0.16; R2 0.65). The fit only covers
agreement shares 21-95% at 8K pairs / LR 8e-6, so predictions are a ranking, not a forecast for full-scale DPO; and
dropping agreed pairs may cost capability (untested).

Usage: python fix_preview.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
"""
import collections
import math
import random
import sys

import pyarrow.parquet as pq
import torch

D, CACHE = sys.argv[1:3]
c = torch.load(CACHE, map_location="cpu")
margin = (c["chosen_logps"] - c["rejected_logps"]).float().tolist()
meta = pq.read_table(f"{D}/instruct_meta.parquet").to_pylist()
perm = list(range(len(meta)))
random.Random(0).shuffle(perm)
info = [meta[i] for i in perm[:65000]]
N = len(margin)
LOG_LEN = math.log(330)  # median chosen length of the random 8K sets


def pred_T(agree):
    return math.exp(0.91 * agree + 0.06 * LOG_LEN - 0.16)


def report(name, keep):
    k = [i for i in range(N) if keep(i)]
    agree = sum(margin[i] > 0 for i in k) / len(k)
    types = collections.Counter(info[i]["preference_type"][:5] for i in k)
    print(f"{name:<44} keeps {len(k) / N:6.1%}  agree {agree:5.1%}  predicted T* {pred_T(agree):.2f}  "
          f"mix {', '.join(f'{t} {v / len(k):.0%}' for t, v in types.most_common())}")


print(f"{'filter':<44}")
report("none (q25_all)", lambda i: True)
for tau in (1.0, 0.5, 0.3, 0.1, 0.0):
    report(f"drop pairs with SFT margin > {tau}", lambda i, t=tau: margin[i] <= t)
report("GPT-judged pairs only", lambda i: info[i]["preference_type"] == "llm_judged")
report("delta pairs only", lambda i: info[i]["preference_type"] == "delta_learning")
rng = random.Random(0)
agreed = [i for i in range(N) if margin[i] > 0]
dis = [i for i in range(N) if margin[i] <= 0]
for target in (0.5, 0.4):
    n_keep = int(target / (1 - target) * len(dis))
    keep = set(dis) | set(rng.sample(agreed, min(n_keep, len(agreed))))
    report(f"subsample agreed pairs to {target:.0%} agreement", lambda i, s=keep: i in s)
