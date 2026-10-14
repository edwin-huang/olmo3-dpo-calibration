"""Data for the fix wave: tests of the agreement account with standard DPO variants (run on the box).

  filt8               8,192 random q25_all pairs with Instruct-SFT margin <= 0.1 (the agreement filter; fix_preview.py
                      predicts T* ~1.5 at 23% agreement). Drawn from the 65K pool, so it trains as many steps as s8_a.
  heldout_pairs       1,000 single-turn Instruct pairs that are NOT in q25_all (instruct_full order after the q25 cut,
                      prompts not in q25_all): eval_ckpt.py's "pairs" part (does the model prefer the chosen response?)
  judge_prompts.jsonl 300 further held-out single-turn prompts (<= 1024 tokens): eval_ckpt.py's "gen" part, judged
                      locally against Instruct-SFT's responses (judge_winrate.py)
Also prints DPO_AGREE_NORM for agreement-weighted DPO: the mean of sigmoid(-k * SFT margin) over s8_a, so the weights
average 1 and the total update size stays comparable to plain DPO.

Usage: python make_fix_sets.py /workspace/dpo_data /workspace/ref_cache/92f0e85ab3b1d79a.pt
"""
import json
import math
import random
import statistics as st
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

D, CACHE = sys.argv[1], sys.argv[2]
Q25 = 65000
q25 = pq.read_table(f"{D}/q25_all.parquet").to_pylist()
c = torch.load(CACHE, map_location="cpu")
margin = (c["chosen_logps"] - c["rejected_logps"]).float().tolist()
assert len(margin) == len(q25) == Q25

# filt8
rng = random.Random(81)
low = [i for i in range(Q25) if margin[i] <= 0.1]
ix = rng.sample(low, 8192)
print(f"filt8: pool {len(low)} ({len(low) / Q25:.0%} of q25_all); set agree {sum(margin[i] > 0 for i in ix) / 8192:.0%}, "
      f"mean margin {st.mean(margin[i] for i in ix):+.3f}")
pq.write_table(pa.Table.from_pylist([{"chosen": q25[i]["chosen"], "rejected": q25[i]["rejected"]} for i in ix]),
               f"{D}/filt8.parquet")

# DPO_AGREE_NORM over s8_a (make_small_sets.py: s8_a = the first 8192 of a Random(11) shuffle of q25_all)
idx = list(range(Q25))
random.Random(11).shuffle(idx)
s8a = idx[:8192]
for k in (3, 5, 10):
    w = [1 / (1 + math.exp(k * margin[i])) for i in s8a]
    agreed = [x for x, i in zip(w, s8a) if margin[i] > 0]
    print(f"DPO_AGREE_K={k}: DPO_AGREE_NORM={st.mean(w):.4f}  (agreed pairs carry {sum(agreed) / sum(w):.0%} of the "
          f"weight vs {len(agreed) / len(w):.0%} of the pairs)")

# held-out pairs and prompts: instruct_full in make_dpo_data.py's Random(0) shuffle order, after the q25 cut
full = pq.read_table(f"{D}/instruct_full.parquet", columns=["chosen", "rejected"]).to_pylist()
order = list(range(len(full)))
random.Random(0).shuffle(order)
user = lambda msgs: msgs[0]["content"] or ""
seen = {user(r["chosen"]) for r in q25}
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
pairs, prompts = [], []
for i in order[Q25:]:
    r = full[i]
    if len(r["chosen"]) != 2 or len(r["rejected"]) != 2 or r["chosen"][0]["role"] != "user":
        continue
    u = user(r["chosen"])
    if u in seen or not u.strip() or not (r["chosen"][1]["content"] or "").strip() or not (r["rejected"][1]["content"] or "").strip():
        continue
    seen.add(u)
    if len(pairs) < 1000:
        pairs.append({"chosen": r["chosen"], "rejected": r["rejected"]})
    elif len(tok(u, add_special_tokens=False).input_ids) <= 1024:
        prompts.append({"id": len(prompts), "prompt": u})
        if len(prompts) == 300:
            break
pq.write_table(pa.Table.from_pylist(pairs), f"{D}/heldout_pairs.parquet")
with open(f"{D}/judge_prompts.jsonl", "w") as f:
    f.writelines(json.dumps(p) + "\n" for p in prompts)
print(f"heldout_pairs {len(pairs)}, judge_prompts {len(prompts)}")
