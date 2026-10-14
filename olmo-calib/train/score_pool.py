"""Brief 2, Experiment 5 input: Selective-DPO-style difficulty scores (GPU).

Selective DPO ranks preference pairs by how easy they are for a model already trained with DPO, via the implicit
reward margin h = (log pi(y_c) - log ref(y_c)) - (log pi(y_r) - log ref(y_r)), and trains on the easiest. Here pi is
our 25%-scale plain DPO model (q25_plain), ref is Instruct-SFT, both scored with mean per-token log-probs (as in
dpo_norm) on the 16,384 pool pairs, which neither model trained on. Keeps the 8,192 easiest (largest h): a 50% keep
ratio. Writes why_easy8.parquet and why_pool_scores.jsonl; prints the easy set's SFT agreement share.

Usage: CUDA_VISIBLE_DEVICES=0 python score_pool.py /workspace/dpo_data /workspace/runs/q25_all
"""
import json
import statistics as st
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

D, POLICY = sys.argv[1], sys.argv[2]
REF = "allenai/Olmo-3-7B-Instruct-SFT"
import os
LIMIT = int(os.environ.get("SCORE_LIMIT", "0"))  # smoke test: score only the first N pairs, write *_smoke files
pool = pq.read_table(f"{D}/why_pool.parquet").to_pylist()[: LIMIT or None]
SUFFIX = "_smoke" if LIMIT else ""
tok = AutoTokenizer.from_pretrained(REF)


def encode(msgs, max_prompt=1024, max_resp=2048):
    msgs = [{"role": m["role"], "content": m["content"] or ""} for m in msgs]
    prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p = tok(prefix, add_special_tokens=False).input_ids[-max_prompt:]
    r = tok(full[len(prefix):], add_special_tokens=False).input_ids[:max_resp]
    return torch.tensor([p + r]), len(p)


enc = [(encode(r["chosen"]), encode(r["rejected"])) for r in pool]


@torch.no_grad()
def mean_logps(name):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for c, r in enc:
        pair = []
        for ids, start in (c, r):
            ids = ids.cuda()
            z = model(ids).logits[0, start - 1:-1].float()
            pair.append(F.log_softmax(z, -1).gather(1, ids[0, start:, None]).mean().item())
        out.append(pair)
    del model
    torch.cuda.empty_cache()
    return out


ref = mean_logps(REF)
pol = mean_logps(POLICY)
h = [(pc - rc) - (pr - rr) for (pc, pr), (rc, rr) in zip(pol, ref)]
sft_margin = [rc - rr for rc, rr in ref]
with open(f"{D}/why_pool_scores{SUFFIX}.jsonl", "w") as f:
    for r, a, b, x in zip(pool, ref, pol, h):
        f.write(json.dumps({"row": r["row"], "ref": a, "policy": b, "implicit_margin": x}) + "\n")
order = sorted(range(len(pool)), key=lambda i: -h[i])
K = len(pool) // 2
easy = order[:K]
print(f"pool {len(pool)}: SFT agreed {sum(m > 0 for m in sft_margin) / len(pool):.3f}, implicit margin mean {st.mean(h):+.3f}")
print(f"easy {K} (50% keep): SFT agreed {sum(sft_margin[i] > 0 for i in easy) / K:.3f}, mean SFT margin "
      f"{st.mean(sft_margin[i] for i in easy):+.3f}, implicit margin >= {h[easy[-1]]:+.3f}")
rest = order[K:]
print(f"rest {len(rest)}: SFT agreed {sum(sft_margin[i] > 0 for i in rest) / len(rest):.3f}")
pq.write_table(pa.Table.from_pylist([{"chosen": pool[i]["chosen"], "rejected": pool[i]["rejected"]} for i in easy]),
               f"{D}/why_easy8{SUFFIX}.parquet")
