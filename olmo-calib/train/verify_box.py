"""Box-side checks for VERIFICATION.md (CPU only; safe to run while a GPU job trains).

  d1  Held-out pairs and judge prompts vs every training parquet: exact (prompt, chosen, rejected) hash overlap, and
      prompt-text overlap. Expected: no overlap with any small-test or 25%-scale set; full overlap with instruct_full
      (the full run and Ai2's released DPO trained on all pairs).
  d2  The 3,000-question MMLU subset: identical (subject, answer) sequence in every results_ckpt/**/mmlu5_3000 file,
      and identical full-MMLU sequence in every mmlu5_full file.
  e   Agreement share recomputed from q25's cached reference log-probs on 500 random pairs (seed 2026), with 5
      example pairs; cross-check against the SFT margins eval_ckpt computed itself on the 1,000 held-out pairs.
  c   The diff of /workspace/open-instruct against its pinned commit, saved to verify_out/open_instruct.diff.

Usage: /workspace/open-instruct/.venv/bin/python verify_box.py > /workspace/verify_out/verify_box.log
"""
import glob
import hashlib
import json
import os
import random
import subprocess

import pyarrow.parquet as pq
import torch

WS = "/workspace"
D = f"{WS}/dpo_data"
OUT = f"{WS}/verify_out"
os.makedirs(OUT, exist_ok=True)


def text(msgs):
    return json.dumps([[m["role"], m["content"] or ""] for m in msgs], ensure_ascii=False)


def key(r):
    return hashlib.sha1((text(r["chosen"]) + "\x00" + text(r["rejected"])).encode()).hexdigest()


def prompt(r):
    return hashlib.sha1(text(r["chosen"][:-1]).encode()).hexdigest()


print("== d1. held-out pairs / judge prompts vs training sets")
held = pq.read_table(f"{D}/heldout_pairs.parquet").to_pylist()
held_keys, held_prompts = {key(r) for r in held}, {prompt(r) for r in held}
judge = [json.loads(l) for l in open(f"{D}/judge_prompts.jsonl")]
judge_prompts = {hashlib.sha1(json.dumps([["user", j["prompt"]]], ensure_ascii=False).encode()).hexdigest() for j in judge}
print(f"held-out pairs {len(held)} (unique {len(held_keys)}), judge prompts {len(judge)}")
for path in sorted(glob.glob(f"{D}/*.parquet")):
    name = os.path.basename(path)[:-8]
    if name in ("heldout_pairs", "instruct_meta", "think_full", "summary"):
        continue
    t = pq.read_table(path, columns=["chosen", "rejected"]).to_pylist()
    ks, ps = {key(r) for r in t}, {prompt(r) for r in t}
    print(f"  {name:<14} n {len(t):>7}  pair overlap {len(held_keys & ks):>5}  held-out prompt overlap {len(held_prompts & ps):>5}"
          f"  judge prompt overlap {len(judge_prompts & ps):>4}")

print("== d2. MMLU subsets identical across runs")
for sub in ("mmlu5_3000", "mmlu5_full"):
    seqs = {}
    for f in glob.glob(f"{WS}/results_ckpt/**/{sub}/letters.jsonl", recursive=True):
        rows = [json.loads(l) for l in open(f)]
        h = hashlib.sha1(json.dumps([(r["subject"], r["answer"]) for r in rows]).encode()).hexdigest()[:12]
        seqs.setdefault((len(rows), h), []).append(os.path.relpath(f, f"{WS}/results_ckpt"))
    print(f"  {sub}: {sum(len(v) for v in seqs.values())} files, {len(seqs)} distinct question sequences")
    for (n, h), files in seqs.items():
        print(f"    n {n} hash {h}: {len(files)} files, e.g. {files[:3]}")

print("== e. agreement share from q25's reference cache")
rows = pq.read_table(f"{D}/q25_all.parquet").to_pylist()
c = torch.load(f"{WS}/ref_cache/92f0e85ab3b1d79a.pt", map_location="cpu")
m = (c["chosen_logps"] - c["rejected_logps"]).float()
print(f"  all 65,000: agreed {(m > 0).float().mean().item():.3f}, mean margin {m.mean().item():+.3f}")
ix = random.Random(2026).sample(range(len(rows)), 500)
ms = m[ix]
print(f"  random 500 (seed 2026): agreed {(ms > 0).float().mean().item():.3f}, mean margin {ms.mean().item():+.3f}")
for i in ix[:5]:
    r = rows[i]
    clip = lambda s: (s or "").replace("\n", " ")[:160]
    print(f"  --- pair {i}: SFT margin {m[i].item():+.3f} (chosen {c['chosen_logps'][i].item():.3f}, rejected {c['rejected_logps'][i].item():.3f} per token)")
    print(f"      prompt:   {clip(r['chosen'][-2]['content'])}")
    print(f"      chosen:   {clip(r['chosen'][-1]['content'])}")
    print(f"      rejected: {clip(r['rejected'][-1]['content'])}")
ref = f"{WS}/results_ckpt/_ref/pairs__Olmo-3-7B-Instruct-SFT__2000.json"  # 1,000 pairs x 2 (the __20 file is a smoke test)
if os.path.exists(ref):
    v = json.load(open(ref))
    mm = [v[2 * i] - v[2 * i + 1] for i in range(len(v) // 2)]
    print(f"  cross-check, eval_ckpt's own SFT forward on the 1,000 held-out pairs: agreed {sum(x > 0 for x in mm) / len(mm):.3f}")

print("== c. open-instruct diff")
oi = f"{WS}/open-instruct"
head = subprocess.run(["git", "-C", oi, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
diff = subprocess.run(["git", "-C", oi, "diff"], capture_output=True, text=True).stdout
open(f"{OUT}/open_instruct.diff", "w").write(diff)
print(f"  HEAD {head}; diff {len(diff.splitlines())} lines -> {OUT}/open_instruct.diff")
grep = subprocess.run(["grep", "-n", "DPO_AGREE\\|ref_margin\\|pair_w\\|losses = losses\\|loss = losses.mean\\|reference_cache\\[batch\\|average_log_prob=args.loss_type.is_average_loss",
                       f"{oi}/open_instruct/dpo_tune_cache.py"], capture_output=True, text=True).stdout
print(grep)
