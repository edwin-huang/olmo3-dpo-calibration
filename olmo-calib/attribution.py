"""Which DPO training pairs push Olmo-3-7B-Instruct-SFT toward listed-option overconfidence?

First-order (TracIn-style) attribution at the start of DPO, where the policy equals the
reference. There, Ai2's length-normalized DPO loss -log σ(β(r_c - r_r)) has update direction
  u_i ∝ ∇ mean log π(y_chosen) − ∇ mean log π(y_rejected)
for every pair i (σ(0) weights all pairs equally). A pair's push on a metric M is <u_i, ∇M>.

Metrics, on 5-shot MMLU with options listed A-D (renormalized letter distribution p):
  conf = mean Σ p log p   (negative entropy: sharper = more confident)
  acc  = mean log p[correct]
Sanity check: the actual SFT->DPO weight change Δ should point along ∇conf, and <Δ, ∇M>
should roughly predict the measured change in M.

Usage: python attribution.py --n-mc 2000 --n-other 2000 --out results_attr
"""
import argparse
import json
import os
import random
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_eval import build_prompts, letter_token_ids, load_task

SFT = "allenai/Olmo-3-7B-Instruct-SFT"
DPO = "allenai/Olmo-3-7B-Instruct-DPO"
DATA = "allenai/dolci-3-instruct-dpo-with-metadata"
OPT = re.compile(r"(?m)^\s*(?:\(?[A-Ea-e][\).:]|\(?[1-5][\).])\s+\S")


# ------------------------------------------------------------ MMLU metric and its gradient


def mmlu_batches(tok, n_q, bs):
    test, shots = load_task("mmlu")
    idx = list(range(len(test)))
    random.Random(1).shuffle(idx)  # seed differs from the eval subsets
    test = [test[i] for i in idx[:n_q]]
    prompts = build_prompts(test, shots, 5, tok, chat=False)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = []
    for s in range(0, len(order), bs):
        ids = order[s : s + bs]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False)
        out.append((enc, torch.tensor([test[i]["answer"] for i in ids])))
    return out


def mmlu_metrics(model, enc, answers, letter_ids):
    """Per-batch sums of negative entropy and log p[correct] over the A-D distribution."""
    logits = model(**{k: v.cuda() for k, v in enc.items()}, logits_to_keep=1).logits[:, -1].float()
    full = torch.log_softmax(logits, -1)
    lp = torch.stack([torch.logsumexp(full[:, letter_ids[l]], -1) for l in "ABCD"], -1)
    lp = torch.log_softmax(lp, -1)  # renormalize over the four letters
    neg_ent = (lp.exp() * lp).sum(-1)
    log_correct = lp.gather(1, answers.cuda()[:, None])[:, 0]
    return neg_ent.sum(), log_correct.sum()


def metric_value(model, batches, letter_ids):
    n, c, a = 0, 0.0, 0.0
    with torch.no_grad():
        for enc, ans in batches:
            nc, la = mmlu_metrics(model, enc, ans, letter_ids)
            c, a, n = c + nc.item(), a + la.item(), n + len(ans)
    return c / n, a / n


def metric_grad(model, params, batches, letter_ids, which):
    """∇ of the mean metric, accumulated in fp32 and returned in bf16 to save memory."""
    n = sum(len(a) for _, a in batches)
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    for enc, ans in batches:
        model.zero_grad(set_to_none=True)
        nc, la = mmlu_metrics(model, enc, ans, letter_ids)
        ((nc if which == "conf" else la) / n).backward()
        for g, p in zip(acc, params):
            if p.grad is not None:
                g += p.grad.float()
    model.zero_grad(set_to_none=True)
    out = [g.to(torch.bfloat16) for g in acc]
    del acc
    torch.cuda.empty_cache()
    return out


def dot(xs, ys):
    return sum((x.float() * y.float()).sum().item() for x, y in zip(xs, ys))


# ------------------------------------------------------------ DPO pairs


def load_pairs(n_mc, n_other, seed=0):
    from huggingface_hub import snapshot_download
    import glob
    import pyarrow.parquet as pq

    d = snapshot_download(DATA, repo_type="dataset", allow_patterns=["*.parquet"])
    cols = ["chosen", "rejected", "preference_type", "prompt_id", "chosen_model", "rejected_model"]
    rows = []
    for f in sorted(glob.glob(f"{d}/**/*.parquet", recursive=True)):
        rows += pq.read_table(f, columns=cols).to_pylist()
    for i, r in enumerate(rows):
        user = [m["content"] or "" for m in r["chosen"] if m["role"] == "user"]
        r["idx"] = i
        r["is_mc"] = len(OPT.findall(user[-1] if user else "")) >= 3
    rng = random.Random(seed)
    mc = [r for r in rows if r["is_mc"]]
    other = [r for r in rows if not r["is_mc"]]
    rng.shuffle(mc)
    rng.shuffle(other)
    mc, other = mc[:n_mc], other[:n_other]
    print(f"pairs: {len(rows)} total, {sum(r['is_mc'] for r in rows)} option-listing; sampled {len(mc)} + {len(other)}")
    out = []
    for i in range(max(len(mc), len(other))):  # interleave so partial results stay balanced
        out += [x[i] for x in (mc, other) if i < len(x)]
    return out


def encode_side(tok, msgs, max_prompt=1024, max_resp=1024):
    msgs = [{"role": m["role"], "content": m["content"] or ""} for m in msgs]
    prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    if not full.startswith(prefix):
        return None
    p_ids = tok(prefix, add_special_tokens=False).input_ids[-max_prompt:]
    r_ids = tok(full[len(prefix):], add_special_tokens=False).input_ids[:max_resp]
    if not r_ids:
        return None
    return torch.tensor([p_ids + r_ids]), len(p_ids)


def mean_logp(model, ids, start):
    ids = ids.cuda()
    logits = model(ids, logits_to_keep=ids.shape[1] - start + 1).logits[0, :-1].float()
    return F.log_softmax(logits, -1).gather(1, ids[0, start:, None]).mean()


# ------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--n-other", type=int, default=2000)
    ap.add_argument("--n-q", type=int, default=1000)
    ap.add_argument("--out", default="results_attr")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pairs_path = os.path.join(args.out, "instruct_pairs.jsonl")
    dir_path = os.path.join(args.out, "instruct_direction.json")

    tok = AutoTokenizer.from_pretrained(SFT)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()  # needed for gradient checkpointing; Olmo 3 has no dropout
    params = [p for p in model.parameters()]
    letter_ids = letter_token_ids(tok)
    batches = mmlu_batches(tok, args.n_q, 8)

    print("computing ∇conf and ∇acc on MMLU ...", flush=True)
    g_conf = metric_grad(model, params, batches, letter_ids, "conf")
    g_acc = metric_grad(model, params, batches, letter_ids, "acc")
    n_conf, n_acc = dot(g_conf, g_conf) ** 0.5, dot(g_acc, g_acc) ** 0.5
    cos_ca = dot(g_conf, g_acc) / (n_conf * n_acc)

    if not os.path.exists(dir_path):
        # Sanity check: does the real SFT->DPO change move along these directions?
        print("sanity check against the actual SFT->DPO weight change ...", flush=True)
        conf_sft, acc_sft = metric_value(model, batches, letter_ids)
        sft_cpu = [p.detach().to("cpu", copy=True) for p in params]
        dpo = AutoModelForCausalLM.from_pretrained(DPO, dtype=torch.bfloat16, device_map="cpu")
        dpo_params = list(dpo.parameters())
        d_conf = d_acc = d_norm2 = 0.0
        for p, q, gc, ga in zip(params, dpo_params, g_conf, g_acc):
            delta = q.detach().cuda().float() - p.detach().float()
            d_conf += (delta * gc.float()).sum().item()
            d_acc += (delta * ga.float()).sum().item()
            d_norm2 += (delta * delta).sum().item()
        with torch.no_grad():
            for p, q in zip(params, dpo_params):
                p.copy_(q.detach().cuda())
        conf_dpo, acc_dpo = metric_value(model, batches, letter_ids)
        with torch.no_grad():
            for p, s in zip(params, sft_cpu):
                p.copy_(s.cuda())
        del dpo, dpo_params, sft_cpu
        info = {
            "n_q": args.n_q, "cos_gconf_gacc": cos_ca,
            "conf_sft": conf_sft, "conf_dpo": conf_dpo, "conf_change": conf_dpo - conf_sft,
            "conf_change_first_order": d_conf,
            "acc_sft": acc_sft, "acc_dpo": acc_dpo, "acc_change": acc_dpo - acc_sft,
            "acc_change_first_order": d_acc,
            "cos_delta_gconf": d_conf / (d_norm2 ** 0.5 * n_conf),
            "cos_delta_gacc": d_acc / (d_norm2 ** 0.5 * n_acc),
        }
        json.dump(info, open(dir_path, "w"), indent=1)
        print(json.dumps(info, indent=1), flush=True)

    done = set()
    if os.path.exists(pairs_path):
        done = {json.loads(l)["idx"] for l in open(pairs_path)}
    pairs = [p for p in load_pairs(args.n_mc, args.n_other) if p["idx"] not in done]
    print(f"{len(done)} pairs already done, {len(pairs)} to go", flush=True)

    with open(pairs_path, "a") as f:
        for k, r in enumerate(pairs):
            c, rj = encode_side(tok, r["chosen"]), encode_side(tok, r["rejected"])
            if c is None or rj is None:
                continue
            model.zero_grad(set_to_none=True)
            (mean_logp(model, *c) - mean_logp(model, *rj)).backward()
            grads = [p.grad for p in params]
            push_conf, push_acc, norm2 = 0.0, 0.0, 0.0
            for g, gc, ga in zip(grads, g_conf, g_acc):
                if g is None:
                    continue
                gf = g.float()
                push_conf += (gf * gc.float()).sum().item()
                push_acc += (gf * ga.float()).sum().item()
                norm2 += (gf * gf).sum().item()
            f.write(json.dumps({
                "idx": r["idx"], "is_mc": r["is_mc"], "preference_type": r["preference_type"],
                "prompt_id": r["prompt_id"], "chosen_model": r["chosen_model"],
                "rejected_model": r["rejected_model"],
                "len_chosen": c[0].shape[1] - c[1], "len_rejected": rj[0].shape[1] - rj[1],
                "push_conf": push_conf, "push_acc": push_acc, "grad_norm": norm2 ** 0.5,
            }) + "\n")
            if k % 100 == 0:
                f.flush()
                print(f"{k}/{len(pairs)}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
