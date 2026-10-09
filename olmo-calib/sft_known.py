"""Causal test of teacher imitation: does SFT on answers the base model doesn't know make it overconfident?

LoRA SFT from Olmo-3-1025-7B on trivia QA pairs, identical runs except for WHICH pairs:
  known   = the base model already answers them (label_trivia.py)
  unknown = the base model never answers them
  mixed   = half of each
After every epoch: MMLU (2000 questions, 5-shot, letters) and ARC-Challenge calibration (acc,
conf, ECE, T*, AUROC), plus greedy accuracy on held-out known / unknown trivia and on a sample of
the training questions (how much of the training set it has memorized).
Teacher imitation predicts: unknown-only raises T* and lowers AUROC on MMLU/ARC; known-only doesn't.

Usage: python sft_known.py --cond unknown --seed 0
"""
import argparse
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_eval import load_task
from format_check import LABELS, score_listed, summarize
from label_trivia import MODEL, correct

N_HELDOUT = 500


def split(path):
    rows = [json.loads(l) for l in open(path)]
    known = [r for r in rows if r["label"] == "known"]
    unknown = [r for r in rows if r["label"] == "unknown"]
    rng = random.Random(0)  # fixed: every run shares the same held-out questions
    rng.shuffle(known)
    rng.shuffle(unknown)
    return {"known": known[N_HELDOUT:], "unknown": unknown[N_HELDOUT:],
            "heldout_known": known[:N_HELDOUT], "heldout_unknown": unknown[:N_HELDOUT]}


def prose_rows(n, seed, k=3):
    """Control: Wikipedia sentences, trained to predict their last k words (about the length of a
    trivia answer). Same context -> short completion structure, but no question and no answer."""
    from datasets import load_dataset

    rng = random.Random(seed)
    text = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    sents = []
    for line in text:
        for sent in line.strip().split(" . "):
            w = sent.split()
            if 15 <= len(w) <= 60 and not sent.startswith("="):
                sents.append(w)
    return [{"prompt": " ".join(w[:-k]), "answer": " ".join(w[-k:])} for w in rng.sample(sents, n)]


def make_train(pools, cond, n, seed):
    rng = random.Random(seed)
    if cond == "prose":
        return prose_rows(n, seed)
    if cond == "mixed":
        data = rng.sample(pools["known"], n // 2) + rng.sample(pools["unknown"], n - n // 2)
    else:
        data = rng.sample(pools[cond], n)
    rng.shuffle(data)
    return data


def encode(tok, r):
    # Lengths are capped: one very long example padded a whole batch and ran out of GPU memory.
    prompt = r["prompt"] if "prompt" in r else f"Q: {r['question']}\nA:"
    p = tok.encode(prompt, add_special_tokens=False)[-200:]
    a = tok.encode(f" {r['answer']}\n", add_special_tokens=False)[:48]
    return p + a, [-100] * len(p) + a


def batches(tok, data, bs):
    for s in range(0, len(data), bs):
        enc = [encode(tok, r) for r in data[s : s + bs]]
        L = max(len(x) for x, _ in enc)
        ids = torch.full((len(enc), L), tok.pad_token_id)
        lab = torch.full((len(enc), L), -100)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, (x, y) in enumerate(enc):
            ids[j, : len(x)], lab[j, : len(y)], att[j, : len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids.cuda(), lab.cuda(), att.cuda()


@torch.no_grad()
def trivia_acc(model, tok, rows, bs=64):
    tok.padding_side = "left"
    ok = 0
    for s in range(0, len(rows), bs):
        part = rows[s : s + bs]
        enc = tok([f"Q: {r['question']}\nA:" for r in part], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, o in zip(part, out[:, enc["input_ids"].shape[1] :]):
            ok += correct(tok.decode(o, skip_special_tokens=True), r["aliases"])
    return ok / len(rows)


def evaluate(model, tok, evals, pools, train):
    model.eval()
    tok.padding_side = "left"
    res = {name: summarize(score_listed(model, tok, test, shots, LABELS["letters"]))
           for name, (test, shots) in evals.items()}
    for name in ("heldout_known", "heldout_unknown"):
        res[f"trivia_{name}"] = trivia_acc(model, tok, pools[name])
    qa_train = [r for r in train[:500] if "question" in r]
    res["trivia_train_fit"] = trivia_acc(model, tok, qa_train) if qa_train else float("nan")
    model.train()
    return res


def show(tag, r):
    print(f"{tag:<14}" + "  ".join(f"{k}: acc {r[k]['acc']:.3f} conf {r[k]['conf']:.3f} ECE {r[k]['ece']:.3f} "
                                   f"T* {r[k]['t_star']:.2f} AUROC {r[k]['auroc']:.3f}" for k in ("mmlu", "arc"))
          + f"  | trivia known {r['trivia_heldout_known']:.3f} unknown {r['trivia_heldout_unknown']:.3f}"
          f" train-fit {r['trivia_train_fit']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["known", "unknown", "mixed", "prose", "base"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data_trivia/labeled.jsonl")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n-q", type=int, default=2000)
    ap.add_argument("--out", default="results_sftk")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.cond}_s{args.seed}.json")

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

    pools = split(args.data)
    mmlu, mmlu_shots = load_task("mmlu")
    idx = list(range(len(mmlu)))
    random.Random(0).shuffle(idx)  # the same 2000 questions as format_check.py / mechanism.py
    evals = {"mmlu": ([mmlu[i] for i in sorted(idx[: args.n_q])], mmlu_shots), "arc": load_task("arc")}
    print({k: len(v) for k, v in pools.items()}, flush=True)

    if args.cond == "base":
        res = {"epoch_0": evaluate(model, tok, evals, pools, pools["known"][:500])}
        show("base", res["epoch_0"])
        json.dump(res, open(path, "w"), indent=1)
        print("ALL DONE")
        return

    train = make_train(pools, args.cond, args.n_train, args.seed)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = (len(train) + args.bs - 1) // args.bs
    warmup = max(1, int(0.03 * steps_per_epoch * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))

    res = {"config": vars(args)}
    model.train()
    step = 0
    for ep in range(1, args.epochs + 1):
        rng = random.Random(args.seed * 100 + ep)
        order = train[:]
        rng.shuffle(order)
        tok.padding_side = "right"
        losses = []
        for ids, lab, att in batches(tok, order, args.bs):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
        r = evaluate(model, tok, evals, pools, train)
        r["train_loss"] = sum(losses) / len(losses)
        res[f"epoch_{ep}"] = r
        show(f"{args.cond} ep{ep} (loss {r['train_loss']:.3f})", r)
        json.dump(res, open(path, "w"), indent=1)
        model.save_pretrained(os.path.join(args.out, f"lora_{args.cond}_s{args.seed}_ep{ep}"))
    print("ALL DONE")


if __name__ == "__main__":
    main()
