"""Multiple-choice calibration eval for one model checkpoint.

Confidence = probability mass on the predicted option letter, renormalized over
A/B/C/D. Every checkpoint gets the same plain-text prompt so stages are compared
apples-to-apples; post-trained models can also be run with --chat.

The model is loaded once and run on every eval in --evals, each written to
<out>/<eval name>/<tag>.{summary.json,jsonl}. An eval is task:n_shot:limit, e.g.
  mmlu:5:3000   5-shot, fixed random 3000-question subset (the original setup)
  mmlu:0:0      0-shot, all 14,042 questions
  arc:5:0       ARC-Challenge test, 5-shot, all 4-option questions

Usage:
  python calib_eval.py --model allenai/Olmo-3-7B-Instruct
  python calib_eval.py --model allenai/Olmo-3-1025-7B --revision stage1-step1413814 \
      --evals mmlu:5:0,arc:5:0 --out results_robust
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]


def load_task(task):
    """Returns (test rows, few-shot pool by subject). Rows: question, choices, answer, subject."""
    if task == "mmlu":
        test = list(load_dataset("cais/mmlu", "all", split="test"))
        shots = defaultdict(list)
        for row in load_dataset("cais/mmlu", "all", split="dev"):
            shots[row["subject"]].append(row)
        return test, shots
    if task == "arc":
        def norm(split):
            rows = []
            for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split):
                labels = r["choices"]["label"]
                if len(labels) != 4 or r["answerKey"] not in labels:
                    continue  # a few questions have 3 or 5 options
                rows.append({
                    "question": r["question"],
                    "choices": r["choices"]["text"],
                    "answer": labels.index(r["answerKey"]),
                    "subject": "science",
                })
            return rows
        train = norm("train")
        random.Random(0).shuffle(train)
        return norm("test"), {"science": train[:5]}
    raise ValueError(f"unknown task {task}")


def format_q(row, with_answer):
    s = row["question"].strip() + "\n"
    for letter, choice in zip(LETTERS, row["choices"]):
        s += f"{letter}. {choice}\n"
    s += "Answer:"
    if with_answer:
        s += f" {LETTERS[row['answer']]}\n\n"
    return s


def build_prompts(test, shots, n_shot, tok, chat):
    prompts = []
    for row in test:
        subj = row["subject"].replace("_", " ")
        header = f"The following are multiple choice questions (with answers) about {subj}.\n\n"
        fewshot = "".join(format_q(r, True) for r in shots[row["subject"]][:n_shot])
        body = header + fewshot + format_q(row, False)
        if chat:
            body = tok.apply_chat_template(
                [{"role": "user", "content": body + "\n\nRespond with only the letter."}],
                tokenize=False, add_generation_prompt=True,
            )
        prompts.append(body)
    return prompts


def letter_token_ids(tok):
    """Token ids for each letter, both ' A' (after 'Answer:') and 'A' (chat reply start)."""
    ids = {}
    for letter in LETTERS:
        variants = []
        for text in (f" {letter}", letter):
            enc = tok.encode(text, add_special_tokens=False)
            if len(enc) == 1:
                variants.append(enc[0])
        assert variants, f"no single-token encoding for {letter}"
        ids[letter] = sorted(set(variants))
    return ids


def load_config_override(model, revision):
    """Olmo-3-7B-RL-Zero-Mix ships config.json with the pre-release model_type
    'olmo2-retrofit', which transformers doesn't know. Its hyperparameters and all
    355 weight names match Olmo-3-7B-Instruct exactly, so load it as Olmo 3."""
    from huggingface_hub import hf_hub_download
    from transformers import Olmo3Config

    with open(hf_hub_download(model, "config.json", revision=revision)) as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "olmo2-retrofit":
        return {}
    print(f"{model}: config says model_type=olmo2-retrofit; loading as olmo3", flush=True)
    cfg.update(model_type="olmo3", architectures=["Olmo3ForCausalLM"])
    for k in ("transformers_version", "torch_dtype"):
        cfg.pop(k, None)
    return {"config": Olmo3Config(**cfg)}


def ece(confs, correct, n_bins=15):
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confs, correct):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, y))
    total = len(confs)
    err, diagram = 0.0, []
    for i, b in enumerate(bins):
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        err += len(b) / total * abs(avg_c - acc)
        diagram.append({"bin": i, "n": len(b), "conf": avg_c, "acc": acc})
    return err, diagram


def eval_dir(out, spec):
    task, n_shot, limit = spec.split(":")
    return os.path.join(out, f"{task}{int(n_shot)}_{int(limit) or 'full'}")


@torch.no_grad()
def run_eval(model, tok, letter_ids, spec, args, tag):
    task, n_shot, limit = spec.split(":")
    n_shot, limit = int(n_shot), int(limit)
    test, shots = load_task(task)
    if limit and limit < len(test):
        idx = list(range(len(test)))
        random.Random(0).shuffle(idx)  # same subset for every checkpoint
        test = [test[i] for i in sorted(idx[:limit])]
    prompts = build_prompts(test, shots, n_shot, tok, args.chat)

    # Batch by length so little compute goes to padding; results keep the original order.
    lengths = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    records = [None] * len(prompts)
    for start in range(0, len(order), args.batch_size):
        ids = order[start : start + args.batch_size]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        full = torch.log_softmax(logits, dim=-1)
        for j, i in enumerate(ids):
            # logsumexp over the ' A' / 'A' variants of each letter
            lp = [torch.logsumexp(full[j, letter_ids[l]], 0).item() for l in LETTERS]
            letter_mass = sum(math.exp(x) for x in lp)
            m = max(lp)
            probs = [math.exp(x - m) for x in lp]
            z = sum(probs)
            probs = [p / z for p in probs]
            pred = max(range(4), key=lambda k: probs[k])
            row = test[i]
            records[i] = {
                "subject": row["subject"],
                "answer": row["answer"],
                "pred": pred,
                "probs": probs,
                "conf": probs[pred],
                "correct": int(pred == row["answer"]),
                "letter_mass": letter_mass,  # low = model isn't answering in the expected format
            }
        if (start // args.batch_size) % 100 == 0:
            print(f"{tag} {spec}: {start + len(ids)}/{len(prompts)}", flush=True)

    out_dir = eval_dir(args.out, spec)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}.jsonl"), "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    confs = [r["conf"] for r in records]
    correct = [r["correct"] for r in records]
    e, diagram = ece(confs, correct)
    summary = {
        "tag": tag,
        "eval": spec,
        "model": args.model,
        "revision": args.revision,
        "chat": args.chat,
        "n": len(records),
        "acc": sum(correct) / len(correct),
        "avg_conf": sum(confs) / len(confs),
        "ece": e,
        "brier": sum(
            sum((p - (k == r["answer"])) ** 2 for k, p in enumerate(r["probs"])) for r in records
        ) / len(records),
        "avg_letter_mass": sum(r["letter_mass"] for r in records) / len(records),
        "reliability": diagram,
    }
    with open(os.path.join(out_dir, f"{tag}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "reliability"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tag", default=None, help="default: Olmo model name [@revision] [+chat]")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--evals", default="mmlu:5:3000")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    tag = args.tag or (
        args.model.split("/")[-1]
        + (f"@{args.revision}" if args.revision else "")
        + ("+chat" if args.chat else "")
    )

    override = load_config_override(args.model, args.revision)  # tokenizer reads the config too
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, **override)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map="cuda", **override
    ).eval()
    letter_ids = letter_token_ids(tok)

    for spec in args.evals.split(","):
        if os.path.exists(os.path.join(eval_dir(args.out, spec), f"{tag}.summary.json")):
            print(f"skip {tag} {spec} (done)")
            continue
        run_eval(model, tok, letter_ids, spec, args, tag)


if __name__ == "__main__":
    main()
