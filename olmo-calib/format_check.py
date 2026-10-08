"""Is DPO's overconfidence about the A/B/C/D answer format, or about answering in general?

Same MMLU questions (5-shot), three formats:
  letters  options listed A-D, score the next-token probability of each letter (the main setup)
  numbers  options listed 1-4, score the next-token probability of each number
  cloze    options NOT listed; score log P(" <answer text>" | prompt) for each option and
           renormalize over the four (the model's own probability ratio between full answers)

Usage: python format_check.py --models Olmo-3-1025-7B,Olmo-3-7B-Instruct-DPO --out results_format
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_eval import load_task
from temp_scaling import auroc, ece, fit_t

MODELS = [
    "Olmo-3-1025-7B",
    "Olmo-3-7B-Think-SFT",
    "Olmo-3-7B-Think-DPO",
    "Olmo-3-7B-Instruct-SFT",
    "Olmo-3-7B-Instruct-DPO",
]
LABELS = {"letters": ["A", "B", "C", "D"], "numbers": ["1", "2", "3", "4"]}


def header(row):
    return f"The following are multiple choice questions (with answers) about {row['subject'].replace('_', ' ')}.\n\n"


def listed(row, labels, with_answer):
    s = row["question"].strip() + "\n"
    for lab, choice in zip(labels, row["choices"]):
        s += f"{lab}. {choice}\n"
    s += "Answer:"
    return s + (f" {labels[row['answer']]}\n\n" if with_answer else "")


def cloze(row, with_answer):
    s = f"Question: {row['question'].strip()}\nAnswer:"
    return s + (f" {row['choices'][row['answer']]}\n\n" if with_answer else "")


def summarize(rows, t_max=10.0):
    confs = [max(r["probs"]) for r in rows]
    correct = [int(r["probs"].index(max(r["probs"])) == r["answer"]) for r in rows]
    return {"n": len(rows), "acc": sum(correct) / len(rows), "conf": sum(confs) / len(rows),
            "ece": ece(confs, correct), "t_star": fit_t(rows, t_max), "auroc": auroc(confs, correct)}


@torch.no_grad()
def score_listed(model, tok, test, shots, labels, bs=16):
    # " A" is one token, but " 1" is [" ", "1"]: for digits the prompt has to end with the
    # space token, exactly as "Answer: 1" is tokenized in the few-shot examples.
    split_space = len(tok.encode(f" {labels[0]}", add_special_tokens=False)) > 1
    ids = []
    for lab in labels:
        variants = [lab] if split_space else [f" {lab}", lab]
        enc = [tok.encode(t, add_special_tokens=False) for t in variants]
        ids.append(sorted({e[0] for e in enc if len(e) == 1}))
        assert ids[-1], f"no single-token encoding for {lab!r}"
    prompts = [header(r) + "".join(listed(s, labels, True) for s in shots[r["subject"]][:5])
               + listed(r, labels, False) + (" " if split_space else "") for r in test]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    rows = [None] * len(test)
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1].float(), -1)
        for j, i in enumerate(idx):
            v = [torch.logsumexp(lp[j, t], 0).item() for t in ids]
            m = max(v)
            p = [math.exp(x - m) for x in v]
            rows[i] = {"probs": [x / sum(p) for x in p], "answer": test[i]["answer"],
                       "label_mass": sum(math.exp(x) for x in v)}
    return rows


@torch.no_grad()
def seq_logprobs(model, tok, seqs, bs):
    """Sum of log P(continuation | prompt) for each (prompt ids, continuation ids)."""
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]) + len(seqs[k][1]))
    out = [0.0] * len(seqs)
    for s in range(0, len(order), bs):
        ks = order[s : s + bs]
        full = [seqs[k][0] + seqs[k][1] for k in ks]
        L = max(map(len, full))
        inp = torch.full((len(ks), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(ks), L), dtype=torch.long)
        for j, f in enumerate(full):  # right padding: positions are easy to index
            inp[j, : len(f)] = torch.tensor(f)
            att[j, : len(f)] = 1
        logits = model(input_ids=inp.cuda(), attention_mask=att.cuda()).logits
        for j, k in enumerate(ks):
            p_len, cont = len(seqs[k][0]), seqs[k][1]
            lp = F.log_softmax(logits[j, p_len - 1 : p_len - 1 + len(cont)].float(), -1)
            out[k] = lp.gather(1, torch.tensor(cont, device="cuda")[:, None]).sum().item()
        del logits
    return out


@torch.no_grad()
def score_cloze(model, tok, test, shots, bs=8):
    """Per option: log P(answer | question prompt), its token length, and log P(answer | "Answer:")
    for PMI scoring, which removes the preference for short/common answers (Holtzman et al. 2021)."""
    null_prompt = tok.encode("Answer:", add_special_tokens=False)
    cond, null = [], []
    for r in test:
        prompt = header(r) + "".join(cloze(s, True) for s in shots[r["subject"]][:5]) + cloze(r, False)
        p_ids = tok.encode(prompt, add_special_tokens=False)
        for choice in r["choices"]:
            c = tok.encode(f" {choice}", add_special_tokens=False) or [tok.eos_token_id]
            cond.append((p_ids, c))
            null.append((null_prompt, c))
    lp_c = seq_logprobs(model, tok, cond, bs)
    lp_n = seq_logprobs(model, tok, null, 64)
    return [{"answer": r["answer"], "logp": lp_c[4 * i : 4 * i + 4], "logp_null": lp_n[4 * i : 4 * i + 4],
             "len": [len(c) for _, c in cond[4 * i : 4 * i + 4]]} for i, r in enumerate(test)]


def cloze_variants(rows):
    def soft(v):
        m = max(v)
        p = [math.exp(x - m) for x in v]
        return [x / sum(p) for x in p]
    scorings = {
        "cloze_sum": lambda r: r["logp"],
        "cloze_norm": lambda r: [a / n for a, n in zip(r["logp"], r["len"])],
        "cloze_pmi": lambda r: [a - b for a, b in zip(r["logp"], r["logp_null"])],
    }
    return {name: summarize([{"probs": soft(f(r)), "answer": r["answer"]} for r in rows], t_max=1000.0)
            for name, f in scorings.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--formats", default="letters,numbers,cloze")
    ap.add_argument("--n-q", type=int, default=2000)
    ap.add_argument("--out", default="results_format")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    test, shots = load_task("mmlu")
    idx = list(range(len(test)))
    random.Random(0).shuffle(idx)  # same 2000-question subset as mechanism.py
    test = [test[i] for i in sorted(idx[: args.n_q])]

    formats = args.formats.split(",")
    shown = [f for f in formats if f != "cloze"] + (["cloze_sum", "cloze_norm", "cloze_pmi"] if "cloze" in formats else [])
    print(f"{'model':<26}{'format':<11}{'acc':>7}{'conf':>7}{'ECE':>7}{'T*':>7}{'AUROC':>7}{'mass':>7}")
    for name in args.models.split(","):
        path = os.path.join(args.out, f"{name}.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        res.pop("cloze", None)  # old sum-only cloze result
        todo = [f for f in formats if (f == "cloze" and "cloze_pmi" not in res) or (f != "cloze" and f not in res)]
        if todo:
            tok = AutoTokenizer.from_pretrained(f"allenai/{name}")
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                f"allenai/{name}", dtype=torch.bfloat16, device_map="cuda").eval()
            for fmt in todo:
                if fmt == "cloze":
                    rows = score_cloze(model, tok, test, shots)
                    with open(os.path.join(args.out, f"{name}.cloze_rows.jsonl"), "w") as f:
                        f.writelines(json.dumps(r) + "\n" for r in rows)
                    res.update(cloze_variants(rows))
                else:
                    rows = score_listed(model, tok, test, shots, LABELS[fmt])
                    res[fmt] = summarize(rows)
                    res[fmt]["label_mass"] = sum(r["label_mass"] for r in rows) / len(rows)
                json.dump(res, open(path, "w"), indent=1)
            del model
            torch.cuda.empty_cache()
        for fmt in shown:
            r = res[fmt]
            print(f"{name:<26}{fmt:<11}{r['acc']:>7.3f}{r['conf']:>7.3f}{r['ece']:>7.3f}{r['t_star']:>7.2f}"
                  f"{r['auroc']:>7.3f}{r.get('label_mass', float('nan')):>7.3f}", flush=True)


if __name__ == "__main__":
    main()
