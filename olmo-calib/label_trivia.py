"""Label TriviaQA questions as known / unknown by the Olmo 3 base model (for sft_known.py).

known   = greedy answer correct AND at least 2 of 3 samples (T=0.7) correct
unknown = greedy answer wrong AND 0 of 3 samples correct
anything in between is "unsure" and is not used for training.
Prompt: 4 fixed examples from the validation split, then "Q: <question>\nA:".

Usage: python label_trivia.py --n 40000 --out data_trivia/labeled.jsonl
"""
import argparse
import json
import os
import random
import re
import string

from datasets import load_dataset

MODEL = "allenai/Olmo-3-1025-7B"


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def correct(pred, aliases):
    p = norm(pred.split("\n")[0])
    if not p:
        return False
    return any(a and (a == p or re.search(rf"\b{re.escape(a)}\b", p)) for a in aliases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="data_trivia/labeled.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    val = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    shots = "".join(f"Q: {val[i]['question']}\nA: {val[i]['answer']['value']}\n\n" for i in range(4))
    header = "Answer each question with a short answer.\n\n"
    idx = list(range(len(train)))
    random.Random(0).shuffle(idx)
    rows = [train[i] for i in idx[: args.n]]
    prompts = [header + shots + f"Q: {r['question']}\nA:" for r in rows]

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=0)
    greedy = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=24, stop=["\n"]))
    sampled = llm.generate(prompts, SamplingParams(temperature=0.7, n=3, max_tokens=24, stop=["\n"], seed=0))

    counts = {"known": 0, "unknown": 0, "unsure": 0}
    with open(args.out, "w") as f:
        for r, g, s in zip(rows, greedy, sampled):
            aliases = [norm(a) for a in r["answer"]["normalized_aliases"] + r["answer"]["aliases"]]
            g_ok = correct(g.outputs[0].text, aliases)
            n_ok = sum(correct(o.text, aliases) for o in s.outputs)
            label = "known" if g_ok and n_ok >= 2 else "unknown" if not g_ok and n_ok == 0 else "unsure"
            counts[label] += 1
            f.write(json.dumps({"question": r["question"], "answer": r["answer"]["value"],
                                "aliases": sorted(set(a for a in aliases if a)), "greedy": g.outputs[0].text.strip(),
                                "n_samples_correct": n_ok, "label": label}) + "\n")
    print(counts)
    print("ALL DONE")


if __name__ == "__main__":
    main()
