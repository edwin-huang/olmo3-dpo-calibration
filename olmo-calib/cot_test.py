"""Does reasoning before answering shrink DPO's added overconfidence?

Prediction (from answer_locus.py): DPO sharpens answers decided cold, at a single token; after a
chain of thought the answer is already settled by the reasoning, so DPO should add less.

Chat format for every model (user turn = question + listed options + instruction):
  direct : assistant turn = "Answer:" -> next-token probabilities of A-D
  cot    : the model first writes its reasoning (vLLM, greedy, up to --max-tokens); everything
           before its own "Answer:" is kept, then "Answer:" is appended -> probabilities of A-D
Confidence = probability of the chosen letter, renormalized over A-D (as everywhere else).

Usage:
  python cot_test.py --stage gen   --model allenai/Olmo-3-7B-Instruct-SFT
  python cot_test.py --stage score --model allenai/Olmo-3-7B-Instruct-SFT
"""
import argparse
import json
import math
import os
import random
import re

from calib_eval import LETTERS, load_task

INSTR_DIRECT = "Answer with the letter of the correct option."
INSTR_COT = "Think step by step, then finish with a final line of the form 'Answer: X', where X is the letter of the correct option."
CUT = re.compile(r"[\*#\s]*(final\s+)?answer[\*\s]*[:：]", re.I)


def questions(n_mmlu, n_arc):
    mmlu, _ = load_task("mmlu")
    idx = list(range(len(mmlu)))
    random.Random(0).shuffle(idx)  # the same ordering as format_check.py / mechanism.py
    rows = [dict(mmlu[i], src="mmlu") for i in sorted(idx[:n_mmlu])]
    arc, _ = load_task("arc")
    return rows + [dict(r, src="arc") for r in arc[:n_arc]]


def vllm_kwargs(model):
    """vLLM 0.19.1 reads config.rope_parameters as one flat dict and itself applies YaRN only on full-attention layers.
    transformers >= 5.17 gives Olmo 3 a per-layer-type dict instead (KeyError: 'rope_theta'), so pass the
    full-attention entry. Checked against transformers 5.17 HF on 64 MMLU questions: 98% same answers, max |dp| 0.056."""
    from transformers import AutoConfig
    rp = getattr(AutoConfig.from_pretrained(model), "rope_parameters", None)
    if isinstance(rp, dict) and "full_attention" in rp:
        return {"hf_overrides": {"rope_parameters": dict(rp["full_attention"])}}
    return {}


def user_msg(r, instr):
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, r["choices"]))
    return f"{r['question'].strip()}\n{opts}\n\n{instr}"


def gen(args, rows, tok, path):
    from vllm import LLM, SamplingParams
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg(r, INSTR_COT)}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0,
              **vllm_kwargs(args.model))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens))
    with open(path, "w") as f:
        for r, p, o in zip(rows, prompts, outs):
            text = o.outputs[0].text
            m = list(CUT.finditer(text))
            reasoning = text[: m[-1].start()] if m else text
            stated = re.search(r"\b([A-D])\b", text[m[-1].end():]) if m else None
            f.write(json.dumps({"src": r["src"], "answer": r["answer"], "prompt": p, "reasoning": reasoning.rstrip(),
                                "stated": stated.group(1) if stated else None, "finished": bool(m),
                                "n_tokens": len(o.outputs[0].token_ids)}) + "\n")


def stated_letter(text):
    m = list(CUT.finditer(text))
    s = re.search(r"\b([A-D])\b", text[m[-1].end():]) if m else None
    return s.group(1) if s else None


def sample(args, rows, tok, out_path):
    """Confidence after reasoning = agreement across k sampled chains of thought (self-consistency)."""
    from vllm import LLM, SamplingParams
    from temp_scaling import auroc, ece

    prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg(r, INSTR_COT)}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0,
              **vllm_kwargs(args.model))
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=1.0, n=args.k, max_tokens=args.max_tokens, seed=0))
    res, per = {}, {"mmlu": [], "arc": []}
    qf = open(out_path.replace(".json", ".questions.jsonl"), "w")  # per-question record, every chain kept
    for qi, (r, o) in enumerate(zip(rows, outs)):
        votes = [stated_letter(c.text) for c in o.outputs]
        qf.write(json.dumps({"qi": qi, "src": r["src"], "subject": r["subject"], "question": r["question"],
                             "choices": r["choices"], "answer": LETTERS[r["answer"]], "votes": votes,
                             "chains": [c.text for c in o.outputs],
                             "n_tokens": [len(c.token_ids) for c in o.outputs]}) + "\n")
        counts = {l: votes.count(l) for l in LETTERS}
        best = max(LETTERS, key=lambda l: counts[l])
        per[r["src"]].append({"conf": counts[best] / args.k, "correct": int(LETTERS.index(best) == r["answer"]),
                              "no_answer": votes.count(None) / args.k, "distinct": sum(c > 0 for c in counts.values())})
    for src, rr in per.items():
        conf, corr = [x["conf"] for x in rr], [x["correct"] for x in rr]
        s = {"n": len(rr), "k": args.k, "acc": sum(corr) / len(rr), "conf": sum(conf) / len(rr),
             "ece": ece(conf, corr, n_bins=args.k), "auroc": auroc(conf, corr),
             "no_answer": sum(x["no_answer"] for x in rr) / len(rr),
             "mean_distinct": sum(x["distinct"] for x in rr) / len(rr),
             "all_agree": sum(x["conf"] == 1 for x in rr) / len(rr)}
        res[f"sc_{src}"] = s
        print(f"{args.model.split('/')[-1]:<24}self-consistency k={args.k} {src:<5} acc {s['acc']:.3f} conf {s['conf']:.3f} "
              f"ECE {s['ece']:.3f} AUROC {s['auroc']:.3f} all-agree {s['all_agree']:.2f} "
              f"distinct {s['mean_distinct']:.2f} no-answer {s['no_answer']:.3f}", flush=True)
    qf.close()
    json.dump(res, open(out_path, "w"), indent=1)


def score(args, rows, tok, gen_path, out_path):
    import torch
    from transformers import AutoModelForCausalLM
    from temp_scaling import auroc, ece, fit_t

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = {l: sorted({e[0] for e in (tok.encode(t, add_special_tokens=False) for t in (f" {l}", l)) if len(e) == 1})
           for l in LETTERS}
    cot = [json.loads(l) for l in open(gen_path)]

    @torch.no_grad()
    def letter_probs(text):
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[0, -1].float(), -1)
        v = [torch.logsumexp(lp[ids[l]], 0).item() for l in LETTERS]
        mass = sum(math.exp(x) for x in v)
        m = max(v)
        p = [math.exp(x - m) for x in v]
        return [x / sum(p) for x in p], mass

    res = {}
    qf = open(out_path.replace(".json", ".questions.jsonl"), "w")  # per-question letter probabilities
    for mode in ("direct", "cot"):
        per = {"mmlu": [], "arc": []}
        for qi, (r, c) in enumerate(zip(rows, cot)):
            if mode == "direct":
                text = tok.apply_chat_template([{"role": "user", "content": user_msg(r, INSTR_DIRECT)}],
                                               tokenize=False, add_generation_prompt=True) + "Answer:"
            else:
                text = c["prompt"] + c["reasoning"] + "\n\nAnswer:"
            p, mass = letter_probs(text)
            per[r["src"]].append({"probs": p, "answer": r["answer"], "mass": mass, "stated": c["stated"]})
            qf.write(json.dumps({"qi": qi, "mode": mode, "src": r["src"], "subject": r["subject"],
                                 "answer": LETTERS[r["answer"]], "probs": p, "mass": mass,
                                 "greedy_stated": c["stated"]}) + "\n")
        for src, rr in per.items():
            if not rr:
                continue
            conf = [max(x["probs"]) for x in rr]
            corr = [int(x["probs"].index(max(x["probs"])) == x["answer"]) for x in rr]
            s = {"n": len(rr), "acc": sum(corr) / len(rr), "conf": sum(conf) / len(rr), "ece": ece(conf, corr),
                 "t_star": fit_t(rr, 100.0), "auroc": auroc(conf, corr) if 0 < sum(corr) < len(corr) else float("nan"),
                 "mass": sum(x["mass"] for x in rr) / len(rr)}
            if mode == "cot":
                agree = [LETTERS[x["probs"].index(max(x["probs"]))] == x["stated"] for x in rr if x["stated"]]
                s["agrees_with_stated"] = sum(agree) / max(1, len(agree))
                s["finished"] = sum(c["finished"] for c, r in zip(cot, rows) if r["src"] == src) / len(rr)
            res[f"{mode}_{src}"] = s
            print(f"{args.model.split('/')[-1]:<24}{mode:<7}{src:<5} acc {s['acc']:.3f} conf {s['conf']:.3f} "
                  f"ECE {s['ece']:.3f} T* {s['t_star']:.2f} AUROC {s['auroc']:.3f} mass {s['mass']:.3f}"
                  + (f" agree {s['agrees_with_stated']:.2f} finished {s['finished']:.2f}" if mode == "cot" else ""),
                  flush=True)
    qf.close()
    json.dump(res, open(out_path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score", "sample"])
    ap.add_argument("--k", type=int, default=8, help="chains per question for --stage sample")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-mmlu", type=int, default=1000)
    ap.add_argument("--n-arc", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="results_cot")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = questions(args.n_mmlu, args.n_arc)
    name = args.model.split("/")[-1]
    gen_path = os.path.join(args.out, f"{name}.gen.jsonl")
    if args.stage == "gen":
        gen(args, rows, tok, gen_path)
    elif args.stage == "sample":
        sample(args, rows, tok, os.path.join(args.out, f"{name}.sc{args.k}.json"))
    else:
        score(args, rows, tok, gen_path, os.path.join(args.out, f"{name}.json"))
    print("ALL DONE")


if __name__ == "__main__":
    main()
