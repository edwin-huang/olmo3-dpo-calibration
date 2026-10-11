"""Every calibration measure for one checkpoint, so all DPO/SFT reruns get identical treatment.

Parts (each runs in its own subprocess, so a crash or vLLM's memory grab can't take down the others):
  letters  MMLU 5-shot (all 14,042; 3000 with --fast) + ARC-Challenge 5-shot, A-D letter probabilities (calib_eval.py)
  format   the same 2000 MMLU questions as letters / numbers / length-normalized cloze (format_check.py)
  chat     chat-format direct answer, 1000 MMLU + 500 ARC (the cot_test.py questions)
  text     sharpening s* of this checkpoint vs --ref on wikitext and DPO chat responses (decision_points.py)
  sc       self-consistency: agreement of k=8 sampled chains of thought, same questions as chat (vLLM)
  ifeval   IFEval strict prompt- and instruction-level accuracy (vLLM greedy, open-instruct's checker): capability
Each reports acc, mean confidence, ECE, T* (fit on half, as in temp_scaling.py) and AUROC where they apply.
Writes <out>/<tag>/<part>.json (+ per-question files) and <out>/<tag>/summary.json with the headline numbers.

Usage:
  python eval_ckpt.py --ckpt /workspace/runs/d0/snap_step_203 --tag d0/step_203 --ref allenai/Olmo-3-7B-Instruct-SFT
  python eval_ckpt.py --ckpt allenai/Olmo-3-7B-Instruct-DPO --tag released/instruct_dpo --ref allenai/Olmo-3-7B-Instruct-SFT
  --fast for smaller question sets, --smoke for a few questions per part (pipeline check only)
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from types import SimpleNamespace

PARTS = ["letters", "format", "chat", "text", "sc", "ifeval"]
SIZES = {  # letters MMLU limit, ARC limit, format n, chat/sc MMLU, chat/sc ARC, text sequences, sc k, IFEval prompts
    "full": dict(mmlu=0, arc=0, fmt=2000, q_mmlu=1000, q_arc=500, n_seq=256, k=8, ifeval=0),
    "fast": dict(mmlu=3000, arc=0, fmt=1000, q_mmlu=500, q_arc=250, n_seq=128, k=8, ifeval=0),
    "smoke": dict(mmlu=100, arc=50, fmt=50, q_mmlu=20, q_arc=10, n_seq=4, k=4, ifeval=20),
}


def calib(rows, t_max=10.0):
    """rows: dicts with probs, answer. T* is fit on a random half; ECE@T* is measured on the other half."""
    from temp_scaling import auroc, ece, evaluate, fit_t

    confs = [max(r["probs"]) for r in rows]
    correct = [int(r["probs"].index(max(r["probs"])) == r["answer"]) for r in rows]
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    fit, held = [rows[i] for i in idx[: len(rows) // 2]], [rows[i] for i in idx[len(rows) // 2 :]]
    t = fit_t(fit, t_max)
    return {"n": len(rows), "acc": sum(correct) / len(rows), "conf": sum(confs) / len(rows),
            "ece": ece(confs, correct), "t_star": t, "ece_at_t_star_heldout": evaluate(held, t)[0],
            "auroc": auroc(confs, correct) if 0 < sum(correct) < len(rows) else float("nan")}


def load(ckpt):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calib_eval import load_config_override

    override = {} if os.path.isdir(ckpt) else load_config_override(ckpt, None)
    tok = AutoTokenizer.from_pretrained(ckpt, **override)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda", **override).eval()
    return model, tok


def mmlu_subset(n):
    from calib_eval import load_task

    test, shots = load_task("mmlu")
    idx = list(range(len(test)))
    random.Random(0).shuffle(idx)  # the same subset as format_check.py / mechanism.py
    return [test[i] for i in sorted(idx[:n])], shots


def part_letters(args, sz, d):
    from calib_eval import letter_token_ids, run_eval

    model, tok = load(args.ckpt)
    ns = SimpleNamespace(chat=False, batch_size=16, out=d, model=args.ckpt, revision=None)
    res = {}
    for task, limit in (("mmlu", sz["mmlu"]), ("arc", sz["arc"])):
        spec = f"{task}:5:{limit}"
        run_eval(model, tok, letter_token_ids(tok), spec, ns, "letters")
        sub = f"{task}5_{limit or 'full'}"
        rows = [json.loads(l) for l in open(os.path.join(d, sub, "letters.jsonl"))]
        res[task] = calib(rows) | {"letter_mass": sum(r["letter_mass"] for r in rows) / len(rows)}
    return res


def part_format(args, sz, d):
    from format_check import LABELS, cloze_variants, score_cloze, score_listed

    model, tok = load(args.ckpt)
    test, shots = mmlu_subset(sz["fmt"])
    res = {}
    for fmt in ("letters", "numbers"):
        rows = score_listed(model, tok, test, shots, LABELS[fmt])
        res[fmt] = calib(rows) | {"label_mass": sum(r["label_mass"] for r in rows) / len(rows)}
    rows = score_cloze(model, tok, test, shots)
    with open(os.path.join(d, "format.cloze_rows.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    res["cloze_norm"] = cloze_variants(rows)["cloze_norm"]  # the only usable cloze scoring (Run 4)
    return res


def part_chat(args, sz, d):
    import torch

    from calib_eval import LETTERS
    from cot_test import INSTR_DIRECT, questions, user_msg

    model, tok = load(args.ckpt)
    ids = {l: sorted({e[0] for e in (tok.encode(t, add_special_tokens=False) for t in (f" {l}", l)) if len(e) == 1})
           for l in LETTERS}
    rows = questions(sz["q_mmlu"], sz["q_arc"])
    per = {"mmlu": [], "arc": []}
    with torch.no_grad(), open(os.path.join(d, "chat.questions.jsonl"), "w") as qf:
        for qi, r in enumerate(rows):
            text = tok.apply_chat_template([{"role": "user", "content": user_msg(r, INSTR_DIRECT)}],
                                           tokenize=False, add_generation_prompt=True) + "Answer:"
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[0, -1].float(), -1)
            v = [torch.logsumexp(lp[ids[l]], 0).item() for l in LETTERS]
            m = max(v)
            p = [math.exp(x - m) for x in v]
            p = [x / sum(p) for x in p]
            per[r["src"]].append({"probs": p, "answer": r["answer"], "mass": sum(math.exp(x) for x in v)})
            qf.write(json.dumps({"qi": qi, "src": r["src"], "subject": r["subject"], "answer": LETTERS[r["answer"]],
                                 "probs": p}) + "\n")
    return {src: calib(rr, 100.0) | {"mass": sum(x["mass"] for x in rr) / len(rr)} for src, rr in per.items() if rr}


def part_text(args, sz, d):
    import torch
    from transformers import AutoModelForCausalLM

    from decision_points import probe

    model, tok = load(args.ckpt)
    ref = AutoModelForCausalLM.from_pretrained(args.ref, dtype=torch.bfloat16, device_map="cuda").eval()
    by_source = probe(ref, model, tok, sz["n_seq"])
    res = {"ref": args.ref, "bins": by_source}
    for source, rows in by_source.items():  # token-weighted mean of the per-bin sharpening
        n = sum(r["tokens"] for r in rows)
        res[f"s_star_{source}"] = sum(r["s_star"] * r["tokens"] for r in rows) / n
    return res


def part_sc(args, sz, d):
    from transformers import AutoTokenizer

    from cot_test import questions, sample

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    path = os.path.join(d, f"sc{sz['k']}.json")
    sample(SimpleNamespace(model=args.ckpt, k=sz["k"], max_tokens=1024), questions(sz["q_mmlu"], sz["q_arc"]), tok, path)
    res = json.load(open(path))
    for s in res.values():
        s["excess"] = s["conf"] - s["acc"]
    return res


def part_ifeval(args, sz, d):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from cot_test import vllm_kwargs
    from open_instruct.IFEvalG import instructions_registry

    rows = list(load_dataset("google/IFEval", split="train"))
    if sz["ifeval"]:
        rows = rows[: sz["ifeval"]]
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True) for r in rows]
    llm = LLM(model=args.ckpt, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0,
              **vllm_kwargs(args.ckpt))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=2048))
    prompt_ok, inst_ok, unknown = [], [], set()
    with open(os.path.join(d, "ifeval.responses.jsonl"), "w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            follows = []
            for key, kw in zip(r["instruction_id_list"], r["kwargs"]):
                if key not in instructions_registry.INSTRUCTION_DICT:
                    unknown.add(key)
                    continue
                inst = instructions_registry.INSTRUCTION_DICT[key](key)
                inst.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
                if "prompt" in inst.get_instruction_args_keys():
                    inst.build_description(prompt=r["prompt"])
                follows.append(bool(text.strip()) and bool(inst.check_following(text)))
            prompt_ok.append(all(follows))
            inst_ok += follows
            f.write(json.dumps({"key": r["key"], "follows": follows, "response": text}) + "\n")
    return {"n": len(rows), "prompt_strict": sum(prompt_ok) / len(prompt_ok), "inst_strict": sum(inst_ok) / len(inst_ok),
            "unknown_instruction_ids": sorted(unknown)}


def headline(res):
    """The numbers to compare across checkpoints, flat."""
    h = {}
    g = lambda part, *keys: _dig(res.get(part), keys)
    for task in ("mmlu", "arc"):
        for k in ("acc", "ece", "t_star", "auroc"):
            h[f"letters_{task}_{k}"] = g("letters", task, k)
        for k in ("ece", "t_star", "auroc"):
            h[f"chat_{task}_{k}"] = g("chat", task, k)
        h[f"sc_{task}_acc"] = g("sc", f"sc_{task}", "acc")
        h[f"sc_{task}_excess"] = g("sc", f"sc_{task}", "excess")
        h[f"chat_{task}_excess"] = (lambda c: c["conf"] - c["acc"] if c else None)(_dig(res.get("chat"), (task,)))
    for fmt in ("letters", "numbers", "cloze_norm"):
        h[f"format_{fmt}_t_star"] = g("format", fmt, "t_star")
    h["text_s_star_wikitext"] = g("text", "s_star_wikitext")
    h["text_s_star_chat"] = g("text", "s_star_chat")
    h["ifeval_prompt_strict"] = g("ifeval", "prompt_strict")
    h["ifeval_inst_strict"] = g("ifeval", "inst_strict")
    return h


def _dig(x, keys):
    for k in keys:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF model dir or hub id")
    ap.add_argument("--tag", required=True, help="output subdirectory, e.g. d0/step_203")
    ap.add_argument("--ref", default="allenai/Olmo-3-7B-Instruct-SFT", help="model the text probe compares against")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--part", default=None, help="internal: run one part in this process")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results_ckpt")
    args = ap.parse_args()
    size = "smoke" if args.smoke else "fast" if args.fast else "full"
    sz = SIZES[size]
    args.out = os.path.abspath(args.out)  # parts run with cwd = this script's directory
    if os.path.isdir(args.ckpt):
        args.ckpt = os.path.abspath(args.ckpt)
    d = os.path.join(args.out, args.tag)
    os.makedirs(d, exist_ok=True)

    if args.part:
        t = time.time()
        res = globals()[f"part_{args.part}"](args, sz, d)
        res["_meta"] = {"ckpt": args.ckpt, "size": size, "seconds": round(time.time() - t)}
        json.dump(res, open(os.path.join(d, f"{args.part}.json"), "w"), indent=1)
        return

    failed = []
    for part in args.parts.split(","):
        if os.path.exists(os.path.join(d, f"{part}.json")):
            print(f"[{args.tag}] skip {part} (done)", flush=True)
            continue
        print(f"[{args.tag}] {part} ...", flush=True)
        cmd = [sys.executable, __file__, "--ckpt", args.ckpt, "--tag", args.tag, "--ref", args.ref, "--out", args.out,
               "--part", part] + (["--smoke"] if args.smoke else ["--fast"] if args.fast else [])
        with open(os.path.join(d, f"{part}.log"), "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(__file__)))
        if code:
            failed.append(part)
            print(f"[{args.tag}] {part} FAILED (exit {code}), see {d}/{part}.log", flush=True)
    res = {p: json.load(open(os.path.join(d, f"{p}.json"))) for p in PARTS if os.path.exists(os.path.join(d, f"{p}.json"))}
    summary = {"tag": args.tag, "ckpt": args.ckpt, "size": size, "failed": failed, **headline(res)}
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
