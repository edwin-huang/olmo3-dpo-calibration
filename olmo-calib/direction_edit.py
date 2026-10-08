"""Remove DPO's "confidence direction" from Olmo-3-7B-Instruct-DPO and re-evaluate calibration.

Direction (at Instruct-SFT, from a SOURCE benchmark of listed-option questions):
  v = ∇conf − c·∇acc, with c chosen so v ⟂ ∇acc   (more confident, not more correct)
  conf = mean Σ p log p over the A-D answer distribution, acc = mean log p[correct]
Edit:     θ = θ_DPO − k·<Δ, v̂>·v̂,   Δ = θ_DPO − θ_SFT   (remove k× DPO's component along v̂)
Control:  θ = θ_DPO − |<Δ, v̂_arc>|·r̂, a same-size move along a random direction r̂

Sources: "arc" (ARC-Challenge train+validation) and "mmlu" (1000 MMLU test questions, which are
then excluded from the MMLU evaluation). Everything is evaluated on held-out questions:
MMLU letters/numbers (2000), ARC-Challenge test letters (1165), MMLU cloze (1000).

Usage: python direction_edit.py --out results_edit --save-dir /workspace/models
"""
import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution import dot, metric_grad
from calib_eval import build_prompts, letter_token_ids, load_task
from format_check import LABELS, cloze_variants, score_cloze, score_listed, summarize

SFT = "allenai/Olmo-3-7B-Instruct-SFT"
DPO = "allenai/Olmo-3-7B-Instruct-DPO"


def arc_rows(split):
    rows = []
    for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split):
        labels = r["choices"]["label"]
        if len(labels) == 4 and r["answerKey"] in labels:
            rows.append({"question": r["question"], "choices": r["choices"]["text"],
                         "answer": labels.index(r["answerKey"]), "subject": "science"})
    return rows


def sources():
    """Direction-source question sets, plus the MMLU indices they use (to hold out)."""
    train = arc_rows("train")
    random.Random(0).shuffle(train)  # same 5 exemplars as calib_eval's ARC eval
    arc_shots = {"science": train[:5]}
    arc_src = train[5:] + arc_rows("validation")

    mmlu, mmlu_shots = load_task("mmlu")
    idx = list(range(len(mmlu)))
    random.Random(1).shuffle(idx)  # same 1000 questions as attribution.py
    used = set(idx[:1000])
    return {"arc": (arc_src, arc_shots), "mmlu": ([mmlu[i] for i in idx[:1000]], mmlu_shots)}, used


def batches_for(tok, rows, shots, bs=8):
    prompts = build_prompts(rows, shots, 5, tok, chat=False)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = []
    for s in range(0, len(order), bs):
        ids = order[s : s + bs]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False)
        out.append((enc, torch.tensor([rows[i]["answer"] for i in ids])))
    return out


def unit_direction(model, params, tok, rows, shots, letter_ids):
    batches = batches_for(tok, rows, shots)
    g_conf = metric_grad(model, params, batches, letter_ids, "conf")
    g_acc = metric_grad(model, params, batches, letter_ids, "acc")
    c = dot(g_conf, g_acc) / dot(g_acc, g_acc)
    v = [(gc.float() - c * ga.float()).to(torch.bfloat16) for gc, ga in zip(g_conf, g_acc)]
    del g_conf, g_acc
    n = dot(v, v) ** 0.5
    for t in v:
        t.div_(n)
    torch.cuda.empty_cache()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_edit")
    ap.add_argument("--save-dir", default="/workspace/models")
    ap.add_argument("--save", default="", help="variants to save for IFEval, comma-separated")
    ap.add_argument("--smoke", action="store_true", help="tiny question sets, nothing saved")
    args = ap.parse_args()
    lim = (lambda rows: rows[:16]) if args.smoke else (lambda rows: rows)
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "edit_results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}

    tok_dpo = AutoTokenizer.from_pretrained(DPO)  # saved with edited models (chat template)
    tok = AutoTokenizer.from_pretrained(SFT)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()  # gradient checkpointing needs train mode; Olmo 3 has no dropout
    params = list(model.parameters())
    letter_ids = letter_token_ids(tok)

    srcs, mmlu_used = sources()
    dirs = {}
    for name, (rows, shots) in srcs.items():
        print(f"direction from {name} ({len(rows)} questions) ...", flush=True)
        v = unit_direction(model, params, tok, lim(rows), shots, letter_ids)
        dirs[name] = [t.cpu() for t in v]  # keep GPU memory free for the next direction
        del v
        torch.cuda.empty_cache()
    dirs = {k: [t.cuda() for t in v] for k, v in dirs.items()}
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    sft_cpu = [p.detach().to("cpu", copy=True) for p in params]
    dpo_cpu = [p.detach() for p in AutoModelForCausalLM.from_pretrained(
        DPO, dtype=torch.bfloat16, device_map="cpu").parameters()]

    # DPO's component along each direction, and how the two directions relate.
    proj, d_norm2 = {}, 0.0
    for name in dirs:
        proj[name] = 0.0
    for i, (s, d) in enumerate(zip(sft_cpu, dpo_cpu)):
        delta = d.cuda().float() - s.cuda().float()
        d_norm2 += (delta * delta).sum().item()
        for name, v in dirs.items():
            proj[name] += (delta * v[i].float()).sum().item()
    res["directions"] = {
        "delta_norm": d_norm2 ** 0.5,
        "proj": proj,
        "cos_delta": {k: p / d_norm2 ** 0.5 for k, p in proj.items()},
        "cos_arc_mmlu": dot(dirs["arc"], dirs["mmlu"]),
    }
    print(json.dumps(res["directions"], indent=1), flush=True)

    def set_weights(fn):
        with torch.no_grad():
            for i, p in enumerate(params):
                p.copy_(fn(i))

    def random_move(i, seed=1234):
        g = torch.Generator(device="cuda").manual_seed(seed + i)
        return torch.randn(params[i].shape, generator=g, device="cuda", dtype=torch.float32)

    r_norm = sum((random_move(i) ** 2).sum().item() for i in range(len(params))) ** 0.5
    variants = {
        "sft": lambda i: sft_cpu[i].cuda(),
        "dpo": lambda i: dpo_cpu[i].cuda(),
        "edit_arc_k1": lambda i: (dpo_cpu[i].cuda().float() - proj["arc"] * dirs["arc"][i].float()).to(torch.bfloat16),
        "edit_arc_k2": lambda i: (dpo_cpu[i].cuda().float() - 2 * proj["arc"] * dirs["arc"][i].float()).to(torch.bfloat16),
        "edit_mmlu_k1": lambda i: (dpo_cpu[i].cuda().float() - proj["mmlu"] * dirs["mmlu"][i].float()).to(torch.bfloat16),
        "edit_mmlu_k3": lambda i: (dpo_cpu[i].cuda().float() - 3 * proj["mmlu"] * dirs["mmlu"][i].float()).to(torch.bfloat16),
        "control_random": lambda i: (dpo_cpu[i].cuda().float()
                                     - abs(proj["arc"]) * random_move(i) / r_norm).to(torch.bfloat16),
    }

    # Held-out evaluation sets.
    mmlu, mmlu_shots = load_task("mmlu")
    rest = [i for i in range(len(mmlu)) if i not in mmlu_used]
    random.Random(2).shuffle(rest)
    mmlu_eval = lim([mmlu[i] for i in sorted(rest[:2000])])
    arc_test, arc_shots = load_task("arc")
    arc_test = lim(arc_test)
    evals = {
        "mmlu_letters": lambda: summarize(score_listed(model, tok, mmlu_eval, mmlu_shots, LABELS["letters"])),
        "mmlu_numbers": lambda: summarize(score_listed(model, tok, mmlu_eval, mmlu_shots, LABELS["numbers"])),
        "arc_letters": lambda: summarize(score_listed(model, tok, arc_test, arc_shots, LABELS["letters"])),
        "mmlu_cloze_norm": lambda: cloze_variants(score_cloze(model, tok, mmlu_eval[:1000], mmlu_shots))["cloze_norm"],
    }

    print(f"\n{'variant':<16}{'eval':<17}{'acc':>7}{'conf':>7}{'ECE':>7}{'T*':>7}{'AUROC':>7}")
    for vname, fn in variants.items():
        set_weights(fn)
        out = res.setdefault(vname, {})
        for ename, run in evals.items():
            if ename not in out:
                out[ename] = run()
                json.dump(res, open(res_path, "w"), indent=1)
            r = out[ename]
            print(f"{vname:<16}{ename:<17}{r['acc']:>7.3f}{r['conf']:>7.3f}{r['ece']:>7.3f}"
                  f"{r['t_star']:>7.2f}{r['auroc']:>7.3f}", flush=True)
        if vname in args.save.split(",") and not args.smoke:
            path = os.path.join(args.save_dir, f"Olmo-3-7B-Instruct-DPO-{vname}")
            # Ai2's generation_config sets temperature/top_p without do_sample, which
            # transformers refuses to save; keep their sampling defaults by enabling it.
            model.generation_config.do_sample = True
            model.save_pretrained(path)
            tok_dpo.save_pretrained(path)
            print(f"saved {path}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
