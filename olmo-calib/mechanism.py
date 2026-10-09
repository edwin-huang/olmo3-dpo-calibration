"""Where does post-training overconfidence come from? Probes on released Olmo 3 checkpoints.

Each pair is A:B with A the earlier checkpoint (e.g. Instruct-SFT:Instruct-DPO).

  weights  relative weight change per layer/module, and whether the final norm or
           lm_head was simply scaled up (scaling logits by s == temperature 1/s)
  text     next-token predictions of A vs B on generic text (wikitext) and chat
           responses: entropy, KL(A||B), and how much of that KL one temperature
           explains (min_t KL(A || softmax(z_B / t)))
  swap     hybrid models: B with one block of A's weights (does reverting it fix the
           overconfidence?), A with one block of B's (does it alone cause it?), and
           linear interpolation A + a(B - A); each scored on MMLU 5-shot calibration

Usage:
  python mechanism.py --probes weights,text,swap --out results_mech
"""
import argparse
import json
import math
import os
import random
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_eval import build_prompts, letter_token_ids, load_task
from temp_scaling import auroc, ece, fit_t

PAIRS = [
    "Olmo-3-7B-Instruct-SFT:Olmo-3-7B-Instruct-DPO",
    "Olmo-3-7B-Think-SFT:Olmo-3-7B-Think-DPO",
    "Olmo-3-1025-7B:Olmo-3-7B-Think-SFT",
    "Olmo-3-7B-Think-SFT:Olmo-3-7B-Instruct-SFT",
]
T_GRID = [0.5 * (8 ** (i / 47)) for i in range(48)]  # 0.5 .. 4.0, log-spaced
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def load(name):
    from calib_eval import load_config_override  # RL-Zero-Mix ships a pre-release model_type

    return AutoModelForCausalLM.from_pretrained(
        f"allenai/{name}", dtype=torch.bfloat16, device_map="cuda",
        **load_config_override(f"allenai/{name}", None),
    ).eval()


# ---------------------------------------------------------------- weights


@torch.no_grad()
def probe_weights(ma, mb):
    pa, pb = dict(ma.named_parameters()), dict(mb.named_parameters())
    per_layer, other = {}, {}
    for n, a in pa.items():
        a, b = a.float(), pb[n].float()
        rel = ((b - a).norm() / a.norm()).item()
        scale = ((a * b).sum() / (a * a).sum()).item()  # best s in b ≈ s·a
        m = LAYER_RE.search(n)
        if m:
            row = per_layer.setdefault(int(m.group(1)), {"attn": [], "mlp": []})
            if n.endswith("norm.weight"):
                row[n.split(".")[-2]] = scale  # gain ratio of each norm
            elif "self_attn" in n:
                row["attn"].append(rel)
            else:
                row["mlp"].append(rel)
        else:
            other[n] = {"rel_change": rel, "best_scale": scale}
    for row in per_layer.values():
        row["attn"] = sum(row["attn"]) / len(row["attn"])
        row["mlp"] = sum(row["mlp"]) / len(row["mlp"])

    print(f"{'layer':>5}{'attn Δ':>9}{'mlp Δ':>9}   norm gain ratios (B/A)")
    for i in sorted(per_layer):
        r = per_layer[i]
        norms = "  ".join(f"{k.replace('_layernorm', '')}={v:.3f}"
                          for k, v in r.items() if k not in ("attn", "mlp"))
        print(f"{i:>5}{r['attn']:>9.4f}{r['mlp']:>9.4f}   {norms}")
    for n, v in other.items():
        print(f"{n:<32} rel_change={v['rel_change']:.4f}  best_scale={v['best_scale']:.4f}")
    return {"per_layer": per_layer, "other": other}


# ---------------------------------------------------------------- text


def text_batches(tok, n_seq):
    """Yields (input_ids [1, L], score_mask [L-1]) for wikitext chunks and chat responses."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ids = tok("\n".join(t for t in wiki["text"] if t.strip()), add_special_tokens=False).input_ids
    for i in range(n_seq):
        chunk = torch.tensor([ids[i * 512 : (i + 1) * 512]])
        mask = torch.zeros(511, dtype=torch.bool)
        mask[32:] = True  # skip the first tokens, which have little context
        yield "wikitext", chunk, mask

    path = hf_hub_download("allenai/dolci-3-instruct-dpo-with-metadata",
                           "data/train-00000-of-00004.parquet", repo_type="dataset")
    rows = pq.read_table(path, columns=["chosen"]).slice(0, 5000).to_pylist()
    rows = [r["chosen"] for r in rows if len(r["chosen"]) == 2]
    random.Random(0).shuffle(rows)
    for msgs in rows[:n_seq]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
        prefix = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        if not full.startswith(prefix):
            continue
        full_ids = tok(full, add_special_tokens=False).input_ids[:1024]
        start = len(tok(prefix, add_special_tokens=False).input_ids)
        if start >= len(full_ids) - 8:
            continue
        mask = torch.zeros(len(full_ids) - 1, dtype=torch.bool)
        mask[start - 1 :] = True  # positions that predict response tokens
        yield "chat", torch.tensor([full_ids]), mask


@torch.no_grad()
def probe_text(ma, mb, tok, n_seq):
    acc = {}
    for source, ids, mask in text_batches(tok, n_seq):
        ids = ids.cuda()
        za = ma(ids).logits[0, :-1][mask.cuda()].float()
        zb = mb(ids).logits[0, :-1][mask.cuda()].float()
        tgt = ids[0, 1:][mask.cuda()]
        s = acc.setdefault(source, {"n": 0, "H_A": 0.0, "H_B": 0.0, "nll_A": 0.0, "nll_B": 0.0,
                                    "std_A": 0.0, "std_B": 0.0, "kl": [0.0] * len(T_GRID),
                                    "nll_A_t": [0.0] * len(T_GRID), "nll_B_t": [0.0] * len(T_GRID)})
        for c in range(0, za.shape[0], 512):
            a, b, y = za[c : c + 512], zb[c : c + 512], tgt[c : c + 512]
            lpa, lpb = F.log_softmax(a, -1), F.log_softmax(b, -1)
            pa = lpa.exp()
            s["n"] += a.shape[0]
            s["H_A"] += -(pa * lpa).sum().item()
            s["H_B"] += -(lpb.exp() * lpb).sum().item()
            s["nll_A"] += -lpa.gather(1, y[:, None]).sum().item()
            s["nll_B"] += -lpb.gather(1, y[:, None]).sum().item()
            s["std_A"] += a.std(-1).sum().item()
            s["std_B"] += b.std(-1).sum().item()
            for k, t in enumerate(T_GRID):
                lqb = F.log_softmax(b / t, -1)
                s["kl"][k] += (pa * (lpa - lqb)).sum().item()
                s["nll_B_t"][k] += -lqb.gather(1, y[:, None]).sum().item()
                s["nll_A_t"][k] += -F.log_softmax(a / t, -1).gather(1, y[:, None]).sum().item()

    out = {}
    print(f"{'source':<10}{'tokens':>8}{'H_A':>7}{'H_B':>7}{'std_A':>7}{'std_B':>7}{'nll_A':>7}{'nll_B':>7}"
          f"{'KL':>8}{'t*':>6}{'KL@t*':>8}{'expl':>6}{'tnll_A':>8}{'tnll_B':>8}")
    for source, s in acc.items():
        n = s["n"]
        kl = [x / n for x in s["kl"]]
        k1 = min(range(len(T_GRID)), key=lambda k: abs(T_GRID[k] - 1.0))
        kbest = min(range(len(T_GRID)), key=lambda k: kl[k])
        r = {
            "tokens": n,
            "entropy_A": s["H_A"] / n, "entropy_B": s["H_B"] / n,
            "logit_std_A": s["std_A"] / n, "logit_std_B": s["std_B"] / n,
            "nll_A": s["nll_A"] / n, "nll_B": s["nll_B"] / n,
            "kl_A_B": kl[k1], "t_star": T_GRID[kbest], "kl_at_t_star": kl[kbest],
            "explained_by_temperature": 1 - kl[kbest] / kl[k1],
            # temperature that best fits the actual next tokens (text calibration)
            "t_nll_A": T_GRID[min(range(len(T_GRID)), key=lambda k: s["nll_A_t"][k])],
            "t_nll_B": T_GRID[min(range(len(T_GRID)), key=lambda k: s["nll_B_t"][k])],
        }
        out[source] = r
        print(f"{source:<10}{n:>8}{r['entropy_A']:>7.3f}{r['entropy_B']:>7.3f}{r['logit_std_A']:>7.2f}"
              f"{r['logit_std_B']:>7.2f}{r['nll_A']:>7.3f}{r['nll_B']:>7.3f}{r['kl_A_B']:>8.4f}{r['t_star']:>6.2f}"
              f"{r['kl_at_t_star']:>8.4f}{r['explained_by_temperature']:>6.0%}{r['t_nll_A']:>8.2f}{r['t_nll_B']:>8.2f}")
    return out


# ---------------------------------------------------------------- swap


def components(n_layers):
    block = n_layers // 4
    comps = {"embed": lambda n: n.startswith("model.embed_tokens")}
    for i in range(4):
        lo, hi = i * block, (i + 1) * block if i < 3 else n_layers
        comps[f"L{lo}-{hi - 1}"] = (lambda lo, hi: lambda n: bool(
            (m := LAYER_RE.search(n)) and lo <= int(m.group(1)) < hi))(lo, hi)
    comps["head"] = lambda n: n.startswith("model.norm") or n.startswith("lm_head")
    return comps


@torch.no_grad()
def mc_eval(model, batches, letter_ids, test):
    rows = [None] * len(test)
    for idx, enc in batches:
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        full = torch.log_softmax(logits, dim=-1)
        for j, i in enumerate(idx):
            lp = [torch.logsumexp(full[j, letter_ids[l]], 0).item() for l in "ABCD"]
            m = max(lp)
            p = [math.exp(x - m) for x in lp]
            z = sum(p)
            rows[i] = {"probs": [x / z for x in p], "answer": test[i]["answer"]}
    confs = [max(r["probs"]) for r in rows]
    correct = [int(r["probs"].index(max(r["probs"])) == r["answer"]) for r in rows]
    return {"acc": sum(correct) / len(rows), "conf": sum(confs) / len(rows),
            "ece": ece(confs, correct), "t_star": fit_t(rows), "auroc": auroc(confs, correct)}


@torch.no_grad()
def probe_swap(ma, mb, tok, n_q):
    test, shots = load_task("mmlu")
    idx = list(range(len(test)))
    random.Random(0).shuffle(idx)
    test = [test[i] for i in sorted(idx[:n_q])]
    prompts = build_prompts(test, shots, 5, tok, chat=False)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    batches = []
    for s in range(0, len(order), 16):
        ids = order[s : s + 16]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        batches.append((ids, enc))
    letter_ids = letter_token_ids(tok)

    sa = {n: p.detach() for n, p in ma.named_parameters()}
    sb = {n: p.detach().clone() for n, p in mb.named_parameters()}
    params = dict(mb.named_parameters())

    def set_weights(pick):
        for n, p in params.items():
            p.data.copy_(pick(n))

    comps = components(mb.config.num_hidden_layers)
    if mb.config.tie_word_embeddings:
        print("warning: tied embeddings, so 'embed' and 'head' overlap")
    variants = [("A", lambda n: sa[n]), ("B", lambda n: sb[n])]
    for c, inside in comps.items():
        variants.append((f"B, {c} from A", (lambda inside: lambda n: sa[n] if inside(n) else sb[n])(inside)))
    for c, inside in comps.items():
        variants.append((f"A, {c} from B", (lambda inside: lambda n: sb[n] if inside(n) else sa[n])(inside)))
    for a in (0.25, 0.5, 0.75, 1.25, 1.5):
        variants.append((f"interp a={a}", (lambda a: lambda n: (sa[n].float() + a * (sb[n].float() - sa[n].float())).to(sa[n].dtype))(a)))

    out = {}
    print(f"{'variant':<24}{'acc':>7}{'conf':>7}{'ECE':>7}{'T*':>6}{'AUROC':>7}")
    for name, pick in variants:
        set_weights(pick)
        r = mc_eval(mb, batches, letter_ids, test)
        out[name] = r
        print(f"{name:<24}{r['acc']:>7.3f}{r['conf']:>7.3f}{r['ece']:>7.3f}{r['t_star']:>6.2f}{r['auroc']:>7.3f}", flush=True)
    set_weights(lambda n: sb[n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--probes", default="weights,text,swap")
    ap.add_argument("--n-seq", type=int, default=128, help="sequences per text source")
    ap.add_argument("--n-q", type=int, default=2000, help="MMLU questions for swap")
    ap.add_argument("--out", default="results_mech")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    probes = args.probes.split(",")

    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        path = os.path.join(args.out, f"{a}__{b}.json")
        result = json.load(open(path)) if os.path.exists(path) else {}
        todo = [p for p in probes if p not in result]
        if not todo:
            print(f"skip {pair}")
            continue
        print(f"\n######## {a}  ->  {b}", flush=True)
        tok = AutoTokenizer.from_pretrained(f"allenai/{b}")
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ma, mb = load(a), load(b)
        for p in todo:
            print(f"\n--- {p}", flush=True)
            if p == "weights":
                result[p] = probe_weights(ma, mb)
            elif p == "text":
                result[p] = probe_text(ma, mb, tok, args.n_seq)
            elif p == "swap":
                result[p] = probe_swap(ma, mb, tok, args.n_q)
            json.dump(result, open(path, "w"), indent=1)
        del ma, mb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
