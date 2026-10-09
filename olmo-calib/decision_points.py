"""Does DPO sharpen specifically at decision points?

For every token of ordinary text, the earlier model A's next-token entropy H_A says how torn it
was; k = exp(H_A) is its effective number of options. Tokens are grouped by k, and per group we
fit the sharpening s that best maps A onto B:  min_s KL(p_B || softmax(s * z_A)).
s > 1 = B is sharper than A there. Prediction for DPO: s peaks where A was choosing among a few
options (k ~ 2-4) and stays ~1 where the next token was obvious or wide open.

Usage: python decision_points.py --out results_decision
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mechanism import load, text_batches

PAIRS = [
    "Olmo-3-7B-Instruct-SFT:Olmo-3-7B-Instruct-DPO",
    "Olmo-3-7B-Think-SFT:Olmo-3-7B-Think-DPO",
    "Olmo-3-1025-7B:Olmo-3-7B-Think-SFT",
    "Olmo-3-1025-7B:Olmo-3-7B-RL-Zero-Mix",
]
K_EDGES = [1.1, 1.5, 2.5, 4.0, 8.0, 16.0]  # bins: <1.1, 1.1-1.5, ..., >=16
K_NAMES = ["<1.1", "1.1-1.5", "1.5-2.5", "2.5-4", "4-8", "8-16", ">=16"]
S_GRID = [math.exp(math.log(0.5) + i * (math.log(3.0) - math.log(0.5)) / 40) for i in range(41)]


@torch.no_grad()
def probe(ma, mb, tok, n_seq):
    nb = len(K_NAMES)
    edges = torch.tensor(K_EDGES, device="cuda")
    acc = {}
    for source, ids, mask in text_batches(tok, n_seq):
        ids = ids.cuda()
        m = mask.cuda()
        za = ma(ids).logits[0, :-1][m].float()
        zb = mb(ids).logits[0, :-1][m].float()
        s = acc.setdefault(source, {k: torch.zeros(nb, dtype=torch.float64, device="cuda") for k in
                                    ("n", "H_A", "H_B", "top_A", "top_B", "agree")})
        s.setdefault("kl", torch.zeros(len(S_GRID), nb, dtype=torch.float64, device="cuda"))
        for c in range(0, za.shape[0], 512):
            a, b = za[c : c + 512], zb[c : c + 512]
            lpa, lpb = F.log_softmax(a, -1), F.log_softmax(b, -1)
            pa, pb = lpa.exp(), lpb.exp()
            ha, hb = -(pa * lpa).sum(-1), -(pb * lpb).sum(-1)
            bins = torch.bucketize(ha.exp(), edges, right=True)
            add = lambda key, v: s[key].index_add_(0, bins, v.double())
            add("n", torch.ones_like(ha))
            add("H_A", ha)
            add("H_B", hb)
            add("top_A", pa.max(-1).values)
            add("top_B", pb.max(-1).values)
            add("agree", (pa.argmax(-1) == pb.argmax(-1)).float())
            for i, sc in enumerate(S_GRID):
                kl = (pb * (lpb - F.log_softmax(sc * a, -1))).sum(-1)
                s["kl"][i].index_add_(0, bins, kl.double())

    out = {}
    i1 = min(range(len(S_GRID)), key=lambda i: abs(S_GRID[i] - 1.0))
    print(f"{'source':<9}{'k bin':<9}{'tokens':>8}{'H_A':>7}{'H_B':>7}{'top1 A':>8}{'top1 B':>8}"
          f"{'agree':>7}{'s*':>6}{'KL(s=1)':>9}{'KL(s*)':>8}{'expl':>6}")
    for source, s in acc.items():
        rows = []
        for j in range(nb):
            n = s["n"][j].item()
            if n < 50:
                continue
            kl = [s["kl"][i][j].item() / n for i in range(len(S_GRID))]
            ib = min(range(len(S_GRID)), key=lambda i: kl[i])
            r = {"k_bin": K_NAMES[j], "tokens": int(n),
                 "H_A": s["H_A"][j].item() / n, "H_B": s["H_B"][j].item() / n,
                 "top1_A": s["top_A"][j].item() / n, "top1_B": s["top_B"][j].item() / n,
                 "agree": s["agree"][j].item() / n, "s_star": S_GRID[ib],
                 "kl_s1": kl[i1], "kl_s_star": kl[ib],
                 "explained": 1 - kl[ib] / kl[i1] if kl[i1] > 0 else 0.0}
            rows.append(r)
            print(f"{source:<9}{r['k_bin']:<9}{r['tokens']:>8}{r['H_A']:>7.3f}{r['H_B']:>7.3f}{r['top1_A']:>8.3f}"
                  f"{r['top1_B']:>8.3f}{r['agree']:>7.3f}{r['s_star']:>6.2f}{r['kl_s1']:>9.4f}"
                  f"{r['kl_s_star']:>8.4f}{r['explained']:>6.0%}", flush=True)
        out[source] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--out", default="results_decision")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        path = os.path.join(args.out, f"{a}__{b}.json")
        if os.path.exists(path):
            print(f"skip {pair}")
            continue
        print(f"\n######## {a} -> {b}", flush=True)
        from calib_eval import load_config_override  # RL-Zero-Mix: pre-release model_type in config
        tok = AutoTokenizer.from_pretrained(f"allenai/{b}", **load_config_override(f"allenai/{b}", None))
        ma, mb = load(a), load(b)
        json.dump(probe(ma, mb, tok, args.n_seq), open(path, "w"), indent=1)
        del ma, mb
        torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
