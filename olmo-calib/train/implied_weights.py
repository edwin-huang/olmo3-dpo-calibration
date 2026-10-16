"""Pre-registered wave, experiment 1: implied per-pair gradient weights at the start of training (no GPU).

At step 0 the policy equals the reference (Instruct-SFT), so each method's weight on a pair is a function of SFT's
own per-token-averaged log-probs (open-instruct's reference cache for all 259,922 Instruct DPO pairs). For each method:
the share of total starting weight that lands on agreed pairs (SFT margin > 0; plain DPO = the agreed share itself),
the ratio of mean weight on agreed vs disagreed pairs, and the Spearman correlation with the SFT margin. Methods whose
weight is a monotone function of the margin have correlation -1 by construction; the share and ratio are the
informative columns.

  DPO / IPO      flat at init (DPO: sigma(0) = 0.5; IPO: |0 - 1/(2 tau)| is the same for every pair)
  agreement-w.   ours: sigma(-5 m)
  SimPO          sigma(gamma - beta m), m = SFT's per-token margin; settings from github.com/princeton-nlp/SimPO README
  ORPO           odds-ratio term sigma(-(log odds_c - log odds_r)), odds = P / (1 - P), P = exp(mean per-token log p)
                 (Hong et al., 2024); ORPO's SFT term weights every pair's chosen response equally
  WPO            open-instruct dpo_utils.wpo_loss: min(exp(mean logp_c + mean logp_r), 1) x DPO's 0.5

Usage: python implied_weights.py /workspace/ref_cache/89203f41147b0c40.pt [/workspace/dpo_data/instruct_meta.parquet]
"""
import math
import sys

import torch

c = torch.load(sys.argv[1], map_location="cpu")
lc, lr = c["chosen_logps"].double(), c["rejected_logps"].double()
m = lc - lr
agreed = m > 0
print(f"pairs {len(m)}, agreed {agreed.double().mean():.4f}, mean SFT margin {m.mean():+.3f}")


def log_odds(lp):  # log(P / (1 - P)) with P = exp(lp), lp < 0
    return lp - torch.log1p(-torch.exp(lp).clamp(max=1 - 1e-12))


def rank(x):
    r = torch.empty_like(x)
    r[torch.argsort(x)] = torch.arange(len(x), dtype=x.dtype)
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm())).item()


methods = {
    "DPO (dpo_norm, beta 5)": torch.full_like(m, 0.5),
    "IPO": torch.ones_like(m),
    "agreement-weighted (ours, k 5)": torch.sigmoid(-5 * m),
}
for name, beta, ratio in (("Llama3-Instruct v0.2", 10.0, 0.3), ("Llama3-Instruct", 2.5, 0.55),
                          ("Mistral-Instruct", 2.5, 0.1), ("our small test", 5.0, 0.3)):
    methods[f"SimPO {name}: beta {beta:g}, gamma/beta {ratio:g}"] = torch.sigmoid(beta * ratio - beta * m)
methods["ORPO odds-ratio term"] = torch.sigmoid(-(log_odds(lc) - log_odds(lr)))
methods["WPO (open-instruct)"] = torch.clamp(torch.exp(lc + lr), max=1) * 0.5

print(f"\n{'method':<58}{'share on agreed':>16}{'agreed/disagreed':>18}{'Spearman w~margin':>19}")
rows = []
for name, w in methods.items():
    share = (w[agreed].sum() / w.sum()).item()
    ratio = (w[agreed].mean() / w[~agreed].mean()).item()
    rho = spearman(w, m) if w.std() > 0 else float("nan")
    rows.append((name, share, ratio, rho))
    print(f"{name:<58}{share:>16.3f}{ratio:>18.3f}{rho:>19.3f}")

if len(sys.argv) > 2:  # by preference source: does any method shift weight between delta-learning and judged pairs?
    import pyarrow.parquet as pq
    meta = pq.read_table(sys.argv[2], columns=["preference_type"]).to_pylist()
    assert len(meta) == len(m), (len(meta), len(m))
    kinds = [r["preference_type"] for r in meta]
    print("\nagreed share and weight share by preference type")
    for t in sorted(set(kinds)):
        sel = torch.tensor([k == t for k in kinds])
        line = f"  {t:<30} n {int(sel.sum()):>7}  agreed {agreed[sel].double().mean():.3f}"
        for name in ("DPO (dpo_norm, beta 5)", "agreement-weighted (ours, k 5)", "SimPO our small test: beta 5, gamma/beta 0.3"):
            w = methods[name]
            line += f" | {name.split(' (')[0].split(':')[0][:22]} {(w[sel].sum() / w.sum()).item():.3f}"
        print(line)
