"""Does agreement weighting favor pairs whose chosen response is longer? (no GPU)

Question: if longer responses get lower per-token SFT log-probs, pairs with a longer chosen response would look
"disagreed" (SFT margin < 0) and agreement weighting sigma(-5 m) would up-weight them, teaching length. Uses SFT's
per-token-averaged log-probs from open-instruct's reference cache for all 259,922 Instruct DPO pairs and token counts
of the final chosen/rejected turns.

  1. per-token SFT log-prob vs response length (chosen and rejected separately)
  2. by length-ratio decile (chosen tokens / rejected tokens): agreement share, mean margin, relative agreement weight
  3. starting weight under plain DPO (uniform), agreement weighting (ours) and SimPO (our small-test setting):
     share of weight on chosen-longer pairs, weighted mean log2 length ratio, weighted mean lengths; overall and
     within each preference type (to separate a within-type length shift from a type-mix shift)

Usage: python length_weights.py /workspace/ref_cache/89203f41147b0c40.pt /workspace/dpo_data/instruct_full.parquet \
           /workspace/dpo_data/instruct_meta.parquet
"""
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

CACHE, FULL, META = sys.argv[1:4]
c = torch.load(CACHE, map_location="cpu")
lc, lr = c["chosen_logps"].double().numpy(), c["rejected_logps"].double().numpy()
m = lc - lr
meta = pq.read_table(META, columns=["idx", "preference_type", "rejected_ntok"]).to_pylist()
assert [r["idx"] for r in meta] == list(range(len(m)))
kind = np.array(["multiturn" if r["preference_type"].startswith("multiturn") else r["preference_type"] for r in meta])

tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
tab = pq.read_table(FULL, columns=["chosen", "rejected"])
assert tab.num_rows == len(m)


def ntok(col):
    texts = [conv[-1]["content"] or "" for conv in tab.column(col).to_pylist()]
    out = []
    for i in range(0, len(texts), 20000):
        out += [len(x) for x in tok(texts[i:i + 20000], add_special_tokens=False).input_ids]
    return np.array(out, dtype=float)


nc, nr = ntok("chosen"), ntok("rejected")
check = np.array([r["rejected_ntok"] for r in meta], dtype=float)
print(f"pairs {len(m)}; rejected token counts match instruct_meta for {np.mean(np.abs(nr - check) <= 2):.1%}")
nc, nr = np.maximum(nc, 1), np.maximum(nr, 1)
ratio = np.log2(nc / nr)
agreed = m > 0


def spearman(a, b):
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return np.corrcoef(ra, rb)[0, 1]


print("\n== 1. per-token SFT log-prob vs length")
for name, n, lp in (("chosen", nc, lc), ("rejected", nr, lr)):
    q = np.quantile(n, [0, .2, .4, .6, .8, 1])
    cells = []
    for a, b in zip(q[:-1], q[1:]):
        s = (n >= a) & (n <= b)
        cells.append(f"{int(a)}-{int(b)} tok: {lp[s].mean():+.3f}")
    print(f"  {name:<9} Spearman(length, logp/token) {spearman(n, lp):+.3f} | by quintile: " + " | ".join(cells))
print(f"  Spearman(log2 chosen/rejected length, SFT margin) {spearman(ratio, m):+.3f}")

w_ours = 1 / (1 + np.exp(5 * m))
w_ours /= w_ours.mean()
w_simpo = 1 / (1 + np.exp(-(5 * 0.3 - 5 * m)))
w_simpo /= w_simpo.mean()
W = {"plain DPO": np.ones_like(m), "agreement-weighted": w_ours, "SimPO (test)": w_simpo}

print("\n== 2. by decile of log2(chosen tokens / rejected tokens)")
print(f"  {'ratio range':<16}{'n':>8}{'agreed':>8}{'margin':>9}{'ours rel. weight':>18}{'SimPO rel. weight':>19}")
q = np.quantile(ratio, np.linspace(0, 1, 11))
for a, b in zip(q[:-1], q[1:]):
    s = (ratio >= a) & (ratio <= b)
    print(f"  {a:+6.2f}..{b:+6.2f}{s.sum():>9}{agreed[s].mean():>8.1%}{m[s].mean():>+9.3f}"
          f"{w_ours[s].mean():>18.2f}{w_simpo[s].mean():>19.2f}")


def summary(sel):
    out = []
    for name, w in W.items():
        ww = w[sel] / w[sel].sum()
        out.append((name, (ww * (nc[sel] > nr[sel])).sum(), (ww * ratio[sel]).sum(),
                    (ww * nc[sel]).sum(), (ww * nr[sel]).sum(), np.exp((ww * np.log(nc[sel])).sum())))
    return out


print("\n== 3. what the starting weights emphasize (weighted over pairs)")
groups = {"all": np.ones_like(m, bool)} | {k: kind == k for k in sorted(set(kind))}
for g, sel in groups.items():
    print(f"  -- {g} (n {sel.sum()}, agreed {agreed[sel].mean():.1%})")
    print(f"     {'weighting':<20}{'chosen longer':>14}{'mean log2 ratio':>17}{'chosen tok':>12}{'rejected tok':>14}"
          f"{'chosen geo-mean':>17}")
    for name, longer, lrat, mc, mr, gc in summary(sel):
        print(f"     {name:<20}{longer:>14.1%}{lrat:>+17.3f}{mc:>12.0f}{mr:>14.0f}{gc:>17.0f}")
weight_by_type = {k: w_ours[kind == k].sum() / w_ours.sum() for k in sorted(set(kind))}
print("\n  weight share by type, plain vs ours: " + ", ".join(
    f"{k} {(kind == k).mean():.1%} -> {v:.1%}" for k, v in weight_by_type.items()))
