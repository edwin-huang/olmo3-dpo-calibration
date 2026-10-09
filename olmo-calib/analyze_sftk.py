"""Summary table for sft_known.py runs: calibration and trivia accuracy per condition, seed and epoch.

Usage: python analyze_sftk.py results_sftk
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results_sftk"
runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{d}/*.json"))}
order = ["base_s0"] + [f"{c}_s{s}" for s in (0, 1) for c in ("known", "mixed", "unknown", "prose")]

print(f"{'run':<12}{'ep':>3}{'loss':>7} | {'MMLU acc':>8}{'conf':>6}{'ECE':>6}{'T*':>6}{'AUROC':>7} | "
      f"{'ARC acc':>7}{'conf':>6}{'ECE':>6}{'T*':>6}{'AUROC':>7} | {'tq known':>8}{'unknown':>8}{'train':>7}")
for name in order:
    if name not in runs:
        continue
    for ep, r in sorted((k, v) for k, v in runs[name].items() if k.startswith("epoch_")):
        m, a = r["mmlu"], r["arc"]
        loss = f"{r['train_loss']:.3f}" if "train_loss" in r else "-"
        print(f"{name:<12}{ep[6:]:>3}{loss:>7} | {m['acc']:>8.3f}{m['conf']:>6.3f}{m['ece']:>6.3f}{m['t_star']:>6.2f}"
              f"{m['auroc']:>7.3f} | {a['acc']:>7.3f}{a['conf']:>6.3f}{a['ece']:>6.3f}{a['t_star']:>6.2f}{a['auroc']:>7.3f}"
              f" | {r['trivia_heldout_known']:>8.3f}{r['trivia_heldout_unknown']:>8.3f}{r['trivia_train_fit']:>7.3f}")
