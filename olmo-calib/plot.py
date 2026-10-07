"""Summary table + reliability diagrams for every <dir>/*.summary.json."""
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pipeline order: base stages, RL straight from base, then the Instruct and Think paths.
ORDER = [
    "Olmo-3-1025-7B@stage1-step1413814",
    "Olmo-3-1025-7B@stage2-step47684",
    "Olmo-3-1025-7B",
    "Olmo-3-7B-RL-Zero-Mix",
    "Olmo-3-7B-Instruct-SFT",
    "Olmo-3-7B-Instruct-DPO",
    "Olmo-3-7B-Instruct",
    "Olmo-3-7B-Instruct-SFT+chat",
    "Olmo-3-7B-Instruct-DPO+chat",
    "Olmo-3-7B-Instruct+chat",
    "Olmo-3-7B-Think-SFT",
    "Olmo-3-7B-Think-DPO",
    "Olmo-3-7B-Think",
]


def rank(s):
    return (ORDER.index(s["tag"]), "") if s["tag"] in ORDER else (len(ORDER), s["tag"])


out_dir = sys.argv[1].rstrip("/\\") if len(sys.argv) > 1 else "results"
summaries = sorted((json.load(open(p)) for p in glob.glob(f"{out_dir}/*.summary.json")), key=rank)
if not summaries:
    sys.exit(f"no summaries in {out_dir}/")

print(f"{'tag':<36}{'n':>7}{'acc':>8}{'conf':>8}{'ECE':>8}{'brier':>8}{'mass':>8}")
for s in summaries:
    print(f"{s['tag']:<36}{s['n']:>7}{s['acc']:>8.3f}{s['avg_conf']:>8.3f}"
          f"{s['ece']:>8.3f}{s['brier']:>8.3f}{s['avg_letter_mass']:>8.3f}")

cols = min(4, len(summaries))
rows = -(-len(summaries) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
for ax, s in zip(axes.flat, summaries):
    d = s["reliability"]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.bar([(b["bin"] + 0.5) / 15 for b in d], [b["acc"] for b in d], width=1 / 15, alpha=0.7, edgecolor="black")
    ax.set_title(f"{s['tag']}\nacc={s['acc']:.2f}  ECE={s['ece']:.3f}", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
for ax in list(axes.flat)[len(summaries):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{out_dir}/reliability.png", dpi=120)
print(f"saved {out_dir}/reliability.png")
