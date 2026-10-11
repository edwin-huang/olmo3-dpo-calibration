"""Table of eval_ckpt.py headline numbers across checkpoints (released baselines and run trajectories).

Usage: python compare_ckpts.py results_ckpt [run ...]   (default: every run found)
"""
import glob
import json
import os
import re
import sys

COLS = [("letters_mmlu_acc", "MMLU acc", 3), ("letters_mmlu_ece", "ECE", 3), ("letters_mmlu_t_star", "T*", 2),
        ("letters_mmlu_auroc", "AUROC", 3), ("letters_arc_t_star", "ARC T*", 2), ("format_cloze_norm_t_star", "cloze T*", 2),
        ("chat_mmlu_t_star", "chat T*", 2), ("sc_mmlu_excess", "SC excess", 3), ("text_s_star_wikitext", "s* wiki", 2),
        ("ifeval_prompt_strict", "IFEval", 3)]


def step(tag):
    m = re.search(r"snap_step_(\d+)$", tag)
    return int(m.group(1)) if m else 10 ** 9  # final last


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results_ckpt"
    runs = sys.argv[2:] or sorted(os.listdir(root))
    print(f"{'checkpoint':<38}" + "".join(f"{name:>10}" for _, name, _ in COLS) + "  failed")
    for run in runs:
        paths = glob.glob(os.path.join(root, run, "*", "summary.json"))
        for p in sorted(paths, key=lambda p: (step(os.path.dirname(p)), p)):
            s = json.load(open(p))
            cells = "".join(f"{s.get(k):>10.{d}f}" if isinstance(s.get(k), (int, float)) else f"{'-':>10}" for k, _, d in COLS)
            print(f"{s['tag']:<38}{cells}  {','.join(s['failed'])}")


if __name__ == "__main__":
    main()
