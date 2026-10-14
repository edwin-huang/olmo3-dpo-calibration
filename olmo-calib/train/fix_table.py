"""Fix-wave table: calibration vs learning for each small run (results_ckpt/small/<tag>/summary.json + judge files).

Usage: python fix_table.py results_ckpt/small fx_sft fx_base8 fx_simpo ...   (default: every fx_* tag found)
"""
import json
import os
import sys

COLS = [("letters_mmlu_t_star", "T*", 2), ("letters_mmlu_ece", "ECE", 3), ("letters_arc_t_star", "ARC T*", 2),
        ("chat_mmlu_t_star", "chat T*", 2), ("format_cloze_norm_t_star", "cloze", 2), ("letters_mmlu_acc", "acc", 3),
        ("letters_mmlu_auroc", "AUROC", 3), ("ifeval_prompt_strict", "IFEval", 3), ("pairs_acc", "pairs", 3),
        ("pairs_acc_ref_disagreed", "learned", 3), ("pairs_acc_ref_agreed", "kept", 3), ("gen_mean_tokens", "gen tok", 0),
        ("judge_win", "win", 3), ("judge_win_la", "win-LA", 3)]


def main():
    root = sys.argv[1]
    tags = sys.argv[2:] or sorted(t for t in os.listdir(root) if t.startswith("fx_") and os.path.isdir(os.path.join(root, t)))
    print(f"{'run':<12}" + "".join(f"{n:>9}" for _, n, _ in COLS))
    for t in tags:
        p = os.path.join(root, t, "summary.json")
        s = json.load(open(p)) if os.path.exists(p) else {}
        j = os.path.join(root, t, "judge_vs_small_fx_sft.json")
        if os.path.exists(j):
            jj = json.load(open(j))
            s["judge_win"], s["judge_win_la"] = jj["win"], jj["win_length_adjusted"]
        print(f"{t:<12}" + "".join(f"{s[k]:>9.{d}f}" if isinstance(s.get(k), (int, float)) else f"{'-':>9}" for k, _, d in COLS))


if __name__ == "__main__":
    main()
