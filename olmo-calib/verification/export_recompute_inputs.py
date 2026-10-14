"""Export per-question A-D probabilities for an independent T*/ECE/AUROC recompute (Task 2b). Writes one CSV per
model x benchmark into verification/recompute_inputs/, plus reported_values.json with the numbers our pipeline
(eval_ckpt.py calib()) reported for the same files, so a recompute can be checked against them.

CSV columns: qid (row index in the benchmark's test split as loaded by calib_eval.load_task), subject, answer (0-3),
p_A, p_B, p_C, p_D (softmax over the four letter tokens' logits, i.e. renormalized over A-D).

Usage: python verification/export_recompute_inputs.py   (from olmo-calib/; add a run as NAME=results_ckpt/<dir> args)
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "instruct_sft": "results_ckpt/released/Olmo-3-7B-Instruct-SFT",
    "instruct_dpo_released": "results_ckpt/released/Olmo-3-7B-Instruct-DPO",
    "q25_plain": "results_ckpt/q25_all/final_full",
    "q25_agreement_weighted": "results_ckpt/q25_w5/final_full",
}
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    RUNS[name] = path
OUT = os.path.join(ROOT, "verification", "recompute_inputs")
os.makedirs(OUT, exist_ok=True)
reported = {}
for name, d in RUNS.items():
    s = json.load(open(os.path.join(ROOT, d, "summary.json")))
    letters = json.load(open(os.path.join(ROOT, d, "letters.json")))
    reported[name] = {"source": d, "summary": {k: v for k, v in s.items() if k.startswith("letters_")},
                      "letters_json": {t: {k: v for k, v in r.items() if k != "_meta"} for t, r in letters.items() if t != "_meta"}}
    for bench, sub in (("mmlu", "mmlu5_full"), ("arc", "arc5_full")):
        rows = [json.loads(l) for l in open(os.path.join(ROOT, d, sub, "letters.jsonl"))]
        with open(os.path.join(OUT, f"{name}__{bench}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "subject", "answer", "p_A", "p_B", "p_C", "p_D"])
            for i, r in enumerate(rows):
                w.writerow([i, r["subject"], r["answer"], *[f"{p:.10g}" for p in r["probs"]]])
        print(f"{name:<24} {bench}: {len(rows)} questions")
json.dump(reported, open(os.path.join(OUT, "reported_values.json"), "w"), indent=1)
print("wrote", OUT)
