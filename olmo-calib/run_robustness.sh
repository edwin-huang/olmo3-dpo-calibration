#!/usr/bin/env bash
# Robustness: every checkpoint on full MMLU and ARC-Challenge, 5-shot and 0-shot.
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1

EVALS=mmlu:5:0,mmlu:0:0,arc:5:0,arc:0:0
OUT=results_robust
B=allenai/Olmo-3-1025-7B

run() {  # model [revision]
  echo "=== $1 ${2:-main} $(date +%T) ==="
  python calib_eval.py --model "$1" ${2:+--revision "$2"} --evals $EVALS --out $OUT \
    || echo "FAILED $1 ${2:-}"
}

run $B stage1-step1413814
run $B stage2-step47684
run $B
run allenai/Olmo-3-7B-RL-Zero-Mix
run allenai/Olmo-3-7B-Instruct-SFT
run allenai/Olmo-3-7B-Instruct-DPO
run allenai/Olmo-3-7B-Instruct
run allenai/Olmo-3-7B-Think-SFT
run allenai/Olmo-3-7B-Think-DPO
run allenai/Olmo-3-7B-Think

for d in $OUT/*/; do echo "##### $d"; python plot.py "$d"; done
echo "ALL DONE $(date +%T)"
