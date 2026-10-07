#!/usr/bin/env bash
# First "is it real?" check: MMLU (fixed 3000-question subset, 5-shot) across Olmo 3 7B stages.
# Results are tagged with Olmo model names (plus @branch for intermediate checkpoints).
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1

B=allenai/Olmo-3-1025-7B
run() {  # model [revision] [extra args...]
  local model=$1 rev=${2:-}
  shift 2 2>/dev/null || shift $#
  echo "=== $model ${rev:-main} $* $(date +%T) ==="
  python calib_eval.py --model "$model" ${rev:+--revision "$rev"} --evals mmlu:5:3000 --out results "$@" \
    || echo "FAILED $model $rev"
}

# Base path: pretrain end -> midtrain end -> long-context (final base)
run $B stage1-step1413814
run $B stage2-step47684
run $B ""

# RL straight from base, no SFT/DPO
run allenai/Olmo-3-7B-RL-Zero-Mix ""

# Instruct path, plain prompt and chat template
for m in Olmo-3-7B-Instruct-SFT Olmo-3-7B-Instruct-DPO Olmo-3-7B-Instruct; do
  run allenai/$m ""
  run allenai/$m "" --chat
done

# Think path (plain prompt only: in chat mode these want to reason before answering)
run allenai/Olmo-3-7B-Think-SFT ""
run allenai/Olmo-3-7B-Think-DPO ""
run allenai/Olmo-3-7B-Think ""

python plot.py results/mmlu5_3000
echo "ALL DONE $(date +%T)"
