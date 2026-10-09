#!/usr/bin/env bash
# Rerun of the CoT tests saving per-question data: direct/cot letter probabilities + all 8 sampled chains.
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== $* $(date +%T)"; }
step "smoke"
python cot_test.py --stage sample --model allenai/Olmo-3-7B-Instruct-SFT --n-mmlu 4 --n-arc 4 --k 2 --max-tokens 128 --out smoke_perq > log_perq_smoke.txt 2>&1
if ! grep -q "ALL DONE" log_perq_smoke.txt || [ ! -s smoke_perq/Olmo-3-7B-Instruct-SFT.sc2.questions.jsonl ]; then echo "PERQ SMOKE FAILED"; exit 1; fi
rm -rf smoke_perq
for m in allenai/Olmo-3-7B-Instruct-SFT allenai/Olmo-3-7B-Instruct-DPO; do
  step "score $m"
  python cot_test.py --stage score --model $m > log_perq_score_$(basename $m).txt 2>&1 || echo "FAILED score $m"
  step "sample $m"
  python cot_test.py --stage sample --model $m --k 8 > log_perq_sample_$(basename $m).txt 2>&1 || echo "FAILED sample $m"
done
step "PERQ DONE"
