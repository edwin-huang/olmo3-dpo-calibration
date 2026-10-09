#!/usr/bin/env bash
# "Why" tests: causal SFT (known vs unknown), decision-point sharpening, per-token attribution.
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== $* $(date +%T)"; }

step "label trivia"
[ -f data_trivia/labeled.jsonl ] || python label_trivia.py --n 40000 --out data_trivia/labeled.jsonl > log_label.txt 2>&1
tail -2 log_label.txt

step "sft smoke"
python sft_known.py --cond mixed --seed 99 --n-train 64 --epochs 1 --n-q 50 --out smoke_sftk > log_sft_smoke.txt 2>&1
if grep -q "ALL DONE" log_sft_smoke.txt; then
  rm -rf smoke_sftk
  for c in base known unknown mixed; do
    step "sft $c seed 0"
    python sft_known.py --cond $c --seed 0 > log_sft_${c}_s0.txt 2>&1 || echo "FAILED sft $c s0"
  done
else
  echo "SFT SMOKE FAILED (see log_sft_smoke.txt); skipping SFT runs"
fi

step "decision points"
python decision_points.py --out results_decision > log_decision.txt 2>&1 || echo "FAILED decision"

step "token attribution"
python token_attribution.py --n 2000 --out results_tokens > log_tokens.txt 2>&1 || echo "FAILED tokens"

if grep -q "ALL DONE" log_sft_smoke.txt; then
  for c in known unknown mixed; do
    step "sft $c seed 1"
    python sft_known.py --cond $c --seed 1 > log_sft_${c}_s1.txt 2>&1 || echo "FAILED sft $c s1"
  done
fi
step "QUEUE DONE"
