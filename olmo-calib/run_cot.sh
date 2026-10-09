#!/usr/bin/env bash
# Chain-of-thought test for Instruct SFT vs DPO; waits for the LoRA text probe to free the GPU.
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== $* $(date +%T)"; }

true
step "lora text probe finished"

SFT=allenai/Olmo-3-7B-Instruct-SFT
DPO=allenai/Olmo-3-7B-Instruct-DPO
step "cot smoke"
python cot_test.py --stage gen --model $SFT --n-mmlu 6 --n-arc 2 --max-tokens 256 --out smoke_cot > log_cot_smoke.txt 2>&1 \
  && python cot_test.py --stage score --model $SFT --n-mmlu 6 --n-arc 2 --out smoke_cot >> log_cot_smoke.txt 2>&1
if [ "$(grep -c "ALL DONE" log_cot_smoke.txt)" -lt 2 ]; then echo "COT SMOKE FAILED"; exit 1; fi

for m in $SFT $DPO; do
  step "gen $m"
  python cot_test.py --stage gen --model $m > log_cot_gen_$(basename $m).txt 2>&1 || echo "FAILED gen $m"
done
for m in $SFT $DPO; do
  step "score $m"
  python cot_test.py --stage score --model $m > log_cot_score_$(basename $m).txt 2>&1 || echo "FAILED score $m"
done
step "COT DONE"
