#!/usr/bin/env bash
# Self-consistency confidence after chain-of-thought, Instruct SFT vs DPO.
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== $* $(date +%T)"; }
step "sc smoke"
python cot_test.py --stage sample --model allenai/Olmo-3-7B-Instruct-SFT --n-mmlu 10 --n-arc 10 --k 2 --max-tokens 256 --out smoke_sc > log_sc_smoke.txt 2>&1
if ! grep -q "ALL DONE" log_sc_smoke.txt; then echo "SC SMOKE FAILED"; exit 1; fi
rm -rf smoke_sc
for m in allenai/Olmo-3-7B-Instruct-SFT allenai/Olmo-3-7B-Instruct-DPO; do
  step "sample $m"
  python cot_test.py --stage sample --model $m --k 8 > log_sc_$(basename $m).txt 2>&1 || echo "FAILED sample $m"
done
step "SC DONE"
