#!/usr/bin/env bash
# Second batch: seed-1 SFT reruns (length cap fix), prose control, RL-Zero decision-point control.
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "=== $* $(date +%T)"; }

for c in known mixed; do
  step "sft $c seed 1"
  rm -f results_sftk/${c}_s1.json
  python sft_known.py --cond $c --seed 1 > log_sft_${c}_s1.txt 2>&1 || echo "FAILED sft $c s1"
done

step "prose smoke"
python sft_known.py --cond prose --seed 99 --n-train 64 --epochs 1 --n-q 50 --out smoke_sftk > log_prose_smoke.txt 2>&1
if grep -q "ALL DONE" log_prose_smoke.txt; then
  rm -rf smoke_sftk
  for s in 0 1; do
    step "sft prose seed $s"
    python sft_known.py --cond prose --seed $s > log_sft_prose_s$s.txt 2>&1 || echo "FAILED sft prose s$s"
  done
else
  echo "PROSE SMOKE FAILED (see log_prose_smoke.txt)"
fi

step "decision points (RL-Zero control)"
python decision_points.py --out results_decision > log_decision2.txt 2>&1 || echo "FAILED decision"
step "QUEUE2 DONE"
