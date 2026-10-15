#!/bin/bash
# Smoke test on a small box (4x H100 or 2x H200) before renting 8x H100. Checks, in order:
#   1. open-instruct env + patch + data build (setup_train_box.sh)
#   2. DPO on 200 Instruct pairs incl. the 10 longest (peak memory at 16K tokens), batch 16 -> 13 steps,
#      2 model-only snapshots, one full-state save (at the last step, deleted after success), final model saved
#   3. eval_ckpt.py --smoke on the final model (every part, incl. vLLM self-consistency and IFEval)
#   4. full baseline evals of the 4 released checkpoints, one per GPU (the D0 gate compares against these)
# Writes /workspace/SMOKE_OK or /workspace/SMOKE_FAILED (with the failing step) for the watcher.
# Run: cd /workspace && nohup bash olmo-calib/train/run_smoke.sh > smoke.log 2>&1 &
set -uo pipefail
export WS=${WS:-/workspace}
PY=$WS/open-instruct/.venv/bin/python
fail() { echo "$1" > $WS/SMOKE_FAILED; echo "SMOKE FAILED: $1"; exit 1; }
rm -f $WS/SMOKE_OK $WS/SMOKE_FAILED

bash $WS/olmo-calib/train/setup_train_box.sh || fail "setup"

$PY - <<'EOF' || fail "smoke data"
import os, pyarrow.parquet as pq, pyarrow as pa
t = pq.read_table(os.environ["WS"] + "/dpo_data/q25_all.parquet")
rows = t.to_pylist()
size = lambda r: sum(len(m["content"]) for m in r["chosen"] + r["rejected"])
longest = sorted(rows, key=size)[-10:]
pq.write_table(pa.Table.from_pylist(rows[:190] + longest), os.environ["WS"] + "/dpo_data/smoke200.parquet")
print("smoke200: 190 random + 10 longest, longest pair", size(longest[-1]), "chars")
EOF

# Steps that already passed are skipped on a rerun (runs/smoke.DONE, runs/smoke_lr.CHECKED).
if [ ! -f $WS/runs/smoke.DONE ]; then
  BATCH=16 N_SNAP=2 N_FULL=1 bash $WS/olmo-calib/train/launch_dpo.sh smoke $WS/dpo_data/smoke200.parquet \
    allenai/Olmo-3-7B-Instruct-SFT 1e-6 || fail "train (see runs/smoke.log)"
fi
ls -d $WS/runs/smoke/snap_step_* || fail "no snapshots"
grep -n "Peak memory\|Total optimization steps\|it/s\|s/it" $WS/runs/smoke.log | tail -8

# At LR 1e-6 the first few dozen updates are below bf16 rounding of the weights, so the smoke run's loss stays exactly
# ln 2 and can't show that training works. Same data at LR 5e-5 (4K tokens, no saves) must bring the loss below 0.69.
if [ ! -f $WS/runs/smoke_lr.CHECKED ]; then
  BATCH=16 N_SNAP=1 N_FULL=0 MAX_LEN=4096 bash $WS/olmo-calib/train/launch_dpo.sh smoke_lr $WS/dpo_data/smoke200.parquet \
    allenai/Olmo-3-7B-Instruct-SFT 5e-5 || fail "train at high LR (see runs/smoke_lr.log)"
  LAST_LOSS=$(grep -oP "Step: \d+, LR: [^,]+, Loss: \K[0-9.]+" $WS/runs/smoke_lr.log | tail -1)
  echo "high-LR check: last loss $LAST_LOSS"
  $PY -c "import sys; sys.exit(0 if float('$LAST_LOSS') < 0.69 else 1)" || fail "loss did not move at LR 5e-5 ($LAST_LOSS)"
  rm -rf $WS/runs/smoke_lr
  touch $WS/runs/smoke_lr.CHECKED
fi

cd $WS/olmo-calib
$PY eval_ckpt.py --ckpt $WS/runs/smoke --tag smoke/final --smoke --out $WS/results_ckpt || fail "eval smoke (see results_ckpt/smoke/final/*.log)"

# Released checkpoints, full eval suite. Each GPU works through its share one model at a time (vLLM takes 85% of a
# GPU, so two evals must never share one). Parts already done are skipped on rerun.
MODELS=(Olmo-3-7B-Instruct-SFT Olmo-3-7B-Instruct-DPO Olmo-3-7B-Think-SFT Olmo-3-7B-Think-DPO)
NG=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NG - 1))); do
  (
    for j in "${!MODELS[@]}"; do
      (( j % NG == g )) || continue
      m=${MODELS[$j]}
      ref=allenai/Olmo-3-7B-Instruct-SFT; [[ $m == Olmo-3-7B-Think-* ]] && ref=allenai/Olmo-3-7B-Think-SFT
      CUDA_VISIBLE_DEVICES=$g $PY eval_ckpt.py --ckpt allenai/$m --tag released/$m --ref $ref \
        --out $WS/results_ckpt > $WS/results_ckpt/released_$m.log 2>&1
    done
  ) &
done
wait
for m in "${MODELS[@]}"; do
  $PY -c "import json,sys; s=json.load(open('$WS/results_ckpt/released/$m/summary.json')); sys.exit(1 if s['failed'] else 0)" \
    || fail "baseline eval $m"
done
touch $WS/SMOKE_OK
echo SMOKE OK
