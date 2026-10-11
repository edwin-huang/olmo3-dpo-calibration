#!/bin/bash
# One 25%-scale DPO run with the fast settings, then its evals.
#   bash run_d25.sh NAME DATA_PARQUET
# Ai2's recipe (LR 1e-6, beta 5, dpo_norm, batch 128, seed 42) from Instruct-SFT, with ZeRO-1 + one combined
# chosen/rejected forward: same loss and optimizer math as q25_all's ZeRO-2 + separate forwards, ~3.4x faster on 2xB200
# (per-micro-batch overhead dominated; see bench_bound.sh / bench_z1c.sh). Snapshots every 50 steps get --fast evals
# and are deleted after (PRUNE); the final model gets the full suite. Relaunching resumes (training from its last full
# state, evals skip finished parts). Writes /workspace/runs/NAME.ALL_DONE or NAME.ALL_FAILED.
set -uo pipefail
NAME=$1; DATA=$2
WS=/workspace
T=$WS/olmo-calib/train
PY=$WS/open-instruct/.venv/bin/python
M=allenai/Olmo-3-7B-Instruct-SFT
fail() { echo "$1" > $WS/runs/$NAME.ALL_FAILED; echo "$NAME FAILED: $1"; exit 1; }
rm -f $WS/runs/$NAME.ALL_DONE $WS/runs/$NAME.ALL_FAILED
export LOG_BATCH_INDEX=1
if [ ! -f $WS/runs/$NAME.DONE ]; then
  ZERO=1 CONCAT=True N_FULL=10 bash $T/launch_dpo.sh $NAME $DATA $M 1e-6 || fail "training (see runs/$NAME.log)"
fi
PRUNE=1 bash $T/eval_run.sh $NAME $M --fast || fail "snapshot evals (see runs/$NAME.EVAL_FAILED)"
cd $WS/olmo-calib
$PY eval_ckpt.py --ckpt $WS/runs/$NAME --tag $NAME/final_full --ref $M --out $WS/results_ckpt \
  > $WS/results_ckpt/$NAME/final_full.log 2>&1 || fail "full eval of the final model"
touch $WS/runs/$NAME.ALL_DONE
echo "$NAME ALL DONE"
