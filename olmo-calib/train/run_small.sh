#!/bin/bash
# One small DPO test on one GPU, then focused evals on its final model.
#   bash run_small.sh NAME DATA GPU [KEY=VALUE ...]     e.g. bash run_small.sh neg_only s8_a.parquet 0 DPO_DETACH=chosen
# Defaults: Instruct-SFT start, batch 128, LR 8e-6 (override with LR=...), ZeRO-1 on 1 GPU, combined forward,
# MAX_LEN 8192, no snapshots or full-state saves. Small runs trade data for learning rate: q25 showed log T* tracks the
# cumulative learning rate, so 8K pairs at 8e-6 aims at the same movement as 65K at 1e-6 (pilot-checked).
# Evals (--fast): letters (MMLU 3000 + ARC), format (letters/numbers/cloze, 1000 MMLU), chat (500 MMLU + 250 ARC),
# written to /workspace/results_ckpt/small/NAME. The model is deleted after (KEEP=1 keeps it).
# Writes /workspace/runs/NAME.SMALL_DONE or NAME.SMALL_FAILED.
set -uo pipefail
NAME=$1; DATA=$2; GPU=$3; shift 3
WS=/workspace
T=$WS/olmo-calib/train
PY=$WS/open-instruct/.venv/bin/python
M=allenai/Olmo-3-7B-Instruct-SFT
for kv in "$@"; do export "$kv"; done
fail() { echo "$1" > $WS/runs/$NAME.SMALL_FAILED; echo "$NAME FAILED: $1"; exit 1; }
rm -f $WS/runs/$NAME.SMALL_DONE $WS/runs/$NAME.SMALL_FAILED
[[ "$DATA" = /* ]] || DATA=$WS/dpo_data/$DATA
if [ ! -f $WS/runs/$NAME.DONE ]; then
  rm -rf $WS/runs/$NAME
  CUDA_VISIBLE_DEVICES=$GPU PORT=$((29500 + GPU)) BATCH=${BATCH:-128} N_SNAP=1 N_FULL=0 ZERO=1 CONCAT=True \
    MAX_LEN=${MAX_LEN:-8192} bash $T/launch_dpo.sh $NAME $DATA $M ${LR:-8e-6} || fail "training (see runs/$NAME.log)"
fi
cd $WS/olmo-calib
mkdir -p $WS/results_ckpt/small
CUDA_VISIBLE_DEVICES=$GPU $PY eval_ckpt.py --ckpt $WS/runs/$NAME --tag small/$NAME --ref $M --out $WS/results_ckpt \
  --parts letters,format,chat --fast > $WS/results_ckpt/small/$NAME.log 2>&1 || fail "evals (see results_ckpt/small/$NAME/)"
grep -m1 -oE "Step: 1, LR: [^,]+, Loss: [0-9.]+" $WS/runs/$NAME.log > $WS/results_ckpt/small/$NAME/train_first_step.txt
grep -oE "Step: [0-9]+, LR: [^,]+, Loss: [0-9.]+" $WS/runs/$NAME.log > $WS/results_ckpt/small/$NAME/train_losses.txt
[ "${KEEP:-0}" = 1 ] || rm -rf $WS/runs/$NAME
touch $WS/runs/$NAME.SMALL_DONE
echo "$NAME SMALL DONE"
