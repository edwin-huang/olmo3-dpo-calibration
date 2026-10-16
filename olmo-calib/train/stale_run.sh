#!/bin/bash
# Experiment A: download full_w5 snapshots from HF, score 4,000 training pairs under each model (2 GPUs), analyze.
set -uo pipefail
export WS=${WS:-/workspace}; PY=$WS/open-instruct/.venv/bin/python; S=$WS/olmo-calib/train/stale_weights.py
SNAP=$WS/stale_snaps; mkdir -p $SNAP $WS/results_why/stale
$PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
for s in (406, 812, 1218, 1624):
    snapshot_download("ehuang5/olmo3-calibration-models", allow_patterns=[f"full_weighted/snap_step_{s}/*"],
                      local_dir=os.environ["WS"] + "/stale_snaps")
    print("downloaded", s, flush=True)
EOF
score() { CUDA_VISIBLE_DEVICES=$1 $PY $S score $2 $3; }
( score 0 sft allenai/Olmo-3-7B-Instruct-SFT; score 0 q25_weighted $WS/runs/q25_w5; score 0 full_s406 $SNAP/full_weighted/snap_step_406
  score 0 full_s1218 $SNAP/full_weighted/snap_step_1218; score 0 q25_plain $WS/runs/q25_all ) > $WS/results_why/stale/gpu0.log 2>&1 &
( score 1 full_s812 $SNAP/full_weighted/snap_step_812; score 1 full_s1624 $SNAP/full_weighted/snap_step_1624
  score 1 full_s2030 $WS/runs/full_w5; score 1 dpo_released allenai/Olmo-3-7B-Instruct-DPO ) > $WS/results_why/stale/gpu1.log 2>&1 &
wait
$PY $S analyze sft q25_weighted full_s406 full_s812 full_s1218 full_s1624 full_s2030 q25_plain dpo_released \
  > $WS/results_why/stale/analysis.txt 2>&1
echo STALE_DONE
