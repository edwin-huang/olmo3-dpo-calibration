#!/bin/bash
# Full-scale agreement-weighted DPO (2026-10-12): all 259,922 Instruct DPO pairs, Ai2's hyperparameters (LR 1e-6,
# beta 5, 16K tokens, ~2,031 steps) on 2xB200, each pair's loss x sigmoid(-5 x SFT margin) / 0.4135 (mean weight over
# the 25% sample; the full-data mean is within ~1% of it). Control: the released Olmo-3-7B-Instruct-DPO.
# 5 snapshots (~14 GB each, pruned after their evals) and 2 full states (~96 GB; two exist briefly) to fit the disk.
# Waits for the transplant probe on GPU 0 to finish first.
WS=/workspace
while pgrep -f "[w]hy.py transplant" > /dev/null; do sleep 30; done
DPO_AGREE_K=5 DPO_AGREE_NORM=0.4135 N_SNAP=5 N_FULL=2 bash $WS/olmo-calib/train/run_d25.sh full_w5 $WS/dpo_data/instruct_full.parquet
