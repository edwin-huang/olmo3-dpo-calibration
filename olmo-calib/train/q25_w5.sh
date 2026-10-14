#!/bin/bash
# 25%-scale confirmation of agreement-weighted DPO (fix wave, 2026-10-12): q25_all's data and recipe (LR 1e-6, 508
# steps, 2xB200), each pair's loss weighted by sigmoid(-5 x SFT margin) / 0.4135 (mean weight over q25_all = 1).
# Control: q25_all (T* 1.85, held-out learned 27%). Waits for the mechanism probes (fix_why.sh) to free the GPUs.
WS=/workspace
until grep -q FIX_WHY_DONE $WS/fix_why.log 2>/dev/null; do sleep 30; done
DPO_AGREE_K=5 DPO_AGREE_NORM=0.4135 N_FULL=2 bash $WS/olmo-calib/train/run_d25.sh q25_w5 $WS/dpo_data/q25_all.parquet
