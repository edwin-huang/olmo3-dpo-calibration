#!/bin/bash
# Fix wave (2026-10-12): standard DPO variants as tests of the agreement account, all scored on calibration AND
# learning (held-out pairs, judge responses). First the no-training references (Instruct-SFT, released Instruct-DPO,
# our q25_all final model) on both GPUs, then 8 small runs through the small-test queue (2 workers).
#   nohup bash olmo-calib/train/fix_wave.sh > fix_wave.log 2>&1 &
export WS=${WS:-/workspace}
PY=$WS/open-instruct/.venv/bin/python
M=allenai/Olmo-3-7B-Instruct-SFT
P=letters,format,chat,pairs,gen,ifeval
cd $WS/olmo-calib
ev() { GPU=$1; CK=$2; TAG=$3; PARTS=$4
  CUDA_VISIBLE_DEVICES=$GPU $PY eval_ckpt.py --ckpt $CK --tag small/$TAG --ref $M --out $WS/results_ckpt --parts $PARTS --fast \
    > $WS/results_ckpt/small/$TAG.log 2>&1; echo "$(date -u +%H:%M) eval $TAG exit $?"; }
# SFT first on GPU 0: its pairs part caches the reference log-probs every later run reuses
ev 0 $M fx_sft pairs,gen,letters,format,chat,ifeval &
(sleep 600; ev 1 allenai/Olmo-3-7B-Instruct-DPO fx_rel_dpo pairs,gen; ev 1 $WS/runs/q25_all fx_q25 pairs,gen) &
wait
cat > $WS/small_queue.txt <<EOF
fx_base8 s8_a.parquet KEEP=1 PARTS=$P
fx_simpo s8_a.parquet KEEP=1 PARTS=$P EXTRA=--loss_type+simpo+--gamma_beta_ratio+0.3
fx_w5 s8_a.parquet KEEP=1 PARTS=$P DPO_AGREE_K=5 DPO_AGREE_NORM=0.4106
fx_nll1 s8_a.parquet KEEP=1 PARTS=$P DPO_NLL=1
fx_filt filt8.parquet KEEP=1 PARTS=$P
fx_ls1 s8_a.parquet KEEP=1 PARTS=$P EXTRA=--label_smoothing+0.1
fx_base4 s8_a.parquet KEEP=1 PARTS=$P LR=4e-6
fx_base2 s8_a.parquet KEEP=1 PARTS=$P LR=2e-6
EOF
rm -f $WS/small_STOP $WS/small_PAUSE
for g in 0 1; do nohup bash $WS/olmo-calib/train/small_worker.sh $g > $WS/worker$g.log 2>&1 & done
echo "$(date -u +%H:%M) workers started"
