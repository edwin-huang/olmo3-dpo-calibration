#!/bin/bash
# One DPO run with Ai2's Olmo 3 7B recipe (scripts/train/olmo3/7b_instruct_dpo.sh in open-instruct), on one node.
# That script's --gradient_checkpointing flag no longer exists; in dpo_tune_cache.py any --activation_memory_budget < 1
# now calls model.gradient_checkpointing_enable(), i.e. the same full checkpointing.
# --concatenated_forward False: chosen and rejected get separate forward passes (same loss). The default concatenated
# pass builds one fp32 logits tensor of 2 x 16K tokens x 100K vocab = 12.2 GiB, which ran out of memory in the smoke test.
#   bash launch_dpo.sh NAME DATA MODEL LR [extra open-instruct flags...]
#   NAME   run name -> /workspace/runs/NAME (model snapshots snap_step_*, final model, log)
#   DATA   a parquet from make_dpo_data.py, e.g. /workspace/dpo_data/instruct_full.parquet
#   MODEL  allenai/Olmo-3-7B-Instruct-SFT or allenai/Olmo-3-7B-Think-SFT
#   LR     1e-6 (Instruct recipe) or 8e-8 (Think recipe)
# Env overrides: BATCH (effective batch, default 128 as in Ai2's runs), CHAT_TEMPLATE (default tokenizer_default;
# Ai2's Think DPO used olmo_thinker), MAX_LEN (16384), N_SNAP (model-only snapshots per run, default 10, ~15 GB each),
# N_FULL (full-state saves per run, default 4; each ~100 GB and briefly two exist while the older one is replaced).
# Ends by writing /workspace/runs/NAME.DONE or NAME.FAILED (with the exit code) for watchers.
set -uo pipefail
NAME=$1; DATA=$(realpath "$2"); MODEL=$3; LR=$4; shift 4   # absolute: we cd into open-instruct below
[ -f "$DATA" ] || { echo "no data file $DATA"; exit 1; }
WS=/workspace
OI=$WS/open-instruct
BATCH=${BATCH:-128}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-tokenizer_default}
MAX_LEN=${MAX_LEN:-16384}
N_SNAP=${N_SNAP:-10}
# DeepSpeed ZeRO stage. Ai2 used 3, which also shards the weights and re-gathers them every micro-batch. 2 keeps a full
# bf16 copy per GPU and shards only gradients + optimizer state: same optimizer math, 1.45x faster in the smoke test
# (7.7 vs 11.2 s/step, 4xH100, same batches), and it fit at 16K tokens on 4 GPUs (8 GPUs hold less state per GPU).
ZERO=${ZERO:-2}
CONCAT=${CONCAT:-False}  # True = one forward for chosen+rejected; its 12.2 GiB logits tensor OOMs on 4xH100 with ZeRO-2 or 3
PDB=${PDB:-1}   # pairs per GPU per micro-batch (Ai2: 1)
# GPU count: the GPUs listed in CUDA_VISIBLE_DEVICES if set (lets two runs share a box), else all of them.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .); else NGPU=$(nvidia-smi -L | wc -l); fi
PORT=${PORT:-29500}   # accelerate's rendezvous port; give concurrent runs different ports
(( BATCH % (NGPU * PDB) == 0 )) || { echo "BATCH $BATCH not divisible by $NGPU GPUs x $PDB"; exit 1; }
ACCUM=$(( BATCH / (NGPU * PDB) ))

N_PAIRS=$($OI/.venv/bin/python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$DATA').metadata.num_rows)")
for a in "$@"; do  # honour --max_train_samples if given
  if [ "${prev:-}" = "--max_train_samples" ]; then N_PAIRS=$(( a < N_PAIRS ? a : N_PAIRS )); fi; prev=$a
done
STEPS=$(( (N_PAIRS + BATCH - 1) / BATCH ))
SNAP=$(( STEPS / N_SNAP > 0 ? STEPS / N_SNAP : 1 ))
N_FULL=${N_FULL:-4}   # full accelerate state (~100 GB) for crash resume, N_FULL times per run (0 = never); only the last is kept
if (( N_FULL == 0 )); then FULL=1000000000; else FULL=$(( STEPS / N_FULL > 0 ? STEPS / N_FULL : 1 )); fi

mkdir -p $WS/runs $WS/ref_cache
rm -f $WS/runs/$NAME.DONE $WS/runs/$NAME.FAILED
echo "$NAME: $N_PAIRS pairs, $NGPU GPUs x $PDB x accum $ACCUM = batch $BATCH, $STEPS steps, LR $LR, snapshot every $SNAP, full state every $FULL, ZeRO-$ZERO, concat $CONCAT" | tee $WS/runs/$NAME.log

cd $OI
export SNAPSHOT_EVERY=$SNAP
export REFERENCE_LOGPROBS_CACHE_PATH=$WS/ref_cache   # keyed on model+data, not LR, so D1 reuses D0's cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True
.venv/bin/accelerate launch \
    --mixed_precision bf16 \
    --num_processes $NGPU \
    --main_process_port $PORT \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage${ZERO}_no_offloading_accelerate.conf \
    open_instruct/dpo_tune_cache.py \
    --exp_name $NAME \
    --output_dir $WS/runs/$NAME \
    --do_not_randomize_output_dir \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_slow_tokenizer False \
    --mixer_list $DATA 1.0 \
    --max_seq_length $MAX_LEN \
    --per_device_train_batch_size $PDB \
    --gradient_accumulation_steps $ACCUM \
    --zero_hpz_partition_size 1 \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --beta 5 \
    --loss_type dpo_norm \
    --logging_steps 1 \
    --activation_memory_budget 0.5 \
    --concatenated_forward $CONCAT \
    --chat_template_name $CHAT_TEMPLATE \
    --checkpointing_steps $FULL \
    --keep_last_n_checkpoints 1 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --try_auto_save_to_beaker false \
    "$@" >> $WS/runs/$NAME.log 2>&1
CODE=$?
if [ $CODE -eq 0 ] && [ -f $WS/runs/$NAME/config.json ]; then
  # the full-state dirs are only for resuming; drop them once the final model is saved
  rm -rf $WS/runs/$NAME/step_*
  echo ok > $WS/runs/$NAME.DONE
else
  echo $CODE > $WS/runs/$NAME.FAILED
fi
exit $CODE
