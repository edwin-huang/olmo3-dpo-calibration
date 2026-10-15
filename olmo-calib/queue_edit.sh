#!/usr/bin/env bash
# Runs after attribution.py: confidence-direction edit experiment, then IFEval capability check.
set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=1
WS=${WS:-/workspace}

until grep -qE "ALL DONE|Traceback" attr.log; do sleep 60; done
echo "attribution finished $(date +%T)"

python direction_edit.py --smoke --out smoke_edit > edit_smoke.log 2>&1
if ! grep -q "ALL DONE" edit_smoke.log; then echo "SMOKE FAILED"; exit 1; fi
rm -rf smoke_edit
echo "smoke ok $(date +%T)"

python direction_edit.py --out results_edit --save-dir $WS/models > edit.log 2>&1
if ! grep -q "ALL DONE" edit.log; then echo "EDIT FAILED"; exit 1; fi
echo "edit done $(date +%T)"

mkdir -p results_ifeval
for m in allenai/Olmo-3-7B-Instruct-SFT allenai/Olmo-3-7B-Instruct-DPO \
         $WS/models/Olmo-3-7B-Instruct-DPO-edit_arc_k1 \
         $WS/models/Olmo-3-7B-Instruct-DPO-edit_mmlu_k1; do
  name=$(basename "$m")
  lm_eval --model vllm \
    --model_args "pretrained=$m,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192" \
    --tasks ifeval --apply_chat_template --batch_size auto \
    --output_path "results_ifeval/$name" > "ifeval_$name.log" 2>&1 \
    && echo "ifeval ok $name $(date +%T)" || echo "IFEVAL FAILED $name"
done
echo "QUEUE DONE $(date +%T)"
