#!/bin/bash
# Task 1 add-on (2026-10-13): BigBenchHard on SFT, released DPO, our full agreement-weighted run, and the two 25%-scale
# models (plain vs weighted, uncontaminated). Starts after full_w5's own evals finish; writes results_ckpt/bbh/<name>/.
WS=/workspace; PY=$WS/open-instruct/.venv/bin/python; M=allenai/Olmo-3-7B-Instruct-SFT
until [ -f $WS/runs/full_w5.ALL_DONE ] || [ -f $WS/runs/full_w5.ALL_FAILED ]; do sleep 60; done
cd $WS/olmo-calib
ev() { GPU=$1; shift; for spec in "$@"; do name=${spec%%=*}; ck=${spec#*=}
  CUDA_VISIBLE_DEVICES=$GPU $PY eval_ckpt.py --ckpt $ck --tag bbh/$name --ref $M --out $WS/results_ckpt --parts bbh \
    > $WS/results_ckpt/bbh_$name.log 2>&1; echo "$(date -u +%H:%M) bbh $name exit $?"; done; }
mkdir -p $WS/results_ckpt/bbh
ev 0 sft=$M dpo_released=allenai/Olmo-3-7B-Instruct-DPO q25_plain=$WS/runs/q25_all &
[ -f $WS/runs/full_w5.ALL_DONE ] && ev 1 full_weighted=$WS/runs/full_w5 q25_weighted=$WS/runs/q25_w5 &
wait; echo BBH_DONE
