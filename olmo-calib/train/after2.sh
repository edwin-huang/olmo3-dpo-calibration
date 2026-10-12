#!/bin/bash
# After q25_delta's evals: normalization analyses (why.py norm) for SFT->released DPO, SFT->q25, SFT->D4, then start
# the small-test workers. Replaces after_analyses.sh's worker start. Writes /workspace/AFTER2_DONE.
WS=/workspace
PY=$WS/open-instruct/.venv/bin/python
until [ -f $WS/runs/q25_delta.ALL_DONE ] || [ -f $WS/runs/q25_delta.ALL_FAILED ]; do sleep 30; done
cd $WS/olmo-calib
CUDA_VISIBLE_DEVICES=0 $PY why.py norm --n 1000 --out $WS/results_why > $WS/why_norm0.log 2>&1 &
(CUDA_VISIBLE_DEVICES=1 $PY why.py norm --n 1000 --b $WS/runs/q25_all --out $WS/results_why
 CUDA_VISIBLE_DEVICES=1 $PY why.py norm --n 1000 --b $WS/runs/q25_d4 --out $WS/results_why) > $WS/why_norm1.log 2>&1 &
wait
rm -f $WS/small_STOP $WS/small_PAUSE
for g in 0 1; do
  nohup bash $WS/olmo-calib/train/small_worker.sh $g > $WS/worker$g.log 2>&1 < /dev/null &
done
touch $WS/AFTER2_DONE
