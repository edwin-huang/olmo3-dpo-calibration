#!/bin/bash
# After analyses.sh: q25_delta's evals (its training finished; run_d25.sh skips training when runs/NAME.DONE exists),
# then restart the small-test workers (one per GPU). Writes /workspace/AFTER_DONE.
WS=/workspace
until [ -f $WS/ANALYSES_DONE ]; do sleep 30; done
bash $WS/olmo-calib/train/run_d25.sh q25_delta $WS/dpo_data/q25_delta.parquet > $WS/q25_delta_evals.log 2>&1
rm -f $WS/small_STOP $WS/small_PAUSE
for g in 0 1; do
  nohup bash $WS/olmo-calib/train/small_worker.sh $g > $WS/worker$g.log 2>&1 < /dev/null &
done
touch $WS/AFTER_DONE
