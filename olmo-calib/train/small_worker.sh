#!/bin/bash
# Worker for the small-test queue: pulls lines from /workspace/small_queue.txt ("NAME DATA [KEY=VALUE ...]") under a
# lock and runs each with run_small.sh on its GPU. Waits while the queue is empty; exits when /workspace/small_STOP
# exists. Start one per GPU:  nohup bash olmo-calib/train/small_worker.sh 0 > worker0.log 2>&1 &
GPU=$1
WS=/workspace
Q=$WS/small_queue.txt
touch $Q
while [ ! -f $WS/small_STOP ]; do
  if [ -f $WS/small_PAUSE ]; then sleep 60; continue; fi   # e.g. while inference analyses use the GPUs
  line=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q")
  if [ -z "$line" ]; then sleep 60; continue; fi
  echo "$(date -u +%H:%M) GPU $GPU start: $line"
  set -- $line
  name=$1; data=$2; shift 2
  bash $WS/olmo-calib/train/run_small.sh $name $data $GPU "$@" >> $WS/runs/$name.small.log 2>&1
  echo "$(date -u +%H:%M) GPU $GPU end: $name $([ -f $WS/runs/$name.SMALL_DONE ] && echo DONE || echo FAILED)"
done
