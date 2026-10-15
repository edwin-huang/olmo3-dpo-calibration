#!/bin/bash
# The 25%-scale queue, in order: control replicate (same data and seed as q25_all, fast settings), D4 (answer-first /
# degenerate rejected responses removed), D3 delta-learning only, D3 GPT-judged only. Stops at the first failure.
# Relaunching skips finished runs. Run: cd /workspace && nohup bash olmo-calib/train/queue_d25.sh > queue.log 2>&1 &
export WS=${WS:-/workspace}
for spec in "q25_all_r2 q25_all" "q25_d4 q25_d4" "q25_delta q25_delta" "q25_judged q25_judged"; do
  set -- $spec
  [ -f $WS/runs/$1.ALL_DONE ] && { echo "skip $1 (done)"; continue; }
  echo "== $1 ($2) $(date -u +%H:%M) UTC"
  bash $WS/olmo-calib/train/run_d25.sh $1 $WS/dpo_data/$2.parquet || { echo "QUEUE STOPPED at $1"; exit 1; }
done
echo "QUEUE DONE $(date -u +%H:%M) UTC"
