#!/bin/bash
# Evaluate every snapshot + the final model of one training run, spread over all GPUs (one eval per GPU at a time).
#   bash eval_run.sh RUN REF [eval_ckpt flags, e.g. --fast]
#   RUN  name under /workspace/runs;  REF  model the text probe compares against (the run's starting SFT model)
#   PRUNE=1 deletes each snapshot after its eval succeeds (keeps disk to ~2 runs' worth on the 8x box)
# Snapshots are taken in order of step so the trajectory fills in from the start. Results: /workspace/results_ckpt/RUN/.
# Ends by writing /workspace/runs/RUN.EVAL_DONE or RUN.EVAL_FAILED (listing failed checkpoints) for the watcher.
set -uo pipefail
RUN=$1; REF=$2; shift 2
export WS=${WS:-/workspace}
PY=$WS/open-instruct/.venv/bin/python
cd $WS/olmo-calib
CKPTS=($(ls -d $WS/runs/$RUN/snap_step_* 2>/dev/null | sort -t_ -k3 -n))
[ -f $WS/runs/$RUN/config.json ] && CKPTS+=($WS/runs/$RUN)
(( ${#CKPTS[@]} )) || { echo "no checkpoints for $RUN"; echo none > $WS/runs/$RUN.EVAL_FAILED; exit 1; }
NG=$(nvidia-smi -L | wc -l)
mkdir -p $WS/results_ckpt/$RUN
rm -f $WS/runs/$RUN.EVAL_DONE $WS/runs/$RUN.EVAL_FAILED
echo "$RUN: ${#CKPTS[@]} checkpoints on $NG GPUs"
for g in $(seq 0 $((NG - 1))); do
  (
    for j in "${!CKPTS[@]}"; do
      (( j % NG == g )) || continue
      c=${CKPTS[$j]}
      tag=$RUN/$(basename $c); [ "$c" = "$WS/runs/$RUN" ] && tag=$RUN/final
      if CUDA_VISIBLE_DEVICES=$g $PY eval_ckpt.py --ckpt $c --tag $tag --ref $REF --out $WS/results_ckpt "$@" \
          > $WS/results_ckpt/$tag.log 2>&1; then
        # PRUNE=1: free the ~15 GB snapshot once it's fully evaluated (the final model is always kept)
        [ "${PRUNE:-0}" = 1 ] && [ "$c" != "$WS/runs/$RUN" ] && rm -rf $c
      else
        echo $tag >> $WS/runs/$RUN.eval_failures
      fi
    done
  ) &
done
wait
if [ -s $WS/runs/$RUN.eval_failures ]; then
  mv $WS/runs/$RUN.eval_failures $WS/runs/$RUN.EVAL_FAILED; exit 1
fi
touch $WS/runs/$RUN.EVAL_DONE
