#!/bin/bash
# Run why.py neurons on GPU 1 next to the small training test there, only once that test is past step 3 (its memory
# peak) and >= 40 GB are free; the analysis is capped at 15% of the GPU (~27 GB) so it, not the trainer, fails if
# memory runs out. Writes /workspace/why_neurons.log and NEURONS_DONE.
WS=/workspace
PY=$WS/open-instruct/.venv/bin/python
until grep -q "Step: 3," $WS/runs/pilot8a.log 2>/dev/null; do sleep 30; done
sleep 60
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
echo "GPU 1 free: $free MiB"
if [ "$free" -lt 40000 ]; then echo "not enough memory; skipped"; touch $WS/NEURONS_SKIPPED; exit 0; fi
cd $WS/olmo-calib
CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True $PY why.py neurons --n 1000 --bs 4 --mem-frac 0.15 \
  --out $WS/results_why > $WS/why_neurons.log 2>&1
echo "exit $?"
touch $WS/NEURONS_DONE
