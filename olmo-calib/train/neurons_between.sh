#!/bin/bash
# Run why.py neurons on GPU 1 between small tests: small_PAUSE is set (workers finish their current test, then wait),
# this waits for pilot8a (GPU 1) to finish, runs the analysis on the free GPU, then removes small_PAUSE.
# Writes /workspace/why_neurons.log and NEURONS_DONE.
export WS=${WS:-/workspace}
PY=$WS/open-instruct/.venv/bin/python
touch $WS/small_PAUSE
until [ -f $WS/runs/pilot8a.SMALL_DONE ] || [ -f $WS/runs/pilot8a.SMALL_FAILED ]; do sleep 30; done
sleep 20
cd $WS/olmo-calib
CUDA_VISIBLE_DEVICES=1 $PY why.py neurons --n 1000 --bs 8 --out $WS/results_why > $WS/why_neurons.log 2>&1
echo "neurons exit $?"
rm -f $WS/small_PAUSE
touch $WS/NEURONS_DONE
