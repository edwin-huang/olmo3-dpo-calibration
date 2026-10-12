#!/bin/bash
# Inference-only root-cause analyses (why.py), two GPUs in parallel, then unpause the small training wave.
# GPU 0: decomposition (SFT->released DPO, SFT->our q25, SFT->D4) and activation patching (SFT->DPO)
# GPU 1: squeezing signature and first-token sharpening (SFT->DPO, SFT->q25)
# Writes /workspace/results_why/*.json, /workspace/why_gpu{0,1}.log, and ANALYSES_DONE; removes small_PAUSE.
WS=/workspace
PY=$WS/open-instruct/.venv/bin/python
SFT=allenai/Olmo-3-7B-Instruct-SFT
cd $WS/olmo-calib
rm -f $WS/ANALYSES_DONE
(
  export CUDA_VISIBLE_DEVICES=0
  $PY why.py decomp --n 1000 --out $WS/results_why
  $PY why.py decomp --n 1000 --b $WS/runs/q25_all --out $WS/results_why
  $PY why.py decomp --n 1000 --b $WS/runs/q25_d4 --out $WS/results_why
  $PY why.py patch --n 400 --out $WS/results_why
) > $WS/why_gpu0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  $PY why.py squeeze --n 400 --out $WS/results_why
  $PY why.py firsttok --n 300 --out $WS/results_why
  $PY why.py squeeze --n 400 --b $WS/runs/q25_all --out $WS/results_why
  $PY why.py firsttok --n 300 --b $WS/runs/q25_all --out $WS/results_why
) > $WS/why_gpu1.log 2>&1 &
wait
touch $WS/ANALYSES_DONE
rm -f $WS/small_PAUSE
echo "ANALYSES DONE $(date -u +%H:%M)"
