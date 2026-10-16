#!/bin/bash
# Pre-registered wave (why_tests/PREDICTIONS.md). 2 workers (1 GPU each) pull from /workspace/small_queue.txt.
#   Phase 1 (12 runs): plain, agreement-weighted, SimPO, SimPO-with-reference x seeds 1-3, interleaved by seed.
#   GPU 1 first runs the difficulty scoring (score_pool.py, ~1.5 h) for the easy subset, then joins.
#   Gate: train/why_gate.py applies the pre-registered stop rule once phase 1 is done. REFUTED -> nothing more is
#   queued (/workspace/WHY_STOPPED); otherwise phase 2 (15 runs): agreement contrast, easy subset, IPO on the contrast.
#   When every queued run has finished, the workers are stopped (/workspace/small_STOP) and WHY_WAVE_DONE is logged.
export WS=${WS:-/workspace}; T=$WS/olmo-calib/train; PY=$WS/open-instruct/.venv/bin/python; Q=$WS/small_queue.txt
# Relaunched 2026-10-13 ~22 UTC after the BBH harness investigation: all 27 runs from scratch, Ai2-style BBH on the
# first 40 questions per task (bbh_ai2, BBH_PER_TASK=40), each model uploaded to ehuang5/olmo3-calibration-models wave/NAME.
C="KEEP=0 UPLOAD=1 LR=4e-6 PARTS=letters,format,chat,pairs,gen,bbh_ai2 BBH_PER_TASK=40"
LOG=$WS/why_wave.log
P1=(); P2=()
for s in 1 2 3; do
  P1+=("wy_plain_s$s s8_a.parquet $C EXTRA=--seed+$s")
  P1+=("wy_w5_s$s s8_a.parquet $C DPO_AGREE_K=5 DPO_AGREE_NORM=0.4106 EXTRA=--seed+$s")
  P1+=("wy_simpo_s$s s8_a.parquet $C EXTRA=--loss_type+simpo+--beta+10+--gamma_beta_ratio+0.3+--seed+$s")
  P1+=("wy_simporef_s$s s8_a.parquet $C DPO_GAMMA=0.3 EXTRA=--beta+10+--seed+$s")
done
for s in 1 2 3; do
  P2+=("wy_hi_s$s why_hi8.parquet $C EXTRA=--seed+$s")
  P2+=("wy_lo_s$s why_lo8.parquet $C EXTRA=--seed+$s")
done
for s in 1 2 3; do P2+=("wy_easy_s$s why_easy8.parquet $C EXTRA=--seed+$s"); done
for s in 1 2 3; do
  P2+=("wy_ipohi_s$s why_hi8.parquet $C DPO_IPO_TAU=1.0 EXTRA=--seed+$s")
  P2+=("wy_ipolo_s$s why_lo8.parquet $C DPO_IPO_TAU=1.0 EXTRA=--seed+$s")
done
names() { for l in "$@"; do echo "${l%% *}"; done; }
finished() { for n in $(names "$@"); do [ -f $WS/runs/$n.SMALL_DONE ] || [ -f $WS/runs/$n.SMALL_FAILED ] || return 1; done; }

echo "$(date -u +%H:%M) wave start" >> $LOG
rm -f $WS/small_STOP $WS/small_PAUSE $WS/WHY_STOPPED
printf "%s\n" "${P1[@]}" > $Q
nohup bash $T/small_worker.sh 0 > $WS/worker0.log 2>&1 &
if [ -f $WS/dpo_data/why_easy8.parquet ]; then  # scoring already done (first launch): both workers start now
  echo "$(date -u +%H:%M) scoring exit 0 (why_easy8 exists, skipped)" >> $LOG
  nohup bash $T/small_worker.sh 1 > $WS/worker1.log 2>&1 &
else ( $PY $T/score_pool.py $WS/dpo_data $WS/runs/q25_all > $WS/score_pool.log 2>&1; echo "$(date -u +%H:%M) scoring exit $?" >> $LOG
  nohup bash $T/small_worker.sh 1 > $WS/worker1.log 2>&1 & ) &
fi
until finished "${P1[@]}"; do sleep 120; done
echo "$(date -u +%H:%M) phase 1 done" >> $LOG
$PY $T/why_gate.py $WS/results_ckpt/small >> $LOG 2>&1
if [ $? = 3 ]; then
  echo "$(date -u +%H:%M) STOP RULE: refuted, phase 2 not queued" >> $LOG; touch $WS/WHY_STOPPED
  until finished "${P1[@]}"; do sleep 60; done; touch $WS/small_STOP; echo "WHY_WAVE_DONE" >> $LOG; exit 0
fi
until grep -q "scoring exit" $LOG; do sleep 60; done
if [ ! -f $WS/dpo_data/why_easy8.parquet ]; then
  echo "$(date -u +%H:%M) no why_easy8.parquet (scoring failed): easy-subset runs dropped" >> $LOG
  NP=(); for l in "${P2[@]}"; do [[ $l == wy_easy* ]] || NP+=("$l"); done; P2=("${NP[@]}")
fi
flock $Q.lock bash -c "printf '%s\n' \"\$@\" >> $Q" _ "${P2[@]}"
echo "$(date -u +%H:%M) phase 2 queued (${#P2[@]} runs)" >> $LOG
until finished "${P2[@]}"; do sleep 120; done
touch $WS/small_STOP
echo "$(date -u +%H:%M) WHY_WAVE_DONE" >> $LOG
