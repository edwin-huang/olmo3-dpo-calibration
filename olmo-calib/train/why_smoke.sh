#!/bin/bash
# Smoke test for the pre-registered wave, then the wave. Waits for the full run's BBH wave to free the GPUs, then:
#   1. applies patch 6 (SimPO-with-reference, IPO) to open-instruct
#   2. 4-step training runs on smoke64 (GPU 0): plain DPO with --seed, SimPO beta 10, SimPO-with-reference, IPO;
#      checks every run exits 0 and the first-step losses equal their analytic values at init
#      (plain ln 2 = 0.693; SimPO-with-reference -log sigmoid(-10 x 0.3) = 3.049; IPO (0 - 0.5)^2 = 0.25)
#   3. scoring smoke on GPU 1: score_pool.py on 16 pool pairs
# If everything passes it starts train/why_wave.sh; otherwise it writes /workspace/WHY_SMOKE_FAILED and stops.
export WS=${WS:-/workspace}; T=$WS/olmo-calib/train; PY=$WS/open-instruct/.venv/bin/python; M=allenai/Olmo-3-7B-Instruct-SFT
LOG=$WS/why_smoke.log
until grep -q BBH_DONE $WS/bbh_wave.log 2>/dev/null; do sleep 60; done
while [ -f $WS/WHY_HOLD ]; do sleep 60; done  # touch /workspace/WHY_HOLD to keep the wave from starting
fail() { echo "SMOKE FAIL: $1" | tee -a $LOG; echo "$1" > $WS/WHY_SMOKE_FAILED; exit 1; }
rm -f $WS/WHY_SMOKE_FAILED $WS/WHY_SMOKE_OK
echo "$(date -u +%H:%M) smoke start" >> $LOG
(cd $WS/open-instruct && $PY $T/patch_open_instruct.py $WS/open-instruct) >> $LOG 2>&1 || fail "patch"
grep -q "DPO_IPO_TAU" $WS/open-instruct/open_instruct/dpo_tune_cache.py || fail "patch 6 not applied"
(CUDA_VISIBLE_DEVICES=1 SCORE_LIMIT=16 $PY $T/score_pool.py $WS/dpo_data $WS/runs/q25_all > $WS/why_smoke_score.log 2>&1; echo "score exit $?" >> $WS/why_smoke_score.log) &
run() { NAME=$1; EXPECT=$2; shift 2
  CUDA_VISIBLE_DEVICES=0 PORT=29521 BATCH=16 N_SNAP=1 N_FULL=0 ZERO=1 CONCAT=True MAX_LEN=4096 \
    bash $T/launch_dpo.sh $NAME $WS/dpo_data/smoke64.parquet $M 4e-6 "$@" >> $LOG 2>&1
  code=$?; rm -rf $WS/runs/$NAME
  first=$(grep -m1 -oE "Step: 1, LR: [^,]+, Loss: [0-9.e+-]+" $WS/runs/$NAME.log | sed 's/.*Loss: //')
  echo "$NAME exit $code first-step loss $first (expected $EXPECT)" | tee -a $LOG
  [ $code = 0 ] || fail "$NAME exit $code"
  [ -n "$first" ] || fail "$NAME no loss logged"
  if [ "$EXPECT" != any ]; then
    awk -v a="$first" -v b="$EXPECT" 'BEGIN{d=a-b; if (d<0) d=-d; exit !(d<0.02)}' || fail "$NAME first-step loss $first != $EXPECT"
  fi; }
run wys_plain 0.693 --seed 1
run wys_simpo any --loss_type simpo --beta 10 --gamma_beta_ratio 0.3 --seed 1
DPO_GAMMA=0.3 run wys_simporef 3.049 --beta 10 --seed 1
DPO_IPO_TAU=1.0 run wys_ipo 0.25 --seed 1
wait
grep -q "score exit 0" $WS/why_smoke_score.log || fail "score_pool smoke (see why_smoke_score.log)"
grep -E "^pool|^easy|^rest" $WS/why_smoke_score.log >> $LOG
echo "$(date -u +%H:%M) SMOKE OK" | tee -a $LOG; touch $WS/WHY_SMOKE_OK
exec bash $T/why_wave.sh
