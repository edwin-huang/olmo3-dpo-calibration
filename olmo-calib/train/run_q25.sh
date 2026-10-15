#!/bin/bash
# The 25% Instruct control run (q25_all: 65K random Instruct pairs, Ai2's recipe at LR 1e-6), the control for D3/D4
# and an early check that DPO's sharpening reproduces in our pipeline (T* rising from Instruct-SFT's 1.52).
# Starts only after test_resume.sh passed. Snapshots: --fast evals, deleted after (PRUNE); final model: full evals.
# Writes /workspace/Q25_DONE or Q25_FAILED. Run: cd /workspace && nohup bash olmo-calib/train/run_q25.sh > q25.log 2>&1 &
set -uo pipefail
export WS=${WS:-/workspace}
T=$WS/olmo-calib/train
PY=$WS/open-instruct/.venv/bin/python
M=allenai/Olmo-3-7B-Instruct-SFT
fail() { echo "$1" > $WS/Q25_FAILED; echo "Q25 FAILED: $1"; exit 1; }
rm -f $WS/Q25_DONE $WS/Q25_FAILED
[ -f $WS/RTEST_OK ] || fail "test_resume.sh has not passed"

# free ~45 GB: released checkpoints only needed for the baselines (re-downloadable)
rm -rf ~/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct-DPO \
       ~/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Think-SFT \
       ~/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Think-DPO
df -h $WS | tail -1

# Data order after a resume. test_resume.sh can't settle it: run-to-run loss noise (up to 0.02 in early steps, from
# GPU nondeterminism amplified by Adam's first updates) is larger than any order effect. So log the example indices
# of every micro-batch and require the resumed run to continue exactly the uninterrupted run's sequence.
export LOG_BATCH_INDEX=1   # kept on for q25_all too, to check data coverage after any real interruption
if [ ! -f $WS/runs/order_check.OK ]; then
  D=$WS/dpo_data/smoke200.parquet
  rm -rf $WS/runs/idx $WS/runs/idx_ref
  BATCH=16 N_SNAP=1 N_FULL=4 bash $T/launch_dpo.sh idx_ref $D $M 1e-6 || fail "order check: idx_ref"
  BATCH=16 N_SNAP=1 N_FULL=4 bash $T/launch_dpo.sh idx $D $M 1e-6 &
  LP=$!
  until [ -f $WS/runs/idx/step_6/COMPLETED ] || ! kill -0 $LP 2>/dev/null; do sleep 2; done
  [ -f $WS/runs/idx/step_6/COMPLETED ] || fail "order check: idx exited before step 6"
  sleep 3
  pkill -9 -f -- "--exp_name idx --output_dir"
  wait $LP
  sleep 15
  wc -l $WS/runs/idx/batch_index_rank0.txt > $WS/runs/idx_part1_lines.txt
  BATCH=16 N_SNAP=1 N_FULL=4 bash $T/launch_dpo.sh idx $D $M 1e-6 || fail "order check: resumed idx"
  $PY - <<'EOF' || fail "order check: resumed run did not continue the same data order"
import glob, os, sys
WS = os.environ["WS"]
part1 = int(open(WS + "/runs/idx_part1_lines.txt").read().split()[0])
ok = True
for f in sorted(glob.glob(WS + "/runs/idx_ref/batch_index_rank*.txt")):
    ref = open(f).read().splitlines()
    got = open(f.replace("/idx_ref/", "/idx/")).read().splitlines()
    skip = 6 * (16 // len(glob.glob(WS + "/runs/idx_ref/batch_index_rank*.txt")))  # 6 steps x (batch 16 / n GPUs) micro-batches per rank
    first, resumed = got[:part1], got[part1:]
    same_start = first[:skip] == ref[:skip]
    same_rest = resumed == ref[skip:]
    print(f"{f.split('/')[-1]}: ref {len(ref)} micro-batches; killed run logged {part1}, resumed {len(resumed)}; "
          f"first {skip} match {same_start}, resumed == ref[{skip}:] {same_rest}")
    ok &= same_start and same_rest
sys.exit(0 if ok else 1)
EOF
  rm -rf $WS/runs/idx $WS/runs/idx_ref
  touch $WS/runs/order_check.OK
fi

if [ ! -f $WS/runs/q25_all.DONE ]; then   # relaunching resumes from the last full state
  # N_FULL=10: full state every ~50 steps (~40 min). This box was interrupted twice in 3.5 h; with the default 4 the
  # second stop (at step 106, before the first save at 127) lost all progress.
  N_FULL=10 bash $T/launch_dpo.sh q25_all $WS/dpo_data/q25_all.parquet $M 1e-6 || fail "training (see runs/q25_all.log)"
fi
PRUNE=1 bash $T/eval_run.sh q25_all $M --fast || fail "snapshot evals (see runs/q25_all.EVAL_FAILED)"
cd $WS/olmo-calib
$PY eval_ckpt.py --ckpt $WS/runs/q25_all --tag q25_all/final_full --ref $M --out $WS/results_ckpt \
  > $WS/results_ckpt/q25_all/final_full.log 2>&1 || fail "full eval of the final model"
$PY compare_ckpts.py $WS/results_ckpt released q25_all
touch $WS/Q25_DONE
echo "Q25 DONE"
