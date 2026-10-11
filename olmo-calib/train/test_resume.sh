#!/bin/bash
# Pre-flight checks before a long run, on 200 smoke pairs with the real settings (ZeRO-2, separate forwards):
#   1. rt_ref: an uninterrupted 13-step run, snapshots + full state every 3 steps
#   2. rt: the same run, SIGKILLed right after its step-6 full state is complete, then relaunched -> must log
#      "Resumed from checkpoint", and its learning rates, losses and final weights must match rt_ref
#   3. rt_lr: same data at another learning rate -> must load the existing reference-logprob cache
#   4. eval_run.sh on rt with PRUNE=1 (--smoke evals): every checkpoint evaluated, snapshots deleted, final kept
# Writes /workspace/RTEST_OK or RTEST_FAILED. Run: cd /workspace && nohup bash olmo-calib/train/test_resume.sh > rtest.log 2>&1 &
set -uo pipefail
WS=/workspace
T=$WS/olmo-calib/train
PY=$WS/open-instruct/.venv/bin/python
D=$WS/dpo_data/smoke200.parquet
M=allenai/Olmo-3-7B-Instruct-SFT
fail() { echo "$1" > $WS/RTEST_FAILED; echo "RTEST FAILED: $1"; exit 1; }
rm -f $WS/RTEST_OK $WS/RTEST_FAILED
rm -rf $WS/runs/rt $WS/runs/rt_ref $WS/runs/rt_lr $WS/runs/rt*.DONE $WS/runs/rt*.FAILED $WS/results_ckpt/rt
export BATCH=16 N_SNAP=4 N_FULL=4   # 13 steps: snapshots and full state at steps 3, 6, 9, 12

echo "== 1. uninterrupted reference run"
bash $T/launch_dpo.sh rt_ref $D $M 1e-6 || fail "rt_ref training"
ls -d $WS/runs/rt_ref/snap_step_* | wc -l

echo "== 2. killed after the step-6 full state, then resumed"
bash $T/launch_dpo.sh rt $D $M 1e-6 &
LP=$!
until [ -f $WS/runs/rt/step_6/COMPLETED ] || ! kill -0 $LP 2>/dev/null; do sleep 2; done
[ -f $WS/runs/rt/step_6/COMPLETED ] || fail "rt exited before its step-6 full state was complete"
sleep 3
pkill -9 -f -- "--exp_name rt --output_dir"
wait $LP
sleep 15
echo "killed; GPU memory now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
grep -oP "Step: \d+" $WS/runs/rt.log | tail -1
cp $WS/runs/rt.log $WS/runs/rt_part1.log
bash $T/launch_dpo.sh rt $D $M 1e-6 || fail "resumed rt training"
grep -m1 "Resumed from checkpoint" $WS/runs/rt.log || fail "rt did not resume from a checkpoint"

$PY - <<'EOF' || fail "resumed run differs from the uninterrupted run"
import re, glob, sys
from safetensors import safe_open
def steps(path):
    return {int(s): (float(lr), float(loss)) for s, lr, loss in
            re.findall(r"Step: (\d+), LR: ([^,]+), Loss: ([0-9.]+)", open(path).read())}
ref, part1, res = steps("/workspace/runs/rt_ref.log"), steps("/workspace/runs/rt_part1.log"), steps("/workspace/runs/rt.log")
print("resumed log covers steps", min(res), "-", max(res), "| part 1 covered", min(part1), "-", max(part1))
bad = False
for s in sorted(res):
    (lr0, l0), (lr1, l1) = ref[s], res[s]
    flag = abs(lr0 - lr1) > 1e-12 or abs(l0 - l1) > 0.01
    bad |= flag
    print(f"  step {s:>2}: LR {lr0:.3e} vs {lr1:.3e}   loss {l0:.4f} vs {l1:.4f}" + ("   <-- MISMATCH" if flag else ""))
maxd, n, ndiff = 0.0, 0, 0
for fa in sorted(glob.glob("/workspace/runs/rt_ref/*.safetensors")):
    fb = fa.replace("/rt_ref/", "/rt/")
    with safe_open(fa, "pt") as a, safe_open(fb, "pt") as b:
        for k in a.keys():
            x, y = a.get_tensor(k).float(), b.get_tensor(k).float()
            maxd = max(maxd, (x - y).abs().max().item()); n += x.numel(); ndiff += (x != y).sum().item()
print(f"final weights: max |diff| {maxd:.2e}, {ndiff / n:.4%} of {n / 1e9:.2f}B params differ")
sys.exit(1 if bad or maxd > 1e-2 else 0)
EOF

echo "== 3. reference-cache reuse at a different learning rate"
N_SNAP=1 N_FULL=0 bash $T/launch_dpo.sh rt_lr $D $M 3e-6 || fail "rt_lr training"
grep -m1 "Loading reference logprobs cache" $WS/runs/rt_lr.log || fail "rt_lr recomputed the reference cache"
rm -rf $WS/runs/rt_lr $WS/runs/rt_ref

echo "== 4. eval_run.sh with PRUNE=1"
PRUNE=1 bash $T/eval_run.sh rt $M --smoke || fail "eval_run (see runs/rt.EVAL_FAILED)"
[ -f $WS/runs/rt.EVAL_DONE ] || fail "no EVAL_DONE marker"
left=$(ls -d $WS/runs/rt/snap_step_* 2>/dev/null | wc -l)
[ "$left" -eq 0 ] || fail "PRUNE left $left snapshots"
[ -f $WS/runs/rt/config.json ] || fail "PRUNE deleted the final model"
ls $WS/results_ckpt/rt
for s in $WS/results_ckpt/rt/*/summary.json; do
  $PY -c "import json,sys; s=json.load(open('$s')); print(s['tag'], 'failed:', s['failed'], 'MMLU T*', s['letters_mmlu_t_star']); sys.exit(1 if s['failed'] else 0)" || fail "eval parts failed in $s"
done
rm -rf $WS/runs/rt
touch $WS/RTEST_OK
echo "RTEST OK"
