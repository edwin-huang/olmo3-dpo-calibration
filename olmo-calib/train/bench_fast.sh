#!/bin/bash
# Checks before running D3/D4 with more pairs per GPU per micro-batch (PDB):
#   1. memory: smoke200 (contains the 10 longest pairs, up to 16K tokens) at PDB=4 with separate and with combined
#      chosen/rejected forwards; reports s/step or OOM
#   2. equivalence: the q25_all run's first 20 steps at PDB=4 (separate forwards, so q25's reference cache is reused),
#      killed at step 20; per-step losses compared with q25_all's log (same seed -> same 128 pairs per step)
# Writes /workspace/bench_fast.txt and BENCHF_DONE.
WS=/workspace
T=$WS/olmo-calib/train
PY=$WS/open-instruct/.venv/bin/python
M=allenai/Olmo-3-7B-Instruct-SFT
rm -f $WS/BENCHF_DONE $WS/bench_fast.txt
steps() {  # log -> "N steps, s/step"
  $PY - "$1" <<'EOF'
import re, sys, datetime
ts = {}
for l in open(sys.argv[1]):
    m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - INFO - .*Step: (\d+),", l)
    if m: ts[int(m.group(2))] = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
s = sorted(ts)
oom = "out of memory" in open(sys.argv[1]).read().lower()
dt = (ts[s[-1]] - ts[s[2]]).total_seconds() / (s[-1] - s[2]) if len(s) > 4 else float("nan")
print(f"{len(s)} steps, {dt:.1f} s/step" + ("  OOM" if oom else ""))
EOF
}
for cfg in "4 False" "4 True"; do
  set -- $cfg
  name=m_pdb$1_c$2; rm -rf $WS/runs/$name
  BATCH=16 N_SNAP=1 N_FULL=0 PDB=$1 CONCAT=$2 bash $T/launch_dpo.sh $name $WS/dpo_data/smoke200.parquet $M 1e-6 > /dev/null 2>&1
  echo "memory test smoke200 PDB=$1 CONCAT=$2: exit $? | $(steps $WS/runs/$name.log)" | tee -a $WS/bench_fast.txt
  rm -rf $WS/runs/$name
done

# equivalence: q25_all settings except PDB=4; kill after step 20
name=eq_pdb4; rm -rf $WS/runs/$name
PDB=4 N_SNAP=1 N_FULL=0 bash $T/launch_dpo.sh $name $WS/dpo_data/q25_all.parquet $M 1e-6 > /dev/null 2>&1 &
LP=$!
until grep -q "Step: 20," $WS/runs/$name.log 2>/dev/null || ! kill -0 $LP 2>/dev/null; do sleep 5; done
pkill -9 -f -- "--exp_name $name --output_dir"; wait $LP 2>/dev/null; sleep 10
echo "equivalence run: $(steps $WS/runs/$name.log); cache: $(grep -c 'Loading reference logprobs cache' $WS/runs/$name.log) loads" | tee -a $WS/bench_fast.txt
$PY - <<'EOF' | tee -a /workspace/bench_fast.txt
import re
def losses(p):
    return {int(s): float(l) for s, l in re.findall(r"Step: (\d+), LR: [^,]+, Loss: ([0-9.]+)", open(p).read())}
a, b = losses("/workspace/runs/q25_all.log"), losses("/workspace/runs/eq_pdb4.log")
common = sorted(set(a) & set(b))[:20]
for s in common:
    print(f"  step {s:>2}: q25 (PDB=1) {a[s]:.4f}   PDB=4 {b[s]:.4f}   diff {b[s] - a[s]:+.4f}")
d = [abs(b[s] - a[s]) for s in common]
print(f"  mean |diff| {sum(d) / len(d):.4f}, max {max(d):.4f} over {len(d)} steps")
EOF
rm -rf $WS/runs/$name
touch $WS/BENCHF_DONE
