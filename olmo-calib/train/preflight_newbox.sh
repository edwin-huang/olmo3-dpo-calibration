#!/bin/bash
# Preflight on a new box/GPU type before a long run (after setup_train_box.sh):
#   1. Instruct-SFT MMLU/ARC letters must reproduce the earlier results (ECE .115 +- .01, T* 1.52 +- .05, acc .608 +- .01)
#   2. 13-step smoke training with the auto-selected attention and with sdpa (--attn_implementation torch), s/step each
#   3. every eval part (--smoke) on the auto-attention smoke model, incl. vLLM self-consistency and IFEval
# Writes /workspace/PREFLIGHT_OK or PREFLIGHT_FAILED. Run: cd /workspace && nohup bash olmo-calib/train/preflight_newbox.sh > preflight.log 2>&1 &
set -uo pipefail
export WS=${WS:-/workspace}
T=$WS/olmo-calib/train
PY=$WS/open-instruct/.venv/bin/python
M=allenai/Olmo-3-7B-Instruct-SFT
fail() { echo "$1" > $WS/PREFLIGHT_FAILED; echo "PREFLIGHT FAILED: $1"; exit 1; }
rm -f $WS/PREFLIGHT_OK $WS/PREFLIGHT_FAILED
cd $WS/olmo-calib

echo "== 1. letters sanity"
CUDA_VISIBLE_DEVICES=0 $PY eval_ckpt.py --ckpt $M --tag newbox/Olmo-3-7B-Instruct-SFT --parts letters --out $WS/results_ckpt \
  > $WS/results_ckpt_sanity.log 2>&1 || fail "letters eval"
$PY - <<'EOF' || fail "letters do not reproduce earlier results"
import json, os, sys
r = json.load(open(os.environ["WS"] + "/results_ckpt/newbox/Olmo-3-7B-Instruct-SFT/letters.json"))["mmlu"]
print(f"Instruct-SFT MMLU acc {r['acc']:.3f} ECE {r['ece']:.3f} T* {r['t_star']:.2f} AUROC {r['auroc']:.3f}  (expected .608 / .115 / 1.52 / .768)")
sys.exit(0 if abs(r["acc"] - .608) < .01 and abs(r["ece"] - .115) < .01 and abs(r["t_star"] - 1.52) < .05 else 1)
EOF

echo "== 2. smoke training, auto attention vs sdpa"
D=$WS/dpo_data/smoke200.parquet
[ -f $D ] || $PY - <<'EOF'
import os, pyarrow.parquet as pq, pyarrow as pa
rows = pq.read_table(os.environ["WS"] + "/dpo_data/q25_all.parquet").to_pylist()
size = lambda r: sum(len(m["content"]) for m in r["chosen"] + r["rejected"])
pq.write_table(pa.Table.from_pylist(rows[:190] + sorted(rows, key=size)[-10:]), os.environ["WS"] + "/dpo_data/smoke200.parquet")
EOF
rm -rf $WS/runs/pf_auto $WS/runs/pf_sdpa
BATCH=16 N_SNAP=2 N_FULL=1 bash $T/launch_dpo.sh pf_auto $D $M 1e-6 || fail "smoke training (auto attention)"
BATCH=16 N_SNAP=1 N_FULL=0 bash $T/launch_dpo.sh pf_sdpa $D $M 1e-6 --attn_implementation torch || fail "smoke training (sdpa)"
grep -h "Auto-detected attention" $WS/runs/pf_auto.log | head -1
for r in pf_auto pf_sdpa; do
  $PY - $WS/runs/$r.log <<'EOF'
import re, sys, datetime
ts = {}
for l in open(sys.argv[1]):
    m = re.match(r"(\S+ \S+) - INFO - .*Step: (\d+),.*Loss: ([0-9.]+)", l)
    if m: ts[int(m.group(2))] = (datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), float(m.group(3)))
d = [(ts[s][0] - ts[s - 1][0]).total_seconds() for s in sorted(ts) if s - 1 in ts and s not in (7, 13)]
print(f"{sys.argv[1].split('/')[-1]}: mean {sum(d) / len(d):.1f} s/step over {len(d)} steps; losses " +
      " ".join(f"{ts[s][1]:.4f}" for s in sorted(ts)[:6]))
EOF
done
rm -rf $WS/runs/pf_sdpa

echo "== 3. all eval parts on the smoke model"
$PY eval_ckpt.py --ckpt $WS/runs/pf_auto --tag newbox/pf_auto --smoke --out $WS/results_ckpt > $WS/results_ckpt_pf.log 2>&1 \
  || fail "smoke evals (see results_ckpt/newbox/pf_auto/*.log)"
cat $WS/results_ckpt/newbox/pf_auto/summary.json | head -30
rm -rf $WS/runs/pf_auto
touch $WS/PREFLIGHT_OK
echo "PREFLIGHT OK"
