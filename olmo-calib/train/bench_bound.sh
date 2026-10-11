#!/bin/bash
# What limits DPO training speed? 10-step runs (batch 16, no saves) on length-matched sets of 160 real Instruct pairs
# (~500 / ~2K / ~8K tokens per pair), at the current settings (1 pair per GPU per micro-batch, separate chosen/rejected
# forwards), plus two fixes: several pairs per micro-batch (PDB=4) and one combined forward (CONCAT=True).
# If short pairs cost nearly as much per step as 2K pairs, the run is overhead-bound, not compute-bound.
# Writes /workspace/bench.txt and BENCH_DONE. Run: cd /workspace && nohup bash olmo-calib/train/bench_bound.sh > bench.log 2>&1 &
WS=/workspace
T=$WS/olmo-calib/train
PY=$WS/open-instruct/.venv/bin/python
M=allenai/Olmo-3-7B-Instruct-SFT
rm -f $WS/BENCH_DONE $WS/bench.txt
$PY - <<'EOF'
import pyarrow.parquet as pq, pyarrow as pa, random
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
rows = pq.read_table("/workspace/dpo_data/q25_all.parquet").slice(0, 20000).to_pylist()
n = lambda ms: len(tok(tok.apply_chat_template(ms, tokenize=False), add_special_tokens=False).input_ids)
for r in rows:
    r["_c"], r["_r"] = n(r["chosen"]), n(r["rejected"])
for name, lo, hi in (("short", 400, 700), ("mid", 1800, 2400), ("long", 7000, 9000)):
    pick = [r for r in rows if lo <= r["_c"] + r["_r"] <= hi and max(r["_c"], r["_r"]) <= 0.65 * (r["_c"] + r["_r"])][:160]
    print(name, len(pick), "pairs, mean tokens/pair", sum(r["_c"] + r["_r"] for r in pick) / len(pick))
    pq.write_table(pa.Table.from_pylist([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in pick]),
                   f"/workspace/dpo_data/bench_{name}.parquet")
EOF
run() {  # name data [env...]
  local name=$1 data=$2; shift 2
  rm -rf $WS/runs/$name
  env BATCH=16 N_SNAP=1 N_FULL=0 "$@" bash $T/launch_dpo.sh $name $WS/dpo_data/bench_$data.parquet $M 1e-6 > /dev/null 2>&1
  $PY - $WS/runs/$name.log "$name" "$data" "$*" <<'EOF' | tee -a /workspace/bench.txt
import re, sys, datetime
ts = {}
for l in open(sys.argv[1]):
    m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - INFO - .*Step: (\d+),", l)
    if m: ts[int(m.group(2))] = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
s = sorted(ts)
if len(s) < 6:
    print(f"{sys.argv[2]:<16} {sys.argv[3]:<6} {sys.argv[4]:<22} FAILED ({len(s)} steps logged)"); sys.exit()
dt = (ts[s[-1]] - ts[s[2]]).total_seconds() / (s[-1] - s[2])
print(f"{sys.argv[2]:<16} {sys.argv[3]:<6} {sys.argv[4]:<22} {dt:6.1f} s/step")
EOF
  rm -rf $WS/runs/$name
}
run b_short short
run b_mid mid
run b_long long
run b_short_pdb4 short PDB=4
run b_mid_concat mid CONCAT=True
run b_mid_pdb4_concat mid PDB=4 CONCAT=True
touch $WS/BENCH_DONE
