#!/bin/bash
# ZeRO-1 + combined chosen/rejected forward: speed on ~2K-token pairs and memory on the longest pairs (smoke200).
# Needs configs/ds_configs/stage1_no_offloading_accelerate.conf in open-instruct (setup_train_box.sh creates it).
WS=/workspace; T=$WS/olmo-calib/train; M=allenai/Olmo-3-7B-Instruct-SFT
rm -f $WS/BENCHZ1C_DONE
for d in mid smoke200; do
  name=z1c_$d; rm -rf $WS/runs/$name
  if [ $d = smoke200 ]; then data=$WS/dpo_data/smoke200.parquet; else data=$WS/dpo_data/bench_$d.parquet; fi
  BATCH=16 N_SNAP=1 N_FULL=0 ZERO=1 CONCAT=True bash $T/launch_dpo.sh $name $data $M 1e-6 > /dev/null 2>&1
  echo "ZeRO-1 + concat, $d: exit $?, $(grep -c 'Step: ' $WS/runs/$name.log) steps, OOM lines: $(grep -ci 'out of memory' $WS/runs/$name.log)"
  grep -oE "^[0-9-]+ [0-9:]+ - INFO - .*Step: [0-9]+," $WS/runs/$name.log | sed "s/ - INFO.*Step: / step /" | tr "\n" " "; echo
  rm -rf $WS/runs/$name
done
touch $WS/BENCHZ1C_DONE
