#!/bin/bash
P=$HOME/.claude-science-scratch/fusion3d; RV=$P/runs_v5; PD=$P/pods
cfgs=""; for m in alloc0.5 uniL1_0.5 uniFPGM_0.5 globalL1_0.5; do for s in 0 1 2; do cfgs="$cfgs ft_${m}_s$s"; done; done
t0=$(date +%s)
while true; do
  left=0
  for c in $cfgs; do
    [ -f $RV/deploy_$c/deploy_eval_int8safe_val.json ] && continue
    M=$RV/$c/model_final.pt
    if [ -f $M ] && [ -f $RV/$c/eval_val.json ]; then
      a=$(stat -c %s $M); sleep 20; b=$(stat -c %s $M)
      if [ "$a" = "$b" ]; then echo "$(date -u +%H:%M) deploy $c"; bash $PD/deploy_final.sh $M $RV/deploy_$c > $RV/dep_$c.log 2>&1 || echo "FAIL $c"; fi
    fi
    [ -f $RV/deploy_$c/deploy_eval_int8safe_val.json ] || left=$((left+1))
  done
  [ $left = 0 ] && break
  [ $(( $(date +%s) - t0 )) -gt 12600 ] && { echo "timeout, left=$left"; break; }
  sleep 60
done
echo "$(date -u +%H:%M) bench"; bash $PD/bench_gb10.sh
