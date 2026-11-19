#!/bin/bash
# Prints DONE x/y, FAILED list, running tasks, last epoch line of running tasks, GPU state, ALL_DONE when finished.
B=${B:-/workspace/fusion3d}
tot=0; done=0; failed=""
while IFS=$'\t' read -r name deps cmd; do
  [ -z "$name" ] && continue; tot=$((tot+1))
  [ -f $B/runs/$name/DONE ] && done=$((done+1))
  [ -f $B/runs/$name/FAILED ] && failed="$failed $name"
done < $B/tasks.tsv
echo "DONE $done/$tot"; echo "FAILED:${failed:- none}"
for d in $B/runs/*/.lock; do [ -d "$d" ] || continue; r=$(dirname $d); echo "RUNNING $(basename $r): $(grep -E '^\{"epoch' $r/stdout.log 2>/dev/null | tail -1 | cut -c1-160)"; done
n=$(pgrep -fc "pod_worker.sh") ; echo "WORKERS_ALIVE $n"
nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
[ $done -eq $tot ] && echo ALL_DONE
[ -n "$failed" ] && [ $((done + $(echo $failed | wc -w))) -eq $tot ] && echo FINISHED_WITH_FAILURES
