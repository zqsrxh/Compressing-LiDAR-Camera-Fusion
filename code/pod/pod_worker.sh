#!/bin/bash
# One worker per GPU. Task file: name<TAB>deps(comma-separated or -)<TAB>command. Runs in runs/<name>/code.
G=$1
B=${B:-/workspace/fusion3d}
export CUDA_VISIBLE_DEVICES=$G
while true; do
  pending=0; ran=0
  while IFS=$'\t' read -r name deps cmd; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue;; esac
    R=$B/runs/$name
    [ -f $R/DONE ] && continue
    [ -f $R/FAILED ] && continue
    pending=1
    ok=1
    if [ "$deps" != "-" ]; then
      for d in ${deps//,/ }; do [ -f $B/runs/$d/DONE ] || ok=0; [ -f $B/runs/$d/FAILED ] && { mkdir -p $R; touch $R/FAILED; echo "dep $d failed" > $R/FAILED; ok=0; }; done
    fi
    [ $ok = 1 ] || continue
    mkdir -p $R
    if mkdir $R/.lock 2>/dev/null; then
      echo $$ > $R/.lock/pid
    else
      p=$(cat $R/.lock/pid 2>/dev/null); if [ -n "$p" ] && kill -0 $p 2>/dev/null; then continue; fi
      rm -rf $R/.lock; mkdir $R/.lock 2>/dev/null || continue; echo $$ > $R/.lock/pid
    fi
    [ -d $R/code ] || cp -r $B/code $R/code
    cd $R/code
    echo "$(date -u +%FT%TZ) gpu$G start $name" >> $B/queue.log
    if bash -c "$cmd" < /dev/null >> $R/stdout.log 2>&1; then
      touch $R/DONE; echo "$(date -u +%FT%TZ) gpu$G done $name" >> $B/queue.log
    else
      n=$(cat $R/attempts 2>/dev/null || echo 0); n=$((n+1)); echo $n > $R/attempts
      echo "$(date -u +%FT%TZ) gpu$G fail $name attempt $n" >> $B/queue.log
      [ $n -ge 2 ] && touch $R/FAILED
    fi
    rm -rf $R/.lock
    ran=1
    break
  done < $B/tasks.tsv
  [ $pending = 0 ] && { echo "$(date -u +%FT%TZ) gpu$G exit: nothing pending" >> $B/queue.log; exit 0; }
  [ $ran = 0 ] && sleep 60
done
