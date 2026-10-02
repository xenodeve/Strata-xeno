#!/bin/bash
# #56: A = the 0.1.26 line (src-126 + strata-live-s7.json), B = the 0.1.30 merge (src-dyn + strata-live-130.json).
# Order A B B A, one boot each. Each boot logs its cache sizing (upstream #199 / WDDM steps may change the slot count).
S=/c/Users/xenod/AppData/Local/Temp/claude/C--AI/37b6d362-d11e-4264-aed6-72fd7cd554f3/scratchpad/live
OUT=$S/ab-130m.jsonl
OUTS=$S/ab-130m-sampled.jsonl
cd $S
boot() {   # $1 = launcher
  cmd //c "D:\Github\Strata\stop-flash-next.cmd x" > /dev/null
  for i in $(seq 1 60); do powershell -NoProfile -Command "if (Get-Process strata* -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" && break; sleep 2; done
  n=$(ls server-s7-130m*.log 2>/dev/null | wc -l)
  [ -f server-s7.log ] && mv server-s7.log server-s7-130m$n.log && mv engine-s7.log engine-s7-130m$n.log
  rm -f sse-s7.jsonl server-s7.log.err
  powershell -NoProfile -File $1 > /dev/null
  for i in $(seq 1 120); do grep -qE "ready:" server-s7.log 2>/dev/null && break; grep -qE "Error|Traceback" server-s7.log.err 2>/dev/null && break; sleep 5; done
}
for arm in A B B A; do
  if [ $arm = A ]; then boot start-live-s7.ps1; else boot start-live-130.ps1; fi
  echo "=== boot $arm: $(grep -hE 'expert cache [0-9]+ slots|staged [0-9]+ next|draft head over' engine-s7.log | cut -c1-110 | tr '\n' '|') err: $(tail -c 300 server-s7.log.err 2>/dev/null)"
  nvidia-smi --query-gpu=index,pstate,clocks.sm,clocks.mem,clocks_throttle_reasons.active,memory.used --format=csv,noheader | tr '\n' ' '; echo
  python bench_arm.py $arm $OUT
  BENCH_SAMPLED=1 python bench_arm.py $arm $OUTS
done
echo DONE
