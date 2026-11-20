#!/bin/zsh
# Kill a training or evaluation process that has stopped computing, so train_all.sh can retry it.
# Seen once: loading the weights deadlocked inside safetensors and the process sat at 0% CPU for good.
seconds() { ps -o cputime= -p $1 | awk -F: '{ s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; print int(s) }' }
last_pid=0; last_cpu=0
while pgrep -f "scripts/train_all.sh" > /dev/null; do
  sleep 300
  pid=$(pgrep -f "python.* -m gongoozler.tslm" | tail -1)
  [ -z "$pid" ] && continue
  cpu=$(seconds $pid)
  # A working run burns over a minute of CPU in five; the deadlocked one burned about a second.
  if [ "$pid" = "$last_pid" ] && [ $((cpu - last_cpu)) -lt 5 ]; then
    echo "$(date) killing stalled process $pid" >> checkpoints/watchdog.log
    kill -9 $pid
  fi
  last_pid=$pid; last_cpu=$cpu
done
