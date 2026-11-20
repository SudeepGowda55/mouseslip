#!/bin/zsh
# Train and evaluate both models, the headline one first: about four hours each on an M1.
# Safe to run again: a finished model is skipped and an interrupted one resumes from its last checkpoint.
cd "$(dirname "$0")/.."
mkdir -p checkpoints
for mode in clocks+engine clocks; do
  [ -f "results/tslm-$mode.json" ] && continue
  for attempt in 1 2 3 4 5; do
    resume=""; [ -f "checkpoints/last-$mode.pt" ] && resume="--resume"
    uv run python -m gongoozler.tslm train --mode $mode --lora --every 1000 --log-every 50 $resume \
      >> "checkpoints/stdout-$mode.log" 2>&1 && break
    sleep 30
  done
  uv run python -m gongoozler.tslm eval --mode $mode >> "checkpoints/stdout-$mode.log" 2>&1
done
