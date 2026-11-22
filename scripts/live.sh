#!/bin/zsh
# Keeps the live watcher running. A GPU error can end the process; the scoreboard carries across restarts.
cd "$(dirname "$0")/.."
while true; do
  .venv/bin/python -m mouseslip.live "$@"
  echo "$(date) watcher exited with $?, restarting in 10 s" >&2
  sleep 10
done
