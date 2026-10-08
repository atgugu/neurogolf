#!/bin/bash
# keepalive supervisor: relaunches the orchestrator if it dies WITHOUT an operator stop.
# The 2026-07-08 07:46 silent death (no traceback, no OOM, heartbeat then nothing) showed
# the wave needs an outer life-support layer.
#
# START: setsid nohup bash runner/run_wave.sh > /dev/null 2>&1 & echo $! > results/keepalive.pid
# STOP : touch results/STOP_WAVE && pkill -INT -f "[r]unner/orchestrate.py"
#        (STOP_WAVE first, or the keepalive resurrects it; second pkill -INT = hard.)
cd "$(dirname "$0")/.."
mkdir -p results
exec 9>results/run_wave.lock
if ! flock -n 9; then
  echo "$(date -u +%Y%m%dT%H%M%S) another run_wave supervisor holds results/run_wave.lock" \
    >> results/keepalive.log
  exit 1
fi
rm -f results/STOP_WAVE

# ORT-profiling temp hygiene (2026-07-09: 27.9 GB of orphaned /tmp/tmp* traces took the
# root disk to 97% and broke tooling — a SIGKILLed scoring/gate subprocess never cleans its
# TemporaryDirectory). Route all wave temp into ONE managed dir and sweep it. 6h age floor:
# detached workers can survive an orchestrator restart, so never sweep anything younger
# than the longest attempt lifetime (2.5h + grace).
[ -n "${NEUROGOLF_SCORING_BIN:-}" ] && export PATH="$NEUROGOLF_SCORING_BIN:$PATH"  # grader venv (ORT 1.24.4) if provided
# Thread caps (2026-07-12): this box is 96 physical cores, so every numpy/onnx/BLAS
# OpenMP pool defaults to ~96 threads PER process. With ~12 seats + deciders each
# spawning scoring/pricing procs, total threads hit the container cgroup pids.max
# (11776) and `fork` fails (EAGAIN), killing the wave. These caps + the scoring-venv
# sitecustomize (ORT intra_op_num_threads) cut each scorer 159->~3 threads. Scores are
# thread-independent (params/shape-based) — verified byte-identical on 12 real ONNX.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TMPDIR=/tmp/ngolf_ort
mkdir -p "$TMPDIR"
find "$TMPDIR" -mindepth 1 -maxdepth 1 -mmin +360 -exec rm -rf {} + 2>/dev/null
find /tmp -maxdepth 1 -name 'tmp*' -user "$(whoami)" -mmin +360 -exec rm -rf {} + 2>/dev/null  # legacy orphans
( while [ ! -f results/STOP_WAVE ]; do
    sleep 900
    find "$TMPDIR" -mindepth 1 -maxdepth 1 -mmin +360 -exec rm -rf {} + 2>/dev/null
  done ) 9>&- &
JANITOR_PID=$!
cleanup() { kill "$JANITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

RESTARTS=0; WINDOW_START=$(date +%s)
while true; do
  if [ -f results/STOP_WAVE ]; then
    echo "$(date -u +%Y%m%dT%H%M%S) STOP_WAVE present — keepalive exiting" >> results/keepalive.log
    exit 0
  fi
  NOW=$(date +%s)
  if [ $((NOW - WINDOW_START)) -gt 3600 ]; then RESTARTS=0; WINDOW_START=$NOW; fi
  if [ "$RESTARTS" -ge 5 ]; then
    echo "$(date -u +%Y%m%dT%H%M%S) 5 restarts in 1h — crash loop, keepalive giving up" >> results/keepalive.log
    exit 1
  fi
  echo "$(date -u +%Y%m%dT%H%M%S) launching orchestrator (restart #$RESTARTS)" >> results/keepalive.log
  # A deliberately created marker releases inherited idle cooldowns for this one
  # orchestrator incarnation only.  Consume it before launch so any later keepalive
  # recovery restores the normal backoff policy.
  RELEASE_ALL_IDLE_ON_START=0
  if [ -f results/RELEASE_ALL_IDLE_COOLDOWNS ]; then
    RELEASE_ALL_IDLE_ON_START=1
    rm -f results/RELEASE_ALL_IDLE_COOLDOWNS
    echo "$(date -u +%Y%m%dT%H%M%S) one-shot idle cooldown release armed" >> results/keepalive.log
  fi
  RELEASE_ALL_IDLE_ON_START="$RELEASE_ALL_IDLE_ON_START" \
    python3 runner/orchestrate.py --workers "${WORKERS:-10}" ${LANES:+--lanes "$LANES"} \
    >> results/ORCHESTRATOR.log 2>&1
  RC=$?
  echo "$(date -u +%Y%m%dT%H%M%S) orchestrator exited rc=$RC" >> results/keepalive.log
  [ -f results/STOP_WAVE ] && exit 0
  # A clean exit normally ends the wave.  The one exception is an armed graceful
  # cooldown release: immediately loop once so the marker is consumed and the
  # replacement scheduler starts, after every in-flight job has drained.
  if [ "$RC" -eq 0 ] && [ ! -f results/RELEASE_ALL_IDLE_COOLDOWNS ]; then
    exit 0
  fi
  RESTARTS=$((RESTARTS+1))
  sleep 30
done
