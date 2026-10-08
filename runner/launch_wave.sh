#!/bin/bash
# Canonical wave launch. Fail closed before detaching; reset the persistent fleet file so
# a prior drain cannot silently launch zero useful builders. The systemd scope avoids the
# observed terminal OOM-kill failure mode.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f runner/ngolf_env.sh ] && source runner/ngolf_env.sh
exec 8>results/LAUNCH.lock
if ! flock -n 8; then
  echo "another launch is already in preflight — refusing"
  exit 1
fi

UNIT="${WAVE_UNIT:-ngolf-wave-next}"
INITIAL_SEATS="${INITIAL_SEATS:-6}"
CODEX_SEATS="${CODEX_SEATS:-12}"

HAVE_USER_SYSTEMD=0
if systemctl --user show-environment >/dev/null 2>&1; then HAVE_USER_SYSTEMD=1; fi
if [ "$HAVE_USER_SYSTEMD" -eq 1 ] && systemctl --user is-active --quiet "$UNIT"; then
  echo "$UNIT already active — refusing a double launch"
  exit 1
fi
if [ -f results/orchestrator.pid ] && kill -0 "$(cat results/orchestrator.pid)" 2>/dev/null; then
  echo "an orchestrator is already running (pid $(cat results/orchestrator.pid)) — refusing a double launch"
  exit 1
fi

KAGGLE_AUTOSUBMIT="${KAGGLE_AUTOSUBMIT:-0}" \
RUNTIME_ROI=1 RUNTIME_ROI_COVERAGE_EVERY=10 \
ROUTE_GPT_MODEL=gpt-5.5 DECIDER_SOL_SHARE=.50 \
DECIDER_SOL_MODEL=gpt-5.6-sol ADMIN_PRIMARY_PROVIDER=codex \
ADMIN_SPARK_MODEL=gpt-5.3-codex-spark \
  python3 runner/preflight.py
pytest -q
# QUEUE.csv and every queued pack are one signed repin generation. Never rewrite the
# queue after preflight: doing so would invalidate pack freshness between check and launch.
# STOP_WAVE is deliberately retained while parked; only this explicit, post-preflight
# operator launch may remove it. No detached supervisor is alive at readiness handoff.
rm -f results/STOP_WAVE
python3 runner/fleet.py set "codex=${INITIAL_SEATS}"

WAVE_ENV=(
  "PATH=$PATH" "KAGGLE_AUTOSUBMIT=${KAGGLE_AUTOSUBMIT:-0}"
  "AUTOSUBMIT_DELTA=${AUTOSUBMIT_DELTA:-10}"
  "AUTOSUBMIT_FLUSH_HRS=${AUTOSUBMIT_FLUSH_HRS:-12}"
  "GPT55_SHARE=0" "ROUTE_GPT_MODEL=gpt-5.5"
  "CODEX_MAX_CONC=$CODEX_SEATS"
  "RUNTIME_ROI=1"
  "RUNTIME_ROI_COVERAGE_EVERY=10" "RUNTIME_ROI_SCORE_CACHE_SECS=60"
  # Shared with the Codex pool: this is a ceiling, not extra agents.
  "DECIDER_MAX_CONC=${DECIDER_MAX_CONC:-12}"
  "DECIDER_SOL_SHARE=${DECIDER_SOL_SHARE:-.50}"
  "DECIDER_SOL_MODEL=${DECIDER_SOL_MODEL:-gpt-5.6-sol}"
  "DECIDER_SOL_EFFORT=xhigh" "DECIDER_BASE_MODEL=gpt-5.5"
  "DECIDER_BASE_EFFORT=xhigh" "ADMIN_PRIMARY_PROVIDER=codex"
  "ADMIN_SPARK_MODEL=${ADMIN_SPARK_MODEL:-gpt-5.3-codex-spark}"
  "ADMIN_FALLBACK_MODEL=gpt-5.5"
  "PS_MAX_CONC=${PS_MAX_CONC:-2}"
  "WORKERS=$CODEX_SEATS"
)
[ -n "${NEUROGOLF_CLEAN:-}" ] && WAVE_ENV+=("NEUROGOLF_CLEAN=$NEUROGOLF_CLEAN")
[ -n "${NEUROGOLF_ARCHIVE:-}" ] && WAVE_ENV+=("NEUROGOLF_ARCHIVE=$NEUROGOLF_ARCHIVE")

if [ "$HAVE_USER_SYSTEMD" -eq 1 ]; then
  SYSTEMD_ENV=()
  for item in "${WAVE_ENV[@]}"; do SYSTEMD_ENV+=("--setenv=$item"); done
  systemd-run --user --unit="$UNIT" --collect \
    -p ManagedOOMPreference=omit -p MemoryHigh="${MEM_HIGH:-32G}" \
    --working-directory="$(pwd)" "${SYSTEMD_ENV[@]}" bash runner/run_wave.sh
  sleep 2
  systemctl --user show -p MainPID --value "$UNIT" > results/keepalive.pid
else
  setsid nohup env "${WAVE_ENV[@]}" bash runner/run_wave.sh \
    >> results/keepalive.launch.log 2>&1 < /dev/null 8>&- &
  echo $! > results/keepalive.pid
  sleep 2
  kill -0 "$(cat results/keepalive.pid)"
fi

RAMP_UNIT="${UNIT}-ramp"
RAMP_ARGS=(--initial "$INITIAL_SEATS" --target "$CODEX_SEATS"
  --seconds "${CANARY_SECONDS:-3600}"
  --min-attempts "${CANARY_MIN_ATTEMPTS:-3}"
  --heartbeat-max-age "${CANARY_HEARTBEAT_MAX_AGE:-900}")
if [ "$HAVE_USER_SYSTEMD" -eq 1 ]; then
  systemd-run --user --unit="$RAMP_UNIT" --collect --working-directory="$(pwd)" \
    --setenv=PATH="$PATH" python3 runner/canary_ramp.py --unit "$UNIT" "${RAMP_ARGS[@]}"
else
  setsid nohup python3 runner/canary_ramp.py --pid-file results/keepalive.pid \
    "${RAMP_ARGS[@]}" >> results/canary_ramp.log 2>&1 < /dev/null 8>&- &
  echo $! > results/canary_ramp.pid
fi
echo "wave launched: unit=$UNIT pid=$(cat results/keepalive.pid) codex=$INITIAL_SEATS/$CODEX_SEATS"
if [ "${KAGGLE_AUTOSUBMIT:-0}" = "1" ]; then
  echo "submission: AUTOMATIC — auto_submit.py will bank/submit/repin unattended; ramp guard=$RAMP_UNIT"
else
  echo "submission: manual banking (KAGGLE_AUTOSUBMIT=0); ramp guard=$RAMP_UNIT"
fi
echo "watch: results/STATUS.txt · results/ORCHESTRATOR.log · results/AUTOSUBMIT.log"
