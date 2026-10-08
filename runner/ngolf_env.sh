# neurogolf operator environment — source this in any shell that runs the harness.
#
# NEUROGOLF_SCORING_BIN: bin/ dir of the grader venv (python3 -> ORT 1.24.4).
# When unset, the current python3 on PATH is used — fine if your active venv
# already satisfies docs/requirements-grader-core.txt.
if [ -n "${NEUROGOLF_SCORING_BIN:-}" ]; then
  export PATH="$NEUROGOLF_SCORING_BIN:$PATH"
fi
export TMPDIR="${TMPDIR:-/tmp/ngolf_ort}"
mkdir -p "$TMPDIR"
