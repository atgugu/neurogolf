# Setup & running

Be honest about what a fresh clone can do. The repository is the **orchestration half** of a
two-part system; the other half (the operator's `neurogolf_clean` data checkout, a logged-in
`codex` CLI, and an open Kaggle competition) is not public. The table below is the map.

## What runs where

| Capability | Command | Needs |
|---|---|---|
| **Test suite** | `pytest -q` | clone only (`pip install -r requirements.txt`) |
| **Price any ONNX** (the compression metric) | `python3 runner/price.py <model.onnx>` | clone only — vendored Apache-2.0 grader |
| **Dry-run the orchestrator** | `python3 runner/orchestrate.py --dry-run --tasks 32` | clone only (prints prompts + commands, spawns nothing) |
| **Build a full prompt pack** | `python3 runner/make_pack.py 32` | `NEUROGOLF_CLEAN` (without it, you get a *thin* pack — placeholders where the pin anatomy / generator source would be) |
| **Price + gate a task vs its pin** | `python3 runner/fast_verify.py 32 <model.onnx>` | `NEUROGOLF_CLEAN` (ground-truth grids + gates) |
| **Live overnight wave** | `bash runner/launch_wave.sh` | the operator's `codex` CLI (non-public model ids) + `NEUROGOLF_CLEAN` + a pin-fresh base |
| **Submit to Kaggle** | (inside the wave / `auto_submit.py`) | Kaggle credentials **and an open competition** — neurogolf-2026 closed 2026-07-08 |

So: the **core mechanic — compress a graph and score it with the real grader — reproduces
from a clean clone**. The live wave and the Kaggle submission need the original operator's
environment and, for submission, a competition that has since closed.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy + onnx + onnxruntime
```

`onnxruntime==1.24.4` reproduces the Kaggle score **exactly** — memory is read from that
runtime's profiler, and several op×dtype legality facts (`data/ORT124_LEGALITY.md`) are
version-specific. Pricing runs on any recent ORT (`params` is exact; `memory` approximate).
`docs/requirements-grader-core.txt` holds the fuller grader-venv pin set for reference.

## 2. Price a graph with the real grader (no external data)

```bash
python3 runner/price.py data/exemplars/task365.onnx
```

This runs the competition's own scoring framework (`runner/grader/`, Apache-2.0 — see
`runner/grader/NOTICE.md`) over a synthetic demo grid and prints
`params / memory / cost / points = 25 − ln(cost)`. It is the same code and the same number
the Kaggle grader produces.

## 3. External data roots (`runner/config.py`)

Everything beyond pricing is reached through `runner/config.py`, overridable by env var:

| Env var | Default | Unlocks |
|---------|---------|---------|
| `NEUROGOLF_CLEAN` | `~/Projects/ARC/neurogolf_clean` | full **gating** (`candidates/gate124.py`, `gate_pair.py`), ARC ground-truth grids (`extracted/task*.json`), generator sources (`external/ARC-GEN`), the live pin (`submission.zip`), and the Kaggle submitter (`candidates/probe_submit.py`) — plus the pin anatomy / generator source that a full `make_pack` embeds |
| `NEUROGOLF_ARCHIVE` | `/media/…/neurogolf` | optional pack enrichment (reasoning traces, gate-pass catalog, dossiers) — packs degrade gracefully without it |
| `NEUROGOLF_RESOURCES` | `~/Projects/ARC/neurogolf_resources` | optional audit pricing (`score_bundle.py` replica) |
| `NEUROGOLF_VENV_PY` | *(current interpreter)* | which python preflight treats as the grader venv |

The orchestrator exports `NEUROGOLF_CLEAN` into every worker's environment, so prompts can
reference `$NEUROGOLF_CLEAN/…` and workers resolve it in their shells.

## 4. Running a wave (operator environment)

Builders, deciders, the analyst, and the watchdog all run through a logged-in `codex` CLI
whose model ids (`gpt-5.5`, `gpt-5.6-sol`, `gpt-5.3-codex-spark`) are not public, so a live
wave is not reproducible outside the original operator's account. `runner/preflight.py`
live-tests every dependency and fails closed.

```bash
bash runner/launch_wave.sh    # preflight → pytest → canary-ramped systemd/nohup launch
python3 runner/webui.py       # read-only live console on http://127.0.0.1:18765
python3 runner/fleet.py set codex=12               # resize a LIVE wave, ~10 s pickup
touch results/STOP_WAVE && pkill -INT -f "[r]unner/orchestrate.py"   # graceful stop
```

Notes:
- Host-resource minimums are tuned for the production box (64 GiB RAM, 40 GiB free disk);
  override with `RESOURCE_MIN_MEM_GB` / `RESOURCE_MIN_DISK_GB` on smaller machines.
- If the live pin (`$NEUROGOLF_CLEAN/submission.zip`) moved since the recorded generation,
  preflight reports pin-freshness failures by design — regenerate with `runner/repin.py`.

## 5. Banking / submission

Manual by default (`KAGGLE_AUTOSUBMIT=0`):

```bash
python3 runner/bank.py --dry-run   # review the plan
python3 runner/bank.py --regate    # reprice + re-gate + razor-scan + chunk into zips
```

Launch with `KAGGLE_AUTOSUBMIT=1` for the unattended
bank → submit → recon → repin → **reprioritize → continue** loop (the running orchestrator
ingests the new pin and re-orders the queue live, no restart). Submission itself needs Kaggle
credentials and an open competition — neurogolf-2026 has closed.
