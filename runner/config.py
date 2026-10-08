"""Single configuration point for every external path the system touches.

All roots are overridable via environment variables so the repo runs on any
machine where the companion data exists:

  NEUROGOLF_CLEAN      the neurogolf_clean checkout — grader-exact scorer
                       (neurogolf/scoring_v2.py), trusted gates (candidates/
                       gate124.py, gate_pair.py), ARC ground-truth grids
                       (extracted/task*.json) and generator sources
                       (external/ARC-GEN).  REQUIRED for verification.
  NEUROGOLF_ARCHIVE    optional deep-history archive (reasoning traces,
                       gate-pass catalog, dossiers).  Pack building degrades
                       gracefully when absent — packs just carry less context.
  NEUROGOLF_RESOURCES  optional audit bundle (score_bundle replica).
  NEUROGOLF_VENV_PY    python interpreter of the grader venv (ORT 1.24.4).
                       Empty = use the current interpreter.
"""
import os

CLEAN = os.environ.get("NEUROGOLF_CLEAN", "/home/atgu/Projects/ARC/neurogolf_clean")
ARCHIVE = os.environ.get("NEUROGOLF_ARCHIVE", "/media/atgu/Elements/neurogolf")
RESOURCES = os.environ.get(
    "NEUROGOLF_RESOURCES", "/home/atgu/Projects/ARC/neurogolf_resources")
VENV_PY = os.environ.get("NEUROGOLF_VENV_PY", "")
