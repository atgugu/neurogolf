from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "extracted"
TASKS_DIR = EXTRACTED_DIR
FRAMEWORK_FILE = EXTRACTED_DIR / "neurogolf_utils" / "neurogolf_utils.py"

PACKAGE_DIR = PROJECT_ROOT / "neurogolf"
SOLVERS_DIR = PROJECT_ROOT / "solvers"
SOLVERS_BEST = SOLVERS_DIR / "best"
CANDIDATES_DIR = PROJECT_ROOT / "candidates"
LOGS_DIR = PROJECT_ROOT / "logs"
SUBMISSIONS_LOG = LOGS_DIR / "submissions.jsonl"

DEFAULT_MANIFEST = PROJECT_ROOT / "solvers_manifest.json"
DEFAULT_ZIP = PROJECT_ROOT / "submission.zip"

FILESIZE_LIMIT_BYTES = 1.44 * 1024 * 1024
EXCLUDED_OPS = ("LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS")
GRID_SHAPE = (1, 10, 30, 30)
INPUT_NAME = "input"
OUTPUT_NAME = "output"


def task_json(task_num: int) -> Path:
    return TASKS_DIR / f"task{task_num:03d}.json"
