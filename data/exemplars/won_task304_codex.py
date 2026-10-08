"""Rebuild task304's selector-Einsum candidate."""

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).parent / "experiments" / "task304" / "build_selector_einsum.py"),
    run_name="__main__",
)
