#!/usr/bin/env python3
"""Build the att12 runtime-period-aware explicit 2-D OR probe."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("att12") / "build.py"), run_name="__main__")
