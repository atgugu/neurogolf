#!/usr/bin/env python3
"""Build the current task330 candidate."""
from __future__ import annotations

from pathlib import Path

import build_thresh9_valid


def main() -> None:
    out = Path(__file__).resolve().with_name("task330.onnx")
    build_thresh9_valid.build(out)
    print(out)


if __name__ == "__main__":
    main()
