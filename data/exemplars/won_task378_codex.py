from __future__ import annotations

from pathlib import Path

import onnx

from build_att27 import build


OUT = Path(__file__).with_name("task378.onnx")


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
