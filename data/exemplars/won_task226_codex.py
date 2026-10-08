#!/usr/bin/env python3
"""Build the attempt-9 exact separable pair-product model."""

from attempt9_model import OUT, build
import onnx


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
