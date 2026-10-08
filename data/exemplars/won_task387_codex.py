#!/usr/bin/env python3
"""Static, certified 18x18 renderer floor for task387.

Budget: the static scalar native renderer retains its uint8[18,18] state and
terminal uint8[30,30] Pad input.  Its charged total is 3,037 B (2,917 B
memory + 120 parameter elements), above the live Lane-A ceiling of 2,904 B.
The only attempted mechanical reduction, bool ArgMax in place of two 30-byte
casts, is invalid in ONNX/ORT 1.24.  This script builds the last static,
value-exact native-size candidate for the required priced record.
"""
from pathlib import Path
import onnx
from onnx import helper

HERE = Path(__file__).resolve().parent
BASE = HERE / "task387_scalaropt_validbase.onnx"
OUT = HERE / "task387.onnx"


def build():
    m = onnx.load(BASE)
    # This static scalar member is a distinct prior draft; retain it intact so
    # the live verifier can price the certified native-size floor.
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
