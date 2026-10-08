#!/usr/bin/env python3
import onnx

from build_att07_masked_grid_relower import build


if __name__ == "__main__":
    onnx.save(build(), "task029.onnx")
