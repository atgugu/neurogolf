#!/usr/bin/env python3
"""Task 086 attempt 18: disjoint u8 regions + dynamic terminal kernel."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper as H, numpy_helper


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "refs" / "kaggle_proven_best.onnx"


def initializer(name, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def build(destination: Path) -> None:
    reference = onnx.load(REFERENCE)
    ref_init = {x.name: numpy_helper.to_array(x) for x in reference.graph.initializer}

    # Quantize the signed reference threshold directly to 0/1.  The +512 shift
    # sends score 0 to the round-to-even tie at 0 while every positive integer
    # reaches 1.  The stencil's absolute positive bound is only 516, so no
    # binary 11x11 patch can round onward to 2 at output scale 1024.
    outer_weight = ref_init["outer_weight"].astype(np.int8)

    initializers = [
        initializer("crop_start", [0, 0, 0, 0], np.int64),
        initializer("crop_end", [1, 1, 12, 12], np.int64),
        initializer("zero", [0], np.int64),
        initializer("one", [1], np.int64),
        initializer("axis_h", [2], np.int64),
        initializer("axis_w", [3], np.int64),
        initializer("inner_pick", 2, np.int64),
        initializer("outer_pick", 1, np.int64),
        initializer("outer_weight", outer_weight),
        initializer("outer_bias", [484], np.int32),
        initializer("unit_scale", [1.0], np.float32),
        initializer("binary_scale", [1024.0], np.float32),
        initializer("zero_u8", [0], np.uint8),
        initializer("zero_i8", [0], np.int8),
        initializer("palette", np.arange(10, dtype=np.int64).reshape(10, 1, 1, 1)),
        initializer("background_kernel", (2 * np.eye(10, 1, dtype=np.int8)).reshape(10, 1, 1, 1)),
        initializer("render_bias", -np.ones(10, dtype=np.int32)),
    ]

    nodes = [
        H.make_node("Slice", ["input", "crop_start", "crop_end"], ["background_f32"]),
        H.make_node("Cast", ["background_f32"], ["background"], to=TensorProto.UINT8),
        H.make_node("Slice", ["background", "zero", "one", "axis_h"], ["valid_cols"]),
        H.make_node("Slice", ["background", "zero", "one", "axis_w"], ["valid_rows"]),
        H.make_node("Mul", ["valid_rows", "valid_cols"], ["valid"]),
        H.make_node("Sub", ["valid", "background"], ["occupancy"]),
        H.make_node(
            "QLinearConv",
            ["occupancy", "unit_scale", "zero_u8", "outer_weight", "unit_scale", "zero_i8",
             "binary_scale", "zero_u8", "outer_bias"],
            ["outer"], kernel_shape=[11, 11], pads=[5, 5, 5, 5],
        ),
        # Reverse subtraction intentionally: ring cells become 255, added-frame
        # cells become 1.  The biased terminal separates those values cheaply.
        H.make_node("Sub", ["outer", "occupancy"], ["inner"]),
        H.make_node("GlobalAveragePool", ["input"], ["color_counts"]),
        H.make_node("TopK", ["color_counts", "axis_w"], ["top_counts", "top_indices"],
                    axis=1, largest=1, sorted=1),
        H.make_node("Gather", ["top_indices", "inner_pick"], ["inner_index"], axis=1),
        H.make_node("Gather", ["top_indices", "outer_pick"], ["outer_index"], axis=1),
        # ORT 1.24 has no i8 OneHot kernel. Equal+Cast is the smallest portable
        # standard-ONNX realization of the same two dynamic kernel columns.
        H.make_node("Equal", ["palette", "inner_index"], ["inner_hot"]),
        H.make_node("Cast", ["inner_hot"], ["inner_kernel"], to=TensorProto.INT8),
        H.make_node("Equal", ["palette", "outer_index"], ["outer_hot"]),
        H.make_node("Cast", ["outer_hot"], ["outer_kernel"], to=TensorProto.INT8),
        H.make_node("Add", ["outer_kernel", "outer_kernel"], ["outer_kernel_2"]),
        # Source-background overlaps the generated frame.  A -2 outer-column
        # coefficient on output channel zero cancels that overlap terminally.
        H.make_node("Sub", ["outer_kernel_2", "background_kernel"], ["outer_kernel_adjusted"]),
        H.make_node("Concat", ["background_kernel", "inner_kernel", "outer_kernel_adjusted"],
                    ["dynamic_kernel"], axis=1),
        H.make_node("Concat", ["background", "inner", "outer"], ["selectors"], axis=1),
        H.make_node(
            "QLinearConv",
            ["selectors", "unit_scale", "zero_u8", "dynamic_kernel", "unit_scale", "zero_i8",
             "unit_scale", "zero_u8", "render_bias"],
            ["output"], kernel_shape=[1, 1], pads=[0, 0, 18, 18],
        ),
    ]

    graph = H.make_graph(
        nodes,
        "task086_u8_regions_dynamic_kernel",
        [H.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [H.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, destination)
    print(f"wrote {destination}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / "task086.onnx")
    args = parser.parse_args()
    build(args.candidate.resolve())


if __name__ == "__main__":
    main()
