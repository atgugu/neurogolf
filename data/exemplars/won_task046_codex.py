#!/usr/bin/env python3
"""task046 compact terminal rebuilds.

Rule: remove black separator columns, repaint gray connectors with the segment
color, and vertically align later segments by their gray boundary anchors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


POLISHED = Path(__file__).with_name("task046_polished.onnx")
OUT = Path(__file__).with_name("task046.onnx")
ATT08 = Path(__file__).with_name("task046_att08.onnx")
QZP = Path(__file__).with_name("task046_qlinear_zp.onnx")
QZP_POOL = Path(__file__).with_name("task046_qlinear_zp_pool.onnx")
CONV_POOL = Path(__file__).with_name("task046_convint_pool.onnx")
WIDTH_ARITH = Path(__file__).with_name("task046_width_arith.onnx")
CROP19 = Path(__file__).with_name("task046_crop19.onnx")


ATT12_CHARGED_TENSORS = [
    ("f32", [1, 1, 3, 19], 1, 228),
    ("u8", [1, 1, 3, 19], 2, 114),
    ("u8", [19], 2, 38),
    ("u8", [16], 1, 16),
    ("i64", [1, 16], 1, 128),
    ("bool", [16], 1, 16),
    ("u8", [3, 16], 5, 240),
    ("bool", [3, 16], 1, 48),
    ("i8", [16], 4, 64),
    ("i8", [15], 8, 120),
    ("bool", [15], 1, 15),
    ("u8", [16], 2, 32),
    ("u8", [1, 16], 10, 160),
    ("u8", [7, 16], 1, 112),
    ("u8", [15], 3, 45),
    ("u8", [10], 1, 10),
]


def init(name: str, arr, dtype=None) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name)


def terminal_weights() -> np.ndarray:
    """Weights for logits: valid*(1-k^2) + 2*k*code - code^2."""
    # Internal color codes used by the polished body. Channel 5 is deliberately
    # mapped to an absent sentinel so gray never appears in the output.
    encoded = np.array([0, 2, 3, 4, 5, 11, 6, 7, 8, 9], dtype=np.int16)
    w = np.zeros((10, 3, 1, 1), dtype=np.int8)
    w[:, 0, 0, 0] = (1 - encoded * encoded).astype(np.int8)
    w[:, 1, 0, 0] = (2 * encoded).astype(np.int8)
    w[:, 2, 0, 0] = -1
    return w


def qlinear_terminal_inits() -> list[onnx.TensorProto]:
    """QLinearConv terminal with padding-safe zero-points.

    The feature channels are encoded color and encoded color squared.  QLinearConv
    subtracts x_zero_point=[10, 100], so spatial padding contributes zero to the
    dot product.  Per-output y_zero_point supplies the quadratic bias:
    1 - (code - k)^2, which is positive only for the matching encoded color.
    """
    encoded = np.array([0, 2, 3, 4, 5, 11, 6, 7, 8, 9], dtype=np.int16)
    w = np.zeros((10, 2, 1, 1), dtype=np.int8)
    w[:, 0, 0, 0] = (2 * encoded).astype(np.int8)
    w[:, 1, 0, 0] = -1
    y_zp = (1 - (encoded - 10) * (encoded - 10)).astype(np.int8)
    return [
        init("q_x_scale", np.array(1.0, dtype=np.float32)),
        init("q_x_zp", np.array([10, 100], dtype=np.uint8)),
        init("q_w", w),
        init("q_w_scale", np.array(1.0, dtype=np.float32)),
        init("q_w_zp", np.array(0, dtype=np.int8)),
        init("q_y_scale", np.array(1.0, dtype=np.float32)),
        init("q_y_zp", y_zp),
    ]


def append_qlinear_terminal(nodes: list[onnx.NodeProto]) -> None:
    nodes.extend(
        [
            helper.make_node(
                "Where",
                ["width_present", "shifted_color", "outside_u8"],
                ["output_code"],
            ),
            helper.make_node("Mul", ["output_code", "output_code"], ["color_sq"]),
            helper.make_node(
                "Concat",
                ["output_code", "color_sq"],
                ["output_feat"],
                axis=1,
            ),
            helper.make_node(
                "QLinearConv",
                [
                    "output_feat",
                    "q_x_scale",
                    "q_x_zp",
                    "q_w",
                    "q_w_scale",
                    "q_w_zp",
                    "q_y_scale",
                    "q_y_zp",
                ],
                ["output"],
                kernel_shape=[1, 1],
                pads=[0, 0, 27, 14],
            ),
        ]
    )


def append_convinteger_terminal(nodes: list[onnx.NodeProto]) -> None:
    nodes.extend(
        [
            helper.make_node(
                "Where",
                ["width_present", "valid_one_u8", "zero_u8"],
                ["valid_cells"],
            ),
            helper.make_node("Mul", ["shifted_color", "shifted_color"], ["color_sq"]),
            helper.make_node(
                "Concat",
                ["valid_cells", "shifted_color", "color_sq"],
                ["output_feat"],
                axis=1,
            ),
            helper.make_node(
                "ConvInteger",
                ["output_feat", "terminal_w"],
                ["output"],
                kernel_shape=[1, 1],
                pads=[0, 0, 27, 14],
            ),
        ]
    )


def make_model_slice_shift_concat() -> onnx.ModelProto:
    """Keep the verified slice/shift body and replace the terminal renderer."""
    model = onnx.load(POLISHED)
    graph = model.graph

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.output and node.output[0] in {"output_ids", "output_top", "output"}:
            if node.output[0] == "output_ids":
                append_convinteger_terminal(new_nodes)
            continue
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    keep = [
        tensor
        for tensor in graph.initializer
        if tensor.name not in {"outside_u8", "color_values", "output_pads"}
    ]
    keep.extend(
        [
            init("valid_one_u8", np.ones((1, 1, 3, 1), dtype=np.uint8)),
            init("terminal_w", terminal_weights()),
        ]
    )
    del graph.initializer[:]
    graph.initializer.extend(keep)

    del graph.value_info[:]
    graph.output[0].CopyFrom(
        helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def make_model_convint_pooled_recolor() -> onnx.ModelProto:
    """Legal ConvInteger terminal with neighborhood recolor instead of directional fill."""
    model = onnx.load(POLISHED)
    graph = model.graph

    recolor_outputs = {
        "visible_color",
        "visible_left_core",
        "visible_left_core_gated",
        "visible_left_gated",
        "visible_right_core",
        "visible_right_core_gated",
        "visible_right_gated",
        "segment_color_cols",
        "filled_comp_color",
    }
    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        out0 = node.output[0] if node.output else ""
        if out0 in recolor_outputs:
            if out0 == "visible_color":
                new_nodes.extend(
                    [
                        helper.make_node(
                            "MaxPool",
                            ["comp_color"],
                            ["pooled_color"],
                            kernel_shape=[3, 3],
                            pads=[1, 1, 1, 1],
                            strides=[1, 1],
                        ),
                        helper.make_node(
                            "Where",
                            ["gray_cells", "pooled_color", "comp_color"],
                            ["filled_comp_color"],
                        ),
                    ]
                )
            continue
        if out0 in {"output_ids", "output_top", "output"}:
            if out0 == "output_ids":
                append_convinteger_terminal(new_nodes)
            continue
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    keep = [
        tensor
        for tensor in graph.initializer
        if tensor.name not in {"outside_u8", "color_values", "output_pads"}
    ]
    keep.extend(
        [
            init("valid_one_u8", np.ones((1, 1, 3, 1), dtype=np.uint8)),
            init("terminal_w", terminal_weights()),
        ]
    )
    del graph.initializer[:]
    graph.initializer.extend(keep)

    del graph.value_info[:]
    graph.output[0].CopyFrom(
        helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def make_model_crop19_fallback() -> onnx.ModelProto:
    """Crop the faithful compact path front to the generator-certified live box.

    A generated example has at most 16 output columns and three separator
    columns.  The optional twentieth input column is always blank, so only the
    first 19 columns can carry path data.  A 2-wide dilated Conv performs that
    crop without materializing a charged Slice output.  The remaining body is
    the value-exact uint8/float16 compact implementation, with the exact pooled
    gray recolor and int32 terminal renderer from the prior verified draft.
    """
    model = onnx.load(Path(__file__).with_name("refs") / "kaggle_proven_best.onnx")
    graph = model.graph

    # Widen the color projection kernel with an all-zero tap.  Dilation 11
    # makes 30 - 11 = 19 live columns while preserving every selected value.
    for tensor in graph.initializer:
        if tensor.name == "color_weights":
            old = numpy_helper.to_array(tensor)
            wide = np.zeros((1, 10, 2, 2), dtype=np.float32)
            wide[:, :, :, 0] = old[:, :, :, 0]
            tensor.CopyFrom(init("color_weights", wide))
        elif tensor.name == "path_weights":
            tensor.CopyFrom(
                init("path_weights", np.arange(19, 0, -1, dtype=np.int8))
            )

    for node in graph.node:
        if node.op_type == "Conv" and node.output == ["color_grid_f_full"]:
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    attr.ints[:] = [2, 2]
                elif attr.name == "dilations":
                    attr.ints[:] = [27, 11]

    # Rank present source columns entirely in int8.  ORT 1.24 has no uint8
    # TopK kernel; all positive scores fit exactly in signed int8.  Descending scores retain
    # the faithful left-to-right TopK order while eliminating the float16
    # Where output and float16 TopK values from att11.
    ranked: list[onnx.NodeProto] = []
    for node in graph.node:
        out0 = node.output[0] if node.output else ""
        if out0 == "path_score":
            ranked.extend(
                [
                    helper.make_node(
                        "Cast", ["path_bool"], ["path_present_i8"],
                        to=TensorProto.INT8,
                    ),
                    helper.make_node(
                        "Mul", ["path_present_i8", "path_weights"], ["path_score"]
                    ),
                ]
            )
        elif out0 == "width_present":
            ranked.append(
                helper.make_node(
                    "Greater", ["topk_values", "zero_i8"], ["width_present"]
                )
            )
        else:
            ranked.append(node)
    del graph.node[:]
    graph.node.extend(ranked)

    graph.initializer.extend([init("zero_i8", np.array(0, dtype=np.int8))])

    # Recolor gray before compaction.  In generator inputs every pair of
    # segments is separated by a black column, so a 3x3 pool cannot cross from
    # one segment to another.  Collapsing the three rows and dilating one
    # column therefore yields the exact segment color in one small tensor.
    directional = {
        "visible_color",
        "visible_left_core",
        "visible_left_core_gated",
        "visible_left_gated",
        "visible_right_core",
        "visible_right_core_gated",
        "visible_right_gated",
        "segment_color_cols",
        "filled_comp_color",
    }
    rewritten: list[onnx.NodeProto] = []
    for node in graph.node:
        out0 = node.output[0] if node.output else ""
        if out0 == "visible_color":
            rewritten.extend(
                [
                    helper.make_node(
                        "MaxPool",
                        ["color_grid_u"],
                        ["source_segment_color"],
                        kernel_shape=[3, 3],
                        pads=[0, 1, 0, 1],
                        strides=[1, 1],
                    ),
                    helper.make_node(
                        "Gather",
                        ["source_segment_color", "source_idx64"],
                        ["segment_color_cols"],
                        axis=3,
                    ),
                    helper.make_node(
                        "Where",
                        ["gray_cells", "segment_color_cols", "comp_color"],
                        ["filled_comp_color"],
                    ),
                ]
            )
        if out0 in directional:
            continue
        rewritten.append(node)
    del graph.node[:]
    graph.node.extend(rewritten)

    # Drop constants made dead by the pooled recolor rewrite.
    used = {name for node in graph.node for name in node.input if name}
    keep = [tensor for tensor in graph.initializer if tensor.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)

    del graph.value_info[:]
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def make_model_width_arith_compactor() -> onnx.ModelProto:
    """TopK-free compactor using certified segment widths in {2,3,4}.

    The input path has three or four contiguous runs separated by one black
    column.  Once the run widths are known, source column j is just
    compact_j + (# previous separators), so a small int8/int32 index vector can
    replace TopK's int64 index plane.
    """

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def add_init(name: str, arr, dtype=None) -> str:
        inits.append(init(name, arr, dtype))
        return name

    def n(op: str, inputs: list[str], outputs: list[str] | str, **attrs) -> None:
        if isinstance(outputs, str):
            outputs = [outputs]
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))

    # Shared constants.
    add_init(
        "color_weights",
        np.array(
            [[[[0], [0]], [[2], [0]], [[3], [0]], [[4], [0]], [[5], [0]],
              [[1], [0]], [[6], [0]], [[7], [0]], [[8], [0]], [[9], [0]]]],
            dtype=np.float32,
        ),
    )
    add_init("zero_f", np.array(0, dtype=np.float16))
    add_init("zero_u8", np.array(0, dtype=np.uint8))
    add_init("zero_i8", np.array([0], dtype=np.int8))
    add_init("one_i8", np.array([1], dtype=np.int8))
    add_init("two_i8", np.array([2], dtype=np.int8))
    add_init("three_i8", np.array([3], dtype=np.int8))
    add_init("gray_u8", np.array(1, dtype=np.uint8))
    add_init("row_ids_u8", np.arange(3, dtype=np.uint8).reshape(1, 1, 3, 1))
    add_init("axes_012", np.array([0, 1, 2], dtype=np.int64))
    add_init("axis0", np.array(0, dtype=np.int64))
    add_init("starts0", np.array([0], dtype=np.int64))
    add_init("starts1", np.array([1], dtype=np.int64))
    add_init("ends15", np.array([15], dtype=np.int64))
    add_init("ends16", np.array([16], dtype=np.int64))
    add_init("pad_pre1", np.array([1, 0], dtype=np.int64))
    add_init("offset_m1", np.array(-1, dtype=np.float16))
    add_init("offset_1", np.array(1, dtype=np.float16))
    add_init("offset_2", np.array(2, dtype=np.float16))
    add_init("valid_one_u8", np.ones((1, 1, 3, 1), dtype=np.uint8))
    add_init("terminal_w", terminal_weights())
    add_init("idx2_i32", np.array([2], dtype=np.int32))
    add_init("idx3_i32", np.array([3], dtype=np.int32))
    add_init("arange16_i8", np.arange(16, dtype=np.int8))
    add_init("arange16p1_i8", np.arange(1, 17, dtype=np.int8))

    # One-hot color projection for the certified top 3 rows.
    n(
        "Conv",
        ["input", "color_weights"],
        "color_grid_f_full",
        kernel_shape=[2, 1],
        dilations=[27, 1],
    )
    n("Cast", ["color_grid_f_full"], "color_grid_u", to=TensorProto.UINT8)
    n("ReduceMax", ["color_grid_u", "axes_012"], "path_color", keepdims=0)
    n("Greater", ["path_color", "zero_u8"], "path_bool")

    def keep_bit(name: str, idx: str) -> str:
        n("Gather", ["path_bool", idx], f"{name}_b", axis=0)
        n("Cast", [f"{name}_b"], f"{name}_i8", to=TensorProto.INT8)
        return f"{name}_i8"

    def idx_for(name: str, base: str, delta: str) -> str:
        n("Add", [base, delta], f"{name}_i8")
        n("Cast", [f"{name}_i8"], f"{name}_i32", to=TensorProto.INT32)
        return f"{name}_i32"

    # w = 2, 3, or 4.  The second probe only counts if start+2 is
    # still inside the same run; otherwise it may be the next segment.
    k2 = keep_bit("k2", "idx2_i32")
    k3 = keep_bit("k3", "idx3_i32")
    n("Add", ["one_i8", k3], "w0_ext_base")
    n("Mul", [k2, "w0_ext_base"], "w0_ext")
    n("Add", ["two_i8", "w0_ext"], "w0")

    # Subsequent widths are read at dynamic starts.  Padding is black, so the
    # fourth width naturally becomes zero when there are only three segments.
    n("Add", ["w0", "one_i8"], "start1")
    k12 = keep_bit("k12", idx_for("idx12", "start1", "two_i8"))
    k13 = keep_bit("k13", idx_for("idx13", "start1", "three_i8"))
    n("Add", ["one_i8", k13], "w1_ext_base")
    n("Mul", [k12, "w1_ext_base"], "w1_ext")
    n("Add", ["two_i8", "w1_ext"], "w1")

    n("Add", ["start1", "w1"], "start2_a")
    n("Add", ["start2_a", "one_i8"], "start2")
    k22 = keep_bit("k22", idx_for("idx22", "start2", "two_i8"))
    k23 = keep_bit("k23", idx_for("idx23", "start2", "three_i8"))
    n("Add", ["one_i8", k23], "w2_ext_base")
    n("Mul", [k22, "w2_ext_base"], "w2_ext")
    n("Add", ["two_i8", "w2_ext"], "w2")

    n("Add", ["start2", "w2"], "start3_a")
    n("Add", ["start3_a", "one_i8"], "start3")
    n("Cast", ["start3"], "start3_i32", to=TensorProto.INT32)
    has4 = keep_bit("has4", "start3_i32")
    k32 = keep_bit("k32", idx_for("idx32", "start3", "two_i8"))
    k33 = keep_bit("k33", idx_for("idx33", "start3", "three_i8"))
    n("Add", ["one_i8", k33], "w3_ext_base")
    n("Mul", [k32, "w3_ext_base"], "w3_ext")
    n("Add", ["two_i8", "w3_ext"], "w3_raw")
    n("Mul", ["w3_raw", has4], "w3")

    n("Add", ["w0", "w1"], "b2")
    n("Add", ["b2", "w2"], "b3")
    n("Add", ["b3", "w3"], "total_w")

    # Build source_idx = arange + (j>=w0) + (j>=w0+w1) + (j>=w0+w1+w2).
    n("Less", ["w0", "arange16p1_i8"], "after_b1")
    n("Less", ["b2", "arange16p1_i8"], "after_b2")
    n("Less", ["b3", "arange16p1_i8"], "after_b3")
    n("Cast", ["after_b1"], "after_b1_i8", to=TensorProto.INT8)
    n("Cast", ["after_b2"], "after_b2_i8", to=TensorProto.INT8)
    n("Cast", ["after_b3"], "after_b3_i8", to=TensorProto.INT8)
    n("Add", ["after_b1_i8", "after_b2_i8"], "shift12")
    n("Add", ["shift12", "after_b3_i8"], "shift123")
    n("Add", ["arange16_i8", "shift123"], "source_idx_i8")
    n("Cast", ["source_idx_i8"], "source_idx_i32", to=TensorProto.INT32)
    n("Less", ["arange16_i8", "total_w"], "width_present")

    n("Gather", ["color_grid_u", "source_idx_i32"], "comp_color", axis=3)
    n("Slice", ["source_idx_i8", "starts0", "ends15", "starts0"], "src_idx_head")
    n("Slice", ["source_idx_i8", "starts1", "ends16", "starts0"], "src_idx_tail")
    n("Sub", ["src_idx_tail", "src_idx_head"], "src_idx_gap")
    n("Greater", ["src_idx_gap", "one_i8"], "boundary_tail")

    # Pooled recolor and verified Hodel vertical offset renderer.
    n("Equal", ["comp_color", "gray_u8"], "gray_cells")
    n("Where", ["gray_cells", "row_ids_u8", "zero_u8"], "gray_row_cells")
    n("ReduceMax", ["gray_row_cells", "axes_012"], "gray_row_u8", keepdims=0)
    n("Cast", ["gray_row_u8"], "gray_row_i8", to=TensorProto.INT8)
    n("Slice", ["gray_row_i8", "starts0", "ends15", "starts0"], "gray_head")
    n("Slice", ["gray_row_i8", "starts1", "ends16", "starts0"], "gray_tail")
    n("Sub", ["gray_tail", "gray_head"], "offset_delta_tail")
    n("Cast", ["boundary_tail"], "boundary_tail_i8", to=TensorProto.INT8)
    n("Mul", ["offset_delta_tail", "boundary_tail_i8"], "offset_delta_tail_gated")
    n("Pad", ["offset_delta_tail_gated", "pad_pre1"], "offset_delta_i8", mode="constant")
    n("Cast", ["offset_delta_i8"], "offset_delta", to=TensorProto.FLOAT16)
    n("CumSum", ["offset_delta", "axis0"], "offset_by_col")
    n(
        "MaxPool",
        ["comp_color"],
        "pooled_color",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    n("Where", ["gray_cells", "pooled_color", "comp_color"], "filled_comp_color")
    n("Equal", ["offset_by_col", "offset_m1"], "offset_m1_mask")
    n("Equal", ["offset_by_col", "zero_f"], "offset_0_mask")
    n("Equal", ["offset_by_col", "offset_1"], "offset_1_mask")
    n("Equal", ["offset_by_col", "offset_2"], "offset_2_mask")
    n(
        "Split",
        ["filled_comp_color"],
        ["src_row_0", "src_row_1", "src_row_2"],
        axis=2,
        num_outputs=3,
    )
    n("Where", ["offset_0_mask", "src_row_0", "zero_u8"], "shifted_row_0_off_0")
    n("Where", ["offset_1_mask", "src_row_1", "shifted_row_0_off_0"], "shifted_row_0_off_1")
    n("Where", ["offset_2_mask", "src_row_2", "shifted_row_0_off_1"], "shifted_row_0_off_2")
    n("Where", ["offset_m1_mask", "src_row_0", "zero_u8"], "shifted_row_1_off_m1")
    n("Where", ["offset_0_mask", "src_row_1", "shifted_row_1_off_m1"], "shifted_row_1_off_0")
    n("Where", ["offset_1_mask", "src_row_2", "shifted_row_1_off_0"], "shifted_row_1_off_1")
    n("Where", ["offset_m1_mask", "src_row_1", "src_row_0"], "shifted_row_2_off_m1")
    n("Where", ["offset_0_mask", "src_row_2", "shifted_row_2_off_m1"], "shifted_row_2_off_0")
    n(
        "Concat",
        ["shifted_row_0_off_2", "shifted_row_1_off_1", "shifted_row_2_off_0"],
        "shifted_color",
        axis=2,
    )
    append_convinteger_terminal(nodes)

    graph = helper.make_graph(
        nodes,
        "task046_width_arith",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def make_model_qlinear_zp(*, pooled_recolor: bool = False) -> onnx.ModelProto:
    """QLinearConv zero-point terminal; optionally replace directional recolor."""
    model = onnx.load(POLISHED)
    graph = model.graph

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        out0 = node.output[0] if node.output else ""
        if out0 in {
            "visible_color",
            "visible_left_core",
            "visible_left_core_gated",
            "visible_left_gated",
            "visible_right_core",
            "visible_right_core_gated",
            "visible_right_gated",
            "segment_color_cols",
            "filled_comp_color",
        }:
            if pooled_recolor and out0 == "visible_color":
                new_nodes.extend(
                    [
                        helper.make_node(
                            "MaxPool",
                            ["comp_color"],
                            ["pooled_color"],
                            kernel_shape=[3, 3],
                            pads=[1, 1, 1, 1],
                            strides=[1, 1],
                        ),
                        helper.make_node(
                            "Where",
                            ["gray_cells", "pooled_color", "comp_color"],
                            ["filled_comp_color"],
                        ),
                    ]
                )
            if pooled_recolor:
                continue
        if out0 in {"output_ids", "output_top", "output"}:
            if out0 == "output_ids":
                append_qlinear_terminal(new_nodes)
            continue
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    drop = {
        "color_values",
        "output_pads",
        "terminal_w",
        "valid_one_u8",
    }
    keep = [tensor for tensor in graph.initializer if tensor.name not in drop]
    keep.extend(qlinear_terminal_inits())
    del graph.initializer[:]
    graph.initializer.extend(keep)

    del graph.value_info[:]
    graph.output[0].CopyFrom(
        helper.make_tensor_value_info("output", TensorProto.INT8, [1, 10, 30, 30])
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    charged = sum(item[3] for item in ATT12_CHARGED_TENSORS)
    print("att12 charged-tensor manifest (dtype shape count bytes):")
    for dtype, shape, count, size in ATT12_CHARGED_TENSORS:
        print(f"  {dtype:4s} {shape!s:16s} x{count:<2d} = {size} B")
    print(f"planned charged={charged} B params=112 total={charged + 112} B")
    crop19 = make_model_crop19_fallback()
    onnx.save(crop19, CROP19)
    onnx.save(crop19, OUT)
    print(f"saved {OUT} and {CROP19}")


if __name__ == "__main__":
    main()
