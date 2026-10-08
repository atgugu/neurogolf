"""Build task383 with side-classified markers and a terminal palette renderer.

Paper price: 1848 bytes of charged tensors + 128 initializer elements = 1976.
The graph deliberately uses raw ONNX only: there are no ngolf graph calls.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent


def init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name)


def vi(name: str, elem_type: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    value_info: list[onnx.ValueInfoProto] = []

    def add(
        op_type: str,
        inputs: list[str],
        output: str,
        elem_type: int,
        shape: list[int],
        **attrs,
    ) -> str:
        nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        if output != "output":
            value_info.append(vi(output, elem_type, shape))
        return output

    # The nonzero box is a solid rectangle. Its row/column projections therefore
    # have constant positive plateaus, whose moments recover the rectangle origin.
    add(
        "Einsum",
        ["input", "nonbg_ch", "ones_30"],
        "row_sum",
        TensorProto.FLOAT,
        [1, 30],
        equation="bchw,c,xw->bh",
    )
    add(
        "Einsum",
        ["input", "nonbg_ch", "ones_30"],
        "col_sum",
        TensorProto.FLOAT,
        [1, 30],
        equation="bchw,c,xh->bw",
    )
    add(
        "Einsum",
        ["row_sum", "ones_30"],
        "area",
        TensorProto.FLOAT,
        [1],
        equation="bh,xh->b",
    )
    add("ReduceMax", ["row_sum"], "width_f", TensorProto.FLOAT, [1], axes=[1], keepdims=0)
    add("ReduceMax", ["col_sum"], "height_f", TensorProto.FLOAT, [1], axes=[1], keepdims=0)
    add(
        "Einsum",
        ["row_sum", "idx_f"],
        "row_moment",
        TensorProto.FLOAT,
        [1],
        equation="bh,h->b",
    )
    add(
        "Einsum",
        ["col_sum", "idx_f"],
        "col_moment",
        TensorProto.FLOAT,
        [1],
        equation="bw,w->b",
    )
    add("Div", ["row_moment", "area"], "row_center", TensorProto.FLOAT, [1])
    add("Div", ["col_moment", "area"], "col_center", TensorProto.FLOAT, [1])
    add("Sub", ["height_f", "one_f"], "height_m1", TensorProto.FLOAT, [1])
    add("Sub", ["width_f", "one_f"], "width_m1", TensorProto.FLOAT, [1])
    add("Mul", ["height_m1", "half_f"], "row_radius", TensorProto.FLOAT, [1])
    add("Mul", ["width_m1", "half_f"], "col_radius", TensorProto.FLOAT, [1])
    add("Sub", ["row_center", "row_radius"], "top_f", TensorProto.FLOAT, [1])
    add("Sub", ["col_center", "col_radius"], "left_f", TensorProto.FLOAT, [1])
    add("Add", ["top_f", "height_m1"], "bottom_f", TensorProto.FLOAT, [1])
    add("Add", ["left_f", "width_m1"], "right_f", TensorProto.FLOAT, [1])

    # The generator encodes vertical markers on rows top+1/bottom-1 and
    # horizontal markers on columns left+1/right-1.
    add("Add", ["top_f", "one_f"], "top1_f", TensorProto.FLOAT, [1])
    add("Sub", ["bottom_f", "one_f"], "bottom1_f", TensorProto.FLOAT, [1])
    add("Add", ["left_f", "one_f"], "left1_f", TensorProto.FLOAT, [1])
    add("Sub", ["right_f", "one_f"], "right1_f", TensorProto.FLOAT, [1])

    # Integer coordinates are needed only for the two one-cell palette slices.
    add("Cast", ["top_f"], "top_i", TensorProto.INT64, [1], to=TensorProto.INT64)
    add("Cast", ["left_f"], "left_i", TensorProto.INT64, [1], to=TensorProto.INT64)
    add("Cast", ["top1_f"], "top1_i", TensorProto.INT64, [1], to=TensorProto.INT64)
    add("Cast", ["left1_f"], "left1_i", TensorProto.INT64, [1], to=TensorProto.INT64)
    add("Add", ["top_i", "two_i"], "top2_i", TensorProto.INT64, [1])
    add("Add", ["left_i", "two_i"], "left2_i", TensorProto.INT64, [1])
    add("Add", ["top_i", "three_i"], "top3_i", TensorProto.INT64, [1])
    add("Add", ["left_i", "three_i"], "left3_i", TensorProto.INT64, [1])

    add("Concat", ["start_prefix", "top_i", "left_i"], "frame_starts", TensorProto.INT64, [4], axis=0)
    add("Concat", ["end_prefix", "top1_i", "left1_i"], "frame_ends", TensorProto.INT64, [4], axis=0)
    add(
        "Slice",
        ["input", "frame_starts", "frame_ends", "axes_4"],
        "frame_vec",
        TensorProto.FLOAT,
        [1, 10, 1, 1],
    )
    add("Concat", ["start_prefix", "top2_i", "left2_i"], "fill_starts", TensorProto.INT64, [4], axis=0)
    add("Concat", ["end_prefix", "top3_i", "left3_i"], "fill_ends", TensorProto.INT64, [4], axis=0)
    add(
        "Slice",
        ["input", "fill_starts", "fill_ends", "axes_4"],
        "fill_vec",
        TensorProto.FLOAT,
        [1, 10, 1, 1],
    )

    add("Equal", ["idx_f", "top1_f"], "eq_top", TensorProto.BOOL, [30])
    add("Equal", ["idx_f", "bottom1_f"], "eq_bottom", TensorProto.BOOL, [30])
    add("Or", ["eq_top", "eq_bottom"], "side_rows_b", TensorProto.BOOL, [30])
    add("Equal", ["idx_f", "left1_f"], "eq_left", TensorProto.BOOL, [30])
    add("Equal", ["idx_f", "right1_f"], "eq_right", TensorProto.BOOL, [30])
    add("Or", ["eq_left", "eq_right"], "side_cols_b", TensorProto.BOOL, [30])
    add("Cast", ["side_rows_b"], "side_rows_f", TensorProto.FLOAT, [30], to=TensorProto.FLOAT)
    add("Cast", ["side_cols_b"], "side_cols_f", TensorProto.FLOAT, [30], to=TensorProto.FLOAT)

    add(
        "Einsum",
        ["input", "fill_vec", "side_cols_f"],
        "row_markers",
        TensorProto.FLOAT,
        [1, 30],
        equation="bchw,bcxy,w->bh",
    )
    add(
        "Einsum",
        ["input", "fill_vec", "side_rows_f"],
        "col_markers",
        TensorProto.FLOAT,
        [1, 30],
        equation="bchw,bcxy,h->bw",
    )
    add("Concat", ["ones_30", "row_markers"], "row_factors", TensorProto.FLOAT, [2, 30], axis=0)
    add("Concat", ["ones_30", "col_markers"], "col_factors", TensorProto.FLOAT, [2, 30], axis=0)

    # One palette basis is reused as both the source and target basis. The
    # coefficient polynomial implements I + (R + C - R*C) * (M - I), where M
    # maps background->fill and both non-background colors->frame.
    add("Concat", ["bg_vec", "frame_vec", "fill_vec"], "palette", TensorProto.FLOAT, [3, 10, 1, 1], axis=0)
    add(
        "Einsum",
        ["input", "row_factors", "col_factors", "coeff", "palette", "palette"],
        "output",
        TensorProto.FLOAT,
        [1, 10, 30, 30],
        equation="bdhw,kh,lw,klqt,qdxy,tcxy->bchw",
    )

    identity = np.eye(3, dtype=np.float32)
    on = np.array(
        [
            [0.0, 0.0, 1.0],  # background -> fill
            [0.0, 1.0, 0.0],  # frame -> frame
            [0.0, 1.0, 0.0],  # fill -> frame
        ],
        dtype=np.float32,
    )
    delta = on - identity
    coeff = np.empty((2, 2, 3, 3), dtype=np.float32)
    coeff[0, 0] = identity
    coeff[1, 0] = delta
    coeff[0, 1] = delta
    coeff[1, 1] = -delta

    nonbg = np.ones(10, dtype=np.float32)
    nonbg[0] = 0.0
    bg = np.zeros((1, 10, 1, 1), dtype=np.float32)
    bg[0, 0, 0, 0] = 1.0
    initializers = [
        init("nonbg_ch", nonbg),
        init("ones_30", np.ones((1, 30), dtype=np.float32)),
        init("idx_f", np.arange(30, dtype=np.float32)),
        init("one_f", np.array([1.0], dtype=np.float32)),
        init("half_f", np.array([0.5], dtype=np.float32)),
        init("two_i", np.array([2], dtype=np.int64)),
        init("three_i", np.array([3], dtype=np.int64)),
        init("start_prefix", np.array([0, 0], dtype=np.int64)),
        init("end_prefix", np.array([1, 10], dtype=np.int64)),
        init("axes_4", np.array([0, 1, 2, 3], dtype=np.int64)),
        init("bg_vec", bg),
        init("coeff", coeff),
    ]

    graph = helper.make_graph(
        nodes,
        "task383_side_classified_palette_basis",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    path = HERE / "task383.onnx"
    onnx.save(build(), path)
    print(path)
