"""Task 359: signed-bit modal-stripe renderer.

Budget (all non-output tensors are charged):

  Design A, exact simplified gather tail: 3329 B
    two f32 10x30 histograms 2400; two i64 modes 480; tail 419; params 30.
  Design B, emitted signed-bit projection renderer: 2436 B
    signed profiles f32 [1,30,4] x2                         960
    ReduceL1 orientation scalars + flag                        9
    bit flags, selected bits, u8 casts, MMI code, u8 code    870
    compact-column squeeze                                    30
    active masks/max/clip + orientation selects              420
    terminal target                                           300
    parameters (sign table, sum vector, code weights, maps)  103

The generator uses colours 1..9 and replaces only a sparse random subset of
each long uniform stripe.  Four signed colour-bit projections recover that
stripe's bit-majority; their L1 line score selects the uniform stripe axis.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path("task359.onnx")


def value(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def init(a, name):
    return numpy_helper.from_array(np.asarray(a), name)


def main():
    # Channel zero is the all-zero padded canvas, not a valid stripe colour.
    # For valid colours, each column is the signed representation of one bit.
    signs = np.zeros((10, 4), dtype=np.float32)
    for colour in range(1, 10):
        signs[colour] = [1.0 if colour & (1 << bit) else -1.0 for bit in range(4)]

    nodes = [
        # One [30,4] profile for each axis.  This is the plane kill: 960 B
        # replaces the pin's two 1200-B ten-colour AveragePool planes.
        helper.make_node("Einsum", ["input", "signs", "ones"], ["row_signed"],
                         equation="bcyx,ck,x->byk"),
        helper.make_node("Einsum", ["input", "signs", "ones"], ["col_signed"],
                         equation="bcyx,ck,y->bxk"),
        # Long, nearly uniform stripe lines have the greater total bit bias.
        helper.make_node("ReduceL1", ["row_signed"], ["row_strength"],
                         axes=[1, 2], keepdims=0),
        helper.make_node("ReduceL1", ["col_signed"], ["col_strength"],
                         axes=[1, 2], keepdims=0),
        helper.make_node("Greater", ["row_strength", "col_strength"], ["use_row"]),
        # A positive signed sum is that bit's majority value.
        helper.make_node("Greater", ["row_signed", "zero_f"], ["row_bits"]),
        helper.make_node("Greater", ["col_signed", "zero_f"], ["col_bits"]),
        # ORT 1.24 has no bool-Where kernel at this opset, so lower the two
        # compact bit strips before selecting one.
        helper.make_node("Cast", ["row_bits"], ["row_bits_u8"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["col_bits"], ["col_bits_u8"], to=TensorProto.UINT8),
        helper.make_node("Where", ["use_row", "row_bits_u8", "col_bits_u8"], ["chosen_u8"]),
        helper.make_node("MatMulInteger", ["chosen_u8", "bit_value"], ["chosen_i32"]),
        helper.make_node("Gather", ["code_map", "chosen_i32"], ["chosen_value"], axis=0),
        # The selected code is a row vector when horizontal stripes win and a
        # column vector otherwise.  Keep both broadcast views compact.
        helper.make_node("Squeeze", ["chosen_value", "squeeze_axis"], ["chosen_col"]),
        # Exact active-rectangle clips; no sampled/padded pixels can paint.
        helper.make_node("ReduceMax", ["input"], ["row_active_f"],
                         axes=[1, 3], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["col_active_f"],
                         axes=[1, 2], keepdims=1),
        helper.make_node("Greater", ["row_active_f", "zero_f"], ["row_active"]),
        helper.make_node("Greater", ["col_active_f", "zero_f"], ["col_active"]),
        helper.make_node("Where", ["row_active", "zero_u8", "sentinel"], ["row_clip"]),
        helper.make_node("Where", ["col_active", "zero_u8", "sentinel"], ["col_clip"]),
        helper.make_node("Where", ["use_row", "chosen_value", "row_clip"], ["row_value"]),
        helper.make_node("Where", ["use_row", "col_clip", "chosen_col"], ["col_value"]),
        # Free terminal expansion to the required ten boolean planes.
        helper.make_node("Sub", ["colour_ids", "col_value"], ["target"]),
        helper.make_node("Equal", ["row_value", "target"], ["output"]),
    ]

    initializers = [
        init(signs, "signs"),
        init(np.ones((30,), dtype=np.float32), "ones"),
        init(np.array(0.0, dtype=np.float32), "zero_f"),
        init(np.array([[1], [2], [4], [8]], dtype=np.int8), "bit_value"),
        init(np.array([16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 16, 16, 16, 16, 16], dtype=np.uint8),
             "code_map"),
        # The compact states use 16 as sentinel; the final channel IDs retain
        # the pin's native 0..9 arithmetic so 16 never aliases channel zero.
        init(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1),
             "colour_ids"),
        init(np.array(0, dtype=np.uint8), "zero_u8"),
        init(np.array(16, dtype=np.uint8), "sentinel"),
        init(np.array([2], dtype=np.int64), "squeeze_axis"),
    ]
    graph = helper.make_graph(
        nodes,
        "task359_signed_bit_modal",
        [value("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [value("output", TensorProto.BOOL, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT)


if __name__ == "__main__":
    main()
