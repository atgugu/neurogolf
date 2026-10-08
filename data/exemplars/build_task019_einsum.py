"""Task019 Einsum attempt — fuse tile+stencil via contraction (fallback if fails)."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neurogolf.builders.common import bool_output, fp32_input, make_model

I = TensorProto


def const(name, arr, dtype):
    return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name)


selector = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
selector[0, 8, 0, 0] = 255

# row/col index grids for 12x12 output
row_grid = np.arange(12, dtype=np.int32).reshape(1, 1, 12, 1)
col_grid = np.arange(12, dtype=np.int32).reshape(1, 1, 1, 12)

inits = [
    const("arange12", np.arange(12), np.int32),
    const("chan_selector", selector, np.uint8),
    const("zero_u8", np.uint8(0), np.uint8),
    const("ten_u8", np.uint8(10), np.uint8),
    const("row_grid", row_grid, np.int32),
    const("col_grid", col_grid, np.int32),
    const(
        "diag_kernel",
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8).reshape(1, 1, 3, 3),
        np.uint8,
    ),
    const("one_scale", 1.0, np.float32),
    const("sat_scale", 0.001, np.float32),
    const("q_zp", np.uint8(0), np.uint8),
    const("ch0_starts", [0, 0, 0, 0], np.int64),
    const("ch0_ends", [1, 1, 6, 6], np.int64),
    const("pad_to_30", [0, 0, 0, 0, 0, 0, 18, 18], np.int64),
    const("axes_23", [2, 3], np.int64),
    const("axes_ch", [1], np.int64),
]

nodes = [
    helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends"], ["ch0_f"]),
    helper.make_node("Cast", ["ch0_f"], ["ch0_u8"], to=I.UINT8),
    helper.make_node("ReduceMax", ["ch0_u8"], ["row_present"], axes=[3], keepdims=1),
    helper.make_node("ReduceMax", ["ch0_u8"], ["col_present"], axes=[2], keepdims=1),
    helper.make_node("Cast", ["row_present"], ["row_i32"], to=I.INT32),
    helper.make_node("Cast", ["col_present"], ["col_i32"], to=I.INT32),
    helper.make_node("ReduceSum", ["row_i32"], ["h"], keepdims=0),
    helper.make_node("ReduceSum", ["col_i32"], ["w"], keepdims=0),
    helper.make_node("Add", ["h", "h"], ["two_h"]),
    helper.make_node("Add", ["w", "w"], ["two_w"]),
    helper.make_node("Equal", ["ch0_u8", "zero_u8"], ["colored_seed"]),
    helper.make_node("Mod", ["row_grid", "h"], ["row_src"]),
    helper.make_node("Mod", ["col_grid", "w"], ["col_src"]),
    helper.make_node("Less", ["row_grid", "two_h"], ["valid_r"]),
    helper.make_node("Less", ["col_grid", "two_w"], ["valid_c"]),
    helper.make_node("Gather", ["colored_seed", "row_src"], ["colored_r"], axis=2),
    helper.make_node("Gather", ["colored_r", "col_src"], ["is_colored_raw"], axis=3),
    helper.make_node("And", ["valid_r", "valid_c"], ["in_valid"]),
    helper.make_node("And", ["is_colored_raw", "in_valid"], ["is_colored"]),
    helper.make_node("Cast", ["is_colored"], ["colored_t"], to=I.UINT8),
    helper.make_node(
        "QLinearConv",
        [
            "colored_t",
            "one_scale",
            "q_zp",
            "diag_kernel",
            "one_scale",
            "q_zp",
            "sat_scale",
            "q_zp",
        ],
        ["diag_sat"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    ),
    helper.make_node("ReduceMax", ["input"], ["ch_present_f"], axes=[2, 3], keepdims=0),
    helper.make_node(
        "ArgMax",
        ["ch_present_f"],
        ["C_i64"],
        axis=1,
        keepdims=1,
        select_last_index=1,
    ),
    helper.make_node("Cast", ["C_i64"], ["C_scalar"], to=I.UINT8),
    helper.make_node("Where", ["is_colored", "C_scalar", "diag_sat"], ["inside_idx"]),
    helper.make_node("Where", ["in_valid", "inside_idx", "ten_u8"], ["final_idx"]),
    helper.make_node("Pad", ["final_idx", "pad_to_30", "ten_u8"], ["idx_pad"], mode="constant"),
    helper.make_node("Equal", ["idx_pad", "chan_selector"], ["output"]),
]

model = make_model(nodes, [fp32_input()], [bool_output()], inits, name="task019_einsum")
onnx.checker.check_model(model, full_check=True)
out = "candidates/task019_einsum.onnx"
onnx.save(model, out)
print(f"saved {out}", len(model.SerializeToString()), "bytes")