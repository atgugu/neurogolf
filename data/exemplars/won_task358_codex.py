#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def tensor(name, arr, dtype=None):
    arr = np.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return numpy_helper.from_array(arr, name)


def node(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


def build(path="task358.onnx"):
    init = []
    nodes = []

    def c(name, arr, dtype=None):
        t = tensor(name, arr, dtype)
        init.append(t)
        return name

    c("axes_chw", np.array([1, 2, 3], np.int64))
    c("nz_weight", np.array([0] + [1] * 9, np.float32))
    c("color_weight", np.arange(10, dtype=np.float32))
    c("one_u8", np.array(1, np.uint8))
    c("zero_u8", np.array(0, np.uint8))
    c("depth30", np.array(30, np.int64))
    c("onehot_values", np.array([0.0, 1.0], np.float32))
    c("coords0", np.arange(30, dtype=np.uint8).reshape(1, 30))
    c("coords60", (np.arange(21, dtype=np.uint8) + np.uint8(60)).reshape(1, 21))
    c("pad_21_to_30", np.array([0, 0, 0, 9], np.int64))

    k = np.arange(10, dtype=np.float16)
    coeff = np.zeros((10, 5), dtype=np.float16)
    coeff[:, 0] = np.float16(1) - k * k
    coeff[:, 1] = np.float16(2) * k
    coeff[:, 2] = np.float16(2) * k
    coeff[:, 3] = np.float16(-1)
    coeff[:, 4] = np.float16(-1)
    c("coeff", coeff)
    c("one_n", np.ones((1,), dtype=np.float16))

    # Certified dimensions: height is width or width+1, and area = width * height.
    nodes += [
        node("ReduceSum", ["input", "axes_chw"], ["area"], keepdims=0),
        node("Sqrt", ["area"], ["width_f"]),
        node("Cast", ["width_f"], ["width_u8"], to=TensorProto.UINT8),
        node("Cast", ["area"], ["area_i32"], to=TensorProto.INT32),
        node("Cast", ["width_u8"], ["width_i32"], to=TensorProto.INT32),
        node("Mul", ["width_i32", "width_i32"], ["width_sq_i32"]),
        node("Greater", ["area_i32", "width_sq_i32"], ["height_extra_b"]),
        node("Where", ["height_extra_b", "one_u8", "zero_u8"], ["height_extra_u8"]),
        node("Add", ["width_u8", "height_extra_u8"], ["height_u8"]),
    ]

    # Row occupancy of non-background cells.  The column arm is decoded from the
    # center-row palette plus the observed orientation, avoiding a second color-vector
    # extraction.
    nodes += [
        node("Einsum", ["input", "nz_weight"], ["row_counts"], equation="nchw,c->nh"),
        node("Cast", ["row_counts"], ["row_counts_u8"], to=TensorProto.UINT8),
        node("ReduceMax", ["row_counts_u8"], ["period_raw"], axes=[1], keepdims=1),
        node("Max", ["period_raw", "one_u8"], ["period_u8"]),
        node("Min", ["row_counts_u8", "one_u8"], ["row_has_u8"]),
        node("ArgMax", ["row_has_u8"], ["r_start_i64"], axis=1, keepdims=1),
        node("Cast", ["r_start_i64"], ["r_start_u8"], to=TensorProto.UINT8),
        node("ArgMax", ["row_counts"], ["row_idx_i64"], axis=1, keepdims=0),
        node("Cast", ["row_idx_i64"], ["row_idx_u8"], to=TensorProto.UINT8),
    ]

    # Extract the visible center-row color sequence as scalar color IDs.
    nodes += [
        node("OneHot", ["row_idx_i64", "depth30", "onehot_values"], ["row_sel"], axis=-1),
        node("Einsum", ["input", "color_weight", "row_sel"], ["row_ids_f"], equation="nchw,c,nh->nw"),
        node("Cast", ["row_ids_f"], ["row_ids_u8"], to=TensorProto.UINT8),
        node("Min", ["row_ids_u8", "one_u8"], ["row_nonzero_u8"]),
        node("ArgMax", ["row_nonzero_u8"], ["row_start_i64"], axis=1, keepdims=1),
        node("Cast", ["row_start_i64"], ["row_start_u8"], to=TensorProto.UINT8),
    ]

    # Repeat each observed sequence from its own visible start.  coords60 prevents uint8
    # underflow from corrupting modulo-3 phases because 60 is divisible by 3 and 4.
    nodes += [
        node("Sub", ["coords60", "row_start_u8"], ["row_delta"]),
        node("Mod", ["row_delta", "period_u8"], ["row_phase"], fmod=0),
        node("Add", ["row_start_u8", "row_phase"], ["row_source_u8"]),
        node("Cast", ["row_source_u8"], ["row_source_i32"], to=TensorProto.INT32),
        node("GatherElements", ["row_ids_u8", "row_source_i32"], ["row_colors21_u8"], axis=1),
        node("Pad", ["row_colors21_u8", "pad_21_to_30"], ["row_colors_u8"], mode="constant"),
        node("Sub", ["row_idx_u8", "r_start_u8"], ["r_offset_u8"]),
        node("Sub", ["period_u8", "one_u8"], ["period_m1_u8"]),
        node("Equal", ["r_offset_u8", "zero_u8"], ["offset_is_zero"]),
        node("Where", ["offset_is_zero", "period_m1_u8", "zero_u8"], ["sample_p_u8"]),
        node("Add", ["r_start_u8", "sample_p_u8"], ["sample_row_u8"]),
        node("Cast", ["sample_row_u8"], ["sample_row_i64"], to=TensorProto.INT64),
        node("OneHot", ["sample_row_i64", "depth30", "onehot_values"], ["sample_row_sel"], axis=-1),
        node("Einsum", ["input", "color_weight", "sample_row_sel"], ["actual_sample_f"], equation="nchw,c,nah->na"),
        node("Cast", ["actual_sample_f"], ["actual_sample_u8"], to=TensorProto.UINT8),
        node("Add", ["row_start_u8", "sample_p_u8"], ["sample_src_fwd_u8"]),
        node("Cast", ["sample_src_fwd_u8"], ["sample_src_fwd_i32"], to=TensorProto.INT32),
        node("GatherElements", ["row_ids_u8", "sample_src_fwd_i32"], ["sample_color_fwd"], axis=1),
        node("Equal", ["actual_sample_u8", "sample_color_fwd"], ["not_flipped"]),
        node("Sub", ["period_m1_u8", "r_offset_u8"], ["r_offset_rev_u8"]),
        node("Where", ["not_flipped", "r_offset_u8", "r_offset_rev_u8"], ["center_phase_u8"]),
        node("Add", ["row_start_u8", "center_phase_u8"], ["col_idx_u8"]),
        node("Sub", ["coords60", "r_start_u8"], ["col_delta"]),
        node("Mod", ["col_delta", "period_u8"], ["col_phase"], fmod=0),
        node("Sub", ["period_m1_u8", "col_phase"], ["col_phase_rev"]),
        node("Where", ["not_flipped", "col_phase", "col_phase_rev"], ["col_phase_oriented"]),
        node("Add", ["row_start_u8", "col_phase_oriented"], ["col_source_u8"]),
        node("Cast", ["col_source_u8"], ["col_source_i32"], to=TensorProto.INT32),
        node("GatherElements", ["row_ids_u8", "col_source_i32"], ["col_colors21_u8"], axis=1),
        node("Pad", ["col_colors21_u8", "pad_21_to_30"], ["col_colors_u8"], mode="constant"),
    ]

    nodes += [
        node("Less", ["coords0", "height_u8"], ["row_valid_b"]),
        node("Less", ["coords0", "width_u8"], ["col_valid_b"]),
        node("Equal", ["coords0", "row_idx_u8"], ["is_row_b"]),
        node("Equal", ["coords0", "col_idx_u8"], ["is_col_b"]),
        node("Cast", ["row_valid_b"], ["row_valid"], to=TensorProto.UINT8),
        node("Cast", ["col_valid_b"], ["col_valid"], to=TensorProto.UINT8),
        node("Cast", ["is_row_b"], ["is_row"], to=TensorProto.UINT8),
        node("Cast", ["is_col_b"], ["is_col"], to=TensorProto.UINT8),
        node("Sub", ["row_valid", "is_row"], ["row_not"]),
        node("Mul", ["row_colors_u8", "row_colors_u8"], ["row_colors_sq"]),
        node("Mul", ["col_colors_u8", "col_colors_u8"], ["col_colors_sq"]),
        node("Mul", ["row_colors_u8", "col_valid"], ["h_color"]),
        node("Mul", ["row_colors_sq", "col_valid"], ["h_color_sq"]),
        node("Mul", ["col_colors_u8", "row_not"], ["v_color"]),
        node("Mul", ["col_colors_sq", "row_not"], ["v_color_sq"]),
        node(
            "Concat",
            ["row_valid", "is_row", "v_color", "is_row", "v_color_sq"],
            ["row_terms_u8"],
            axis=0,
        ),
        node(
            "Concat",
            ["col_valid", "h_color", "is_col", "h_color_sq", "is_col"],
            ["col_terms_u8"],
            axis=0,
        ),
        node("Cast", ["row_terms_u8"], ["row_terms"], to=TensorProto.FLOAT16),
        node("Cast", ["col_terms_u8"], ["col_terms"], to=TensorProto.FLOAT16),
        node("Einsum", ["row_terms", "col_terms", "coeff", "one_n"], ["output"], equation="fr,fc,kf,n->nkrc"),
    ]

    graph = helper.make_graph(
        nodes,
        "task358_periodic_cross_logits",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        init,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)


if __name__ == "__main__":
    build()
