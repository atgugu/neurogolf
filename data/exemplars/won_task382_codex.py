"""task382 attempt 9: rank-factor terminal renderer.

True rule: cyan edge markers sweep inward, and each red side marker crossed
increases the cyan shift by one cell away from the red side; flips/gravity only
choose which two edges carry those vectors.

Priced before build:

A. Binding rank-factor terminal-Einsum family, with M_proof=20 vector core:
   - f32 row/col signed profiles/slices: 2 x [1,30] + 2 x [1,20] = 400 B
   - u8/bool row/col endpoint/profile masks and small selectors: about 1200 B
   - fp16 terminal factor banks: 2 x [1,7,20] = 560 B
   - params: profile weights, shift gather indices, coeffs, shared 20->30 extender ~= 920
   Paper total: about 3500 B, below the 3610 registration bar.  This deviates
   from the audited six-term budget because the source generator can produce
   four red markers, which requires five cyan sweep segments plus red plus
   background.

B. Prior scalar mask-bank/Pad tail:
   - u8 [1,20,20] x3 = 1200 B
   - charged scalar Pad [1,30,30] = 900 B
   - profile/cumsum/vector tail and params ~= 2095 B
   Measured total: 4195 B, correct but cost-rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


OUT = Path("task382.onnx")
T = 7
N = 20


def tensor(name: str, arr) -> onnx.TensorProto:
    arr = np.asarray(arr)
    return helper.make_tensor(
        name,
        helper.np_dtype_to_tensor_dtype(arr.dtype),
        arr.shape,
        arr.reshape(-1).tolist(),
    )


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def n(op: str, inputs: list[str], output: str, **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, [output], name=output, **attrs)


def select_bool(nodes: list[onnx.NodeProto], cond: str, not_cond: str,
                a: str, b: str, out: str) -> None:
    nodes.extend(
        [
            n("And", [cond, a], out + "_a"),
            n("And", [not_cond, b], out + "_b"),
            n("Or", [out + "_a", out + "_b"], out),
        ]
    )


def build() -> None:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def add(name: str, arr) -> str:
        inits.append(tensor(name, arr))
        return name

    # Signed profile: inside-background rows/cols are small positive, red rows
    # are negative, and cyan rows are large positive.
    prof_w = np.zeros(10, dtype=np.float32)
    prof_w[0] = 0.01
    prof_w[2] = -1.0
    prof_w[8] = 1.0
    add("prof_w", prof_w)

    add("zero_f", np.array(0.0, dtype=np.float32))
    add("half_f", np.array(0.5, dtype=np.float32))
    add("one5_h", np.array(1.5, dtype=np.float16))
    add("zero_u8", np.array(0, dtype=np.uint8))
    add("axis1", np.array([1], dtype=np.int64))
    add("axis1_scalar", np.array(1, dtype=np.int64))
    add("starts0", np.array([0], dtype=np.int64))
    add("ends1", np.array([1], dtype=np.int64))
    add("ends20", np.array([N], dtype=np.int64))
    add("steps1", np.array([1], dtype=np.int64))
    add("unsq1", np.array([1], dtype=np.int64))

    ks = np.arange(5, dtype=np.float16).reshape(1, 5, 1)
    add("ks_h", ks)

    base = np.arange(N, dtype=np.int64)
    idx_pos = np.stack([np.clip(base - k, 0, N - 1) for k in range(5)], axis=0)
    idx_neg = np.stack([np.clip(base + k, 0, N - 1) for k in range(5)], axis=0)
    add("idx_pos", idx_pos)
    add("idx_neg", idx_neg)

    coeff = np.zeros((1, T, 10), dtype=np.float16)
    coeff[0, 0, 0] = 1.0
    coeff[0, 1, 0] = -1.0
    coeff[0, 1, 2] = 1.0
    coeff[0, 2:, 0] = -1.0
    coeff[0, 2:, 8] = 1.0
    add("coeff", coeff)
    pad20_30 = np.zeros((N, 30), dtype=np.float16)
    pad20_30[np.arange(N), np.arange(N)] = 1.0
    add("pad20_30", pad20_30)

    nodes.extend(
        [
            n("Einsum", ["input", "prof_w"], "row_prof", equation="nchw,c->nh"),
            n("Einsum", ["input", "prof_w"], "col_prof", equation="nchw,c->nw"),
            n("Slice", ["row_prof", "starts0", "ends20", "axis1", "steps1"], "row_prof20"),
            n("Slice", ["col_prof", "starts0", "ends20", "axis1", "steps1"], "col_prof20"),
            n("Cast", ["row_prof20"], "row_in", to=TensorProto.BOOL),
            n("Cast", ["col_prof20"], "col_in", to=TensorProto.BOOL),
            n("Less", ["row_prof20", "zero_f"], "row_red"),
            n("Less", ["col_prof20", "zero_f"], "col_red"),
            n("Greater", ["row_prof20", "half_f"], "row_cyan"),
            n("Greater", ["col_prof20", "half_f"], "col_cyan"),
            n("Cast", ["row_in"], "row_in_u8", to=TensorProto.UINT8),
            n("Cast", ["col_in"], "col_in_u8", to=TensorProto.UINT8),
            n("Cast", ["row_red"], "row_red_u8", to=TensorProto.UINT8),
            n("Cast", ["col_red"], "col_red_u8", to=TensorProto.UINT8),
            n("Cast", ["row_cyan"], "row_cyan_u8", to=TensorProto.UINT8),
            n("Cast", ["col_cyan"], "col_cyan_u8", to=TensorProto.UINT8),
            n("Cast", ["row_cyan"], "row_cyan_h", to=TensorProto.FLOAT16),
            n("ReduceSum", ["row_cyan_h", "axis1"], "row_cyan_count_h", keepdims=0),
            n("Less", ["row_cyan_count_h", "one5_h"], "horizontal"),
            n("Not", ["horizontal"], "not_horizontal"),
        ]
    )

    # Canonical flow/source vectors.  Use uint8 Where because bool-branch Where
    # is not implemented in ORT 1.24.
    nodes.extend(
        [
            n("Where", ["horizontal", "row_red_u8", "col_red_u8"], "flow_red_u8"),
            n("Where", ["horizontal", "row_in_u8", "col_in_u8"], "flow_in_u8"),
            n("Where", ["horizontal", "col_in_u8", "row_in_u8"], "source_in_u8"),
            n("Where", ["horizontal", "col_cyan_u8", "row_cyan_u8"], "source_axis_u8"),
        ]
    )

    nodes.extend(
        [
            n("Slice", ["row_cyan_u8", "starts0", "ends1", "axis1", "steps1"], "row_cyan0_u8"),
            n("Slice", ["col_cyan_u8", "starts0", "ends1", "axis1", "steps1"], "col_cyan0_u8"),
            n("Where", ["horizontal", "row_cyan0_u8", "col_cyan0_u8"], "low_source_u8"),
            n("Greater", ["low_source_u8", "zero_u8"], "low_source"),
        ]
    )

    nodes.extend(
        [
            n("Slice", ["col_red_u8", "starts0", "ends1", "axis1", "steps1"], "col_red0_u8"),
            n("Slice", ["row_red_u8", "starts0", "ends1", "axis1", "steps1"], "row_red0_u8"),
            n("Where", ["horizontal", "col_red0_u8", "row_red0_u8"], "low_red_u8"),
            n("Greater", ["low_red_u8", "zero_u8"], "low_red"),
        ]
    )

    nodes.extend(
        [
            n("Cast", ["flow_red_u8"], "flow_red_h", to=TensorProto.FLOAT16),
            n("CumSum", ["flow_red_h", "axis1_scalar"], "shift_fwd"),
            n("CumSum", ["flow_red_h", "axis1_scalar"], "shift_rev", reverse=1),
            n("Where", ["low_source", "shift_fwd", "shift_rev"], "shift_count"),
            n("Equal", ["shift_count", "ks_h"], "flow_eq"),
            n("Where", ["flow_eq", "flow_in_u8", "zero_u8"], "flow_terms"),
            n("Gather", ["source_axis_u8", "idx_pos"], "src_pos", axis=1),
            n("Gather", ["source_axis_u8", "idx_neg"], "src_neg", axis=1),
            n("Where", ["low_red", "src_pos", "src_neg"], "source_shift_raw"),
            n("Mul", ["source_shift_raw", "source_in_u8"], "source_shift"),
            n("Where", ["horizontal", "flow_terms", "source_shift"], "cyan_rows"),
            n("Where", ["horizontal", "source_shift", "flow_terms"], "cyan_cols"),
            n("Unsqueeze", ["row_in_u8", "unsq1"], "row_bg_u"),
            n("Unsqueeze", ["row_red_u8", "unsq1"], "row_red_u"),
            n("Unsqueeze", ["col_in_u8", "unsq1"], "col_bg_u"),
            n("Unsqueeze", ["col_red_u8", "unsq1"], "col_red_u"),
            n("Concat", ["row_bg_u", "row_red_u", "cyan_rows"], "row_terms20_u8", axis=1),
            n("Concat", ["col_bg_u", "col_red_u", "cyan_cols"], "col_terms20_u8", axis=1),
            n("Cast", ["row_terms20_u8"], "row_terms", to=TensorProto.FLOAT16),
            n("Cast", ["col_terms20_u8"], "col_terms", to=TensorProto.FLOAT16),
            n(
                "Einsum",
                ["coeff", "row_terms", "col_terms", "pad20_30", "pad20_30"],
                "output",
                equation="btc,btr,bts,ru,sv->bcuv",
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task382_rank_factor",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
