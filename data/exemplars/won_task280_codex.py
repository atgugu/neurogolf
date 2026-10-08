#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


OUT = Path("task280.onnx")
PIN = Path("pin_current.onnx")

DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT64: 8,
    TensorProto.INT32: 4,
    TensorProto.UINT32: 4,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.BOOL: 1,
}


def init(name: str, array) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name)


def cast(name: str, x: str, to: int) -> onnx.NodeProto:
    return helper.make_node("Cast", [x], [name], name=name, to=to)


def node(op: str, name: str, inputs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, [name], name=name, **attrs)


def replace_or_add_init(model: onnx.ModelProto, tensor: onnx.TensorProto) -> None:
    for i, old in enumerate(model.graph.initializer):
        if old.name == tensor.name:
            model.graph.initializer[i].CopyFrom(tensor)
            return
    model.graph.initializer.append(tensor)


def prune_initializers(model: onnx.ModelProto) -> None:
    used = {name for n in model.graph.node for name in n.input if name}
    kept = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def validate(model: onnx.ModelProto, output_type: int = TensorProto.FLOAT) -> onnx.ModelProto:
    model.ir_version = 10
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = 14
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out = inferred.graph.output[0]
    out.name = "output"
    out.type.tensor_type.elem_type = output_type
    dims = out.type.tensor_type.shape.dim
    del dims[:]
    for d in [1, 10, 30, 30]:
        dims.add().dim_value = d
    return inferred


def audit_cost(model: onnx.ModelProto) -> tuple[int, int, int]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    params = sum(math.prod(t.dims) for t in inferred.graph.initializer)
    value_types = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        tt = v.type.tensor_type
        shape = [d.dim_value for d in tt.shape.dim]
        value_types[v.name] = (tt.elem_type, shape)
    outputs = {o.name for o in inferred.graph.output}
    mem = 0
    for n in inferred.graph.node:
        for out in n.output:
            if out in outputs:
                continue
            elem_type, shape = value_types[out]
            mem += math.prod(shape) * DTYPE_BYTES[elem_type]
    return int(params + mem), int(params), int(mem)


def assert_pin_contract(model: onnx.ModelProto) -> None:
    out = model.graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    if out.name != "output" or dims != [1, 10, 30, 30]:
        raise AssertionError(f"bad output contract: {out.name} {dims}")
    for n in model.graph.node:
        if n.output and n.output[0] == "output":
            if n.op_type != "Einsum":
                raise AssertionError(f"terminal changed to {n.op_type}")
            if "spatial_select" not in n.input:
                raise AssertionError("pin spatial_select terminal was not preserved")
            return
    raise AssertionError("output node not found")


def assert_no_full_3030_intermediate(model: onnx.ModelProto) -> None:
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    value_types = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        tt = v.type.tensor_type
        value_types[v.name] = [d.dim_value for d in tt.shape.dim]
    outputs = {o.name for o in inferred.graph.output}
    offenders = []
    for n in inferred.graph.node:
        for out in n.output:
            if out in outputs:
                continue
            shape = value_types.get(out, [])
            if shape.count(30) >= 2:
                offenders.append((out, n.op_type, shape))
    if offenders:
        raise AssertionError(f"full 30x30 intermediate(s): {offenders[:5]}")


def replace_index_prefix_with_i8(model: onnx.ModelProto) -> None:
    """Keep the pin terminal, but compute prefix coordinate walks as int8.

    GatherND itself still requires int64 indices in ORT 1.24, so each final
    index tensor is cast back to int64 immediately before its GatherND.
    """

    for name, arr in {
        "zero2_col_i8": np.zeros((2, 1), dtype=np.int8),
        "three2_col_i8": np.full((2, 1), 3, dtype=np.int8),
        "off_p0_p1_i8": np.array([0, 0, 0, 1], dtype=np.int8),
        "off_p0_m1_i8": np.array([0, 0, 0, -1], dtype=np.int8),
        "off_p1_p0_i8": np.array([0, 0, 1, 0], dtype=np.int8),
        "off_m1_p0_i8": np.array([0, 0, -1, 0], dtype=np.int8),
    }.items():
        replace_or_add_init(model, init(name, arr))

    nodes = list(model.graph.node)
    start = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "src_rows64")
    end = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "in5")

    repl: list[onnx.NodeProto] = [
        cast("src_rows8", "rows_f_mom", TensorProto.INT8),
        cast("src_cols8", "cols_f_mom", TensorProto.INT8),
        node("Unsqueeze", "idx_rows", ["src_rows8", "axis1"]),
        node("Unsqueeze", "idx_cols", ["src_cols8", "axis1"]),
        helper.make_node(
            "Concat",
            ["zero2_col_i8", "three2_col_i8", "idx_rows", "idx_cols"],
            ["base_idx8"],
            name="base_idx8",
            axis=1,
        ),
    ]

    for tag, off in [
        ("g_right1", "off_p0_p1_i8"),
        ("g_left1", "off_p0_m1_i8"),
        ("g_down1", "off_p1_p0_i8"),
        ("g_up1", "off_m1_p0_i8"),
    ]:
        repl.extend(
            [
                node("Add", f"{tag}_idx8", ["base_idx8", off]),
                cast(f"{tag}_idx", f"{tag}_idx8", TensorProto.INT64),
                node("GatherND", f"{tag}_f", ["input", f"{tag}_idx"]),
                cast(tag, f"{tag}_f", TensorProto.BOOL),
            ]
        )

    repl.extend(
        [
            node("Unsqueeze", "g_right_col", ["g_right1", "axis1"]),
            node("Unsqueeze", "g_left_col", ["g_left1", "axis1"]),
            node("Unsqueeze", "g_down_col", ["g_down1", "axis1"]),
            node("Unsqueeze", "g_up_col", ["g_up1", "axis1"]),
            node("Xor", "h_dir", ["g_left_col", "g_right_col"]),
            node("Xor", "v_dir", ["g_up_col", "g_down_col"]),
            cast("g_right8", "g_right_col", TensorProto.INT8),
            cast("g_left8", "g_left_col", TensorProto.INT8),
            cast("g_down8", "g_down_col", TensorProto.INT8),
            cast("g_up8", "g_up_col", TensorProto.INT8),
            node("Sub", "step_c_col", ["g_right8", "g_left8"]),
            node("Sub", "step_r_col", ["g_down8", "g_up8"]),
            node("Add", "in1_rows", ["idx_rows", "step_r_col"]),
            node("Add", "in1_cols", ["idx_cols", "step_c_col"]),
            node("Add", "in2_rows", ["in1_rows", "step_r_col"]),
            node("Add", "in2_cols", ["in1_cols", "step_c_col"]),
        ]
    )

    for n in range(2, 6):
        repl.extend(
            [
                helper.make_node(
                    "Concat",
                    ["zero2_col_i8", "three2_col_i8", f"in{n}_rows", f"in{n}_cols"],
                    [f"in{n}_idx8"],
                    name=f"in{n}_idx8",
                    axis=1,
                ),
                cast(f"in{n}_idx", f"in{n}_idx8", TensorProto.INT64),
                node("GatherND", f"in{n}_f", ["input", f"in{n}_idx"]),
                cast(f"in{n}", f"in{n}_f", TensorProto.BOOL),
            ]
        )
        if n < 5:
            repl.extend(
                [
                    node("Add", f"in{n + 1}_rows", [f"in{n}_rows", "step_r_col"]),
                    node("Add", f"in{n + 1}_cols", [f"in{n}_cols", "step_c_col"]),
                ]
            )

    del nodes[start : end + 1]
    nodes[start:start] = repl
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    prune_initializers(model)


def replace_index_prefix_with_batched_i8(model: onnx.ModelProto) -> None:
    """Use the input's singleton batch as GatherND's batch dimension.

    The pin has eight dynamic GatherND index tensors of shape [2, 4]: the
    first coordinate is invariably zero.  ``batch_dims=1`` removes that
    coordinate without changing a single gathered value: indices become
    [1, 2, 3] for (channel, row, column).  All geometry remains in the
    certified 20x20 source box; the original terminal still renders 30x30.
    """

    for name, arr in {
        "three2_b_i8": np.full((1, 2, 1), 3, dtype=np.int8),
        "off_p0_p1_b_i8": np.array([[[0, 0, 1]]], dtype=np.int8),
        "off_p0_m1_b_i8": np.array([[[0, 0, -1]]], dtype=np.int8),
        "off_p1_p0_b_i8": np.array([[[0, 1, 0]]], dtype=np.int8),
        "off_m1_p0_b_i8": np.array([[[0, -1, 0]]], dtype=np.int8),
        "axes02": np.array([0, 2], dtype=np.int64),
        "axis0": np.array([0], dtype=np.int64),
        "axis2": np.array([2], dtype=np.int64),
    }.items():
        replace_or_add_init(model, init(name, arr))

    nodes = list(model.graph.node)
    start = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "src_rows64")
    end = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "in5")

    repl: list[onnx.NodeProto] = [
        cast("src_rows8", "rows_f_mom", TensorProto.INT8),
        cast("src_cols8", "cols_f_mom", TensorProto.INT8),
        # The [2,1] versions feed the 20-wide renderer tail.  The [1,2,1]
        # versions feed the batched gathers and retain only three indices.
        node("Unsqueeze", "idx_rows", ["src_rows8", "axis1"]),
        node("Unsqueeze", "idx_cols", ["src_cols8", "axis1"]),
        node("Unsqueeze", "idx_rows_b", ["src_rows8", "axes02"]),
        node("Unsqueeze", "idx_cols_b", ["src_cols8", "axes02"]),
        helper.make_node(
            "Concat",
            ["three2_b_i8", "idx_rows_b", "idx_cols_b"],
            ["base_idx8"],
            name="base_idx8",
            axis=2,
        ),
    ]

    for tag, off in [
        ("g_right1", "off_p0_p1_b_i8"),
        ("g_left1", "off_p0_m1_b_i8"),
        ("g_down1", "off_p1_p0_b_i8"),
        ("g_up1", "off_m1_p0_b_i8"),
    ]:
        repl.extend(
            [
                node("Add", f"{tag}_idx8", ["base_idx8", off]),
                cast(f"{tag}_idx", f"{tag}_idx8", TensorProto.INT64),
                helper.make_node(
                    "GatherND",
                    ["input", f"{tag}_idx"],
                    [f"{tag}_f"],
                    name=f"{tag}_f",
                    batch_dims=1,
                ),
                # The source is one-hot, so its green-channel values are
                # exactly 0/1; direct relowering avoids a bool->i8 staging
                # output while preserving the signed direction arithmetic.
                cast(f"{tag}_i8", f"{tag}_f", TensorProto.INT8),
                node("Unsqueeze", f"{tag}_col", [f"{tag}_i8", "axis2"]),
            ]
        )

    repl.extend(
        [
            node("Sub", "step_c_b", ["g_right1_col", "g_left1_col"]),
            node("Sub", "step_r_b", ["g_down1_col", "g_up1_col"]),
            node("Squeeze", "step_c_col", ["step_c_b", "axis0"]),
            node("Squeeze", "step_r_col", ["step_r_b", "axis0"]),
            node("Add", "in1_rows_b", ["idx_rows_b", "step_r_b"]),
            node("Add", "in1_cols_b", ["idx_cols_b", "step_c_b"]),
            node("Add", "in2_rows_b", ["in1_rows_b", "step_r_b"]),
            node("Add", "in2_cols_b", ["in1_cols_b", "step_c_b"]),
        ]
    )

    for n in range(2, 6):
        repl.extend(
            [
                helper.make_node(
                    "Concat",
                    ["three2_b_i8", f"in{n}_rows_b", f"in{n}_cols_b"],
                    [f"in{n}_idx8"],
                    name=f"in{n}_idx8",
                    axis=2,
                ),
                cast(f"in{n}_idx", f"in{n}_idx8", TensorProto.INT64),
                helper.make_node(
                    "GatherND",
                    ["input", f"in{n}_idx"],
                    [f"in{n}_f"],
                    name=f"in{n}_f",
                    batch_dims=1,
                ),
                # These are also literal 0/1 green samples.  Keeping the
                # contiguous-run arithmetic in u8 is value-exact.
                cast(f"in{n}_u8", f"in{n}_f", TensorProto.UINT8),
            ]
        )
        if n < 5:
            repl.extend(
                [
                    node("Add", f"in{n + 1}_rows_b", [f"in{n}_rows_b", "step_r_b"]),
                    node("Add", f"in{n + 1}_cols_b", [f"in{n}_cols_b", "step_c_b"]),
                ]
            )

    del nodes[start : end + 1]
    nodes[start:start] = repl
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    prune_initializers(model)


def replace_tail_with_batched_golf(model: onnx.ModelProto) -> None:
    """Keep the pin renderer but delete value-redundant tail staging.

    A zero signed step denotes the non-shooting axis.  ``signed < 0`` is
    therefore precisely the old ``(signed <= 0) AND direction_present``;
    at the excluded source cell the existing band/equality terms are already
    true.  This removes the two direction masks and two 20-wide comparisons.
    """

    nodes = list(model.graph.node)
    start = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "contig3")
    end = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "v_range")
    repl: list[onnx.NodeProto] = [
        node("Mul", "contig3_u8", ["in2_u8", "in3_u8"]),
        node("Mul", "contig4_u8", ["contig3_u8", "in4_u8"]),
        node("Mul", "contig5_u8", ["contig4_u8", "in5_u8"]),
        node("Add", "count_u8_23", ["in2_u8", "contig3_u8"]),
        node("Add", "count_u8_234", ["count_u8_23", "contig4_u8"]),
        node("Add", "count_u8_2345", ["count_u8_234", "contig5_u8"]),
        node("Add", "thickness_b_u8", ["count_u8_2345", "two_u8_tail"]),
        cast("thickness_b_i8", "thickness_b_u8", TensorProto.INT8),
        helper.make_node("Transpose", ["thickness_b_i8"], ["thick_col"], name="thick_col", perm=[1, 0]),
        node("Sub", "row_diff", ["coords20_i8_tail", "idx_rows"]),
        node("Abs", "row_abs", ["row_diff"]),
        node("Less", "row_band", ["row_abs", "thick_col"]),
        node("Equal", "row_eq", ["coords20_i8_tail", "idx_rows"]),
        node("Sub", "col_diff", ["coords20_i8_tail", "idx_cols"]),
        node("Abs", "col_abs", ["col_diff"]),
        node("Less", "col_band", ["col_abs", "thick_col"]),
        node("Equal", "col_eq", ["coords20_i8_tail", "idx_cols"]),
        node("Mul", "h_signed", ["col_diff", "step_c_col"]),
        node("Mul", "v_signed", ["row_diff", "step_r_col"]),
        node("Less", "h_range", ["h_signed", "zero_i8_tail"]),
        node("Less", "v_range", ["v_signed", "zero_i8_tail"]),
    ]
    del nodes[start : end + 1]
    nodes[start:start] = repl
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    prune_initializers(model)


def build_cast_golf() -> onnx.ModelProto:
    if not PIN.exists():
        raise FileNotFoundError(f"missing local floor model: {PIN}")
    model = onnx.load(PIN)
    assert_pin_contract(model)

    # The fleet plan asks for bool/i8 terminal constants. That is not a legal
    # ONNX/ORT surface with the fp32 input Einsum, so the terminal fp32 factors
    # stay pinned and only the dynamic prefix index arithmetic is relowered.
    replace_index_prefix_with_i8(model)
    model = validate(model)
    cost, params, mem = audit_cost(model)
    print(f"local audit: cost={cost} params={params} mem={mem}")
    return model


def build_batch_golf() -> onnx.ModelProto:
    if not PIN.exists():
        raise FileNotFoundError(f"missing local floor model: {PIN}")
    model = onnx.load(PIN)
    assert_pin_contract(model)
    replace_index_prefix_with_batched_i8(model)
    replace_tail_with_batched_golf(model)
    model = validate(model)
    cost, params, mem = audit_cost(model)
    print(f"local audit: cost={cost} params={params} mem={mem}")
    return model


def replace_terminal_with_integer_probe(model: onnx.ModelProto) -> None:
    """Measured integer terminal probe.

    This is the binding family instantiated on the att12 prefix: bool row/column
    terms are cast to int8, projected to 30-wide with MatMulInteger, and a final
    MatMulInteger expands compact state to the full output. It deliberately does
    not claim correctness without the old fp32 input-copy term; the measured
    scorer line documents why this family does not close.
    """

    init_map = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    spatial_select_i8 = init_map["spatial_select"].astype(np.int8)
    channel_coeff_i8 = np.zeros((1, 10, 5, 1), dtype=np.int8)
    # Coefficients induced by channel_tf when the source cell is black. The old
    # terminal also depends on the actual fp32 input color; MatMulInteger cannot
    # consume that float input without a full charged cast.
    channel_coeff_i8[0, 0, :, 0] = [1, -1, -1, 0, 0]
    channel_coeff_i8[0, 2, :, 0] = [0, 0, 0, 2, 2]
    channel_coeff_i8[0, 3, :, 0] = [0, 1, 1, -2, -2]

    for name, arr in {
        "spatial_select_i8": spatial_select_i8,
        "channel_coeff_i8": channel_coeff_i8,
        "shape_colchan_i8": np.array([1, 1, 5, 30], dtype=np.int64),
    }.items():
        replace_or_add_init(model, init(name, arr))

    nodes = list(model.graph.node)
    start = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "row_terms")
    end = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == "output")
    repl = [
        cast("row_terms_i8", "row_terms20", TensorProto.INT8),
        cast("col_terms_i8", "col_terms20", TensorProto.INT8),
        node("MatMulInteger", "row30_i32", ["row_terms_i8", "spatial_select_i8"]),
        node("MatMulInteger", "col30_i32", ["col_terms_i8", "spatial_select_i8"]),
        cast("row30_i8", "row30_i32", TensorProto.INT8),
        cast("col30_i8", "col30_i32", TensorProto.INT8),
        helper.make_node("Transpose", ["row30_i8"], ["rowA_i8"], name="rowA_i8", perm=[1, 0]),
        node("Reshape", "col30_11530_i8", ["col30_i8", "shape_colchan_i8"]),
        node("Mul", "colB_i8", ["channel_coeff_i8", "col30_11530_i8"]),
        node("MatMulInteger", "output", ["rowA_i8", "colB_i8"]),
    ]
    del nodes[start : end + 1]
    nodes[start:start] = repl
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    out = model.graph.output[0]
    out.name = "output"
    out.type.tensor_type.elem_type = TensorProto.INT32
    prune_initializers(model)


def build_integer_terminal() -> onnx.ModelProto:
    if not PIN.exists():
        raise FileNotFoundError(f"missing local floor model: {PIN}")
    model = onnx.load(PIN)
    assert_pin_contract(model)
    replace_index_prefix_with_i8(model)
    replace_terminal_with_integer_probe(model)
    model = validate(model, output_type=TensorProto.INT32)
    assert_no_full_3030_intermediate(model)
    cost, params, mem = audit_cost(model)
    print(f"local audit: cost={cost} params={params} mem={mem}")
    return model


def build_pin() -> onnx.ModelProto:
    model = onnx.load(PIN)
    assert_pin_contract(model)
    return validate(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["batch-golf", "cast_golf", "integer-terminal", "pin"], default="batch-golf")
    args = ap.parse_args()

    if args.variant == "batch-golf":
        model = build_batch_golf()
    elif args.variant == "cast_golf":
        model = build_cast_golf()
    elif args.variant == "integer-terminal":
        model = build_integer_terminal()
    else:
        model = build_pin()
    onnx.save(model, OUT)
    print(f"saved {OUT} ({args.variant})")


if __name__ == "__main__":
    main()
