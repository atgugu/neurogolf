from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


OUT = Path(__file__).with_name("task213.onnx")


def T(name: str, dtype: int, dims: list[int], vals: list[int | float | bool]) -> onnx.TensorProto:
    return helper.make_tensor(name, dtype, dims, vals)


MODULUS = 104
OUTSIDE_SEQ = MODULUS


def build_lut() -> list[int]:
    """Maps sampled weighted stripe-code residues back to ARC color ids."""
    weights = {1: 1, 2: 4, 3: 10, 4: 11, 6: 13, 7: 19, 8: 21, 9: 25}
    lut = [0] * MODULUS
    for color, weight in weights.items():
        for visible_samples in range(3, 10):
            lut[(weight * visible_samples) % MODULUS] = color
    return lut


def quad_renderer_weights() -> list[int]:
    """Final ConvInteger weights: positive only when x == output channel."""
    vals = [0] * (10 * 3)
    for color in range(1, 10):
        base = color * 3
        vals[base + 0] = 2 * color
        vals[base + 1] = -1
        vals[base + 2] = 1 - color * color
    return vals


def build() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])

    sample_weight = [0.0] * 30
    for idx in (0, 1, 2, 6, 7, 8, 12, 13, 14):
        sample_weight[idx] = 1.0

    inits = [
        T("code_weight", TensorProto.FLOAT, [10], [0, 1, 4, 10, 11, 0, 13, 19, 21, 25]),
        T("sample_weight", TensorProto.FLOAT, [30], sample_weight),
        T("code_lut", TensorProto.UINT8, [MODULUS], build_lut()),
        T("quad_w", TensorProto.INT8, [10, 3, 1, 1], quad_renderer_weights()),
        T("mod_outside", TensorProto.UINT8, [], [OUTSIDE_SEQ]),
        T("one_u8", TensorProto.UINT8, [], [1]),
        T("axis1", TensorProto.INT64, [1], [1]),
        T("axes_h_grid", TensorProto.INT64, [2], [1, 3]),
        T("axes_v_grid", TensorProto.INT64, [2], [1, 2]),
        T("step3", TensorProto.INT64, [1], [3]),
        T("start0", TensorProto.INT64, [1], [0]),
        T("start2", TensorProto.INT64, [1], [2]),
        T("end0", TensorProto.INT64, [1], [21]),
        T("end1", TensorProto.INT64, [1], [22]),
        T("end2", TensorProto.INT64, [1], [23]),
        T("zero_u8", TensorProto.UINT8, [], [0]),
    ]

    n = helper.make_node
    nodes: list[onnx.NodeProto] = [
        n("Einsum", ["input", "code_weight", "sample_weight"], ["h_code"], equation="nchw,c,w->nh"),
        n("Einsum", ["input", "code_weight", "sample_weight"], ["v_code"], equation="nchw,c,h->nw"),
        n("Cast", ["h_code"], ["h_code_u8"], to=TensorProto.UINT8),
        n("Cast", ["v_code"], ["v_code_u8"], to=TensorProto.UINT8),
    ]

    def period(prefix: str, code_name: str) -> None:
        nodes.extend(
            [
                n("Slice", [code_name, "start0", "end0", "axis1", "step3"], [f"{prefix}_p0"]),
                n("Slice", [code_name, "axis1", "end1", "axis1", "step3"], [f"{prefix}_p1"]),
                n("Slice", [code_name, "start2", "end2", "axis1", "step3"], [f"{prefix}_p2"]),
                n("Max", [f"{prefix}_p0", f"{prefix}_p1"], [f"{prefix}_m01"]),
                n("Max", [f"{prefix}_m01", f"{prefix}_p2"], [f"{prefix}_selected_code"]),
                n("Greater", [f"{prefix}_selected_code", "zero_u8"], [f"{prefix}_pos_raw"]),
                n("Mod", [f"{prefix}_selected_code", "mod_outside"], [f"{prefix}_selected_residue"]),
                n("Cast", [f"{prefix}_selected_residue"], [f"{prefix}_selected_i32"], to=TensorProto.INT32),
                n("Gather", ["code_lut", f"{prefix}_selected_i32"], [f"{prefix}_ids_selected"], axis=0),
            ]
        )

    period("h", "h_code_u8")
    nodes.extend(
        [
            n("Slice", ["h_p0", "start2", "step3", "axis1"], ["h_slot_p0"]),
            n("Greater", ["h_slot_p0", "zero_u8"], ["h_valid_p0"]),
            n("Slice", ["h_p1", "start2", "step3", "axis1"], ["h_slot_p1"]),
            n("Greater", ["h_slot_p1", "zero_u8"], ["h_valid_p1"]),
            n("And", ["h_valid_p0", "h_valid_p1"], ["h_valid_pair"]),
            n("Not", ["h_valid_pair"], ["h_valid_no_pair"]),
            n("And", ["h_valid_no_pair", "h_pos_raw"], ["h_pos_selected"]),
        ]
    )
    period("v", "v_code_u8")
    nodes.extend(
        [
            n("And", ["h_valid_pair", "v_pos_raw"], ["v_pos_selected"]),
            n("Where", ["v_pos_selected", "mod_outside", "zero_u8"], ["h_sentinel"]),
            n("Where", ["h_pos_selected", "h_ids_selected", "h_sentinel"], ["h_seq_ids"]),
            n("Where", ["h_pos_selected", "mod_outside", "zero_u8"], ["v_sentinel"]),
            n("Where", ["v_pos_selected", "v_ids_selected", "v_sentinel"], ["v_seq_ids"]),
            n("Unsqueeze", ["h_seq_ids", "axes_h_grid"], ["h_seq_rows"]),
            n("Unsqueeze", ["v_seq_ids", "axes_v_grid"], ["v_seq_cols"]),
            n("Min", ["h_seq_rows", "v_seq_cols"], ["color_idx7"]),
            n("Mul", ["color_idx7", "color_idx7"], ["color_sq7"]),
            n("Min", ["color_idx7", "one_u8"], ["valid7"]),
            n("Concat", ["color_idx7", "color_sq7", "valid7"], ["quad_features"], axis=1),
            n("ConvInteger", ["quad_features", "quad_w"], ["output"], pads=[0, 0, 23, 23]),
        ]
    )

    graph = helper.make_graph(nodes, "task213_sample_code_lut_quad_terminal", [x], [y], inits)
    model = helper.make_model(graph, producer_name="codex_task213_sample_code", opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    OUT.write_bytes(build().SerializeToString())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
