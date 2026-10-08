#!/usr/bin/env python3
"""task105 folded exact row-hash shrink.

Starts from the guard-correct row-hash member and keeps the final
Where(paint_mask_30, color_two, input) renderer.  The shrink is graph surgery:
uint8 relative coordinates, exact row-vector sharing, cheaper selected-row bit
logic, Pad axes, and a small-bit veto fold that preserves the strict guard.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from hypo import make_constants


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "tmp_strict_smallveto.onnx"
OUT = ROOT / "task105.onnx"


def replace_init(model: onnx.ModelProto, name: str, arr) -> None:
    kept = [init for init in model.graph.initializer if init.name != name]
    kept.append(numpy_helper.from_array(np.asarray(arr), name))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def add_init(model: onnx.ModelProto, name: str, arr) -> None:
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(arr), name))


def set_nodes(model: onnx.ModelProto, nodes: list[onnx.NodeProto]) -> None:
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def patch_slice_no_axes(model: onnx.ModelProto) -> None:
    replace_init(model, "crop_fg_starts", np.array([0, 1, 1, 2], dtype=np.int64))
    replace_init(model, "crop_fg_ends", np.array([1, 2, 13, 11], dtype=np.int64))
    for node in model.graph.node:
        if node.op_type == "Slice" and node.output[0] == "fg":
            del node.input[:]
            node.input.extend(["input", "crop_fg_starts", "crop_fg_ends"])


def patch_pad_axes(model: onnx.ModelProto) -> None:
    replace_init(model, "pads_mask", np.array([1, 2, 17, 19], dtype=np.int64))
    add_init(model, "pad_axes", np.array([2, 3], dtype=np.int64))
    for node in model.graph.node:
        if node.op_type == "Pad" and node.output[0] == "paint_mask_30":
            del node.input[:]
            node.input.extend(["red_b", "pads_mask", "", "pad_axes"])


def patch_row_in_rect(model: onnx.ModelProto) -> None:
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.output and node.output[0] in {"row_ge_top", "row_le_bottom", "row_in_rect"}:
            continue
        nodes.append(node)
        if node.output and node.output[0] == "side_cols":
            nodes.append(helper.make_node("Or", ["is_h_border_row", "interior_rows_r"], ["row_in_rect"]))
    set_nodes(model, nodes)


def patch_interior_raw_xor(model: onnx.ModelProto) -> None:
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.output and node.output[0] in {
            "row_hash_le_left",
            "row_hash_side_only",
            "interior_row_any_raw_b",
        }:
            continue
        nodes.append(node)
        if node.output and node.output[0] == "row_hash_is_right":
            nodes.append(helper.make_node("Xor", ["row_any_b", "row_hash_is_right"], ["interior_row_any_raw_b"]))
    set_nodes(model, nodes)


def patch_gather_elements(model: onnx.ModelProto) -> None:
    nodes: list[onnx.NodeProto] = []
    for old in model.graph.node:
        node = copy.deepcopy(old)
        out = node.output[0] if node.output else ""
        if out in {"selected_row_hash_gather", "selected_row_hash"}:
            if out == "selected_row_hash_gather":
                nodes.append(helper.make_node("GatherElements", ["row_hash", "interior_min_row_i"], ["selected_row_hash"], axis=2))
            continue
        if out in {"interior_min_row_i", "interior_max_row_i"}:
            for attr in node.attribute:
                if attr.name == "keepdims":
                    attr.i = 1
        nodes.append(node)
    set_nodes(model, nodes)


def patch_i16_bitmask(model: onnx.ModelProto) -> None:
    add_init(model, "low_clear_i16", np.array(-4, dtype=np.int16))
    add_init(
        model,
        "interior_power_values_i16",
        np.array([4, 8, 16, 32, 64], dtype=np.int16).reshape(1, 1, 1, 5),
    )
    nodes: list[onnx.NodeProto] = []
    for old in model.graph.node:
        node = copy.deepcopy(old)
        out = node.output[0] if node.output else ""
        if out in {"selected_row_hash_i32", "right_bit_i32", "selected_low_small_bits_i32"}:
            continue
        if out == "right_bit":
            nodes.append(node)
            nodes.append(helper.make_node("Cast", ["right_bit"], ["right_bit_i16"], to=TensorProto.INT16))
            continue
        if out == "selected_row_hash":
            nodes.append(node)
            nodes.append(helper.make_node("Cast", ["selected_row_hash"], ["selected_row_hash_i16"], to=TensorProto.INT16))
            continue
        if out == "selected_low_bits_i32":
            node.input[0] = "selected_row_hash_i16"
            node.input[1] = "right_bit_i16"
            node.output[0] = "selected_low_bits_i16"
        elif out == "selected_interior_bits_i32":
            nodes.append(helper.make_node("BitwiseAnd", ["selected_low_bits_i16", "low_clear_i16"], ["selected_interior_bits_i16"]))
            continue
        elif out == "interior_power_match":
            node.input[0] = "selected_interior_bits_i16"
            node.input[1] = "interior_power_values_i16"
        else:
            for i, inp in enumerate(node.input):
                if inp == "selected_low_bits_i32":
                    node.input[i] = "selected_low_bits_i16"
                elif inp == "selected_interior_bits_i32":
                    node.input[i] = "selected_interior_bits_i16"
        nodes.append(node)
    set_nodes(model, nodes)


def patch_hline_mask(model: onnx.ModelProto) -> None:
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        out = node.output[0] if node.output else ""
        if out in {"line_row_active_f", "h_line_row"}:
            continue
        nodes.append(node)
        if out == "h_selected":
            nodes.append(helper.make_node("And", ["h_selected", "interior_row_any_b"], ["h_line_row"]))
    set_nodes(model, nodes)


def patch_rel_u8_coords(model: onnx.ModelProto) -> None:
    replace_init(model, "row_coords", np.arange(12, dtype=np.uint8).reshape(1, 1, 12, 1))
    replace_init(model, "col_coords", np.arange(9, dtype=np.uint8).reshape(1, 1, 1, 9))
    for name, value in (("two_u8", 2), ("six_u8", 6), ("seven_u8", 7), ("eight_u8", 8)):
        add_init(model, name, np.array(value, dtype=np.uint8))

    nodes: list[onnx.NodeProto] = []
    for old in model.graph.node:
        node = copy.deepcopy(old)
        out = node.output[0] if node.output else ""
        if out in {
            "right_minus_two_f",
            "small_bits_present",
        }:
            continue
        if out in {"top_f", "bottom_f", "interior_min_row_f", "interior_max_row_f", "interior_min_col_index_f"}:
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.UINT8
        if out == "prefer_v_by_len":
            node.op_type = "Less"
            del node.input[:]
            node.input.extend(["row_span0", "right_f"])
        for i, inp in enumerate(node.input):
            if out == "right_low_f":
                if inp == "nine_f":
                    node.input[i] = "seven_u8"
                elif inp == "eight_f":
                    node.input[i] = "six_u8"
            if out == "right_f" and inp == "ten_f":
                node.input[i] = "eight_u8"
            if out == "interior_min_col_f" and inp == "four_f":
                node.input[i] = "two_u8"
            if out == "is_left_col" and inp == "two_f":
                node.input[i] = "zero_u8"
            if out == "line_col_active_f" and inp in {"zero_f", "neg_one_f"}:
                node.input[i] = "zero_u8"
            if inp == "v_possible_base":
                node.input[i] = "interior_cols_same"
        nodes.append(node)
    set_nodes(model, nodes)


def patch_any_interior(model: onnx.ModelProto) -> None:
    nodes: list[onnx.NodeProto] = []
    for old in model.graph.node:
        node = copy.deepcopy(old)
        out = node.output[0] if node.output else ""
        if out in {"v_col_le", "v_possible_base", "any_interior_u8", "any_interior_b"}:
            continue
        for i, inp in enumerate(node.input):
            if inp == "v_possible_base":
                node.input[i] = "interior_cols_same"
        nodes.append(node)
        if out == "interior_min_row_f":
            nodes.append(helper.make_node("Greater", ["interior_min_row_f", "zero_u8"], ["any_interior_b"]))
    set_nodes(model, nodes)


def patch_fold_small_veto(model: onnx.ModelProto) -> None:
    """Fold `(low & -4 == power) & (low == low & -4)` into `low == power`."""
    nodes: list[onnx.NodeProto] = []
    for old in model.graph.node:
        node = copy.deepcopy(old)
        out = node.output[0] if node.output else ""
        if out in {"selected_interior_bits_i16", "no_small_bits", "v_possible_raw"}:
            continue
        if out == "interior_power_match":
            node.input[0] = "selected_low_bits_i16"
        elif out == "v_possible":
            node.input[0] = "interior_cols_same"
            node.input[1] = "any_interior_b"
        nodes.append(node)
    set_nodes(model, nodes)


def cleanup(model: onnx.ModelProto) -> onnx.ModelProto:
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            opset.version = max(opset.version, 18)
    used = {inp for node in model.graph.node for inp in node.input if inp}
    old_inits = list(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend([init for init in old_inits if init.name in used])
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    model.ir_version = 10
    return model


def cleanup_fresh(model: onnx.ModelProto) -> onnx.ModelProto:
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    model.ir_version = 10
    return model


def build() -> onnx.ModelProto:
    model = onnx.load(BASE)
    patch_slice_no_axes(model)
    patch_pad_axes(model)
    patch_row_in_rect(model)
    patch_interior_raw_xor(model)
    patch_gather_elements(model)
    patch_i16_bitmask(model)
    patch_hline_mask(model)
    patch_rel_u8_coords(model)
    patch_any_interior(model)
    patch_fold_small_veto(model)
    return cleanup(model)


def task105_sentinel_lut() -> onnx.ModelProto:
    c = make_constants()
    inits = [
        numpy_helper.from_array(c["idx20_i64"], "idx20_i64"),
        numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "half_f"),
        numpy_helper.from_array(c["weights_i32"], "weights_i32"),
        numpy_helper.from_array(c["keys_i32"], "keys_i32"),
        numpy_helper.from_array(c["template_codes_u8"], "template_codes_u8"),
        numpy_helper.from_array(np.array([0, 0, 1, 2, 0, 0, 17, 19], dtype=np.int64), "pads_code"),
        numpy_helper.from_array(np.asarray(0, dtype=np.uint8), "zero_u8"),
        numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "classes_u8"),
    ]
    nodes = [
        helper.make_node("GatherND", ["input", "idx20_i64"], ["samples_f"]),
        helper.make_node("Greater", ["samples_f", "half_f"], ["samples_b"]),
        helper.make_node("Cast", ["samples_b"], ["bits_i32"], to=TensorProto.INT32),
        helper.make_node("Mul", ["bits_i32", "weights_i32"], ["weighted_i32"]),
        helper.make_node("ReduceSum", ["weighted_i32"], ["key_i32"], keepdims=0),
        helper.make_node("Equal", ["key_i32", "keys_i32"], ["hit_b"]),
        helper.make_node("Cast", ["hit_b"], ["hit_u8"], to=TensorProto.UINT8),
        helper.make_node("ArgMax", ["hit_u8"], ["template_idx"], axis=0, keepdims=1),
        helper.make_node("Gather", ["template_codes_u8", "template_idx"], ["crop_code"], axis=0),
        helper.make_node("Pad", ["crop_code", "pads_code", "zero_u8"], ["code30"], mode="constant"),
        helper.make_node("Equal", ["code30", "classes_u8"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "task105_sentinel_lut",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return cleanup_fresh(model)


def task105_fillmask_lut_where(*, strict: bool = True) -> onnx.ModelProto:
    c = make_constants(strict=strict)
    two_onehot = np.zeros((1, 10, 1, 1), dtype=np.float32)
    two_onehot[0, 2, 0, 0] = 1.0
    inits = [
        numpy_helper.from_array(c["idx20_i64"], "idx20_i64"),
        numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "half_f"),
        numpy_helper.from_array(c["weights_i32"], "weights_i32"),
        numpy_helper.from_array(c["keys_i32"], "keys_i32"),
        numpy_helper.from_array(c["fill_templates_b"], "fill_templates_b"),
        numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 17, 21], dtype=np.int64), "pads_mask"),
        numpy_helper.from_array(np.asarray(False, dtype=np.bool_), "false_b"),
        numpy_helper.from_array(two_onehot, "two_onehot_f32"),
    ]
    nodes = [
        helper.make_node("GatherND", ["input", "idx20_i64"], ["samples_f"]),
        helper.make_node("Greater", ["samples_f", "half_f"], ["samples_b"]),
        helper.make_node("Cast", ["samples_b"], ["bits_i32"], to=TensorProto.INT32),
        helper.make_node("Mul", ["bits_i32", "weights_i32"], ["weighted_i32"]),
        helper.make_node("ReduceSum", ["weighted_i32"], ["key_i32"], keepdims=0),
        helper.make_node("Equal", ["key_i32", "keys_i32"], ["hit_b"]),
        # ORT 1.24 does not accept bool ArgMax in this harness.
        helper.make_node("Cast", ["hit_b"], ["hit_u8"], to=TensorProto.UINT8),
        helper.make_node("ArgMax", ["hit_u8"], ["template_idx"], axis=0, keepdims=1),
        helper.make_node("Gather", ["fill_templates_b", "template_idx"], ["fill_crop_b"], axis=0),
        helper.make_node("Pad", ["fill_crop_b", "pads_mask", "false_b"], ["paint_mask_30"], mode="constant"),
        helper.make_node("Where", ["paint_mask_30", "two_onehot_f32", "input"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "task105_fillmask_lut_where",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return cleanup_fresh(model)


def main() -> int:
    model = build()
    onnx.save(model, OUT)
    params = sum(math.prod(init.dims) if init.dims else 1 for init in model.graph.initializer)
    print(f"{OUT} params={params} nodes={len(model.graph.node)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
