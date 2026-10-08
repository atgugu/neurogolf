#!/usr/bin/env python3
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "task361.onnx")
SOURCE = os.path.join(HERE, "task361.source.onnx")


def main():
    if not os.path.exists(SOURCE):
        shutil.copyfile(MODEL, SOURCE)

    model = onnx.load(SOURCE)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 17
    graph = model.graph
    init_arrays = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == "safe_top_rows_i8":
                node.input[i] = "top_rows_i8"
            elif name == "safe_top_cols_i8":
                node.input[i] = "top_cols_i8"

    drop_node_outputs = {
        "top_is_bg_i8",
        "top_is_fg_i8",
        "top_is_fg_u8",
        "top_colors_masked_u8",
        "delta_rows_i8",
        "fg_delta_rows_i8",
        "safe_top_rows_i8",
        "delta_cols_i8",
        "fg_delta_cols_i8",
        "safe_top_cols_i8",
    }

    rewritten = []
    inserted_color_einsum = False
    for node in graph.node:
        if node.output and node.output[0] in drop_node_outputs:
            continue

        if node.op_type.startswith("Reduce") and len(node.input) > 1 and node.input[1] in init_arrays:
            axes = [int(x) for x in init_arrays[node.input[1]].reshape(-1)]
            keepdims = 1
            attrs = []
            for attr in node.attribute:
                if attr.name == "keepdims":
                    keepdims = attr.i
                elif attr.name != "axes":
                    attrs.append(attr)
            del node.input[1:]
            del node.attribute[:]
            node.attribute.extend(attrs)
            node.attribute.extend([
                helper.make_attribute("axes", axes),
                helper.make_attribute("keepdims", keepdims),
            ])

        if node.output and node.output[0] == "sampled_scalar_f":
            rewritten.append(helper.make_node(
                "Einsum",
                ["sampled_colors", "sample_color_weights_vec"],
                ["sampled_scalar_f"],
                name="sampled_scalar_f",
                equation="nchw,c->nh",
            ))
            inserted_color_einsum = True
            continue

        if node.output and node.output[0] in {"top_colors_u8_3d", "top_colors_u8"}:
            if node.output[0] == "top_colors_u8_3d":
                rewritten.append(helper.make_node(
                    "Cast",
                    ["sampled_scalar_f"],
                    ["top_colors_u8"],
                    name="top_colors_u8",
                    to=TensorProto.UINT8,
                ))
            continue

        if node.op_type == "Cast" and node.output and node.output[0] in {
            "rot1_idx_i32",
            "rot2_idx_i32",
            "rot3_idx_i32",
        }:
            src = node.input[0]
            clipped = src.replace("_idx_i8", "_idx_clip_i8")
            rewritten.append(helper.make_node(
                "Clip",
                [src, "zero_i8", "eighty_i8"],
                [clipped],
                name=clipped,
            ))
            node.input[0] = clipped

        rewritten.append(node)

    if not inserted_color_einsum:
        raise RuntimeError("did not replace sampled color Conv with Einsum")

    fused = []
    inserted = False
    for node in rewritten:
        if any(out in {"sc0", "sc1", "sc2", "completed_flat"} for out in node.output):
            if not inserted:
                fused.extend([
                    helper.make_node(
                        "Cast",
                        ["top_idx_i64"],
                        ["top_idx_i32"],
                        name="top_idx_i32",
                        to=TensorProto.INT32,
                    ),
                    helper.make_node(
                        "Concat",
                        ["top_idx_i32", "rot1_idx_i32", "rot2_idx_i32", "rot3_idx_i32"],
                        ["all_scatter_idx_i32"],
                        name="all_scatter_idx_i32",
                        axis=1,
                    ),
                    helper.make_node(
                        "Concat",
                        ["top_colors_u8", "top_colors_u8", "top_colors_u8", "top_colors_u8"],
                        ["all_scatter_updates_u8"],
                        name="all_scatter_updates_u8",
                        axis=1,
                    ),
                    helper.make_node(
                        "ScatterElements",
                        ["zero_flat_u8", "all_scatter_idx_i32", "all_scatter_updates_u8"],
                        ["completed_flat"],
                        name="completed_flat_fused",
                        axis=1,
                        reduction="max",
                    ),
                ])
                inserted = True
            continue
        fused.append(node)

    rendered = []
    inserted_poly = False
    for node in fused:
        if node.output and node.output[0] in {"completed_scalar30", "output"}:
            if node.output[0] == "completed_scalar30":
                rendered.extend([
                    helper.make_node(
                        "Mul",
                        ["completed_scalar10", "completed_scalar10"],
                        ["completed_scalar10_sq"],
                        name="completed_scalar10_sq",
                    ),
                    helper.make_node(
                        "Concat",
                        ["completed_scalar10", "completed_scalar10_sq", "valid10_u8"],
                        ["poly_features_u8"],
                        name="poly_features_u8",
                        axis=1,
                    ),
                    helper.make_node(
                        "ConvInteger",
                        ["poly_features_u8", "poly_weights_i8"],
                        ["output"],
                        name="output",
                        pads=[0, 0, 20, 20],
                    ),
                ])
                inserted_poly = True
            continue
        rendered.append(node)

    if not inserted_poly:
        raise RuntimeError("did not replace scalar30 Equal renderer")

    del graph.node[:]
    graph.node.extend(rendered)

    old_initializers = list(graph.initializer)
    old_initializers.append(numpy_helper.from_array(
        np.array(80, dtype=np.int8),
        name="eighty_i8",
    ))
    old_initializers.append(numpy_helper.from_array(
        np.arange(10, dtype=np.float32),
        name="sample_color_weights_vec",
    ))
    old_initializers.append(numpy_helper.from_array(
        np.ones((1, 1, 10, 10), dtype=np.uint8),
        name="valid10_u8",
    ))
    weights = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for color in range(10):
        weights[color, 0, 0, 0] = 2 * color
        weights[color, 1, 0, 0] = -1
        weights[color, 2, 0, 0] = 1 - color * color
    old_initializers.append(numpy_helper.from_array(
        weights,
        name="poly_weights_i8",
    ))

    used = {name for node in graph.node for name in node.input if name}
    used.update(inp.name for inp in graph.input)
    used.update(out.name for out in graph.output)

    del graph.initializer[:]
    seen = set()
    for init in old_initializers:
        if init.name in used and init.name not in seen:
            graph.initializer.append(init)
            seen.add(init.name)

    del graph.value_info[:]
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])
    ])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, MODEL)


if __name__ == "__main__":
    main()
