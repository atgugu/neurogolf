import os
#!/usr/bin/env python3
"""task066 renderer replacement.

The route solver is the current moment-coordinate vector solver, but the charged
20x20 bool painter and 30x30 Pad tail are replaced by a terminal Einsum renderer.
The final op mixes the original one-hot input with three disjoint route segment
row/column factors, so the 10x30x30 output tensor is free under the scorer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PIN = Path(
    os.environ.get("NEUROGOLF_ARCHIVE", "")
    + "/task_candidates_gate_pass/task066/"
    "389d478f__ARC__neurogolf_clean__tasks__task066__task066_cheaper_f16ws.onnx"
)


def tensor(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def node(op: str, ins: list[str], outs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, ins, outs, **attrs)


def prune_initializers(model: onnx.ModelProto) -> None:
    used = {s for n in model.graph.node for s in n.input if s}
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def add_renderer(model: onnx.ModelProto) -> None:
    # Drop the old charged renderer:
    #   3 x 20x20 segment masks -> OR -> Cast/Transpose/Where -> Pad -> final Where.
    cut_outputs = {
        "seg_red_b",
        "seg_green_b",
        "path_b0",
        "seg_mid_b",
        "path_b1",
        "path_u8_2",
        "path_c_t_u8",
        "path_orient",
        "path20",
        "path30",
        "output",
    }
    kept = [n for n in model.graph.node if not any(o in cut_outputs for o in n.output)]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    # New reusable constants. Initializer dtype is free for params, but f32 is needed
    # for the terminal Einsum operands.
    mix = np.zeros((4, 10, 10), dtype=np.float32)
    mix[0] = np.eye(10, dtype=np.float32)
    # Route segments only rewrite black input cells: black -> 0, green -> 1.
    mix[1:, 0, 0] = -1.0
    mix[1:, 3, 0] = 1.0
    model.graph.initializer.extend(
        [
            tensor("seg_row_pad_20_30", np.array([0, 0, 0, 0, 0, 0, 10, 0], dtype=np.int64)),
            tensor("seg_col_pad_20_30", np.array([0, 0, 0, 0, 0, 0, 0, 10], dtype=np.int64)),
            tensor("seg_one_row_u8", np.ones((1, 1, 30, 1), dtype=np.uint8)),
            tensor("seg_one_col_u8", np.ones((1, 1, 1, 30), dtype=np.uint8)),
            tensor("seg_mix", mix),
        ]
    )

    new_nodes: list[onnx.NodeProto] = []

    # Middle segment is strict between red and green rows. The horizontal segments
    # include the two corner cells, avoiding double application in the final sum.
    new_nodes.extend(
        [
            node("Greater", ["row_h", "row_lo"], ["mid_row_gt_lo"]),
            node("Less", ["row_h", "row_hi"], ["mid_row_lt_hi"]),
            node("And", ["mid_row_gt_lo", "mid_row_lt_hi"], ["mid_row_inner"]),
        ]
    )

    def oriented(prefix: str, canon_row: str, canon_col: str) -> tuple[str, str]:
        row30 = f"{prefix}_row30"
        col30 = f"{prefix}_col30"
        col30_t = f"{prefix}_col30_t"
        row30_t = f"{prefix}_row30_t"
        row30_u8 = f"{prefix}_row30_u8"
        col30_u8 = f"{prefix}_col30_u8"
        col30_t_u8 = f"{prefix}_col30_t_u8"
        row30_t_u8 = f"{prefix}_row30_t_u8"
        row_o = f"{prefix}_row_o"
        col_o = f"{prefix}_col_o"
        new_nodes.extend(
            [
                node("Pad", [canon_row, "seg_row_pad_20_30"], [row30], mode="constant"),
                node("Pad", [canon_col, "seg_col_pad_20_30"], [col30], mode="constant"),
                node("Transpose", [col30], [col30_t], perm=[0, 1, 3, 2]),
                node("Transpose", [row30], [row30_t], perm=[0, 1, 3, 2]),
                node("Cast", [row30], [row30_u8], to=TensorProto.UINT8),
                node("Cast", [col30], [col30_u8], to=TensorProto.UINT8),
                node("Cast", [col30_t], [col30_t_u8], to=TensorProto.UINT8),
                node("Cast", [row30_t], [row30_t_u8], to=TensorProto.UINT8),
                node("Where", ["flag_h", row30_u8, col30_t_u8], [row_o]),
                node("Where", ["flag_h", col30_u8, row30_t_u8], [col_o]),
            ]
        )
        return row_o, col_o

    rr, rc = oriented("red_seg", "red_row", "red_cols")
    gr, gc = oriented("green_seg", "green_row", "green_cols")
    mr, mc = oriented("mid_seg", "mid_row_inner", "route_col")

    new_nodes.extend(
        [
            node("Concat", ["seg_one_row_u8", rr, gr, mr], ["seg_rows_u8"], axis=1),
            node("Concat", ["seg_one_col_u8", rc, gc, mc], ["seg_cols_u8"], axis=1),
            node("Cast", ["seg_rows_u8"], ["seg_rows_f"], to=TensorProto.FLOAT),
            node("Cast", ["seg_cols_u8"], ["seg_cols_f"], to=TensorProto.FLOAT),
            node(
                "Einsum",
                ["input", "seg_rows_f", "seg_cols_f", "seg_mix"],
                ["output"],
                equation="nkhw,nshq,nsqw,sck->nchw",
            ),
        ]
    )
    model.graph.node.extend(new_nodes)

    del model.graph.output[:]
    model.graph.output.extend(
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])]
    )
    del model.graph.value_info[:]
    prune_initializers(model)


def replace_cyan_substrate(model: onnx.ModelProto) -> None:
    """Replace the charged 20x20 cyan plane with direct line probes."""

    crop = np.zeros((20, 30), dtype=np.float32)
    crop[np.arange(20), np.arange(20)] = 1.0
    model.graph.initializer.extend(
        [
            tensor("cyan_sel", np.eye(10, dtype=np.float32)[8]),
            tensor("crop20_30", crop),
            tensor("zero_f32_s", np.array(0.0, dtype=np.float32)),
        ]
    )

    def cyan_line(prefix: str, row_mask_f: str, out: str) -> list[onnx.NodeProto]:
        h_f = f"{prefix}_h_f"
        v_f = f"{prefix}_v_f"
        h_u8 = f"{prefix}_h_u8"
        v_u8 = f"{prefix}_v_u8"
        return [
            node(
                "Einsum",
                ["input", "cyan_sel", row_mask_f, "crop20_30", "crop20_30"],
                [h_f],
                equation="nkhw,k,nsrq,rh,cw->nsqc",
            ),
            node(
                "Einsum",
                ["input", "cyan_sel", row_mask_f, "crop20_30", "crop20_30"],
                [v_f],
                equation="nkhw,k,nsrq,rw,ch->nsqc",
            ),
            node("Cast", [h_f], [h_u8], to=TensorProto.UINT8),
            node("Cast", [v_f], [v_u8], to=TensorProto.UINT8),
            node("Where", ["flag_h", h_u8, v_u8], [out]),
        ]

    def cyan_span() -> list[onnx.NodeProto]:
        return [
            node("Cast", ["row_span"], ["row_span_f"], to=TensorProto.FLOAT),
            node(
                "Einsum",
                ["input", "cyan_sel", "row_span_f", "crop20_30", "crop20_30"],
                ["col_bad_h_f"],
                equation="nkhw,k,nsrq,rh,cw->nsqc",
            ),
            node(
                "Einsum",
                ["input", "cyan_sel", "row_span_f", "crop20_30", "crop20_30"],
                ["col_bad_v_f"],
                equation="nkhw,k,nsrq,rw,ch->nsqc",
            ),
            node("Where", ["flag_h", "col_bad_h_f", "col_bad_v_f"], ["col_bad_f"]),
            node("Greater", ["col_bad_f", "zero_f32_s"], ["vbad"]),
            node("Transpose", ["vbad"], ["vbad_vec"], perm=[0, 1, 3, 2]),
        ]

    drop_node_outputs = {
        "ch8",
        "ch8_u8",
        "ch8_u8_t",
        "ch8_c",
        "red_cy_idx_clip",
        "red_cy_idx_i64_4d",
        "red_cy_idx",
        "red_cy_line",
        "green_cy_idx_clip",
        "green_cy_idx_i64_4d",
        "green_cy_idx",
        "green_cy_line",
        "guard_cy_idx_clip",
        "guard_cy_idx_i64_4d",
        "guard_cy_idx",
        "guard_cy_line",
        "red_inside_cy_idx_clip",
        "red_inside_cy_idx_i64_4d",
        "red_inside_cy_idx",
        "red_inside_cy_line",
        "row_span_u8",
        "span_cy",
        "col_bad",
        "vbad",
        "vbad_vec",
        "red_row",
        "green_row",
    }

    new_nodes: list[onnx.NodeProto] = []
    for old in model.graph.node:
        if any(out in drop_node_outputs for out in old.output):
            continue
        new_nodes.append(old)
        outs = set(old.output)
        if "ri" in outs:
            new_nodes.extend(
                [
                    node("Equal", ["row_h", "ri"], ["red_row"]),
                    node("Cast", ["red_row"], ["red_row_f"], to=TensorProto.FLOAT),
                ]
            )
            new_nodes.extend(cyan_line("red_cy", "red_row_f", "red_cy_line"))
        if "gu" in outs:
            new_nodes.extend(
                [
                    node("Equal", ["row_h", "gu"], ["green_row"]),
                    node("Cast", ["green_row"], ["green_row_f"], to=TensorProto.FLOAT),
                ]
            )
            new_nodes.extend(cyan_line("green_cy", "green_row_f", "green_cy_line"))
        if "guard_idx" in outs:
            new_nodes.extend(
                [
                    node("Equal", ["row_h", "guard_idx"], ["guard_row"]),
                    node("Cast", ["guard_row"], ["guard_row_f"], to=TensorProto.FLOAT),
                ]
            )
            new_nodes.extend(cyan_line("guard_cy", "guard_row_f", "guard_cy_line"))
        if "row_span" in outs:
            new_nodes.extend(cyan_span())
        if "red_inside_r" in outs:
            new_nodes.extend(
                [
                    node("Equal", ["row_h", "red_inside_r"], ["red_inside_row"]),
                    node(
                        "Cast",
                        ["red_inside_row"],
                        ["red_inside_row_f"],
                        to=TensorProto.FLOAT,
                    ),
                ]
            )
            new_nodes.extend(
                cyan_line("red_inside_cy", "red_inside_row_f", "red_inside_cy_line")
            )

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.value_info[:]
    prune_initializers(model)


def build() -> onnx.ModelProto:
    model = onnx.load(PIN)
    add_renderer(model)
    replace_cyan_substrate(model)
    model.graph.name = "task066_terminal_einsum_renderer"
    model.producer_name = "neurogolf_task066_terminal_einsum"
    model.ir_version = 8
    return model


def main() -> None:
    model = build()
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task066.onnx")
    print("saved task066.onnx")


if __name__ == "__main__":
    main()
