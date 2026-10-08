#!/usr/bin/env python3
"""Build task174 by removing reverse-slice max-coordinate paths.

The source graph is the exact attempt-2 graph.  This rewrite keeps the same
rule and renderer, but uses ArgMax(select_last_index=1) to get bbox maxima
directly from row/column projections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parent


DROP_OUTPUTS = {
    "oh0_b",
    "oh0",
    "pres1",
    "oh1_b",
    "oh1",
    "pres2",
    "colid_u8",
    "cmaxri_0",
    "cmn_0",
    "cmx_0",
    "f0_0",
    "r0_0",
    "cmaxri_1",
    "cmn_1",
    "cmx_1",
    "f0_1",
    "r0_1",
    "cpr_0",
    "cmaxi_0",
    "cpr_1",
    "cmaxi_1",
    "rpr_sel",
    "cprb_sel",
    "rmni_cid",
    "cmni_cid",
    "rmaxri_sel",
    "cmaxri_sel",
    "rmax_sel_i32",
    "cmaxb_sel_i32",
    "hsub_sel_i32",
    "wsub_sel_i32",
    "sel_h",
    "sel_w",
    "rmn_cid",
    "cmn_cid",
    "ir0_cid",
    "ic0_cid",
    "ir_cid",
    "ic_cid",
}

DROP_INITS = {
    "Wm1",
    "rev_starts",
    "rev_ends",
    "rev_steps",
    "ax_h",
    "ax_w",
    "ramp9_i64",
    "one_i64",
    "one_i32v",
    "clip_min",
    "clip_max",
}

REPLACE_INPUTS = {
    "cmaxi_0": "cmaxri_0",
    "cmaxi_1": "cmaxri_1",
    "rmax_sel_i32": "rmaxri_sel",
    "cmaxb_sel_i32": "cmaxri_sel",
    "colid_i64": "colid_u8",
}


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    out = onnx.NodeProto()
    out.CopyFrom(node)
    for i, name in enumerate(out.input):
        out.input[i] = REPLACE_INPUTS.get(name, name)
    return out


def argmax_last(src: str, dst: str, axis: int) -> onnx.NodeProto:
    return helper.make_node(
        "ArgMax",
        [src],
        [dst],
        axis=axis,
        keepdims=0,
        select_last_index=1,
    )


def argmax_keep1(src: str, dst: str) -> onnx.NodeProto:
    return helper.make_node("ArgMax", [src], [dst], axis=0, keepdims=1)


def scatter_zero(src: str, idx: str, dst: str) -> onnx.NodeProto:
    return helper.make_node("ScatterElements", [src, idx, "zero_u8v"], [dst], axis=0)


def cast(src: str, dst: str, to: int) -> onnx.NodeProto:
    return helper.make_node("Cast", [src], [dst], to=to)


def add(src_a: str, src_b: str, dst: str) -> onnx.NodeProto:
    return helper.make_node("Add", [src_a, src_b], [dst])


def sub(src_a: str, src_b: str, dst: str) -> onnx.NodeProto:
    return helper.make_node("Sub", [src_a, src_b], [dst])


def reshape(src: str, dst: str) -> onnx.NodeProto:
    return helper.make_node("Reshape", [src, "one2"], [dst])


def unsqueeze0(src: str, dst: str) -> onnx.NodeProto:
    return helper.make_node("Unsqueeze", [src, "sq0"], [dst])


def h_index_block(tag: str) -> list[onnx.NodeProto]:
    return [
        cast(f"cmin_{tag}", f"cmini8_{tag}", 3),
        cast(f"cmaxr_{tag}", f"cmaxi8_{tag}", 3),
        reshape(f"cmini8_{tag}", f"cmn_{tag}"),
        reshape(f"cmaxi8_{tag}", f"cmx_{tag}"),
        add(f"cmn_{tag}", "ramp2", f"f0u_{tag}"),
        sub(f"cmx_{tag}", "ramp2", f"r0u_{tag}"),
        cast(f"f0u_{tag}", f"f0_{tag}", 6),
        cast(f"r0u_{tag}", f"r0_{tag}", 6),
    ]


def selected_u8_tail() -> list[onnx.NodeProto]:
    return [
        cast("rmin_sel", "rminu_sel", 2),
        cast("cminb_sel", "cminu_sel", 2),
        cast("rmaxr_sel", "rmaxu_sel", 2),
        cast("cmaxrb_sel", "cmaxu_sel", 2),
        sub("rmaxu_sel", "rminu_sel", "hsub_sel"),
        sub("cmaxu_sel", "cminu_sel", "wsub_sel"),
        add("hsub_sel", "one_u8", "sel_h"),
        add("wsub_sel", "one_u8", "sel_w"),
        reshape("rminu_sel", "rmn_cid_u8"),
        reshape("cminu_sel", "cmn_cid_u8"),
        add("rmn_cid_u8", "ramp5", "ir0_cid_u8"),
        add("cmn_cid_u8", "ramp5", "ic0_cid_u8"),
        helper.make_node("Clip", ["ir0_cid_u8", "clip_min_u8", "clip_max_u8"], ["ir_cid_u8"]),
        helper.make_node("Clip", ["ic0_cid_u8", "clip_min_u8", "clip_max_u8"], ["ic_cid_u8"]),
        cast("ir_cid_u8", "ir_cid", 6),
        cast("ic_cid_u8", "ic_cid", 6),
    ]


def build() -> onnx.ModelProto:
    src = ROOT / "task174_att02.onnx"
    if not src.exists():
        src = ROOT / "task174.onnx"
    model = onnx.load(src)

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        out0 = node.output[0] if node.output else ""
        if out0 in DROP_OUTPUTS:
            continue
        if out0 == "a0":
            nodes.append(argmax_keep1("pres0", "a0"))
            nodes.append(scatter_zero("pres0", "a0", "pres1"))
            continue
        if out0 == "a1":
            nodes.append(argmax_keep1("pres1", "a1"))
            nodes.append(scatter_zero("pres1", "a1", "pres2"))
            continue
        if out0 == "a2":
            nodes.append(argmax_keep1("pres2", "a2"))
            continue
        if out0 == "ch_0":
            nodes.extend([cast("a0", "a0_u8", 2), add("a0_u8", "one_u8", "ch_0")])
            continue
        if out0 == "ch_1":
            nodes.extend([cast("a1", "a1_u8", 2), add("a1_u8", "one_u8", "ch_1")])
            continue
        if out0 == "ch_2":
            nodes.extend([cast("a2", "a2_u8", 2), add("a2_u8", "one_u8", "ch_2")])
            continue
        if out0 == "col12_i64":
            nodes.append(helper.make_node("Where", ["hH_1_b", "ch_1", "ch_2"], ["col12_u8"]))
            continue
        if out0 == "colid_i64":
            nodes.append(helper.make_node("Where", ["hH_0_b", "ch_0", "col12_u8"], ["colid_u8"]))
            continue
        if out0 == "cmaxr_0":
            nodes.append(argmax_last("cp_0", "cmaxr_0", 3))
            continue
        if out0 == "cmini_0":
            nodes.extend(h_index_block("0"))
            continue
        if out0 == "cmaxr_1":
            nodes.append(argmax_last("cp_1", "cmaxr_1", 3))
            continue
        if out0 == "cmini_1":
            nodes.extend(h_index_block("1"))
            continue
        if out0 == "rmaxr_sel":
            nodes.append(argmax_last("rp_sel", "rmaxr_sel", 2))
            continue
        if out0 == "cmaxrb_sel":
            nodes.append(argmax_last("cpb_sel", "cmaxrb_sel", 3))
            nodes.extend(selected_u8_tail())
            continue
        nodes.append(clone_node(node))

    inits: list[onnx.TensorProto] = []
    for init in model.graph.initializer:
        if init.name in DROP_INITS:
            continue
        if init.name == "ramp2":
            inits.append(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.int8), init.name))
        elif init.name in {"ramp5", "row_idx_const", "col_idx_const"}:
            inits.append(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.uint8), init.name))
        else:
            inits.append(init)
    have = {init.name for init in inits}
    if "one_u8" not in have:
        inits.append(numpy_helper.from_array(np.array(1, dtype=np.uint8), "one_u8"))
    if "clip_min_u8" not in have:
        inits.append(numpy_helper.from_array(np.array(0, dtype=np.uint8), "clip_min_u8"))
    if "clip_max_u8" not in have:
        inits.append(numpy_helper.from_array(np.array(9, dtype=np.uint8), "clip_max_u8"))
    if "zero_u8v" not in have:
        inits.append(numpy_helper.from_array(np.array([0], dtype=np.uint8), "zero_u8v"))
    graph = helper.make_graph(
        nodes,
        "task174_argmax_last",
        list(model.graph.input),
        list(model.graph.output),
        inits,
    )
    out = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    out.ir_version = 10
    return out


def main() -> int:
    out = build()
    onnx.save(out, ROOT / "task174.onnx")
    print(ROOT / "task174.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
