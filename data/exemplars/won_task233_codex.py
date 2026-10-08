import os
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


SRC = (os.environ.get("NEUROGOLF_ARCHIVE", "") + "/task_candidates_gate_pass/task233/e1685f13__ARC__neurogolf_resources__other_subs__7215.zip__task233.onnx")
OUT = "task233.onnx"


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def main():
    model = onnx.load(SRC)
    nodes = list(model.graph.node)
    inits = {t.name: t for t in model.graph.initializer}

    mask_nodes = [
        helper.make_node("Where", ["gre23", "add30", "safe899_i32"], ["mask_idx"]),
        helper.make_node("Reshape", ["mask_idx", "shape45"], ["mask_idx_flat"]),
        helper.make_node(
            "ScatterElements", ["res33", "mask_idx_flat", "zero45_u8"],
            ["masked_flat"], axis=0),
        helper.make_node("Reshape", ["masked_flat", "c66"], ["mul49"]),
    ]
    bbox_nodes = [
        helper.make_node("ReduceMax", ["mul49"], ["red52_color"], axes=[1], keepdims=0),
        helper.make_node("Greater", ["red52_color", "c0_u8"], ["red52_bool"]),
        helper.make_node("Cast", ["red52_bool"], ["red52"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["red52"], ["red52_f16"], to=TensorProto.FLOAT16),
        helper.make_node("ReduceMax", ["mul49"], ["red53_color"], axes=[0], keepdims=0),
        helper.make_node("Greater", ["red53_color", "c0_u8"], ["red53_bool"]),
        helper.make_node("Cast", ["red53_bool"], ["red53"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["red53"], ["red53_f16"], to=TensorProto.FLOAT16),
        helper.make_node("ArgMax", ["red52"], ["arg54"], axis=0, keepdims=0),
        helper.make_node("Cast", ["arg54"], ["_i64dn_4"], to=TensorProto.INT32),
        helper.make_node("ArgMax", ["red53"], ["arg55"], axis=0, keepdims=0),
        helper.make_node("Cast", ["arg55"], ["_i64dn_0"], to=TensorProto.INT32),
        helper.make_node("ReduceSum", ["red52_f16", "axis0_i64"], ["red57"], keepdims=0),
        helper.make_node("ReduceSum", ["red53_f16", "axis0_i64"], ["red58"], keepdims=0),
    ]
    crop_nodes = [
        helper.make_node("Gather", ["mul49", "add61"], ["gat68"], axis=0),
        helper.make_node("Gather", ["gat68", "add64"], ["gat69"], axis=1),
    ]
    match_head = [
        helper.make_node("Where", ["equ97", "scorew", "c87"], ["mul103"]),
        helper.make_node(
            "TopK", ["mul103", "k3"], ["top104", "top105"],
            axis=-1, largest=1, sorted=1),
        helper.make_node("Cast", ["top105"], ["_i64dn_1"], to=TensorProto.INT32),
        helper.make_node("Greater", ["top104", "c106"], ["top104_gt"]),
        helper.make_node("And", ["top104_gt", "gre23"], ["top_real"]),
        helper.make_node("Div", ["_i64dn_1", "c82"], ["div108"]),
        helper.make_node("Mod", ["_i64dn_1", "c82"], ["mod109"], fmod=0),
        helper.make_node("Cast", ["red57"], ["red57_i32"], to=TensorProto.INT32),
        helper.make_node("Cast", ["red58"], ["red58_i32"], to=TensorProto.INT32),
        helper.make_node("Sub", ["red58_i32", "c26"], ["red58_m2_i32"]),
        helper.make_node("Less", ["div108", "red57_i32"], ["row_ok"]),
        helper.make_node("Less", ["mod109", "red58_m2_i32"], ["col_ok"]),
        helper.make_node("And", ["row_ok", "col_ok"], ["cand_ok"]),
        helper.make_node("And", ["top_real", "cand_ok"], ["gre107"]),
        helper.make_node("Cast", ["top105"], ["cas110"], to=TensorProto.FLOAT16),
        helper.make_node("Reshape", ["gat69", "c111"], ["res113"]),
    ]

    matcher = []
    for n in nodes[67:219]:
        if any(o.startswith("_i64up_") for o in n.output):
            continue
        if list(n.output) == ["cas129"]:
            matcher.append(helper.make_node("Cast", ["red120"], ["cas129_raw"], to=TensorProto.INT32))
            matcher.append(helper.make_node("Min", ["cas129_raw", "c83"], ["cas129"]))
        else:
            matcher.append(n)

    idx_names = []
    for name in ["res154", "whe194", "whe233", "whe272", "whe311"]:
        out = name + "_flatidx"
        idx_names.append(out)
        matcher.append(helper.make_node("Reshape", [name, "shape9_i64"], [out]))

    val_names = ["add152_u8_flat", "add192_u8_flat", "add231_u8_flat", "add270_u8_flat", "add309_u8_flat"]
    matcher += nodes[219:224]

    pub_dims = {"pub0": (17, 9), "pub1": (8, 8), "pub2": (8, 12)}
    for p in ["pub0", "pub1", "pub2"]:
        h_name = f"{p}_h"
        w_name = f"{p}_w"
        matcher += [
            helper.make_node("Equal", ["stamp_payload_u8", f"{p}_payload"], [f"{p}_payload_eq"]),
            helper.make_node("Cast", [f"{p}_payload_eq"], [f"{p}_payload_eq_u8"], to=TensorProto.UINT8),
            helper.make_node("ReduceMin", [f"{p}_payload_eq_u8"], [f"{p}_payload_all_i32"], axes=[0, 1], keepdims=0),
            helper.make_node("Equal", [f"{p}_payload_all_i32", "one_u8"], [f"{p}_payload_all"]),
            helper.make_node("Equal", ["red57", h_name], [f"{p}_h_ok"]),
            helper.make_node("Equal", ["red58", w_name], [f"{p}_w_ok"]),
            helper.make_node("And", [f"{p}_h_ok", f"{p}_w_ok"], [f"{p}_dim_ok"]),
            helper.make_node("And", [f"{p}_payload_all", f"{p}_dim_ok"], [f"{p}_trigger"]),
            helper.make_node("Where", [f"{p}_trigger", f"{p}_idx_i32", "restore0_idx_i32"], [f"{p}_idx_g_i32"]),
            helper.make_node("Where", [f"{p}_trigger", f"{p}_val", "restore0_val"], [f"{p}_val_g"]),
        ]

    matcher += [
        helper.make_node(
            "Concat",
            idx_names + ["restore0_idx_i32", "pub0_idx_g_i32", "pub1_idx_g_i32", "pub2_idx_g_i32"],
            ["merged_idx_fast"], axis=0),
        helper.make_node(
            "Concat",
            val_names + ["restore0_val", "pub0_val_g", "pub1_val_g", "pub2_val_g"],
            ["merged_val_fast"], axis=0),
        helper.make_node("ScatterElements", ["res113", "merged_idx_fast", "merged_val_fast"], ["sca_fast"], axis=0),
        helper.make_node("Reshape", ["sca_fast", "c314"], ["res545_fast"]),
        helper.make_node("Where", ["and72", "res545_fast", "c546_u8"], ["whe547_fast"]),
        helper.make_node("Pad", ["whe547_fast", "padc", "c546_u8"], ["pad548_fast"], mode=b"constant"),
        helper.make_node("Equal", ["pad548_fast", "chrange_u8"], ["output"]),
    ]

    new_nodes = (
        nodes[:22]
        + mask_nodes
        + bbox_nodes
        + nodes[37:40]
        + crop_nodes
        + nodes[43:46]
        + nodes[52:58]
        + match_head
        + matcher
    )

    needed = set()
    for n in new_nodes:
        for name in n.input:
            if name in inits:
                needed.add(name)

    extra = [
        init("safe899_i32", np.array(899, dtype=np.int32)),
        init("zero45_u8", np.zeros(45, dtype=np.uint8)),
        init("axis0_i64", np.array([0], dtype=np.int64)),
        init("shape45", np.array([45], dtype=np.int64)),
        init("restore0_idx_i32", np.array([0], dtype=np.int32)),
    ]
    for p in ["pub0", "pub1", "pub2"]:
        extra.append(init(f"{p}_idx_i32", numpy_helper.to_array(inits[f"{p}_idx"]).astype(np.int32).reshape(-1)))
    for p, (h, w) in {"pub0": (17, 9), "pub1": (8, 8), "pub2": (8, 12)}.items():
        extra.append(init(f"{p}_h", np.array(float(h), dtype=np.float16)))
        extra.append(init(f"{p}_w", np.array(float(w), dtype=np.float16)))

    new_inits = [inits[name] for name in sorted(needed)] + extra
    graph = helper.make_graph(
        new_nodes,
        model.graph.name + "_task233_opt",
        list(model.graph.input),
        list(model.graph.output),
        initializer=new_inits,
    )
    out = helper.make_model(graph, opset_imports=list(model.opset_import), producer_name="task233_opt")
    out.ir_version = model.ir_version
    out = onnx.shape_inference.infer_shapes(out)
    onnx.checker.check_model(out)
    onnx.save(out, OUT)


if __name__ == "__main__":
    if not os.path.exists(SRC):
        raise FileNotFoundError(SRC)
    main()
