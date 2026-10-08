#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = "task187.onnx"


def c(name: str, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def main() -> None:
    nodes = []
    inits = []

    pack_w = np.zeros((1, 10, 1, 15), dtype=np.float32)
    powers15 = (2.0 ** np.arange(15, dtype=np.float32)).astype(np.float32)
    pack_w[0, 1:, 0, :] = powers15
    border = np.array(
        [0xFFFFFFFF] + [0xFF000001] * 23 + [0xFFFFFFFF],
        dtype=np.uint32,
    ).reshape(1, 1, 25, 1)

    idx_p1 = np.minimum(np.arange(25) + 1, 24).astype(np.int64)
    idx_p2 = np.minimum(np.arange(25) + 2, 24).astype(np.int64)
    idx_p4 = np.minimum(np.arange(25) + 4, 24).astype(np.int64)
    idx_p8 = np.minimum(np.arange(25) + 8, 24).astype(np.int64)
    idx_m1 = np.maximum(np.arange(25) - 1, 0).astype(np.int64)
    idx_m2 = np.maximum(np.arange(25) - 2, 0).astype(np.int64)
    idx_m3 = np.maximum(np.arange(25) - 3, 0).astype(np.int64)
    idx_m4 = np.maximum(np.arange(25) - 4, 0).astype(np.int64)
    idx_m7 = np.maximum(np.arange(25) - 7, 0).astype(np.int64)
    idx_m8 = np.maximum(np.arange(25) - 8, 0).astype(np.int64)

    inits += [
        c("safe_name_0", pack_w),
        c("safe_name_1", np.array(15, np.uint32)),
        c("safe_name_2", np.array(1, np.uint32)),
        c("safe_name_3", np.array(2, np.uint32)),
        c("safe_name_4", np.array(4, np.uint32)),
        c("safe_name_5", np.array(8, np.uint32)),
        c("safe_name_6", border),
        c("safe_name_7", idx_p1),
        c("safe_name_8", idx_p2),
        c("safe_name_9", idx_p4),
        c("safe_name_10", idx_p8),
        c("safe_name_11", idx_m1),
        c("safe_name_12", idx_m2),
        c("safe_name_13", idx_m3),
        c("safe_name_14", idx_m4),
        c("safe_name_15", idx_m7),
        c("safe_name_16", idx_m8),
        c("safe_name_24", np.array([0], np.int64)),
        c("safe_name_25", np.array([1], np.int64)),
        c("safe_name_26", np.array([2], np.int64)),
        c("safe_name_27", np.array([3], np.int64)),
        c("safe_name_29", np.array([25], np.int64)),
    ]

    def n(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    n("Conv", ["input", "safe_name_0"], ["safe_name_30"], strides=[1, 15])
    n("Cast", ["safe_name_30"], ["safe_name_31"], to=TensorProto.UINT32)
    n("Slice", ["safe_name_31", "safe_name_24", "safe_name_25", "safe_name_27"], ["safe_name_32"])
    n("Slice", ["safe_name_31", "safe_name_25", "safe_name_26", "safe_name_27"], ["safe_name_33"])
    n("BitShift", ["safe_name_33", "safe_name_1"], ["safe_name_34"], direction="LEFT")
    n("BitwiseOr", ["safe_name_32", "safe_name_34"], ["safe_name_35"])
    n("Slice", ["safe_name_35", "safe_name_24", "safe_name_29", "safe_name_26"], ["safe_name_36"])
    n("BitwiseNot", ["safe_name_36"], ["safe_name_37"])
    n("BitwiseAnd", ["safe_name_37", "safe_name_6"], ["safe_name_38"])
    n("BitShift", ["safe_name_37", "safe_name_2"], ["safe_name_39"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_37", "safe_name_39"], ["safe_name_40"])
    n("BitShift", ["safe_name_40", "safe_name_3"], ["safe_name_41"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_40", "safe_name_41"], ["safe_name_42"])
    n("BitShift", ["safe_name_42", "safe_name_4"], ["safe_name_43"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_42", "safe_name_43"], ["safe_name_44"])
    n("Gather", ["safe_name_37", "safe_name_7"], ["safe_name_45"], axis=2)
    n("BitwiseAnd", ["safe_name_37", "safe_name_45"], ["safe_name_46"])
    n("Gather", ["safe_name_46", "safe_name_8"], ["safe_name_47"], axis=2)
    n("BitwiseAnd", ["safe_name_46", "safe_name_47"], ["safe_name_48"])
    n("Gather", ["safe_name_48", "safe_name_9"], ["safe_name_49"], axis=2)
    n("BitwiseAnd", ["safe_name_48", "safe_name_49"], ["safe_name_50"])
    n("Gather", ["safe_name_46", "safe_name_11"], ["safe_name_51"], axis=2)
    n("Gather", ["safe_name_48", "safe_name_13"], ["safe_name_52"], axis=2)
    n("Gather", ["safe_name_50", "safe_name_15"], ["safe_name_53"], axis=2)
    n("Add", ["safe_name_37", "safe_name_38"], ["safe_name_54"])
    n("BitwiseXor", ["safe_name_54", "safe_name_37"], ["safe_name_55"])
    n("BitwiseAnd", ["safe_name_37", "safe_name_55"], ["safe_name_56"])
    n("BitwiseOr", ["safe_name_38", "safe_name_56"], ["safe_name_57"])
    n("BitShift", ["safe_name_57", "safe_name_3"], ["safe_name_58"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_40", "safe_name_58"], ["safe_name_59"])
    n("BitwiseOr", ["safe_name_57", "safe_name_59"], ["safe_name_60"])
    n("BitShift", ["safe_name_60", "safe_name_4"], ["safe_name_61"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_42", "safe_name_61"], ["safe_name_62"])
    n("BitwiseOr", ["safe_name_60", "safe_name_62"], ["safe_name_63"])
    n("BitShift", ["safe_name_63", "safe_name_5"], ["safe_name_64"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_44", "safe_name_64"], ["safe_name_65"])
    n("BitwiseOr", ["safe_name_63", "safe_name_65"], ["safe_name_66"])
    n("Gather", ["safe_name_66", "safe_name_7"], ["safe_name_67"], axis=2)
    n("BitwiseAnd", ["safe_name_37", "safe_name_67"], ["safe_name_68"])
    n("BitwiseOr", ["safe_name_66", "safe_name_68"], ["safe_name_69"])
    n("Gather", ["safe_name_69", "safe_name_8"], ["safe_name_70"], axis=2)
    n("BitwiseAnd", ["safe_name_46", "safe_name_70"], ["safe_name_71"])
    n("BitwiseOr", ["safe_name_69", "safe_name_71"], ["safe_name_72"])
    n("Gather", ["safe_name_72", "safe_name_9"], ["safe_name_73"], axis=2)
    n("BitwiseAnd", ["safe_name_48", "safe_name_73"], ["safe_name_74"])
    n("BitwiseOr", ["safe_name_72", "safe_name_74"], ["safe_name_75"])
    n("Gather", ["safe_name_75", "safe_name_10"], ["safe_name_76"], axis=2)
    n("BitwiseAnd", ["safe_name_50", "safe_name_76"], ["safe_name_77"])
    n("BitwiseOr", ["safe_name_75", "safe_name_77"], ["safe_name_78"])
    n("Gather", ["safe_name_78", "safe_name_12"], ["safe_name_79"], axis=2)
    n("BitwiseAnd", ["safe_name_51", "safe_name_79"], ["safe_name_80"])
    n("BitwiseOr", ["safe_name_78", "safe_name_80"], ["safe_name_81"])
    n("Gather", ["safe_name_81", "safe_name_14"], ["safe_name_82"], axis=2)
    n("BitwiseAnd", ["safe_name_52", "safe_name_82"], ["safe_name_83"])
    n("BitwiseOr", ["safe_name_81", "safe_name_83"], ["safe_name_84"])
    n("Gather", ["safe_name_84", "safe_name_16"], ["safe_name_85"], axis=2)
    n("BitwiseAnd", ["safe_name_53", "safe_name_85"], ["safe_name_86"])
    n("BitwiseOr", ["safe_name_84", "safe_name_86"], ["safe_name_87"])
    n("Add", ["safe_name_37", "safe_name_87"], ["safe_name_88"])
    n("BitwiseXor", ["safe_name_88", "safe_name_37"], ["safe_name_89"])
    n("BitwiseAnd", ["safe_name_37", "safe_name_89"], ["safe_name_90"])
    n("BitwiseOr", ["safe_name_87", "safe_name_90"], ["safe_name_91"])
    n("BitShift", ["safe_name_91", "safe_name_2"], ["safe_name_92"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_37", "safe_name_92"], ["safe_name_93"])
    n("BitwiseOr", ["safe_name_91", "safe_name_93"], ["safe_name_94"])
    n("BitShift", ["safe_name_94", "safe_name_3"], ["safe_name_95"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_40", "safe_name_95"], ["safe_name_96"])
    n("BitwiseOr", ["safe_name_94", "safe_name_96"], ["safe_name_97"])
    n("BitShift", ["safe_name_97", "safe_name_4"], ["safe_name_98"], direction="RIGHT")
    n("BitwiseAnd", ["safe_name_42", "safe_name_98"], ["safe_name_99"])
    n("BitwiseOr", ["safe_name_97", "safe_name_99"], ["safe_name_100"])
    n("BitwiseXor", ["safe_name_37", "safe_name_100"], ["safe_name_101"])

    width_w = powers15.reshape(1, 1, 1, 15)
    colmask = np.zeros((1, 1, 1, 30), dtype=np.uint32)
    for col in range(24):
        colmask[0, 0, 0, col] = 1 << col
    colmask[0, 0, 0, 24] = 1 << 24
    inits += [
        c("br_width_w", width_w),
        c("br_axes23", np.array([2, 3], np.int64)),
        c("br_axes12", np.array([1, 2], np.int64)),
        c("br_axes13", np.array([1, 3], np.int64)),
        c("br_zero_5row", np.zeros((1, 1, 5, 1), dtype=np.uint32)),
        c("br_zero_30row", np.zeros((1, 1, 30, 1), dtype=np.uint32)),
        c(
            "br_row25",
            np.array([1] * 25 + [0] * 5, dtype=np.uint32).reshape(1, 1, 30, 1),
        ),
        c("br_colmask", colmask),
        c("br_c1", np.array([1], np.int64)),
        c("br_c4", np.array([4], np.int64)),
        c("br_c5", np.array([5], np.int64)),
        c("br_c6", np.array([6], np.int64)),
        c("br_c7", np.array([7], np.int64)),
        c("br_c8", np.array([8], np.int64)),
        c("br_c9", np.array([9], np.int64)),
    ]

    red25 = "safe_name_101"
    n("Concat", [red25, "br_zero_5row"], ["br_red30"], axis=2)
    n("ReduceMax", ["input", "br_axes12"], ["br_col_present_f"], keepdims=1)
    n("Conv", ["br_col_present_f", "br_width_w"], ["br_width_pack_f"], strides=[1, 15])
    n("Cast", ["br_width_pack_f"], ["br_width_pack"], to=TensorProto.UINT32)
    n("Slice", ["br_width_pack", "safe_name_24", "safe_name_25", "safe_name_27"], ["br_width_lo"])
    n("Slice", ["br_width_pack", "safe_name_25", "safe_name_26", "safe_name_27"], ["br_width_hi0"])
    n("BitShift", ["br_width_hi0", "safe_name_1"], ["br_width_hi"], direction="LEFT")
    n("BitwiseOr", ["br_width_lo", "br_width_hi"], ["br_width_mask"])
    n("ReduceMax", ["input", "br_axes13"], ["br_row_present_f"], keepdims=1)
    n("Cast", ["br_row_present_f"], ["br_row_present"], to=TensorProto.UINT32)
    n("Mul", ["br_row_present", "br_row25"], ["br_row25_present"])
    n("Mul", ["br_row25_present", "br_width_mask"], ["br_extent30"])
    n("BitwiseXor", ["br_extent30", "safe_name_35"], ["br_blackish30"])
    n("BitwiseXor", ["br_blackish30", "br_red30"], ["br_green30"])
    n("ReduceMax", ["input", "br_axes23"], ["br_present_f"], keepdims=1)
    n("Cast", ["br_present_f"], ["br_present_01"], to=TensorProto.UINT32)
    color_state_names = []
    for color in (1, 4, 5, 6, 7, 8, 9):
        scalar = f"br_pres_{color}"
        state = f"br_color_{color}"
        n("Gather", ["br_present_01", f"br_c{color}"], [scalar], axis=1)
        n("Mul", ["safe_name_35", scalar], [state])
        color_state_names.append(state)
    n(
        "Concat",
        [
            "br_zero_30row",
            color_state_names[0],
            "br_red30",
            "br_green30",
            color_state_names[1],
            color_state_names[2],
            color_state_names[3],
            color_state_names[4],
            color_state_names[5],
            color_state_names[6],
        ],
        ["br_state30"],
        axis=1,
    )
    n("BitwiseAnd", ["br_state30", "br_colmask"], ["output"])

    used = {x for node in nodes for x in node.input if x}
    inits = [init for init in inits if init.name in used]

    graph = helper.make_graph(
        nodes,
        "task187_packed_terminal_bitset_p4",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUT)
    print(f"saved {OUT}: nodes={len(nodes)} inits={len(inits)}")


if __name__ == "__main__":
    main()
