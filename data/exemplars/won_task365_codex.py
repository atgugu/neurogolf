#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).resolve().parent / "task365.onnx"


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def node(op: str, inputs: list[str], outputs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, **attrs)


def build() -> onnx.ModelProto:
    cw = np.zeros((1, 10, 2, 2), dtype=np.float32)
    cw[0, :, 0, 0] = np.arange(10, dtype=np.float32)

    tl_kernel = np.zeros((1, 1, 4, 4), dtype=np.float16)
    tl_kernel[0, 0, 0, 1] = -9
    tl_kernel[0, 0, 1, 0] = -9
    tl_kernel[0, 0, 1, 1] = 1

    initializers = [
        init("cw", cw),
        init("two_u8", np.array(2, dtype=np.uint8)),
        init("zero_u8", np.array(0, dtype=np.uint8)),
        init("tl_kernel", tl_kernel),
        init("topk_k", np.array([3], dtype=np.int64)),
        init("c8", np.array([8], dtype=np.int32)),
        init("offs4d", np.array([[[[3, 4, 5]]]], dtype=np.int32)),
        init("ten", np.array(10, dtype=np.int32)),
        init("nine", np.array(9, dtype=np.int32)),
        init("three_i", np.array([[[[3]]]], dtype=np.int32)),
        init("r10", np.arange(10, dtype=np.int32).reshape(1, 10)),
        init("r6", np.arange(6, dtype=np.int32)),
        init("ax0", np.array([0], dtype=np.int64)),
        init("ax1", np.array([1], dtype=np.int64)),
        init("ax01", np.array([0, 1], dtype=np.int64)),
        init("ax013", np.array([0, 1, 3], dtype=np.int64)),
        init("ax012", np.array([0, 1, 2], dtype=np.int64)),
        init("split3", np.array([1, 1, 1], dtype=np.int64)),
        init("col8", np.arange(1, 9, dtype=np.uint8).reshape(8, 1, 1)),
        init("sent", np.array([[[[255]]]], dtype=np.uint8)),
        init("out_pads", np.array([0, 1, 0, 0, 0, 1, 24, 24], dtype=np.int64)),
        init("zero_h", np.array(0, dtype=np.float16)),
    ]

    nodes = [
        node("Conv", ["input", "cw"], ["M"], dilations=[20, 20], kernel_shape=[2, 2], pads=[0, 0, 0, 0]),
        node("Cast", ["M"], ["C"], to=TensorProto.UINT8),
        node("Equal", ["C", "two_u8"], ["red_b"]),
        node("Cast", ["red_b"], ["red_h"], to=TensorProto.FLOAT16),
        node("Cast", ["M"], ["M_h"], to=TensorProto.FLOAT16),
        node("Conv", ["M_h", "tl_kernel"], ["tlsc"], kernel_shape=[4, 4], pads=[1, 1, 0, 0]),
        node("Flatten", ["tlsc"], ["tlflat"], axis=1),
        node("TopK", ["tlflat", "topk_k"], ["tvals", "tidx"], axis=1, largest=1, sorted=1),
        node("Cast", ["tidx"], ["tidx32"], to=TensorProto.INT32),
        node("Squeeze", ["tidx32", "ax0"], ["idx"]),
        node("Div", ["idx", "c8"], ["rows"]),
        node("Mod", ["idx", "c8"], ["cols"]),
        node("Greater", ["tvals", "zero_h"], ["pres2d"]),
        node("Squeeze", ["pres2d", "ax0"], ["present"]),
        node("Unsqueeze", ["rows", "ax1"], ["rows1"]),
        node("Unsqueeze", ["cols", "ax1"], ["cols1"]),
        node("Gather", ["C", "rows"], ["top_rows"], axis=2),
        node("Unsqueeze", ["cols1", "ax01"], ["cols1_4d"]),
        node("Add", ["cols1_4d", "offs4d"], ["colw4"]),
        node("Less", ["colw4", "ten"], ["validw4"]),
        node("Min", ["colw4", "nine"], ["colwc4"]),
        node("GatherElements", ["top_rows", "colwc4"], ["wcells0_u"], axis=3),
        node("Greater", ["wcells0_u", "zero_u8"], ["wcells0"]),
        node("And", ["wcells0", "validw4"], ["wcells"]),
        node("Split", ["wcells", "split3"], ["we0", "we1", "we2"], axis=3),
        node("And", ["we0", "we1"], ["we01"]),
        node("And", ["we01", "we2"], ["we012"]),
        node("Cast", ["we0"], ["wc0"], to=TensorProto.INT32),
        node("Cast", ["we01"], ["wc1"], to=TensorProto.INT32),
        node("Cast", ["we012"], ["wc2"], to=TensorProto.INT32),
        node("Add", ["wc0", "wc1"], ["wsa"]),
        node("Add", ["wsa", "wc2"], ["wsb"]),
        node("Add", ["wsb", "three_i"], ["width4"]),
        node("Squeeze", ["width4", "ax01"], ["width2d"]),
        node("Squeeze", ["width2d", "ax1"], ["width"]),
        node("Gather", ["C", "cols"], ["left_cols"], axis=3),
        node("Transpose", ["left_cols"], ["left_t"], perm=[0, 1, 3, 2]),
        node("Unsqueeze", ["rows1", "ax01"], ["rows1_4d"]),
        node("Add", ["rows1_4d", "offs4d"], ["rowh4"]),
        node("Less", ["rowh4", "ten"], ["validh4"]),
        node("Min", ["rowh4", "nine"], ["rowhc4"]),
        node("GatherElements", ["left_t", "rowhc4"], ["hcells0_u"], axis=3),
        node("Greater", ["hcells0_u", "zero_u8"], ["hcells0"]),
        node("And", ["hcells0", "validh4"], ["hcells"]),
        node("Split", ["hcells", "split3"], ["he0", "he1", "he2"], axis=3),
        node("And", ["he0", "he1"], ["he01"]),
        node("And", ["he01", "he2"], ["he012"]),
        node("Cast", ["he0"], ["hc0"], to=TensorProto.INT32),
        node("Cast", ["he01"], ["hc1"], to=TensorProto.INT32),
        node("Cast", ["he012"], ["hc2"], to=TensorProto.INT32),
        node("Add", ["hc0", "hc1"], ["hsa"]),
        node("Add", ["hsa", "hc2"], ["hsb"]),
        node("Add", ["hsb", "three_i"], ["height4"]),
        node("Squeeze", ["height4", "ax01"], ["height2d"]),
        node("Squeeze", ["height2d", "ax1"], ["height"]),
        node("GreaterOrEqual", ["r10", "rows1"], ["row_ge"]),
        node("Add", ["rows1", "height2d"], ["row_end"]),
        node("Less", ["r10", "row_end"], ["row_lt"]),
        node("And", ["row_ge", "row_lt"], ["row_mask"]),
        node("Cast", ["row_mask"], ["row_mh"], to=TensorProto.FLOAT16),
        node("GreaterOrEqual", ["r10", "cols1"], ["col_ge"]),
        node("Add", ["cols1", "width2d"], ["col_end"]),
        node("Less", ["r10", "col_end"], ["col_lt"]),
        node("And", ["col_ge", "col_lt"], ["col_mask"]),
        node("Cast", ["col_mask"], ["col_mh"], to=TensorProto.FLOAT16),
        node("Einsum", ["row_mh", "red_h", "col_mh"], ["count"], equation="ir,abrc,ic->i"),
        node("Where", ["present", "count", "zero_h"], ["scores"]),
        node("ArgMax", ["scores"], ["chosen"], axis=0, keepdims=1),
        node("Gather", ["rows", "chosen"], ["sr"]),
        node("Gather", ["cols", "chosen"], ["sc"]),
        node("Gather", ["height", "chosen"], ["sh"]),
        node("Gather", ["width", "chosen"], ["sw"]),
        node("Add", ["sr", "r6"], ["ri_raw"]),
        node("Min", ["ri_raw", "nine"], ["ri"]),
        node("Add", ["sc", "r6"], ["ci_raw"]),
        node("Min", ["ci_raw", "nine"], ["ci"]),
        node("Gather", ["C", "ci"], ["Mg"], axis=3),
        node("Gather", ["Mg", "ri"], ["patch"], axis=2),
        node("Less", ["r6", "sh"], ["vr"]),
        node("Less", ["r6", "sw"], ["vc"]),
        node("Unsqueeze", ["vr", "ax013"], ["vr4"]),
        node("Unsqueeze", ["vc", "ax012"], ["vc4"]),
        node("And", ["vr4", "vc4"], ["valid"]),
        node("Where", ["valid", "patch", "sent"], ["box_u"]),
        node("Equal", ["box_u", "col8"], ["oh"]),
        node("Pad", ["oh", "out_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(
        nodes,
        "task365_colorid_crop",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    onnx.save(build(), OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
