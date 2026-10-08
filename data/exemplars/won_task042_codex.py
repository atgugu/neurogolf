#!/usr/bin/env python3
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP
from onnx import helper as H


def init(name, arr, proto=None):
    arr = np.asarray(arr)
    if proto is None:
        proto = {
            np.dtype("float32"): TP.FLOAT,
            np.dtype("uint8"): TP.UINT8,
            np.dtype("int8"): TP.INT8,
            np.dtype("int32"): TP.INT32,
            np.dtype("int64"): TP.INT64,
        }[arr.dtype]
    return H.make_tensor(name, proto, arr.shape, arr.ravel())


def check_convinteger_zero_point_padding():
    """ORT must zero-adjust ConvInteger's implicit padding."""
    graph = H.make_graph(
        [
            H.make_node(
                "ConvInteger",
                ["x", "w", "xz", "wz"],
                ["output"],
                pads=[0, 0, 1, 1],
            )
        ],
        "convinteger_xzp_padding_sentinel",
        [H.make_tensor_value_info("x", TP.UINT8, [1, 1, 1, 1])],
        [H.make_tensor_value_info("output", TP.INT32, [1, 1, 2, 2])],
        initializer=[
            init("w", np.array([[[[1]]]], np.int8)),
            init("xz", np.array(1, np.uint8)),
            init("wz", np.array(0, np.int8)),
        ],
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    got = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"x": np.array([[[[2]]]], np.uint8)})[0]
    expect = np.array([[[[1, 0], [0, 0]]]], np.int32)
    if not np.array_equal(got, expect):
        raise RuntimeError(f"ConvInteger x_zero_point padding sentinel failed: {got}")


def main():
    check_convinteger_zero_point_padding()

    # Certified crop: every green source pixel lives in rows/cols 1..8.
    slice_st = np.array([0, 3, 1, 1], np.int64)
    slice_en = np.array([1, 4, 9, 9], np.int64)

    # Live pin exact m=1 detector/stamper. The 4x4 zero-point detector
    # suppresses magnify=2/3 block corners that a 2x2 detector would stamp.
    w_s1 = np.array(
        [[[[0, 0, 0, 0],
           [0, 65, 0, 0],
           [0, 0, 65, 0],
           [0, 0, 0, 0]]],
         [[[0, 0, 0, 0],
           [0, 0, 65, 0],
           [0, 65, 0, 0],
           [0, 0, 0, 0]]]],
        np.uint8,
    )
    b_s1 = np.array([-1, -1], np.int32)
    wst_s1 = np.array(
        [[[[0, 0, 0, 1],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [1, 0, 0, 0]],
          [[1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]]]],
        np.uint8,
    )
    b_direct23 = np.array([-230], np.int32)

    # Pin m=2/m=3 direct classifier, unchanged.
    w_direct23 = np.array(
        [[[[255, 128, 153, 103, 179, 103, 0, 103, 179, 103, 153, 128, 255],
           [154, 128, 52, 179, 255, 179, 0, 179, 255, 179, 52, 128, 154],
           [255, 128, 179, 103, 179, 102, 0, 102, 179, 103, 179, 128, 255],
           [255, 78, 203, 103, 0, 0, 0, 0, 0, 103, 203, 78, 255],
           [178, 52, 254, 28, 0, 0, 0, 0, 0, 28, 254, 52, 178],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [178, 52, 254, 28, 0, 0, 0, 0, 0, 28, 254, 52, 178],
           [255, 78, 203, 103, 0, 0, 0, 0, 0, 103, 203, 78, 255],
           [255, 128, 179, 103, 179, 102, 0, 102, 179, 103, 179, 128, 255],
           [154, 128, 52, 179, 255, 179, 0, 179, 255, 179, 52, 128, 154],
           [255, 128, 153, 103, 179, 103, 0, 103, 179, 103, 153, 128, 255]]]],
        np.uint8,
    )

    w_green2 = np.array([[[[2]]]], np.uint8)

    # Terminal renderer over [green2, hit2]. Valid black cells become
    # [-1, -1] after x_zero_point=1; implicit padding becomes [0, 0].
    w_render = np.zeros((10, 2, 1, 1), np.int8)
    w_render[0, :, 0, 0] = [-1, -1]
    w_render[3, :, 0, 0] = [1, -1]
    w_render[8, :, 0, 0] = [-1, 1]

    inits = [
        init("slice_st", slice_st),
        init("slice_en", slice_en),
        init("q_scale", np.array(1.0, np.float32)),
        init("det_y_s1", np.array(0.5, np.float32)),
        init("direct_y", np.array(30.0, np.float32)),
        init("zero_u8", np.array(0, np.uint8)),
        init("one_u8", np.array(1, np.uint8)),
        init("zero_i8", np.array(0, np.int8)),
        init("det_w_zero", np.array(64, np.uint8)),
        init("direct_w_zero", np.array(128, np.uint8)),
        init("W_s1", w_s1),
        init("B_s1", b_s1),
        init("Wst_s1", wst_s1),
        init("W_direct23", w_direct23),
        init("B_direct23", b_direct23),
        init("W_green2", w_green2),
        init("W_render", w_render),
    ]

    nodes = [
        H.make_node("Slice", ["input", "slice_st", "slice_en"], ["green8_f"]),
        H.make_node("Cast", ["green8_f"], ["green8_u8"], to=TP.UINT8),
        H.make_node(
            "QLinearConv",
            ["green8_u8", "q_scale", "zero_u8", "W_s1", "q_scale", "det_w_zero",
             "det_y_s1", "zero_u8", "B_s1"],
            ["hit_s1"],
            pads=[1, 1, 1, 1],
        ),
        H.make_node(
            "QLinearConv",
            ["hit_s1", "q_scale", "zero_u8", "Wst_s1", "q_scale", "zero_u8",
             "q_scale", "zero_u8"],
            ["m1_10"],
            pads=[3, 3, 3, 3],
        ),
        H.make_node(
            "QLinearConv",
            ["green8_u8", "q_scale", "zero_u8", "W_direct23", "q_scale", "direct_w_zero",
             "direct_y", "zero_u8", "B_direct23"],
            ["direct23_10"],
            pads=[6, 7, 6, 7],
        ),
        H.make_node("Max", ["m1_10", "direct23_10"], ["hit2_10"]),
        H.make_node(
            "QLinearConv",
            ["green8_u8", "q_scale", "zero_u8", "W_green2", "q_scale", "zero_u8",
             "q_scale", "zero_u8"],
            ["green2_10"],
            pads=[1, 1, 1, 1],
        ),
        H.make_node("Concat", ["green2_10", "hit2_10"], ["feat2"], axis=1),
        H.make_node(
            "ConvInteger",
            ["feat2", "W_render", "one_u8", "zero_i8"],
            ["output"],
            pads=[0, 0, 20, 20],
        ),
    ]

    graph = H.make_graph(
        nodes,
        "task042_att9_zero_point_two_feature_terminal",
        [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [H.make_tensor_value_info("output", TP.INT32, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "task042.onnx")


if __name__ == "__main__":
    main()
