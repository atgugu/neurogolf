#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as h
from onnx import numpy_helper as nh


def const(name: str, arr) -> onnx.TensorProto:
    return nh.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    idx = np.arange(30, dtype=np.float32)
    powers = np.stack([np.ones(30, dtype=np.float32), idx, idx * idx])
    pick_i = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    scalar = lambda x: np.array([[[x]]], dtype=np.float32)

    inits = [
        const("w01", np.array([0.0] + [1.0] * 9, dtype=np.float32).reshape(1, 10, 1, 1)),
        const("T", powers),
        const("pick_i", pick_i),
        const("zero", scalar(0.0)),
        const("one", scalar(1.0)),
        const("neg_one", scalar(-1.0)),
        const("two", scalar(2.0)),
        const("neg_two", scalar(-2.0)),
        const("k_shape", np.array([1, 2, 3, 3], dtype=np.int64)),
        const("score2", np.array([[-1.0, 0.0], [0.0, 0.5]], dtype=np.float32)),
        const("sign", np.array([-1.0] + [1.0] * 9, dtype=np.float32)),
    ]

    nodes = [
        h.make_node("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=0),
        h.make_node(
            "Einsum",
            ["input", "w01", "T", "pick_i"],
            ["seed_row"],
            equation="nchw,pcij,fh,f->nij",
        ),
        h.make_node(
            "Einsum",
            ["input", "w01", "T", "pick_i"],
            ["seed_col"],
            equation="nchw,pcij,fw,f->nij",
        ),
        h.make_node("Mul", ["seed_row", "seed_row"], ["seed_row2"]),
        h.make_node("Mul", ["seed_col", "seed_col"], ["seed_col2"]),
        h.make_node("Sub", ["seed_col2", "seed_row2"], ["seed_diff"]),
        h.make_node("Mul", ["seed_row", "two"], ["two_r"]),
        h.make_node("Mul", ["seed_col", "neg_two"], ["neg_two_c"]),
        h.make_node(
            "Concat",
            [
                "seed_diff",
                "neg_two_c",
                "one",
                "two_r",
                "zero",
                "zero",
                "neg_one",
                "zero",
                "zero",
                "one",
                "zero",
                "zero",
                "zero",
                "zero",
                "zero",
                "zero",
                "zero",
                "zero",
            ],
            ["k_flat"],
            axis=1,
        ),
        h.make_node("Reshape", ["k_flat", "k_shape"], ["K"]),
        h.make_node(
            "Einsum",
            ["present", "input", "K", "K", "T", "T", "T", "T", "score2", "sign"],
            ["output"],
            equation="nq,nsyx,nmac,node,ay,cx,dy,ex,mo,q->nqyx",
        ),
    ]

    graph = h.make_graph(
        nodes,
        "task141_shared_power_terminal",
        [h.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [h.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def main() -> int:
    model = build()
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task141.onnx")
    print("wrote task141.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
