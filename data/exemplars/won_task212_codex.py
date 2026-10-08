from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import relower_onehot_plane  # noqa: E402


W = 10


def init(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def c(name: str, value) -> str:
        inits.append(init(name, value))
        return name

    def relower(name: str, channel: int, crop) -> str:
        rnodes, rinits, _shape = relower_onehot_plane(
            "input",
            name,
            channel=channel,
            crop=crop,
            dtype="u8",
            starts_name=f"{name}_starts",
            ends_name=f"{name}_ends",
        )
        nodes.extend(rnodes)
        for iname, arr in rinits:
            c(iname, arr)
        return name

    # Color-1/blue seeds are guaranteed to be above rows 0:5 or below rows 5:10.
    blue_top = relower("blue_top_seed", 1, ((0, 1), (1, 2), (0, 5), (0, W)))
    blue_bottom = relower("blue_bottom_seed", 1, ((0, 1), (1, 2), (5, W), (0, W)))
    red_seed = relower("red_seed", 2, ((0, 1), (2, 3), (0, W), (0, W)))
    gray = relower("gray", 5, ((0, 1), (5, 6), (0, W), (0, W)))

    ax3 = c("ax3", np.array([3], np.int64))
    st0 = c("st0", np.array([0], np.int64))
    en1 = c("en1", np.array([1], np.int64))
    bias = c("bias_plane", np.ones((1, 1, W, W), dtype=np.uint8))

    weights = np.zeros((10, 4, 1, 1), dtype=np.int8)
    # Feature order: [blue_ray, gray_row, red_ray, constant_bias].
    # Red can overlap the gray horizon row; the gray classifier covers both
    # (0, 1, 0, 1) and (0, 1, 1, 1).
    weights[0, :, 0, 0] = [-5, -5, -5, 1]
    weights[1, :, 0, 0] = [1, -5, -5, 0]
    weights[2, :, 0, 0] = [-5, -5, 1, 0]
    weights[5, :, 0, 0] = [-5, 1, 0, 0]
    c("render_weights", weights)

    nodes.extend(
        [
            helper.make_node(
                "MaxPool",
                [blue_top],
                ["blue_top_pool"],
                kernel_shape=[5, 1],
                pads=[0, 0, 4, 0],
                strides=[1, 1],
            ),
            helper.make_node(
                "MaxPool",
                [blue_bottom],
                ["blue_bottom_pool"],
                kernel_shape=[5, 1],
                pads=[4, 0, 0, 0],
                strides=[1, 1],
            ),
            helper.make_node("Concat", ["blue_top_pool", "blue_bottom_pool"], ["blue"], axis=2),
            helper.make_node("Slice", [gray, st0, en1, ax3], ["gray_col"]),
            helper.make_node(
                "MaxPool",
                ["gray_col"],
                ["gray_up"],
                kernel_shape=[W, 1],
                pads=[0, 0, W - 1, 0],
                strides=[1, 1],
            ),
            helper.make_node("Greater", ["gray_up", "gray_col"], ["above"]),
            helper.make_node(
                "MaxPool",
                [red_seed],
                ["red_above_pool"],
                kernel_shape=[W, 1],
                pads=[W - 1, 0, 0, 0],
                strides=[1, 1],
            ),
            helper.make_node(
                "MaxPool",
                [red_seed],
                ["red_below_pool"],
                kernel_shape=[W, 1],
                pads=[0, 0, W - 1, 0],
                strides=[1, 1],
            ),
            helper.make_node("Where", ["above", "red_above_pool", "red_below_pool"], ["red"]),
            helper.make_node("Concat", ["blue", gray, "red", bias], ["render4"], axis=1),
            helper.make_node(
                "ConvInteger",
                ["render4", "render_weights"],
                ["output"],
                pads=[0, 0, 20, 20],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task212_relower_bias_renderer",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("task212.onnx")
    onnx.save(build(), out)
    print(out)
