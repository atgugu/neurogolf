from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import relower_onehot_plane


OUT = Path(__file__).with_name("task055.onnx")


def init(inits, name: str, arr) -> None:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))


def node(nodes, op: str, inputs: list[str], output: str, **attrs) -> None:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))


def add_relower(nodes, inits, name: str, crop) -> str:
    ns, vals, _shape = relower_onehot_plane(
        "input",
        name,
        channel=8,
        crop=crop,
        dtype="u8",
        starts_name=f"{name}_starts",
        ends_name=f"{name}_ends",
    )
    nodes.extend(ns)
    for n, a in vals:
        init(inits, n, a)
    return name


def low_rank_terms() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four terminal terms: three classify zero panels, one preserves cyan.

    Attempt 10 consumed one-hot row/column band descriptors. This build feeds cumulative
    step descriptors `[idx < sep1, idx < sep2, idx < 30]`; transform the row/column
    vectors so each band sees exactly the same scalar as in the verified renderer.
    """
    r_basis = np.array(
        [
            [1.0, 1.0, 1.0],
            [-0.61341786, -1.6051494, 0.7293494],
            [0.80613935, -0.47637674, 0.16333994],
        ],
        dtype=np.float32,
    )
    c_basis = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.13573073, 2.3103635, -0.78719276],
            [0.5802844, -0.19550583, 0.56581783],
        ],
        dtype=np.float32,
    )
    out_weights = np.array(
        [
            [-0.11984304, 0.06444215, 2.7790515],
            [-0.3875657, 0.29536775, 0.3177597],
            [-7.447775, -2.5974748, -24.533407],
            [-0.37085176, 0.14007626, -1.0902057],
            [-0.73064226, -0.12147164, -2.9091117],
            [-0.1, 0.0, 0.0],
            [-0.22370642, -0.0872882, 0.0],
            [-0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    term_in = np.zeros((4, 10), dtype=np.float32)
    term_out = np.zeros((4, 10), dtype=np.float32)
    term_row = np.ones((4, 3), dtype=np.float32)
    term_col = np.ones((4, 3), dtype=np.float32)

    term_in[:3, 0] = 1.0
    term_out[:3, :] = out_weights.T
    term_row[:3, :] = r_basis
    term_col[:3, :] = c_basis

    term_in[3, 8] = 1.0
    term_out[3, 8] = 1.0
    # If step state rows are [[1,1,1], [0,1,1], [0,0,1]], then
    # step @ [top-mid, mid-bot, bot] == [top, mid, bot].
    term_row = np.stack(
        [
            term_row[:, 0] - term_row[:, 1],
            term_row[:, 1] - term_row[:, 2],
            term_row[:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    term_col = np.stack(
        [
            term_col[:, 0] - term_col[:, 1],
            term_col[:, 1] - term_col[:, 2],
            term_col[:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    return term_in, term_out, term_row, term_col


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    h8 = add_relower(nodes, inits, "h8_edge", ((0, 1), (8, 9), (0, 30), (0, 1)))
    v8 = add_relower(nodes, inits, "v8_edge", ((0, 1), (8, 9), (0, 1), (0, 30)))

    init(inits, "row_idx", np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1))
    init(inits, "col_idx", np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30))
    init(inits, "limit30", np.array([[[[30]]]], dtype=np.int64))

    term_in, term_out, term_row, term_col = low_rank_terms()
    init(inits, "term_in", term_in)
    init(inits, "term_out", term_out)
    init(inits, "term_row", term_row)
    init(inits, "term_col", term_col)

    node(nodes, "ArgMax", [h8], "r1", axis=2, keepdims=1, select_last_index=0)
    node(nodes, "ArgMax", [h8], "r2", axis=2, keepdims=1, select_last_index=1)
    node(nodes, "ArgMax", [v8], "c1", axis=3, keepdims=1, select_last_index=0)
    node(nodes, "ArgMax", [v8], "c2", axis=3, keepdims=1, select_last_index=1)
    node(nodes, "Concat", ["r1", "r2", "limit30"], "row_thresholds", axis=3)
    node(nodes, "Concat", ["c1", "c2", "limit30"], "col_thresholds", axis=2)
    node(nodes, "Less", ["row_idx", "row_thresholds"], "row_step_b")
    node(nodes, "Less", ["col_idx", "col_thresholds"], "col_step_b")
    node(nodes, "Cast", ["row_step_b"], "row_oh", to=TensorProto.FLOAT)
    node(nodes, "Cast", ["col_step_b"], "col_oh", to=TensorProto.FLOAT)
    node(
        nodes,
        "Einsum",
        ["input", "row_oh", "col_oh", "term_in", "term_out", "term_row", "term_col"],
        "output",
        equation="nchw,nqha,nqbw,tc,tk,ta,tb->nkhw",
    )

    graph = helper.make_graph(
        nodes,
        "task055_argmax_separator_rank4",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(OUT)
