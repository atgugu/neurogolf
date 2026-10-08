import os
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path((os.environ.get("NEUROGOLF_CLEAN", "") + ""))
SRC = ROOT / "tasks" / "task208" / "_live_pin.onnx"
OUT = Path(__file__).resolve().parent / "task208.onnx"


def init(name: str, arr, dtype=None):
    a = np.asarray(arr, dtype=dtype)
    return numpy_helper.from_array(a, name)


def node(op: str, ins: list[str], out: str, **attrs):
    return helper.make_node(op, ins, [out], name=out, **attrs)


def bitset_matcher_nodes() -> list[onnx.NodeProto]:
    nodes: list[onnx.NodeProto] = []

    # Pack the black channel into one uint32 word per row.  After shifting by
    # two, bit 0 is candidate interior column 2 and bit 15 is column 17.
    nodes.append(node("Einsum", ["input", "ch0_f32", "col_bits_f32"], "row_words_f32", equation="nchw,c,w->nh"))
    nodes.append(node("Cast", ["row_words_f32"], "row_words", to=TensorProto.UINT32))
    nodes.append(node("BitShift", ["row_words", "two_u32"], "row_words_s2", direction="RIGHT"))
    nodes.append(node("BitwiseAnd", ["row_words_s2", "mask20_u32"], "row_bits20"))

    prev = "row_bits20"
    run_names: list[str] = []
    for k in range(1, 5):
        shifted = f"row_bits_sh{k}"
        run = f"run{k + 1}"
        nodes.append(node("BitShift", ["row_bits20", f"shift{k}_u32"], shifted, direction="RIGHT"))
        nodes.append(node("BitwiseAnd", [prev, shifted], run))
        run_names.append(run)
        prev = run

    nodes.append(node("Equal", ["inner_width_u8", "three_u8"], "is_w3"))
    nodes.append(node("Equal", ["inner_width_u8", "four_u8"], "is_w4"))
    nodes.append(node("Equal", ["inner_width_u8", "five_u8"], "is_w5"))
    nodes.append(node("Where", ["is_w3", "run3", "run2"], "run_w3"))
    nodes.append(node("Where", ["is_w4", "run4", "run_w3"], "run_w4"))
    nodes.append(node("Where", ["is_w5", "run5", "run_w4"], "run_width"))
    nodes.append(node("BitwiseAnd", ["run_width", "mask16_u32"], "run_valid"))

    # Candidate top rows are global rows 2..17.  Every slice below is length 16;
    # rows beyond the 21x21 ARC grid are zero in the padded one-hot input.
    for off in range(5):
        nodes.append(
            helper.make_node(
                "Slice",
                ["run_valid", f"row{off}_start", f"row{off}_end", "axis1_i64"],
                [f"vr{off}"],
                name=f"vr{off}",
            )
        )
    nodes.append(node("BitwiseAnd", ["vr0", "vr1"], "v2"))
    nodes.append(node("BitwiseAnd", ["v2", "vr2"], "v3"))
    nodes.append(node("BitwiseAnd", ["v3", "vr3"], "v4"))
    nodes.append(node("BitwiseAnd", ["v4", "vr4"], "v5"))

    nodes.append(node("Equal", ["inner_height_u8", "three_u8"], "is_h3"))
    nodes.append(node("Equal", ["inner_height_u8", "four_u8"], "is_h4"))
    nodes.append(node("Equal", ["inner_height_u8", "five_u8"], "is_h5"))
    nodes.append(node("Where", ["is_h3", "v3", "v2"], "win_h3"))
    nodes.append(node("Where", ["is_h4", "v4", "win_h3"], "win_h4"))
    nodes.append(node("Where", ["is_h5", "v5", "win_h4"], "win_bits"))

    # Remove the already framed hole, leaving exactly the target h*w all-black
    # window under the generator's accidental-hole rejection.
    nodes.append(node("Cast", ["source_left64"], "source_left_u32", to=TensorProto.UINT32))
    nodes.append(node("BitShift", ["one_u32", "source_left_u32"], "source_bit", direction="LEFT"))
    nodes.append(node("BitwiseXor", ["mask16_u32", "source_bit"], "clear_source_col"))
    nodes.append(node("BitwiseAnd", ["win_bits", "clear_source_col"], "win_bits_col_cleared"))
    nodes.append(node("Equal", ["axis16_i64", "source_top64"], "source_row_sel"))
    nodes.append(node("Where", ["source_row_sel", "win_bits_col_cleared", "win_bits"], "win_no_source"))

    nodes.append(node("ArgMax", ["win_no_source"], "target_rel_top64", axis=1, keepdims=1))
    nodes.append(
        helper.make_node(
            "GatherElements",
            ["win_no_source", "target_rel_top64"],
            ["target_word"],
            name="target_word",
            axis=1,
        )
    )
    nodes.append(node("Equal", ["target_word", "powers16_u32"], "target_col_onehot_b"))
    nodes.append(node("Cast", ["target_col_onehot_b"], "target_col_onehot_u8", to=TensorProto.UINT8))
    nodes.append(node("ArgMax", ["target_col_onehot_u8"], "target_rel_left64", axis=1, keepdims=1))
    nodes.append(node("Add", ["target_rel_top64", "two_i64"], "target_top64"))
    nodes.append(node("Add", ["target_rel_left64", "two_i64"], "target_left64"))
    return nodes


def build() -> onnx.ModelProto:
    model = onnx.load(SRC)
    old = list(model.graph.node)

    new_nodes: list[onnx.NodeProto] = []
    for i, n in enumerate(old):
        if i == 14:
            new_nodes.append(node("ReduceMax", ["source_cols", "axis1_i64"], "outer_height", keepdims=1))
            continue
        if i < 22:
            new_nodes.append(n)
            continue
        if i == 22:
            new_nodes.extend(bitset_matcher_nodes())
            continue
        if 22 < i <= 45:
            continue
        new_nodes.append(n)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    existing = {t.name for t in model.graph.initializer}
    extra = [
        init("ch0_f32", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.float32),
        init("col_bits_f32", [float(1 << c) if c < 21 else 0.0 for c in range(30)], np.float32),
        init("axis1_i64", [1], np.int64),
        init("two_u32", np.array(2, dtype=np.uint32)),
        init("one_u32", np.array(1, dtype=np.uint32)),
        init("mask20_u32", np.array((1 << 20) - 1, dtype=np.uint32)),
        init("mask16_u32", np.array((1 << 16) - 1, dtype=np.uint32)),
        init("shift1_u32", np.array(1, dtype=np.uint32)),
        init("shift2_u32", np.array(2, dtype=np.uint32)),
        init("shift3_u32", np.array(3, dtype=np.uint32)),
        init("shift4_u32", np.array(4, dtype=np.uint32)),
        init("three_u8", np.array(3, dtype=np.uint8)),
        init("four_u8", np.array(4, dtype=np.uint8)),
        init("five_u8", np.array(5, dtype=np.uint8)),
        init("axis16_i64", np.arange(16, dtype=np.int64).reshape(1, 16)),
        init("powers16_u32", (1 << np.arange(16, dtype=np.uint32)).reshape(1, 16), None),
    ]
    for off in range(5):
        extra.append(init(f"row{off}_start", [2 + off], np.int64))
        extra.append(init(f"row{off}_end", [18 + off], np.int64))

    for t in extra:
        if t.name not in existing:
            model.graph.initializer.append(t)

    used = {name for n in model.graph.node for name in n.input if name}
    kept = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    del model.graph.value_info[:]
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = 18
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build(), OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
