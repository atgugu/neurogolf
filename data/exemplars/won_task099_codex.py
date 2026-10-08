#!/usr/bin/env python3
"""task099 attempt 5: scalar rule-circuit replacement.

The certified pin already has the cheapest valid 30x30 bool renderer:
native scalar ids -> Pad(clear=10) -> Equal(color_bank).  This attempt keeps
that tail and replaces the over-broad Hodel-derived middle circuit.  It splits
the six possible marker labels directly from the two existing marker vectors,
then computes the finite generator cases as scalar uint8 expressions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


BASE = Path("task099_pin.onnx")
OUT = Path("task099.onnx")


def _drop_initializers(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [i for i in model.graph.initializer if i.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def _add_init(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    _drop_initializers(model, {name})
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))


class Circuit:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.i = 0

    def _name(self, prefix: str) -> str:
        self.i += 1
        return f"{prefix}_{self.i}"

    def max(self, a: str, b: str, name: str | None = None) -> str:
        out = name or self._name("mx")
        self.nodes.append(helper.make_node("Max", [a, b], [out]))
        return out

    def min(self, a: str, b: str, name: str | None = None) -> str:
        out = name or self._name("mn")
        self.nodes.append(helper.make_node("Min", [a, b], [out]))
        return out

    def mul(self, a: str, b: str, name: str | None = None) -> str:
        out = name or self._name("mul")
        self.nodes.append(helper.make_node("Mul", [a, b], [out]))
        return out

    def sub(self, a: str, b: str, name: str | None = None) -> str:
        out = name or self._name("sub")
        self.nodes.append(helper.make_node("Sub", [a, b], [out]))
        return out

    def pres(self, a: str, name: str | None = None) -> str:
        return self.min(a, "one_u8", name)


def _finish(model: onnx.ModelProto, path: Path) -> None:
    model.ir_version = 10
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 18
    del model.graph.output[:]
    model.graph.output.extend(
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])]
    )
    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)
    print(f"wrote {path}")


def build(path: Path = OUT) -> None:
    model = onnx.load(BASE)
    _add_init(model, "zero_u8", np.zeros((1, 1, 1), dtype=np.uint8))
    _add_init(model, "scalar_shape", np.array([10, 10], dtype=np.int64))
    _add_init(model, "pad_spec_scalar", np.array([0, 0, 20, 20], dtype=np.int64))
    _add_init(
        model,
        "gate_idx",
        np.array([[[1, 1, 0], [1, 1, 1]]], dtype=np.int64),
    )

    # Keep marker extraction (nodes 0..7), gate read (10..11), and tail Pad/Equal.
    by_output = {out: node for node in model.graph.node for out in node.output}
    del by_output["gates2"].attribute[:]
    by_output["gates2"].attribute.extend([helper.make_attribute("batch_dims", 1)])
    keep_outputs = {
        "top_marker_channels",
        "top_marker_label_f",
        "top_marker_label_3d",
        "bottom_marker_channels",
        "bottom_marker_label_f",
        "bottom_marker_label_3d",
        "gates2",
        "gates_u8",
        "scalar_padded",
        "output",
    }
    kept = [node for node in model.graph.node if any(o in keep_outputs for o in node.output)]

    c = Circuit()
    c.nodes.append(
        helper.make_node(
            "Split", ["top_marker_label_3d"], ["top_fixed_row", "top_ambig_row"], axis=1, num_outputs=2
        )
    )
    c.nodes.append(
        helper.make_node(
            "Split", ["top_fixed_row"], ["f0", "f1"], axis=2, num_outputs=2
        )
    )
    c.nodes.append(
        helper.make_node(
            "Split", ["top_ambig_row"], ["a0", "a1"], axis=2, num_outputs=2
        )
    )
    c.nodes.append(
        helper.make_node(
            "Split", ["bottom_marker_label_3d"], ["b0", "b1"], axis=1, num_outputs=2
        )
    )
    c.nodes.append(
        helper.make_node("Split", ["gates_u8"], ["g0", "g1"], axis=1, num_outputs=2)
    )

    common0 = c.max("f0", "a0", "common0")
    common1 = c.max("f1", "a1", "common1")
    t0 = c.mul(common0, "g0", "t0")
    t1 = c.mul(common1, "g1", "t1")
    s0 = c.sub(common0, t0, "s0")
    s1 = c.sub(common1, t1, "s1")
    super_label = c.max(t0, t1, "super_label")
    top_all = c.max(common0, common1, "top_all")

    pa0 = c.pres("a0", "pa0")
    pa1 = c.pres("a1", "pa1")
    pambig = c.max(pa0, pa1, "pambig")
    pc0 = c.pres(common0, "pc0")
    pc1 = c.pres(common1, "pc1")

    c.nodes.append(helper.make_node("Max", ["one_u8", s0, s1], ["top_super_shifted"]))
    top_super_shifted = "top_super_shifted"

    r1c0 = c.max(pc0, s0, "r1c0")
    c.nodes.append(helper.make_node("Max", [t0, "one_u8", s0, s1], ["r1c2"]))
    r1c2 = "r1c2"
    c.nodes.append(helper.make_node("Max", [t1, "one_u8", s0, s1], ["r1c3"]))
    r1c3 = "r1c3"
    r1c5 = c.max(pc1, s1, "r1c5")
    r2c1 = c.max("one_u8", t0, "r2c1")
    c.nodes.append(helper.make_node("Max", [super_label, s0, "one_u8"], ["r2c2"]))
    r2c2 = "r2c2"
    c.nodes.append(helper.make_node("Max", [super_label, s1, "one_u8"], ["r2c3"]))
    r2c3 = "r2c3"
    r2c4 = c.max("one_u8", t1, "r2c4")
    r3c1 = c.max("one_u8", common0, "r3c1")
    r3c4 = c.max("one_u8", common1, "r3c4")
    r4c1 = c.max("one_u8", "a0", "r4c1")
    r4c4 = c.max("one_u8", "a1", "r4c4")
    top_fixed_ambig = c.max(r4c1, r4c4, "top_fixed_ambig")

    pb0 = c.pres("b0", "pb0")
    pb1 = c.pres("b1", "pb1")
    bottom_pair = c.max("b0", "b1", "bottom_pair")
    bottom_pair_presence = c.pres(bottom_pair, "bottom_pair_presence")
    bottom_cross0 = c.max(pb0, "b1", "bottom_cross0")
    bottom_cross1 = c.max(pb1, "b0", "bottom_cross1")

    canvas54 = c.max("b0", pambig, "canvas54")
    canvas55 = c.max("b0", pa1, "canvas55")

    z = "zero_u8"
    flat = [
        # row 0
        t0,
        super_label,
        super_label,
        super_label,
        super_label,
        t1,
        z,
        z,
        z,
        z,
        # row 1
        r1c0,
        top_super_shifted,
        r1c2,
        r1c3,
        top_super_shifted,
        r1c5,
        z,
        z,
        z,
        z,
        # row 2
        pc0,
        r2c1,
        r2c2,
        r2c3,
        r2c4,
        pc1,
        z,
        z,
        z,
        z,
        # row 3
        pc0,
        r3c1,
        top_all,
        top_all,
        r3c4,
        pc1,
        z,
        z,
        z,
        z,
        # row 4
        pc0,
        r4c1,
        top_fixed_ambig,
        top_fixed_ambig,
        r4c4,
        pc1,
        z,
        z,
        z,
        z,
        # row 5
        pa0,
        pambig,
        pambig,
        pambig,
        canvas54,
        canvas55,
        "b0",
        "b0",
        "b0",
        z,
        # row 6
        z,
        z,
        z,
        z,
        bottom_cross0,
        bottom_cross0,
        bottom_pair,
        bottom_cross0,
        bottom_cross0,
        z,
        # row 7
        z,
        z,
        z,
        z,
        bottom_pair_presence,
        bottom_cross1,
        bottom_pair,
        bottom_cross1,
        bottom_pair_presence,
        z,
        # row 8
        z,
        z,
        z,
        z,
        bottom_pair_presence,
        bottom_pair,
        bottom_pair,
        bottom_pair,
        bottom_pair_presence,
        z,
        # row 9
        z,
        z,
        z,
        z,
        bottom_pair_presence,
        bottom_pair_presence,
        bottom_pair_presence,
        bottom_pair_presence,
        bottom_pair_presence,
        z,
    ]
    c.nodes.append(helper.make_node("Concat", flat, ["scalar_flat"], axis=0))
    c.nodes.append(helper.make_node("Reshape", ["scalar_flat", "scalar_shape"], ["scalar_grid"]))

    # Replace everything between marker/gate extraction and Pad.
    pad = by_output["scalar_padded"]
    equal = by_output["output"]
    ordered = []
    for node in model.graph.node:
        if any(o in keep_outputs for o in node.output) and node.output[0] not in {
            "scalar_padded",
            "output",
        }:
            ordered.append(node)
    ordered.extend(c.nodes)
    ordered.extend([pad, equal])
    del model.graph.node[:]
    model.graph.node.extend(ordered)

    used = {name for node in model.graph.node for name in node.input}
    kept_initializers = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    _finish(model, path)


if __name__ == "__main__":
    build()
