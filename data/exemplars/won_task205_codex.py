"""Task 205: pin-equivalent staged filter with a uint8 MaxPool detector."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
BASE = HERE / "refs/kaggle_proven_best.onnx"
OUT = HERE / "task205.onnx"


def node(op, ins, outs, **attrs):
    return helper.make_node(op, ins, outs, **attrs)


def build():
    # Work only from the exact Kaggle-paying member.  The score/Bidx and all
    # staged thresholds stay byte-for-byte equivalent; only exact binary state
    # after Greater is relowered.
    m = onnx.load(BASE)
    m.graph.initializer.extend([
        numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "minus_one"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "u8zero"),
    ])

    out = []
    for n in m.graph.node:
        name = n.output[0]

        # rfl1 must remain float for the following Einsum.  The duplicate rfl2
        # and cfl2 copies exist only for run detection, so carry them as u8.
        if name == "rfl2":
            continue
        if name == "cfl2":
            continue

        # Exact at-least-six detector.  MaxPool(complement)==0 iff all six
        # flags are one; 1-pool is therefore an exact 0/1 score for ArgMax.
        if name == "rfl2u":
            out += [
                node("Unsqueeze", ["rflb1"], ["rfl2u"], axes=[1]),
                node("Unsqueeze", ["cflb2"], ["cfl2u"], axes=[1]),
                node("Not", ["rfl2u"], ["nrfl2ub"]),
                node("Not", ["cfl2u"], ["ncfl2ub"]),
                node("Cast", ["nrfl2ub"], ["nrfl2u"], to=TensorProto.UINT8),
                node("Cast", ["ncfl2ub"], ["ncfl2u"], to=TensorProto.UINT8),
                node("MaxPool", ["nrfl2u"], ["rbad6"], kernel_shape=[6], strides=[1]),
                node("MaxPool", ["ncfl2u"], ["cbad6"], kernel_shape=[6], strides=[1]),
                node("Equal", ["rbad6", "u8zero"], ["rrunb"]),
                node("Equal", ["cbad6", "u8zero"], ["crunb"]),
                node("Cast", ["rrunb"], ["rrun"], to=TensorProto.UINT8),
                node("Cast", ["crunb"], ["crun"], to=TensorProto.UINT8),
            ]
            continue
        if name in {"cfl2u", "nrfl2u", "ncfl2u", "rbad6", "cbad6", "rrun", "crun"}:
            continue

        # Use the one existing int32 coordinate table everywhere.  This also
        # shares the four scalar casts with Gather index construction.
        if name == "r0f":
            out.append(node("Cast", ["r0i"], ["r0i32"], to=TensorProto.INT32))
            continue
        if name == "r1f":
            out.append(node("Cast", ["r1i"], ["r1i32"], to=TensorProto.INT32))
            continue
        if name == "c0f":
            out.append(node("Cast", ["c0i"], ["c0i32"], to=TensorProto.INT32))
            continue
        if name == "c1f":
            out.append(node("Cast", ["c1i"], ["c1i32"], to=TensorProto.INT32))
            continue
        if name == "rgeb":
            out += [
                node("Unsqueeze", ["r0i32"], ["r0i32u"], axes=[1]),
                node("GreaterOrEqual", ["iota30i", "r0i32u"], ["rgeb"]),
            ]
            continue
        if name == "rleb":
            out += [
                node("Unsqueeze", ["r1i32"], ["r1i32u"], axes=[1]),
                node("LessOrEqual", ["iota30i", "r1i32u"], ["rleb"]),
            ]
            continue
        if name == "cgeb":
            out += [
                node("Unsqueeze", ["c0i32"], ["c0i32u"], axes=[1]),
                node("GreaterOrEqual", ["iota30i", "c0i32u"], ["cgeb"]),
            ]
            continue
        if name == "cleb":
            out += [
                node("Unsqueeze", ["c1i32"], ["c1i32u"], axes=[1]),
                node("LessOrEqual", ["iota30i", "c1i32u"], ["cleb"]),
            ]
            continue
        if name in {"r0i32", "c0i32"}:
            continue
        if name == "khf":
            out.append(node("Sub", ["r1i32", "r0i32"], ["khf"]))
            continue
        if name == "kwf":
            out.append(node("Sub", ["c1i32", "c0i32"], ["kwf"]))
            continue
        if name == "rwoB":
            out += [
                node("Unsqueeze", ["khf"], ["khfu"], axes=[1]),
                node("LessOrEqual", ["iota30i", "khfu"], ["rwoB"]),
            ]
            continue
        if name == "cwoB":
            out += [
                node("Unsqueeze", ["kwf"], ["kwfu"], axes=[1]),
                node("LessOrEqual", ["iota30i", "kwfu"], ["cwoB"]),
            ]
            continue

        # Exclude B directly from the in-box color score.  Every non-B count is
        # nonnegative, making -1 an exact sentinel for the existing ArgMax.
        if name == "notB":
            out.append(node("Where", ["selBb", "minus_one", "cntin"], ["cinm"]))
            continue
        if name == "cinm":
            continue

        # Keep row/column factors bool until each packed [1,2,30] tensor is
        # ready.  The final Einsum remains the single fp16 renderer.
        if name in {"rwo", "cwo", "U", "V"}:
            if name == "V":
                out += [
                    node("Unsqueeze", ["rwoB"], ["rwinu"], axes=[1]),
                    node("Unsqueeze", ["Ub"], ["Uu"], axes=[1]),
                    node("Concat", ["rwinu", "Uu"], ["Rb"], axis=1),
                    node("Cast", ["Rb"], ["R"], to=TensorProto.FLOAT16),
                    node("Unsqueeze", ["cwoB"], ["cwinu"], axes=[1]),
                    node("Unsqueeze", ["Vb"], ["Vu"], axes=[1]),
                    node("Concat", ["cwinu", "Vu"], ["Cb"], axis=1),
                    node("Cast", ["Cb"], ["C"], to=TensorProto.FLOAT16),
                ]
            continue
        if name in {"rwinu", "Uu", "R", "cwinu", "Vu", "C"}:
            continue

        out.append(n)

    del m.graph.node[:]
    m.graph.node.extend(out)
    kept = [x for x in m.graph.initializer if x.name not in {"one", "iota30f"}]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)
    del m.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m, full_check=True)
    return m


def budget(m):
    v = {x.name: x for x in list(m.graph.value_info) + list(m.graph.output)}
    mem = 0
    for n in m.graph.node:
        for name in n.output:
            if name and name != "output":
                t = v[name].type.tensor_type
                mem += int(np.prod([d.dim_value for d in t.shape.dim])) * np.dtype(
                    helper.tensor_dtype_to_np_dtype(t.elem_type)
                ).itemsize
    params = sum(numpy_helper.to_array(x).size for x in m.graph.initializer)
    cost = mem + params
    print(f"BUDGET: params={params} memory={mem} cost={cost} pts={25-math.log(cost):.4f}")


if __name__ == "__main__":
    model = build()
    onnx.save(model, OUT)
    budget(model)
    print(f"saved {OUT}")
