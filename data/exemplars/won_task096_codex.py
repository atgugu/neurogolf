import os
#!/usr/bin/env python3
"""Task096 moment-signature decoder rebuild.

True rule: recover each clipped corner/L-ring fragment's radius and arm length,
then redraw the rings concentrically on the runtime background.  This build
keeps the known-good color recovery and Pad->Equal terminal renderer from the
pin, but replaces the row/column profile decoder with translation-invariant
second-moment signatures plus a tiny ambiguity correction layer.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto, numpy_helper


BASE = Path((os.environ.get("NEUROGOLF_CLEAN", "") + "/tasks/task096/_live_pin.onnx"))
OUT = Path("task096.onnx")


PAIRS = []
for idx in range(6):
    lo = min(idx + 1, 2)
    hi = idx + (0 if idx > 1 else 1)
    for length in range(lo, hi + 1):
        PAIRS.append((idx, length))


def ring_points(height, width, row, col, idx, length):
    pts = set()
    quadrants = 0
    for r, c in [(-idx, -idx), (-idx, idx), (idx, -idx), (idx, idx)]:
        qpts = set()
        for i in range(length):
            for dr, dc in [
                (r, c + (i if c < 0 else -i)),
                (r + (i if r < 0 else -i), c),
            ]:
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    pts.add((rr, cc))
                    qpts.add((rr, cc))
        if qpts:
            quadrants += 1
    return pts, quadrants


def moment_key(pts):
    n = len(pts)
    sy = sum(r for r, _ in pts)
    sx = sum(c for _, c in pts)
    sy2 = sum(r * r for r, _ in pts)
    sx2 = sum(c * c for _, c in pts)
    vy = n * sy2 - sy * sy
    vx = n * sx2 - sx * sx
    return n, min(vy, vx), max(vy, vx)


AMBIGUITY_FIXES = [
    (86, "count_i32", 3, 34),
    (224, "count_i32", 5, 43),
    (2560, "count_i32", 16, 45),
    (912, "count_i32", 12, 36),
    (206, "var_min_i32", 6, 45),
    (146, "count_i32", 5, 44),
    (650, "count_i32", 9, 45),
    (272, "count_i32", 8, 45),
    (829, "var_min_i32", 41, 45),
]


def build_moment_tables():
    by_max = defaultdict(set)
    for height in range(13, 20):
        for width in range(13, 20):
            for pair in PAIRS:
                idx, length = pair
                for row in range(-6, height + 7):
                    for col in range(-6, width + 7):
                        pts, quadrants = ring_points(height, width, row, col, idx, length)
                        if not pts:
                            continue
                        if idx and quadrants < 2:
                            continue
                        _, _, vmax = moment_key(pts)
                        by_max[vmax].add(idx * 8 + length)

    cond_code = {vmax: code for vmax, _, _, code in AMBIGUITY_FIXES}
    else_code = {}
    for vmax, codes in by_max.items():
        if len(codes) == 1:
            continue
        other = sorted(codes - {cond_code[vmax]})
        if len(other) != 1:
            raise AssertionError((vmax, sorted(codes)))
        else_code[vmax] = other[0]

    def patch_counts(vmax, codes, base_code):
        if len(codes) == 1:
            return (0, 0) if next(iter(codes)) == base_code else (1, 0)
        # The complex predicate writes cond_code[vmax].  If the slot default is
        # anything else, a simple vmax predicate first restores the else branch.
        return (0, 1) if base_code == else_code[vmax] else (1, 1)

    mod = 379
    slots = defaultdict(list)
    for vmax, codes in by_max.items():
        slots[vmax % mod].append((vmax, codes))

    codes = np.zeros(mod, dtype=np.uint8)
    for slot, vals in slots.items():
        candidates = set()
        for vmax, vmax_codes in vals:
            candidates.update(vmax_codes)
            if vmax in else_code:
                candidates.add(else_code[vmax])
        best = None
        for candidate in candidates:
            simple, complex_ = 0, 0
            for vmax, vmax_codes in vals:
                s_count, c_count = patch_counts(vmax, vmax_codes, candidate)
                simple += s_count
                complex_ += c_count
            cost = 12 * simple + 23 * complex_
            if best is None or cost < best[0]:
                best = cost, candidate
        codes[slot] = best[1]

    simple_fixes = []
    complex_fixes = []
    for vmax, vmax_codes in sorted(by_max.items()):
        base_code = int(codes[vmax % mod])
        if len(vmax_codes) == 1:
            code = next(iter(vmax_codes))
            if code != base_code:
                simple_fixes.append((vmax, code))
        else:
            if base_code != else_code[vmax]:
                simple_fixes.append((vmax, else_code[vmax]))
            complex_fixes.append(next(fix for fix in AMBIGUITY_FIXES if fix[0] == vmax))

    if len(simple_fixes) != 10 or len(complex_fixes) != 9:
        raise AssertionError((len(simple_fixes), len(complex_fixes)))
    return mod, codes, simple_fixes, complex_fixes


def init(arr, name):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def attrs(node):
    return {a.name: h.get_attribute_value(a) for a in node.attribute}


def node(op, inputs, output, **kw):
    return h.make_node(op, inputs, [output], **kw)


def main():
    model = onnx.load(BASE)
    hash_mod, max_codes, simple_fixes, complex_fixes = build_moment_tables()

    coords = np.arange(30, dtype=np.float32)
    new_inits = list(model.graph.initializer)
    new_inits.extend(
        [
            init(coords, "coords30_f32"),
            init(coords * coords, "coords30_sq_f32"),
            init(max_codes, "moment_code_u8"),
            init(np.array(hash_mod, dtype=np.int32), "moment_hash_mod_i32"),
            init(np.array(0, dtype=np.int32), "m_zero_i32"),
            init(np.array([1], dtype=np.int64), "m_axis1_i64"),
        ]
    )

    prefix = []
    suffix = []
    in_prefix = True
    original_nodes = list(model.graph.node)
    for idx, n in enumerate(original_nodes):
        outs = set(n.output)
        if in_prefix:
            prefix.append(h.make_node(n.op_type, list(n.input), list(n.output), **attrs(n)))
            if outs == {"top_color_selector_f32"}:
                in_prefix = False
            continue
        if outs == {"code_i32"}:
            suffix.append(h.make_node(n.op_type, list(n.input), list(n.output), **attrs(n)))
            suffix.extend(
                h.make_node(x.op_type, list(x.input), list(x.output), **attrs(x))
                for x in original_nodes[idx + 1 :]
            )
            break

    moment_nodes = [
        node(
            "Einsum",
            ["input", "top_color_selector_f32", "coords30_f32", "ones30_f32"],
            "sum_y",
            equation="nchw,kc,h,w->k",
        ),
        node(
            "Einsum",
            ["input", "top_color_selector_f32", "ones30_f32", "coords30_f32"],
            "sum_x",
            equation="nchw,kc,h,w->k",
        ),
        node(
            "Einsum",
            ["input", "top_color_selector_f32", "coords30_sq_f32", "ones30_f32"],
            "sum_y2",
            equation="nchw,kc,h,w->k",
        ),
        node(
            "Einsum",
            ["input", "top_color_selector_f32", "ones30_f32", "coords30_sq_f32"],
            "sum_x2",
            equation="nchw,kc,h,w->k",
        ),
        node("Mul", ["top_colors_0", "sum_y2"], "n_sy2"),
        node("Mul", ["sum_y", "sum_y"], "sy_sq"),
        node("Sub", ["n_sy2", "sy_sq"], "var_y_f32"),
        node("Mul", ["top_colors_0", "sum_x2"], "n_sx2"),
        node("Mul", ["sum_x", "sum_x"], "sx_sq"),
        node("Sub", ["n_sx2", "sx_sq"], "var_x_f32"),
        node("Cast", ["top_colors_0"], "count_i32", to=TensorProto.INT32),
        node("Cast", ["var_y_f32"], "var_y_i32", to=TensorProto.INT32),
        node("Cast", ["var_x_f32"], "var_x_i32", to=TensorProto.INT32),
        node("Min", ["var_y_i32", "var_x_i32"], "var_min_i32"),
        node("Max", ["var_y_i32", "var_x_i32"], "var_max_i32"),
        node("Mod", ["var_max_i32", "moment_hash_mod_i32"], "moment_slot_i32", fmod=0),
        node("Gather", ["moment_code_u8", "moment_slot_i32"], "moment_code0", axis=0),
    ]

    cur = "moment_code0"
    for i, (vmax, out_code) in enumerate(simple_fixes):
        vmax_name = f"mh_vmax_{vmax}_i32"
        code_const = f"mh_code_{i}_u8"
        new_inits.extend(
            [
                init(np.array(vmax, dtype=np.int32), vmax_name),
                init(np.array(out_code, dtype=np.uint8), code_const),
            ]
        )
        moment_nodes.extend(
            [
                node("Equal", ["var_max_i32", vmax_name], f"mh_eq_max_{i}"),
                node("Where", [f"mh_eq_max_{i}", code_const, cur], f"moment_code_h{i + 1}"),
            ]
        )
        cur = f"moment_code_h{i + 1}"

    for i, (vmax, rhs_name, rhs_value, out_code) in enumerate(complex_fixes):
        vmax_name = f"m_vmax_{vmax}_i32"
        rhs_const = f"m_rhs_{i}_i32"
        code_const = f"m_code_{i}_u8"
        new_inits.extend(
            [
                init(np.array(vmax, dtype=np.int32), vmax_name),
                init(np.array(rhs_value, dtype=np.int32), rhs_const),
                init(np.array(out_code, dtype=np.uint8), code_const),
            ]
        )
        moment_nodes.extend(
            [
                node("Equal", ["var_max_i32", vmax_name], f"m_eq_max_{i}"),
                node("Equal", [rhs_name, rhs_const], f"m_eq_rhs_{i}"),
                node("And", [f"m_eq_max_{i}", f"m_eq_rhs_{i}"], f"m_fix_{i}"),
                node("Where", [f"m_fix_{i}", code_const, cur], f"moment_code{i + 1}"),
            ]
        )
        cur = f"moment_code{i + 1}"
    moment_nodes.append(node("Identity", [cur], "code_pre"))

    new_nodes = prefix + moment_nodes + suffix
    used = {name for n in new_nodes for name in n.input if name}
    kept_inits = [t for t in new_inits if t.name in used]

    graph = h.make_graph(
        new_nodes,
        "task096_moment_decoder",
        [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [h.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        initializer=kept_inits,
    )
    out = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    out.ir_version = 10
    onnx.checker.check_model(out)
    out = onnx.shape_inference.infer_shapes(out, strict_mode=True)
    onnx.save(out, OUT)

    mem = 0
    vi = {v.name: v for v in list(out.graph.value_info) + list(out.graph.output)}
    for n in out.graph.node:
        for name in n.output:
            if name == "output" or name not in vi:
                continue
            tt = vi[name].type.tensor_type
            shape = [d.dim_value for d in tt.shape.dim]
            dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type))
            mem += int(np.prod(shape)) * dtype.itemsize
    params = sum(numpy_helper.to_array(t).size for t in out.graph.initializer)
    print(f"wrote {OUT} cost={params + mem} params={params} mem={mem} nodes={len(new_nodes)}")


if __name__ == "__main__":
    main()
