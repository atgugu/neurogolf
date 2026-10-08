#!/usr/bin/env python3
"""ngolf_einsum — terminal multi-operand Einsum renderer for NeuroGolf workers.

THE MONEY SHAPE: one terminal Einsum node. The scoring contract charges
  cost = params (element count of initializers/Constants)
       + memory (Σ numel×dtype_bytes over INTERMEDIATE tensors; graph input/output FREE)
so a SINGLE-NODE graph has memory 0 and cost == params. The free graph input
`input` (FLOAT [1,10,30,30] one-hot color planes) may appear as an Einsum operand
ANY number of times (quadratic / relational features cost nothing extra); tiny
factor matrices are charged once each as initializers.

Mirrors ngolf.py conventions so a build.py written against ngolf can switch
renderers with one line:
  input  named "input"  FLOAT [1,10,30,30]     (ngolf G.__init__ / G.save)
  output named "output"                        (ngolf G.n(is_output=True))
  graph  named "g", ir_version 8               (ngolf G.save)
  initializers named t1, t2, ...               (ngolf G._nm)
  save(): checker full_check + strict shape-inference + ORT zeros-probe
          + BUDGET line                        (ngolf G.save)

Two entry points:
  build_einsum_model(equation, operands)  -> onnx.ModelProto   # PRIMARY: single node
  out_einsum(g, equation, operands)       -> T                 # terminal render into a
                                                               # ngolf.G (g.save() works)
  out_einsum(None, equation, operands)    -> onnx.ModelProto   # same as build_einsum_model

Grader environment: opset 18, ORT 1.24.4 with graph_optimization_level =
ORT_DISABLE_ALL. Banned op/tensor-name substrings: Loop/Scan/NonZero/Unique/
Compress/If.
"""
import hashlib
import json
import math
import os

import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP

INPUT_NAME = "input"
INPUT_SHAPE = [1, 10, 30, 30]
OUTPUT_NAME = "output"
OPSET = 18            # standalone models only; the G path keeps the caller's g.opset
                      # (ngolf.G stays at 17: its Reduce* axes-as-attribute is illegal at 18)
IR_VERSION = 8
BANNED = ("Loop", "Scan", "NonZero", "Unique", "Compress", "If")
from config import CLEAN


# ---------- equation / operand plumbing ----------

def _split_equation(equation):
    eq = equation.replace(" ", "")
    if eq.count("->") > 1:
        raise ValueError(f"malformed einsum equation (multiple '->'): {equation!r}")
    lhs = eq.split("->", 1)[0]
    if not lhs:
        raise ValueError(f"malformed einsum equation (empty LHS): {equation!r}")
    return eq, lhs.split(",")


def _check_banned(*strings):
    for s in strings:
        if not s:
            continue
        for b in BANNED:
            if b in s:
                raise ValueError(f"banned substring {b!r} in {s!r} "
                                 f"(grader bans {'/'.join(BANNED)})")


def _normalize_operands(equation, operands):
    """Validate count vs equation LHS; return list of 'input' | float32 C-contiguous array.

    Factor arrays are stored float32 (matches the FLOAT graph input; params are a
    dtype-free element count, so this never changes the price)."""
    eq, terms = _split_equation(equation)
    if len(operands) != len(terms):
        raise ValueError(f"equation {equation!r} has {len(terms)} LHS terms but "
                         f"{len(operands)} operands were given")
    out = []
    for i, o in enumerate(operands):
        if isinstance(o, str):
            if o != INPUT_NAME:
                raise ValueError(f"operand {i}: string operands must be the literal "
                                 f"{INPUT_NAME!r}, got {o!r}")
            out.append(INPUT_NAME)
        else:
            a = np.ascontiguousarray(np.asarray(o, dtype=np.float32))
            out.append(a)
    return eq, out


def _infer_out_shape(eq, ops):
    """Output shape via numpy einsum on zeros of the true operand shapes — also
    validates the equation semantics (dim mismatches raise here, with numpy's error)."""
    dummies = [np.zeros(INPUT_SHAPE, np.float32) if isinstance(o, str) else o
               for o in ops]
    return list(np.einsum(eq, *dummies).shape)


def _scan_model_banned(m):
    names = set()
    for node in m.graph.node:
        names.add(node.op_type)
        names.add(node.name)
        names.update(node.input)
        names.update(node.output)
    for vi in list(m.graph.input) + list(m.graph.output):
        names.add(vi.name)
    for it in m.graph.initializer:
        names.add(it.name)
    _check_banned(*sorted(n for n in names if n))


def _as_model(model):
    if isinstance(model, onnx.ModelProto):
        return model
    if isinstance(model, (bytes, bytearray)):
        return onnx.load_model_from_string(bytes(model))
    return onnx.load(str(model))


# ---------- builders ----------

def build_einsum_model(equation, operands, output_shape=None, *, out_name=None):
    """Complete SINGLE-NODE model: Einsum(operands) -> graph output. The primary path.

    equation : ONNX/numpy einsum string, e.g. "bchw,cd->bdhw".
    operands : list where each item is the literal string "input" (the free graph
               input — may repeat any number of times) or a numpy array (becomes a
               float32 initializer). Operand ORDER IS PRESERVED EXACTLY — callers
               hand a topologically interleaved order on purpose; never reordered,
               never canonicalized. Identical arrays (by content hash) share ONE
               initializer, so a reused factor is charged once.
    output_shape : optional cross-check against the inferred output shape.
    out_name : graph output name (default "output", ngolf convention).

    Returns onnx.ModelProto (opset 18, ir_version 8, input FLOAT [1,10,30,30] named
    "input"), already through onnx.checker full_check. memory == 0 by construction
    (single node → no intermediates), so grader cost == params_of(model).
    """
    eq, ops = _normalize_operands(equation, operands)
    out_name = out_name or OUTPUT_NAME
    _check_banned(eq, out_name)

    inits, in_names, seen = [], [], {}
    n = 0
    for o in ops:
        if isinstance(o, str):
            in_names.append(INPUT_NAME)
            continue
        key = (o.shape, hashlib.sha256(o.tobytes()).hexdigest())
        name = seen.get(key)
        if name is None:
            n += 1
            name = f"t{n}"                       # ngolf G._nm() naming
            seen[key] = name
            inits.append(H.make_tensor(name, TP.FLOAT, o.shape, o.ravel()))
        in_names.append(name)

    out_shape = _infer_out_shape(eq, ops)
    if output_shape is not None and list(output_shape) != out_shape:
        raise ValueError(f"output_shape {list(output_shape)} != inferred {out_shape} "
                         f"for equation {equation!r}")

    node = H.make_node("Einsum", in_names, [out_name], equation=eq)
    graph = H.make_graph(
        [node], "g",
        [H.make_tensor_value_info(INPUT_NAME, TP.FLOAT, INPUT_SHAPE)],
        [H.make_tensor_value_info(out_name, TP.FLOAT, out_shape)],
        initializer=inits)
    m = H.make_model(graph, opset_imports=[H.make_opsetid("", OPSET)])
    m.ir_version = IR_VERSION
    _scan_model_banned(m)
    onnx.checker.check_model(m, full_check=True)
    return m


def out_einsum(g_or_none, equation, operands, out_name=None, output_shape=None):
    """TERMINAL Einsum renderer (the output tensor is FREE).

    g_or_none is None  -> returns build_einsum_model(equation, operands, ...): a
        complete single-node ModelProto — the money shape (memory 0).
    g_or_none is a ngolf.G-like graph -> emits ONE terminal Einsum node into it
        (duck-typed: needs .input/.init/.n) and returns the output T; g.done is set
        so g.save(path) works unchanged. Numpy operands are registered via g.init
        (charged as params; deduped by content within this call — factors the caller
        already registered separately are NOT deduped against). The caller's g.opset
        is left alone (Einsum is legal from opset 12). Output is always named
        "output" (ngolf hard-codes it), so out_name must be None or "output".

    Operand order is preserved exactly in both paths (see build_einsum_model).
    """
    if g_or_none is None:
        return build_einsum_model(equation, operands, output_shape, out_name=out_name)

    g = g_or_none
    if out_name not in (None, OUTPUT_NAME):
        raise ValueError(f"the ngolf.G path always names the graph output "
                         f"{OUTPUT_NAME!r}; got out_name={out_name!r}")
    eq, ops = _normalize_operands(equation, operands)
    _check_banned(eq)
    out_shape = _infer_out_shape(eq, ops)
    if output_shape is not None and list(output_shape) != out_shape:
        raise ValueError(f"output_shape {list(output_shape)} != inferred {out_shape} "
                         f"for equation {equation!r}")
    ins, seen = [], {}
    for o in ops:
        if isinstance(o, str):
            ins.append(g.input)
            continue
        key = (o.shape, hashlib.sha256(o.tobytes()).hexdigest())
        t = seen.get(key)
        if t is None:
            t = g.init(o)
            seen[key] = t
        ins.append(t)
    return g.n("Einsum", ins, out_shape, "f32", is_output=True, equation=eq)


# ---------- pricing (builder-side) ----------

def params_of(model):
    """Element count of all initializers + Constant-node tensors (dtype-free count,
    matching the grader's params definition). BUILDER-SIDE BUDGETING ONLY — the
    authoritative pricer stays fast_verify.py (neurogolf.scoring_v2); always price
    there before gating/banking. Top-level graph only (If-subgraphs are banned anyway).

    model: path / bytes / onnx.ModelProto."""
    m = _as_model(model)
    total = 0
    for it in m.graph.initializer:
        total += int(np.prod(it.dims))          # np.prod([]) == 1: scalar tensor = 1
    for node in m.graph.node:
        if node.op_type != "Constant":
            continue
        for a in node.attribute:
            if a.name == "value":
                total += int(np.prod(a.t.dims))
            elif a.name == "sparse_value":
                total += int(np.prod(a.sparse_tensor.dims))
            elif a.name in ("value_float", "value_int", "value_string"):
                total += 1
            elif a.name == "value_floats":
                total += len(a.floats)
            elif a.name == "value_ints":
                total += len(a.ints)
            elif a.name == "value_strings":
                total += len(a.strings)
    return total


def memory_of(model):
    """Charged intermediate bytes, mirroring ngolf.G.save's static accounting
    (shape-inferred node outputs, graph outputs free). 0 for a single-node model.
    Builder-side estimate only — the authoritative pricer stays fast_verify."""
    m = _as_model(model)
    g = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    out_names = {o.name for o in g.graph.output}
    vi = {t.name: t for t in list(g.graph.value_info) + list(g.graph.output)}
    total = 0
    for node in g.graph.node:
        for o in node.output:
            if not o or o in out_names or o not in vi:
                continue
            tt = vi[o].type.tensor_type
            dims = [d.dim_value for d in tt.shape.dim]
            total += int(np.prod(dims)) * np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)).itemsize
    return total


def save(model, path, ort_check=True):
    """Standalone mirror of ngolf.G.save(path, ort_check=True): checker full_check +
    strict shape-inference + ORT zeros-probe (ORT_DISABLE_ALL, like the grader),
    then save + print the BUDGET line. Returns path."""
    m = _as_model(model)
    onnx.checker.check_model(m, full_check=True)
    onnx.shape_inference.infer_shapes(m, strict_mode=True)
    if ort_check:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        o.log_severity_level = 4
        s = ort.InferenceSession(m.SerializeToString(), o)
        d = np.zeros((1, 10, 30, 30), np.float32); d[0, 0, 0, 0] = 1
        s.run(None, {INPUT_NAME: d})
    onnx.save(m, path)
    params, mem = params_of(m), memory_of(m)
    cost = params + mem
    pts = max(1.0, 25.0 - math.log(max(1, cost)))          # ngolf.G.points()
    opset = next((op.version for op in m.opset_import if op.domain == ""), "?")
    print(f"BUDGET: params={params}  memory={mem}  cost={cost}  → {pts:.4f} pts")
    print(f"saved {path} (opset {opset}) — now run fast_verify")
    return path


# ---------- parity (pre-gate behavioral check) ----------

def onehot(g):
    """Mirror the grader exactly (copied from gate_vs_pin.onehot): cells outside the
    real grid are all-channel zero."""
    x = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(g):
        for c, color in enumerate(row):
            x[0, int(color), r, c] = 1.0
    return x


def _model_bytes(model):
    if isinstance(model, onnx.ModelProto):
        return model.SerializeToString()
    if isinstance(model, (bytes, bytearray)):
        return bytes(model)
    with open(str(model), "rb") as f:
        return f.read()


def parity(model, ref_fn, task=None, draws=40, seed=0):
    """Check the ONNX model against a python reference BEFORE spending gate time.

    model  : path / bytes / onnx.ModelProto.
    ref_fn : rule(grid)->grid per the packs/taskNNN/rule.py convention — takes a HxW
             int grid (values 0-9), returns the expected output grid.
    task   : if given, use real graded grids from
             {CLEAN}/extracted/task{task:03d}.json (train+test+arc-gen, up to
             `draws`); else `draws` random grids (np.random.default_rng(seed),
             H,W in 1..30, values 0..9).

    Decode is the grader's: (raw > 0) per cell, compared against the one-hot of the
    expected grid (NOT argmax — established by gate_vs_pin.run / the grader).

    Returns (ok, n_checked, first_diff_info): first_diff_info is None on success,
    else a dict {case, grid_hw, cells_differ, got_shape, expected_shape}.
    """
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # match grader
    o.log_severity_level = 4
    sess = ort.InferenceSession(_model_bytes(model), o)

    if task is not None:
        d = json.load(open(f"{CLEAN}/extracted/task{task:03d}.json"))
        pool = [(f"{sp}#{i}", e["input"])
                for sp in ("train", "test", "arc-gen")
                for i, e in enumerate(d.get(sp, []))][:draws]
    else:
        rng = np.random.default_rng(seed)
        pool = []
        for i in range(draws):
            h, w = int(rng.integers(1, 31)), int(rng.integers(1, 31))
            pool.append((f"rand#{i}", rng.integers(0, 10, size=(h, w))))

    n = 0
    for case, grid in pool:
        garr = np.asarray(grid, dtype=int)
        x = onehot(garr)
        raw = sess.run(None, {INPUT_NAME: x})[0]
        got = raw > 0                                   # grader decode
        exp = onehot(np.asarray(ref_fn(garr))) > 0
        n += 1
        if got.shape != exp.shape or not (got == exp).all():
            cells = int((got != exp).sum()) if got.shape == exp.shape else -1
            return False, n, {"case": case, "grid_hw": tuple(garr.shape),
                              "cells_differ": cells, "got_shape": tuple(got.shape),
                              "expected_shape": tuple(exp.shape)}
    return True, n, None
