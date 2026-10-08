#!/usr/bin/env python3
"""
factor_sweep.py — Engine A: deterministic READ-ONLY factorization-ticket scanner.

Scans all 400 current pin members (task{NNN}.onnx inside
$NEUROGOLF_CLEAN/submission.zip) and emits
"factorization tickets" describing statically-detectable compression
opportunities. Writes ONLY to tickets/.

Flag classes: SPARSE_INIT, CONV_SHRINK, I64_STATE, STORED_TABLE, FACTOR_PAIR,
TAIL_FOLD, DTYPE_NORM, ATTR_MOVE, CONV_GEOMETRY (+ SCAN_FAIL for members that
could not be scanned — a failure is UNKNOWN, not clean).

Points law: score = 25 - ln(cost). For a member of cost C and value-exact
element saving s: est_pts = ln(C / (C - s)). est_pts is filled only for
value_exact = Y tickets.

READ-ONLY guarantees: the zip, the pricing CSV and the exemplar bytes are
opened read-only; nothing outside tickets/ is written.
"""

import csv
import hashlib
import io
import math
import os
import sys
import zipfile
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLEAN

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference, TensorProto

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ZIP_PATH = os.environ.get(
    "NGOLF_PIN_ZIP", os.path.join(CLEAN, "submission.zip"))
PRICING_CSV = os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv")
EXEMPLAR_DIR = os.environ.get("NGOLF_EXEMPLAR_DIR", "")
OUT_DIR = os.path.join(ROOT, "data", "tickets")

RECIPES = {
    "R1_SPARSE_FACTOR": "refactor: rank-k/trig/radix factorization into terminal contraction",
    "R2_CONV_SHRINK": ("rewrite kernel + pads/dilations attrs so the op computes the "
                       "identical function (value-exact; negative end-pads = free crop)"),
    "R3_I64_TO_I32": ("i64 is 8B/element in charged memory: recast to i32 where legal / "
                      "replace TopK ranking with Sign+CumSum+ScatterElements or count arithmetic"),
    "R4_RUNTIME_SYNTH": "synthesize at runtime from derived scalars — FLOPs are free",
    "R5_OPERAND_REUSE": "mode-dim packing / operand reuse / CSE",
    "R6_TAIL_FOLD": "fold the tail into the terminal contraction/renderer",
    "R7_DTYPE_NORM": "dtype normalization (drop redundant casts, fp64->fp32, fp16 kept native)",
    "R8_SHRINK_BIAS": "Shrink-bias absorbs Add-const for free (bias/lambd attrs are FREE)",
    "R9_CONV_ABSORB": ("Conv attrs absorb adjacent Slice/Pad crop/placement "
                       "(negative pads = free crop, asymmetric pads = placement, strides/group)"),
    "SCAN_FAIL": "member could not be scanned — status UNKNOWN, re-inspect manually",
}

EXEMPLARS = {
    "SPARSE_INIT": ["073", "321", "220", "257", "344", "296"],
    "CONV_SHRINK": ["138", "266", "006"],
    "I64_STATE": ["316", "393", "091"],
    "STORED_TABLE": ["249", "290", "231", "005", "203"],
    "FACTOR_PAIR": ["001", "373", "287", "298"],
    "TAIL_FOLD": ["291", "304", "332", "242"],
    "DTYPE_NORM": ["056", "362", "254", "266"],
    "ATTR_MOVE": ["331", "150"],
    "CONV_GEOMETRY": ["006", "296", "365", "072"],
}

FLAG_RECIPE = {
    "SPARSE_INIT": "R1_SPARSE_FACTOR",
    "CONV_SHRINK": "R2_CONV_SHRINK",
    "I64_STATE": "R3_I64_TO_I32",
    "STORED_TABLE": "R4_RUNTIME_SYNTH",
    "FACTOR_PAIR": "R5_OPERAND_REUSE",
    "TAIL_FOLD": "R6_TAIL_FOLD",
    "DTYPE_NORM": "R7_DTYPE_NORM",
    "ATTR_MOVE": "R8_SHRINK_BIAS",
    "CONV_GEOMETRY": "R9_CONV_ABSORB",
    "SCAN_FAIL": "SCAN_FAIL",
}

TAIL_OPS = {"Pad", "Expand", "Cast", "Reshape", "Where"}

# dtype -> set of dtypes it can be cast to LOSSLESSLY (for Cast->Cast collapse)
_I = TensorProto
LOSSLESS_TO = {
    _I.BOOL: {_I.BOOL, _I.UINT8, _I.INT8, _I.UINT16, _I.INT16, _I.UINT32, _I.INT32,
              _I.UINT64, _I.INT64, _I.FLOAT16, _I.FLOAT, _I.DOUBLE},
    _I.UINT8: {_I.UINT8, _I.UINT16, _I.INT16, _I.UINT32, _I.INT32, _I.UINT64, _I.INT64,
               _I.FLOAT16, _I.FLOAT, _I.DOUBLE},
    _I.INT8: {_I.INT8, _I.INT16, _I.INT32, _I.INT64, _I.FLOAT16, _I.FLOAT, _I.DOUBLE},
    _I.UINT16: {_I.UINT16, _I.UINT32, _I.INT32, _I.UINT64, _I.INT64, _I.FLOAT, _I.DOUBLE},
    _I.INT16: {_I.INT16, _I.INT32, _I.INT64, _I.FLOAT, _I.DOUBLE},
    _I.UINT32: {_I.UINT32, _I.UINT64, _I.INT64, _I.DOUBLE},
    _I.INT32: {_I.INT32, _I.INT64, _I.DOUBLE},
    _I.UINT64: {_I.UINT64},
    _I.INT64: {_I.INT64},
    _I.FLOAT16: {_I.FLOAT16, _I.FLOAT, _I.DOUBLE},
    _I.FLOAT: {_I.FLOAT, _I.DOUBLE},
    _I.DOUBLE: {_I.DOUBLE},
}


def dtype_name(elem_type):
    try:
        return TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


# ----------------------------------------------------------------------------
# Ticket container
# ----------------------------------------------------------------------------
class Ticket:
    __slots__ = ("task", "flag", "recipe", "est_els", "value_exact", "evidence", "detail",
                 "aux")

    def __init__(self, task, flag, evidence, est_els=None, value_exact="heuristic",
                 detail="", aux=None):
        self.task = task                      # "task138"
        self.flag = flag
        self.recipe = FLAG_RECIPE[flag]
        self.est_els = est_els                # int or None
        self.value_exact = value_exact        # "Y" | "N" | "heuristic"
        self.evidence = evidence              # short string for CSV
        self.detail = detail or evidence      # long string for per-task md
        self.aux = aux or {}                  # detector-specific metadata

    def est_pts(self, cost):
        if self.value_exact != "Y" or not self.est_els or cost is None:
            return None
        s = self.est_els
        if 0 < s < cost:
            return math.log(cost / (cost - s))
        return None


# ----------------------------------------------------------------------------
# Tensor collection (initializers + Constant nodes, recursing into subgraphs)
# ----------------------------------------------------------------------------
def collect_graph(graph, tensors, origins, nodes, unresolved, prefix=""):
    """tensors: name -> np.ndarray; origins: name -> 'init'|'const';
    nodes: flat list of (graph_depth_prefix, node). Recurses into subgraphs."""
    for init in graph.initializer:
        try:
            tensors[init.name] = numpy_helper.to_array(init)
            origins[init.name] = "init"
        except Exception as e:  # e.g. exotic dtype
            unresolved.append((init.name, f"init unresolved: {e}"))
    for node in graph.node:
        nodes.append((prefix, node))
        if node.op_type == "Constant" and node.output:
            out = node.output[0]
            got = False
            for attr in node.attribute:
                try:
                    if attr.name == "value":
                        tensors[out] = numpy_helper.to_array(attr.t)
                        got = True
                    elif attr.name == "value_float":
                        tensors[out] = np.array(attr.f, dtype=np.float32)
                        got = True
                    elif attr.name == "value_int":
                        tensors[out] = np.array(attr.i, dtype=np.int64)
                        got = True
                    elif attr.name == "value_floats":
                        tensors[out] = np.array(list(attr.floats), dtype=np.float32)
                        got = True
                    elif attr.name == "value_ints":
                        tensors[out] = np.array(list(attr.ints), dtype=np.int64)
                        got = True
                except Exception as e:
                    unresolved.append((out, f"const unresolved: {e}"))
            if got:
                origins[out] = "const"
        for attr in node.attribute:
            if attr.HasField("g"):
                collect_graph(attr.g, tensors, origins, nodes, unresolved,
                              prefix + node.op_type + "/")
            for g in attr.graphs:
                collect_graph(g, tensors, origins, nodes, unresolved,
                              prefix + node.op_type + "/")


# ----------------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------------
def det_sparse_init(task, tensors, origins, conv_weight_names):
    out = []
    for name, a in sorted(tensors.items()):
        if not np.issubdtype(a.dtype, np.number):
            continue
        n = a.size
        if n < 25:
            continue
        nnz = int(np.count_nonzero(a))
        if nnz / n < 0.3:
            tag = " (also Conv weight -> see CONV_SHRINK)" if name in conv_weight_names else ""
            ev = f"{name} {a.dtype}{list(a.shape)} nnz={nnz}/{n}{tag}"
            out.append(Ticket(task, "SPARSE_INIT", ev, est_els=n - nnz, value_exact="N",
                              detail=(f"Tensor `{name}` ({origins.get(name)}) dtype={a.dtype} "
                                      f"shape={list(a.shape)}: nnz={nnz}/{n} "
                                      f"(density {nnz/n:.3f}). Upper-bound saving numel-nnz = "
                                      f"{n-nnz} elements (NOT exact — factorized form has its "
                                      f"own operand cost).{tag}")))
    return out


def _axis_lattice(taps, dil_old, k_old):
    """taps: sorted unique kernel indices with nonzero content on this axis.
    Returns (k_new, d_new, pad_begin_shift, pad_end_shift) in EFFECTIVE offset
    space (existing dilation folded in)."""
    eff = sorted(t * dil_old for t in taps)
    o0, o1 = eff[0], eff[-1]
    if len(eff) == 1:
        k_new, d_new = 1, 1
    else:
        g = 0
        for o in eff:
            g = math.gcd(g, o - o0)
        # taps must all sit on the stride-g lattice by construction of gcd
        d_new = g
        k_new = (o1 - o0) // g + 1
    extent_old = (k_old - 1) * dil_old + 1
    pb_shift = o0                      # subtract from pad_begin
    pe_shift = (extent_old - 1) - o1   # subtract from pad_end
    return k_new, d_new, pb_shift, pe_shift


def det_conv_shrink(task, nodes, tensors):
    out = []
    for prefix, node in nodes:
        if node.op_type not in ("Conv", "ConvTranspose") or len(node.input) < 2:
            continue
        wname = node.input[1]
        if wname not in tensors:
            continue
        w = tensors[wname]
        if w.ndim < 3:
            continue
        spatial = w.shape[2:]
        nsp = len(spatial)
        nz = np.any(w != 0, axis=(0, 1)) if w.ndim > 2 else None
        if not np.any(nz):
            continue  # all-zero weight — SPARSE_INIT territory
        attrs = {a.name: a for a in node.attribute}
        dil_old = list(attrs["dilations"].ints) if "dilations" in attrs else [1] * nsp
        pads_old = list(attrs["pads"].ints) if "pads" in attrs else [0] * (2 * nsp)
        auto_pad = (attrs["auto_pad"].s.decode() if "auto_pad" in attrs else "NOTSET") or "NOTSET"

        new_shape, new_dil, new_pads_b, new_pads_e = [], [], [], []
        for ax in range(nsp):
            other = tuple(i for i in range(nsp) if i != ax)
            axmap = np.any(nz, axis=other) if other else nz
            taps = [int(i) for i in np.nonzero(axmap)[0]]
            k_new, d_new, pb_s, pe_s = _axis_lattice(taps, dil_old[ax], spatial[ax])
            new_shape.append(k_new)
            new_dil.append(d_new)
            new_pads_b.append(pads_old[ax] - pb_s)
            new_pads_e.append(pads_old[nsp + ax] - pe_s)
        if tuple(new_shape) == tuple(spatial):
            continue  # no shrink possible
        numel_old = int(w.size)
        numel_new = int(w.shape[0] * w.shape[1] * int(np.prod(new_shape)))
        saved = numel_old - numel_new
        if saved <= 0:
            continue
        neg_pads = any(p < 0 for p in new_pads_b + new_pads_e)
        manual = []
        if node.op_type == "ConvTranspose":
            manual.append("ConvTranspose pad semantics differ — needs manual pads check")
        if auto_pad != "NOTSET":
            manual.append(f"auto_pad={auto_pad} — needs manual pads check")
        if neg_pads and node.op_type == "Conv":
            manual.append("rewrite uses negative end/begin pads (free-crop idiom) — "
                          "confirm ORT accepts on this graph")
        exact = "Y" if node.op_type == "Conv" and auto_pad == "NOTSET" else "N"
        # ASCII tap map (2D only)
        tapmap = ""
        if nsp == 2:
            rows = ["".join("#" if nz[r, c] else "." for c in range(spatial[1]))
                    for r in range(spatial[0])]
            tapmap = "\n".join(rows)
        ev = (f"{node.op_type} '{node.name or node.output[0]}' w={wname}"
              f"{list(w.shape)} -> spatial {list(new_shape)} dil={new_dil} "
              f"pads=({new_pads_b},{new_pads_e}) save={saved}")
        detail = (f"{node.op_type} node `{node.name or node.output[0]}` (weight `{wname}`, "
                  f"shape {list(w.shape)}, dilations {dil_old}, pads {pads_old}, "
                  f"auto_pad {auto_pad}).\n"
                  f"Nonzero spatial tap map (rows=kh, cols=kw, '#'=nonzero):\n"
                  f"```\n{tapmap}\n```\n"
                  f"Rewrite: spatial kernel {list(spatial)} -> {new_shape}, "
                  f"dilations -> {new_dil}, pads -> begin {new_pads_b} end {new_pads_e}. "
                  f"EXACT element saving {numel_old} - {numel_new} = {saved}."
                  + ("".join("\nNOTE: " + m for m in manual)))
        out.append(Ticket(task, "CONV_SHRINK", ev, est_els=saved, value_exact=exact,
                          detail=detail))
    return out


def det_i64_state(task, tensors, origins, nodes, model):
    i64 = {n: a for n, a in tensors.items() if a.dtype == np.int64}
    rank_ops = [f"{n.op_type}:{n.name or n.output[0]}" for _, n in nodes
                if n.op_type in ("TopK", "ArgMax", "ArgMin")]
    # i64 intermediates via shape inference (best effort)
    inferred_i64 = []
    try:
        inf = shape_inference.infer_shapes(model)
        for vi in inf.graph.value_info:
            if vi.type.tensor_type.elem_type == TensorProto.INT64:
                inferred_i64.append(vi.name)
    except Exception:
        pass
    if not i64 and not rank_ops and not inferred_i64:
        return []
    # which i64 tensors feed Slice/Pad bound inputs (ATTR_MOVE(a) folded here)
    slice_bounds = set()
    for _, n in nodes:
        if n.op_type in ("Slice", "Pad"):
            for inp in list(n.input)[1:]:
                if inp in i64:
                    slice_bounds.add(inp)
    total = sum(int(a.size) for a in i64.values())
    parts = []
    if i64:
        parts.append(f"{len(i64)} i64 init/const tensors, {total} els")
    if slice_bounds:
        parts.append(f"{len(slice_bounds)} feed Slice/Pad bounds (i64->i32 candidates)")
    if rank_ops:
        parts.append(f"rank ops: {','.join(rank_ops[:4])}")
    if inferred_i64:
        parts.append(f"{len(inferred_i64)} inferred i64 intermediates")
    names = sorted(i64)
    detail = ("i64 state inventory (heuristic — memory charge depends on scorer's "
              "traced peak):\n"
              + "".join(f"- `{n}` shape={list(i64[n].shape)} numel={i64[n].size}"
                        f"{' [Slice/Pad bound]' if n in slice_bounds else ''}\n"
                        for n in names)
              + (f"- rank-producing ops (i64 outputs): {', '.join(rank_ops)}\n" if rank_ops else "")
              + (f"- inferred i64 intermediates: {', '.join(inferred_i64[:12])}"
                 f"{' ...' if len(inferred_i64) > 12 else ''}\n" if inferred_i64 else ""))
    return [Ticket(task, "I64_STATE", "; ".join(parts), est_els=total or None,
                   value_exact="heuristic", detail=detail)]


def _detect_table_form(a):
    """Return short form string or None. a: numeric ndarray, numel>=64."""
    flat = a.astype(np.float64).ravel()
    n = flat.size
    if not np.all(np.isfinite(flat)):
        return None
    if np.all(flat == flat[0]):
        return f"constant({flat[0]:g})"
    d = np.diff(flat)
    if np.all(d == d[0]) and d[0] != 0:
        return f"arange(start={flat[0]:g}, step={d[0]:g})"
    # affine in coordinates of squeezed shape
    shp = [s for s in a.shape if s > 1]
    if 1 <= len(shp) <= 4 and int(np.prod(shp)) == n:
        coords = np.stack(np.meshgrid(*[np.arange(s) for s in shp], indexing="ij"),
                          axis=-1).reshape(n, len(shp)).astype(np.float64)
        X = np.hstack([np.ones((n, 1)), coords])
        try:
            beta, *_ = np.linalg.lstsq(X, flat, rcond=None)
            pred = X @ beta
            if np.max(np.abs(pred - flat)) <= 1e-9 * max(1.0, np.max(np.abs(flat))):
                terms = " + ".join(f"{b:g}*c{i}" for i, b in enumerate(beta[1:]) if abs(b) > 1e-12)
                return f"affine({beta[0]:g} + {terms or '0'}) over shape {shp}"
        except Exception:
            pass
    # periodic tile
    for p in range(1, min(64, n // 2) + 1):
        if n % p == 0 and np.all(flat.reshape(-1, p) == flat[:p]):
            return f"periodic(tile={p})"
    return None


def det_stored_table(task, tensors, origins):
    out = []
    for name, a in sorted(tensors.items()):
        if not np.issubdtype(a.dtype, np.number) or a.size < 64:
            continue
        form = _detect_table_form(a)
        if form:
            ev = f"{name} {a.dtype}{list(a.shape)} numel={a.size} form={form}"
            out.append(Ticket(task, "STORED_TABLE", ev, est_els=int(a.size), value_exact="N",
                              detail=(f"Tensor `{name}` ({origins.get(name)}) dtype={a.dtype} "
                                      f"shape={list(a.shape)} numel={a.size} matches cheap "
                                      f"arithmetic form **{form}** — synthesize at runtime "
                                      f"(Range/Mul/Add/Mod from a couple of scalars); saving "
                                      f"upper bound = {a.size} els minus generator cost.")))
    return out


def det_factor_pair(task, tensors, origins):
    out = []
    items = [(n, a) for n, a in sorted(tensors.items())
             if np.issubdtype(a.dtype, np.number) and a.size >= 1]
    # exact duplicates (same dtype+shape+bytes) — scalars included, matching the
    # historical dup-CSE measurement convention
    groups = defaultdict(list)
    for n, a in items:
        groups[(str(a.dtype), a.shape, a.tobytes())].append(n)
    dup_names = set()
    for (dt, shp, _), names in sorted(groups.items(), key=lambda kv: kv[1][0]):
        if len(names) > 1:
            numel = int(np.prod(shp)) if shp else 1
            saved = (len(names) - 1) * numel
            dup_names.update(names)
            ev = f"exact-dup x{len(names)} {dt}{list(shp)} ({','.join(names[:4])}{'...' if len(names) > 4 else ''}) save={saved}"
            out.append(Ticket(task, "FACTOR_PAIR", ev, est_els=saved, value_exact="Y",
                              aux={"dup_n": len(names)},
                              detail=(f"EXACT DUPLICATE initializers ({len(names)} copies, "
                                      f"dtype={dt}, shape={list(shp)}, numel={numel}): "
                                      f"{', '.join('`'+x+'`' for x in names)}. CSE: keep one, "
                                      f"rewire consumers — value-exact saving {saved} els.")))
    # cross-shape value aliases: same dtype+bytes, different shapes (needs a
    # Reshape/broadcast to share — not drop-in, so N)
    vgroups = defaultdict(list)
    for n, a in items:
        vgroups[(str(a.dtype), a.tobytes())].append((n, a.shape))
    for (dt, _), members in sorted(vgroups.items(), key=lambda kv: kv[1][0][0]):
        shapes = sorted({s for _, s in members})
        if len(shapes) > 1:
            numel = int(np.prod(shapes[0])) if shapes[0] else 1
            saved = (len(shapes) - 1) * numel
            names = [n for n, _ in members]
            dup_names.update(names)
            ev = (f"value-alias {dt} across shapes {[list(s) for s in shapes]} "
                  f"({','.join(names[:4])}{'...' if len(names) > 4 else ''}) save<={saved}")
            out.append(Ticket(task, "FACTOR_PAIR", ev, est_els=saved, value_exact="N",
                              aux={"dup_n": len(names)},
                              detail=(f"Same raw values ({dt}, {numel} els) stored under "
                                      f"{len(shapes)} different shapes: "
                                      + ", ".join(f"`{n}` {list(s)}" for n, s in members)
                                      + f". Reshape/broadcast alias could share one copy — "
                                        f"saving up to {saved} els minus reshape/shape-tensor "
                                        f"cost (NOT drop-in exact).")))
    # near pairs among non-dup float tensors
    fl = [(n, a) for n, a in items if np.issubdtype(a.dtype, np.floating) and a.size >= 8]
    for i in range(len(fl)):
        for j in range(i + 1, len(fl)):
            n1, a1 = fl[i]
            n2, a2 = fl[j]
            if n1 in dup_names and n2 in dup_names:
                continue
            rel, ev = None, None
            if a1.shape == a2.shape:
                f1 = a1.astype(np.float64).ravel()
                f2 = a2.astype(np.float64).ravel()
                nz = np.nonzero(f1)[0]
                if nz.size:
                    k = f2[nz[0]] / f1[nz[0]]
                    if k != 0 and np.allclose(f2, k * f1, rtol=1e-9, atol=1e-12):
                        rel = f"scalar-multiple k={k:g}"
                if rel is None and a1.ndim >= 2:
                    for p in permutations(range(a1.ndim)):
                        if p == tuple(range(a1.ndim)):
                            continue
                        if np.transpose(a1, p).shape == a2.shape and \
                           np.array_equal(np.transpose(a1, p), a2):
                            rel = f"axis-permutation {p}"
                            break
            if rel is None and a1.size == a2.size:
                f1 = a1.astype(np.float64).ravel()
                f2 = a2.astype(np.float64).ravel()
                d1, d2 = np.linalg.norm(f1), np.linalg.norm(f2)
                if d1 > 0 and d2 > 0:
                    cos = float(np.dot(f1, f2) / (d1 * d2))
                    if cos > 0.9 and not np.array_equal(f1, f2):
                        rel = f"cosine={cos:.3f}"
            if rel:
                ve = "N" if ("scalar-multiple" in rel or "axis-permutation" in rel) else "heuristic"
                saved = int(min(a1.size, a2.size))
                ev = f"{n1}{list(a1.shape)} ~ {n2}{list(a2.shape)}: {rel}"
                out.append(Ticket(task, "FACTOR_PAIR", ev, est_els=saved, value_exact=ve,
                                  detail=(f"`{n1}` (shape {list(a1.shape)}) and `{n2}` "
                                          f"(shape {list(a2.shape)}) related by {rel}. "
                                          f"Candidate mode-dim packing / operand reuse; "
                                          f"potential saving up to {saved} els (not exact — "
                                          f"depends on packing rewrite).")))
    return out


def det_tail_fold(task, model, tensors, nodes):
    out = []
    producer = {}
    for _, n in nodes:
        for o in n.output:
            producer[o] = n
    const_like = set(tensors.keys())
    for gout in model.graph.output:
        chain = []
        cur = gout.name
        while len(chain) < 3 and cur in producer:
            nd = producer[cur]
            if nd.op_type not in TAIL_OPS:
                break
            chain.append(nd)
            nxt = None
            for inp in nd.input:
                if inp and inp not in const_like:
                    nxt = inp
                    break
            if nxt is None:
                break
            cur = nxt
        if not chain:
            continue
        if len(chain) == 1 and chain[0].op_type == "Cast":
            continue  # lone terminal Cast — too common to be signal
        tiny = []
        for nd in chain:
            for inp in nd.input:
                if inp in tensors and tensors[inp].size <= 16:
                    tiny.append(f"{inp}{list(tensors[inp].shape)}")
        if not tiny:
            continue
        ops = "->".join(nd.op_type for nd in reversed(chain))
        ev = f"tail {ops} -> {gout.name}; tiny consts: {','.join(tiny[:4])}"
        out.append(Ticket(task, "TAIL_FOLD", ev, est_els=None, value_exact="heuristic",
                          detail=(f"Final node chain before output `{gout.name}`: {ops} "
                                  f"(ops: {', '.join(nd.name or nd.output[0] for nd in reversed(chain))}), "
                                  f"consuming tiny constants {', '.join(tiny)}. Candidate: fold "
                                  f"tail into the terminal contraction/renderer.")))
    return out


def det_dtype_norm(task, tensors, origins, nodes, model):
    out = []
    # dtype knowledge map
    dtype_of = {}
    for n, a in tensors.items():
        try:
            dtype_of[n] = numpy_helper.from_array(np.zeros(1, a.dtype)).data_type
        except Exception:
            pass
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.HasField("tensor_type"):
            dtype_of[vi.name] = vi.type.tensor_type.elem_type
    try:
        inf = shape_inference.infer_shapes(model)
        for vi in inf.graph.value_info:
            if vi.type.HasField("tensor_type"):
                dtype_of.setdefault(vi.name, vi.type.tensor_type.elem_type)
    except Exception:
        pass
    producer = {}
    for _, n in nodes:
        for o in n.output:
            producer[o] = n
    cast_to = {}
    for _, n in nodes:
        if n.op_type == "Cast":
            to = next((a.i for a in n.attribute if a.name == "to"), None)
            cast_to[n.output[0]] = to
            if n.input and n.input[0] in dtype_of and to is not None:
                dtype_of.setdefault(n.output[0], to)

    safe, unsafe = [], []
    for _, n in nodes:
        if n.op_type != "Cast" or not n.input:
            continue
        to = cast_to.get(n.output[0])
        src = n.input[0]
        src_dt = dtype_of.get(src)
        # Cast to same type
        if to is not None and src_dt is not None and to == src_dt:
            safe.append(f"Cast '{n.name or n.output[0]}' {dtype_name(to)}->same (remove)")
            continue
        # Cast->Cast chain
        prev = producer.get(src)
        if prev is not None and prev.op_type == "Cast":
            a_dt = dtype_of.get(prev.input[0]) if prev.input else None
            mid = cast_to.get(prev.output[0])
            chain = (f"Cast '{prev.name or prev.output[0]}'->'{n.name or n.output[0]}' "
                     f"{dtype_name(a_dt) if a_dt else '?'}->{dtype_name(mid) if mid else '?'}"
                     f"->{dtype_name(to) if to else '?'}")
            if a_dt is not None and mid is not None and mid in LOSSLESS_TO.get(a_dt, set()):
                safe.append(chain + " (collapse: lossless)")
            else:
                unsafe.append(chain + " (verify range before collapse)")
    if safe:
        ev = f"{len(safe)} redundant cast(s): {safe[0][:80]}{'...' if len(safe) > 1 else ''}"
        out.append(Ticket(task, "DTYPE_NORM", ev, est_els=None, value_exact="Y",
                          detail="Redundant casts (removal is value-exact):\n"
                                 + "".join(f"- {s}\n" for s in safe)))
    if unsafe:
        ev = f"{len(unsafe)} cast chain(s) needing range check"
        out.append(Ticket(task, "DTYPE_NORM", ev, est_els=None, value_exact="heuristic",
                          detail="Cast->Cast chains (collapse needs range verification):\n"
                                 + "".join(f"- {s}\n" for s in unsafe)))
    # fp64 anywhere
    fp64_inits = {n: a for n, a in tensors.items() if a.dtype == np.float64}
    fp64_casts = [n.name or n.output[0] for _, n in nodes
                  if n.op_type == "Cast" and cast_to.get(n.output[0]) == TensorProto.DOUBLE]
    if fp64_inits or fp64_casts:
        tot = sum(int(a.size) for a in fp64_inits.values())
        ev = (f"fp64: {len(fp64_inits)} init(s) ({tot} els)"
              + (f", {len(fp64_casts)} Cast->DOUBLE" if fp64_casts else ""))
        out.append(Ticket(task, "DTYPE_NORM", ev, est_els=tot or None, value_exact="heuristic",
                          detail=("fp64 present — 8B/element if charged:\n"
                                  + "".join(f"- init `{n}` shape={list(a.shape)} numel={a.size}\n"
                                            for n, a in sorted(fp64_inits.items()))
                                  + "".join(f"- Cast->DOUBLE `{c}`\n" for c in fp64_casts))))
    # fp16 initializers that get cast up
    consumers = defaultdict(list)
    for _, n in nodes:
        for inp in n.input:
            consumers[inp].append(n)
    fp16_up = []
    for n, a in sorted(tensors.items()):
        if a.dtype == np.float16:
            for c in consumers.get(n, []):
                if c.op_type == "Cast":
                    to = cast_to.get(c.output[0])
                    if to in (TensorProto.FLOAT, TensorProto.DOUBLE):
                        fp16_up.append(f"`{n}` ({a.size} els) -> Cast {dtype_name(to)}")
    if fp16_up:
        out.append(Ticket(task, "DTYPE_NORM", f"{len(fp16_up)} fp16 init(s) cast up",
                          est_els=None, value_exact="heuristic",
                          detail="fp16 initializers cast up (check if op can run fp16-native "
                                 "or store fp32 directly):\n"
                                 + "".join(f"- {s}\n" for s in fp16_up)))
    return out


def det_attr_move(task, tensors, nodes):
    out = []
    has_shrink = any(n.op_type == "Shrink" for _, n in nodes)
    for _, n in nodes:
        if n.op_type not in ("Add", "Sub", "Mul"):
            continue
        for inp in n.input:
            a = tensors.get(inp)
            if a is not None and a.size == 1 and np.issubdtype(a.dtype, np.floating):
                ev = (f"{n.op_type} '{n.name or n.output[0]}' by scalar const "
                      f"{inp}={float(a.ravel()[0]):g}"
                      + ("; Shrink already in graph" if has_shrink else ""))
                out.append(Ticket(task, "ATTR_MOVE", ev, est_els=1, value_exact="heuristic",
                                  detail=(f"`{n.op_type}` node `{n.name or n.output[0]}` uses "
                                          f"scalar constant `{inp}` = {float(a.ravel()[0]):g}. "
                                          f"Shrink bias/lambd attrs are FREE: a Shrink node can "
                                          f"absorb an Add/Sub-const (for Mul, check rescale "
                                          f"elsewhere). "
                                          + ("Graph already contains Shrink — bias may absorb "
                                             "this for free." if has_shrink else
                                             "Would need a Shrink inserted — only wins if it "
                                             "replaces this node."))))
                break  # one ticket per node
    return out


def det_conv_geometry(task, nodes):
    out = []
    producer = {}
    consumers = defaultdict(list)
    for _, n in nodes:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers[i].append(n)
    for _, n in nodes:
        if n.op_type not in ("Conv", "ConvTranspose"):
            continue
        pre = [producer[i].op_type for i in n.input[:1]
               if i in producer and producer[i].op_type in ("Slice", "Pad")]
        post = [c.op_type for o in n.output for c in consumers.get(o, [])
                if c.op_type in ("Slice", "Pad")]
        if pre or post:
            pat = (("+".join(pre) + "->" if pre else "") + n.op_type
                   + ("->" + "+".join(post) if post else ""))
            out.append(Ticket(task, "CONV_GEOMETRY",
                              f"{pat} around '{n.name or n.output[0]}'",
                              est_els=None, value_exact="heuristic",
                              detail=(f"Pattern `{pat}` around {n.op_type} "
                                      f"`{n.name or n.output[0]}`: adjacent Slice/Pad doing "
                                      f"crop/placement that Conv attrs could absorb "
                                      f"(negative pads = free crop, large asymmetric pads = "
                                      f"placement, strides/group). Pattern flag only — derive "
                                      f"attrs manually.")))
    return out


# ----------------------------------------------------------------------------
# Per-member scan
# ----------------------------------------------------------------------------
def scan_member(task, data):
    model = onnx.load_from_string(data)
    tensors, origins, nodes, unresolved = {}, {}, [], []
    collect_graph(model.graph, tensors, origins, nodes, unresolved)
    conv_weight_names = {n.input[1] for _, n in nodes
                         if n.op_type in ("Conv", "ConvTranspose") and len(n.input) >= 2}
    tickets = []
    tickets += det_sparse_init(task, tensors, origins, conv_weight_names)
    tickets += det_conv_shrink(task, nodes, tensors)
    tickets += det_i64_state(task, tensors, origins, nodes, model)
    tickets += det_stored_table(task, tensors, origins)
    tickets += det_factor_pair(task, tensors, origins)
    tickets += det_tail_fold(task, model, tensors, nodes)
    tickets += det_dtype_norm(task, tensors, origins, nodes, model)
    tickets += det_attr_move(task, tensors, nodes)
    tickets += det_conv_geometry(task, nodes)
    if unresolved:
        tickets.append(Ticket(task, "SCAN_FAIL",
                              f"{len(unresolved)} tensor(s) unresolved: "
                              f"{unresolved[0][0]}", value_exact="N",
                              detail="Unresolved tensors (partial scan):\n"
                                     + "".join(f"- {n}: {e}\n" for n, e in unresolved)))
    return tickets


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ZIP_PATH, "rb") as pin_fh:
        pin_sha = hashlib.sha256(pin_fh.read()).hexdigest()[:8]
    # pricing
    cost = {}
    with open(PRICING_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                cost[row["task"]] = float(row["cost"])
            except (ValueError, KeyError):
                pass

    zf = zipfile.ZipFile(ZIP_PATH)
    names = sorted(n for n in zf.namelist() if n.startswith("task") and n.endswith(".onnx"))
    all_tickets = []
    for name in names:
        task = name[:-5]  # strip .onnx
        try:
            all_tickets += scan_member(task, zf.read(name))
        except Exception as e:
            all_tickets.append(Ticket(task, "SCAN_FAIL", f"{type(e).__name__}: {e}",
                                      value_exact="N",
                                      detail=f"Member scan failed: {type(e).__name__}: {e}. "
                                             f"Status UNKNOWN, not clean."))

    # ---- TICKETS.csv
    csv_path = os.path.join(OUT_DIR, "TICKETS.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "flag", "recipe", "est_els_saved", "est_pts",
                    "value_exact", "evidence", "exemplars"])
        for t in all_tickets:
            c = cost.get(t.task)
            pts = t.est_pts(c)
            w.writerow([t.task, t.flag, t.recipe,
                        t.est_els if t.est_els is not None else "",
                        f"{pts:.4f}" if pts is not None else "",
                        t.value_exact,
                        t.evidence[:160],
                        ";".join(EXEMPLARS.get(t.flag, []))])

    # ---- per-task md
    by_task = defaultdict(list)
    for t in all_tickets:
        by_task[t.task].append(t)
    for task, tk in sorted(by_task.items()):
        c = cost.get(task)
        lines = [f"# {task} — factorization tickets",
                 "",
                 f"Pin: {pin_sha} (`{ZIP_PATH}:{task}.onnx`). "
                 f"Pin cost: {c if c is not None else 'UNKNOWN'}"
                 f"{f' (score {25 - math.log(c):.4f})' if c else ''}.",
                 ""]
        for i, t in enumerate(tk, 1):
            pts = t.est_pts(c)
            lines.append(f"## Ticket {i}: {t.flag} [{t.value_exact}]"
                         + (f" — est {t.est_els} els" if t.est_els else "")
                         + (f", est {pts:.4f} pts" if pts is not None else ""))
            lines.append("")
            lines.append(t.detail)
            lines.append("")
            lines.append(f"**Recipe** `{t.recipe}`: {RECIPES[t.recipe]}")
            ex = EXEMPLARS.get(t.flag, [])
            if ex:
                lines.append("**Exemplar bytes:** "
                             + ", ".join(f"`{EXEMPLAR_DIR}/task{e}.onnx`" for e in ex))
            lines.append("")
        with open(os.path.join(OUT_DIR, f"{task}.md"), "w") as f:
            f.write("\n".join(lines))

    # ---- SUMMARY.md
    per_flag = defaultdict(lambda: [0, 0, 0.0])  # count, exact count, exact els
    for t in all_tickets:
        per_flag[t.flag][0] += 1
        if t.value_exact == "Y":
            per_flag[t.flag][1] += 1
            per_flag[t.flag][2] += t.est_els or 0
    exact = [(t, t.est_pts(cost.get(t.task))) for t in all_tickets if t.value_exact == "Y"]
    exact = [(t, p) for t, p in exact if p is not None]
    exact.sort(key=lambda tp: -tp[1])
    others = sorted([t for t in all_tickets if t.value_exact != "Y" and t.est_els],
                    key=lambda t: -t.est_els)
    sum_exact_pts = sum(p for _, p in exact)
    # per-task combined exact pts (correct joint accounting per task)
    per_task_exact = defaultdict(float)
    for t, _ in exact:
        per_task_exact[t.task] += t.est_els or 0
    joint = 0.0
    for task, s in per_task_exact.items():
        c = cost.get(task)
        if c and 0 < s < c:
            joint += math.log(c / (c - s))

    lines = ["# Factor sweep summary — Engine A (factor_sweep.py)",
             "",
             f"Pin: {pin_sha}, {len(names)} members scanned from `{ZIP_PATH}`.",
             f"Pricing: `{PRICING_CSV}`. Points law: score = 25 - ln(cost); "
             f"est_pts = ln(C/(C-s)) for value-exact saving s.",
             "",
             "## Totals per flag class",
             "",
             "| flag | tickets | tasks | value-exact (Y) | exact els |",
             "|------|---------|-------|-----------------|-----------|"]
    for flag in ["SPARSE_INIT", "CONV_SHRINK", "I64_STATE", "STORED_TABLE", "FACTOR_PAIR",
                 "TAIL_FOLD", "DTYPE_NORM", "ATTR_MOVE", "CONV_GEOMETRY", "SCAN_FAIL"]:
        cnt, ycnt, yels = per_flag[flag]
        ntasks = len({t.task for t in all_tickets if t.flag == flag})
        lines.append(f"| {flag} | {cnt} | {ntasks} | {ycnt} | {int(yels)} |")
    lines += ["",
              f"**Total tickets:** {len(all_tickets)} across "
              f"{len(by_task)} tasks.",
              f"**Sigma est_pts over value-exact tickets (independent sum):** "
              f"{sum_exact_pts:.4f} pts.",
              f"**Sigma est_pts with per-task joint accounting (savings summed per task "
              f"before ln):** {joint:.4f} pts.",
              "",
              "## Top 30 tickets by est_pts (value-exact first)",
              "",
              "| # | task | flag | est els | est pts | exact | evidence |",
              "|---|------|------|---------|---------|-------|----------|"]
    rank = 0
    for t, p in exact[:30]:
        rank += 1
        lines.append(f"| {rank} | {t.task} | {t.flag} | {t.est_els} | {p:.4f} | Y | "
                     f"{t.evidence[:90].replace('|', '/')} |")
    for t in others[: max(0, 30 - rank)]:
        rank += 1
        lines.append(f"| {rank} | {t.task} | {t.flag} | {t.est_els} | — | {t.value_exact} | "
                     f"{t.evidence[:90].replace('|', '/')} |")
    lines += ["",
              "## Sanity anchors",
              ""]
    t138 = [t for t in all_tickets if t.task == "task138" and t.flag == "CONV_SHRINK"]
    if t138:
        t = t138[0]
        lines.append(f"- task138 CONV_SHRINK (live pin bytes): {t.evidence} "
                     f"[value_exact={t.value_exact}]")
    else:
        lines.append("- task138 CONV_SHRINK: NOT FOUND in live pin — pin member differs "
                     "from expectation.")
    dup_y = [t for t in all_tickets if t.flag == "FACTOR_PAIR" and t.value_exact == "Y"
             and "dup_n" in t.aux]
    alias_n = [t for t in all_tickets if t.flag == "FACTOR_PAIR" and t.value_exact == "N"
               and "dup_n" in t.aux]
    y_tensors = sum(t.aux["dup_n"] for t in dup_y)
    y_els = sum(t.est_els or 0 for t in dup_y)
    a_els = sum(t.est_els or 0 for t in alias_n)
    all_dup_tasks = {t.task for t in dup_y} | {t.task for t in alias_n}
    lines.append(f"- Exact-dup CSE (same dtype+shape+bytes, drop-in, Y): {len(dup_y)} groups, "
                 f"{y_tensors} tensors involved, {y_els} redundant els, across "
                 f"{len({t.task for t in dup_y})} tasks. Cross-shape value-aliases (same "
                 f"dtype+bytes, N): {len(alias_n)} groups, up to {a_els} more els. "
                 f"Combined: {y_els + a_els} els across {len(all_dup_tasks)} tasks "
                 f"(previous older-pin raw-value measurement: ~184 dup tensors / 2553 els / "
                 f"108 tasks — same order of magnitude).")
    fails = [t for t in all_tickets if t.flag == "SCAN_FAIL"]
    lines.append(f"- SCAN_FAIL: {len(fails)} "
                 + (f"({', '.join(sorted({t.task for t in fails}))})" if fails else "(none)"))
    try:
        pin_sha = hashlib.sha256(open(ZIP_PATH, "rb").read()).hexdigest()[:8]
    except Exception:
        pin_sha = "unknown"
    lines.append(f"- Pin sha256[:8] of scanned zip: {pin_sha}.")
    if pin_sha == "b3407fa4":
        lines.append("- CONV_SHRINK rewrite legality: A/B-verified value-exact under "
                     "onnxruntime (CPU EP) on this pin (2026-07-12) for task138 (-410, "
                     "pads 0,0,-5,-6), task177 (-290, pads -1,0,-11,-7 incl. negative "
                     "BEGIN pad), task198 H5 (-120), task322 (-30): shrunk models produced "
                     "byte-identical outputs; ORT accepts negative Conv pads (free crop).")
    lines.append("")
    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"wrote {len(all_tickets)} tickets for {len(by_task)} tasks -> {OUT_DIR}")
    for flag in sorted(per_flag):
        print(f"  {flag}: {per_flag[flag][0]}")


if __name__ == "__main__":
    main()
