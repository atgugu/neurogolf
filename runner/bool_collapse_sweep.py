#!/usr/bin/env python3
"""
bool_collapse_sweep.py — Engine B executor: exact structural rewrites (July-2026
top-bundle meta), value-exact or nothing.

Deterministic passes over every current-pin member (and, with --pending, over the
REAL_WIN artifacts in results/task*/task*.onnx):

  P1  Sub(1, Cast(bool→X))  →  Cast(Not(bool)→X)      (boolean complement collapse;
      kills the ones-initializer and narrows the intermediate to bool)
  P2  Where(c, x, x)        →  rewire consumers to x   (identical-branch elision;
      only when inferred shapes of x and the Where output match exactly)
  P3  Cast(x → dtype(x))    →  rewire consumers to x   (same-dtype cast erasure)
  P4  conservative CSE       —  merge nodes with identical (op_type, inputs, attrs)
      from a side-effect-free whitelist; commutative ops compared input-sorted
  P5  initializer dedupe     —  byte-identical initializers collapse to one
  P6  dead-node / dead-initializer elimination

Passes iterate to fixpoint. Every changed member is verified with apply_conv_shrink's
differential harness — byte-identical outputs vs the PARENT bytes under ORT with
optimizations DISABLED, on all real graded grids (every split) + 40 seeded random
grids; any mismatch/asymmetry ⇒ REFUSE. Then scan_razors must show no NEW finding vs
the parent. Only then is the candidate written to
results/sweep_boolcollapse/task{NNN}/task{NNN}.onnx (+ VERIFY.json) and repriced.

Validity model: outputs byte-identical to an LB-proven parent ⇒ Kaggle validity is
inherited; the generator-correctness gate is NOT re-litigated here (that is what
makes this channel near-zero P(0) — see the 2026-07-13 plan, Channel B).

Writes ONLY under results/sweep_boolcollapse/. Never touches the zip, LEDGER, QUEUE,
tickets, or any pinned tool. Deterministic, no LLM.

  python3 runner/bool_collapse_sweep.py [--tasks 1,2,...] [--pending] [--limit N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict

import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN
ZIP_PATH = os.path.join(CLEAN, "submission.zip")
OUT_ROOT = os.path.join(ROOT, "results", "sweep_boolcollapse")
AUDIT = os.path.join(ROOT, "results", "REPRICE_AUDIT_2026-07-10.csv")

sys.path.insert(0, HERE)
from apply_conv_shrink import (  # noqa: E402 — shared differential harness
    RefuseError, sha8, verify_identical)
from scan_razors import scan as razor_scan  # noqa: E402

# side-effect-free ops safe for exact CSE (no subgraph attrs, deterministic)
_CSE_OPS = {"Cast", "Not", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
            "Add", "Mul", "Sub", "Div", "And", "Or", "Xor", "BitwiseAnd", "BitwiseOr",
            "BitwiseXor", "Gather", "GatherElements", "Slice", "Reshape", "Transpose",
            "Concat", "Squeeze", "Unsqueeze", "ReduceSum", "ReduceMax", "ReduceMin",
            "Abs", "Neg", "Exp", "Sqrt", "Pad", "Shape", "Constant"}
_COMMUTATIVE = {"Add", "Mul", "And", "Or", "Xor", "BitwiseAnd", "BitwiseOr",
                "BitwiseXor", "Equal"}


def _consumers(g):
    cons = defaultdict(list)
    for n in g.node:
        for i, inp in enumerate(n.input):
            cons[inp].append((n, i))
    return cons


def _rewire(g, old, new):
    """Point every consumer of tensor `old` at tensor `new` (never graph outputs)."""
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp == old:
                n.input[i] = new


def _graph_outputs(g):
    return {o.name for o in g.output}


def _shapes(model):
    try:
        inf = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return {}
    out = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
        if vi.type.HasField("tensor_type"):
            dims = vi.type.tensor_type.shape.dim
            if all(d.HasField("dim_value") for d in dims):
                out[vi.name] = (vi.type.tensor_type.elem_type,
                                tuple(d.dim_value for d in dims))
    for i in model.graph.initializer:
        out.setdefault(i.name, (i.data_type, tuple(i.dims)))
    return out


def p1_sub_one_cast(model):
    """Sub(1, Cast(bool→X)) → Cast(Not(bool)→X)."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    node_by_out = {n.output[0]: n for n in g.node if n.output}
    gouts = _graph_outputs(g)
    changed = 0
    for n in list(g.node):
        if n.op_type != "Sub" or len(n.input) != 2:
            continue
        a, b = n.input
        one = inits.get(a)
        cast = node_by_out.get(b)
        if one is None or cast is None or cast.op_type != "Cast":
            continue
        try:
            v = numpy_helper.to_array(one)
        except Exception:
            continue
        if v.size != 1 or float(v.reshape(-1)[0]) != 1.0:
            continue
        src = cast.input[0]
        # Cast source must be bool (one-hot semantics of the collapse)
        src_dt = None
        for vi in list(g.value_info) + list(g.input):
            if vi.name == src and vi.type.HasField("tensor_type"):
                src_dt = vi.type.tensor_type.elem_type
        if src in inits:
            src_dt = inits[src].data_type
        if src_dt != onnx.TensorProto.BOOL:
            continue
        if b in gouts or len([1 for m in g.node for i in m.input if i == b]) != 1:
            continue  # Cast output multiply-consumed or a graph output — leave it
        to_attr = next((a2 for a2 in cast.attribute if a2.name == "to"), None)
        if to_attr is None:
            continue
        not_out = n.output[0] + "__ngolf_not"
        not_node = helper.make_node("Not", [src], [not_out],
                                    name=n.output[0] + "__ngolf_not")
        new_cast = helper.make_node("Cast", [not_out], [n.output[0]],
                                    name=n.output[0], to=to_attr.i)
        idx = list(g.node).index(n)
        g.node.remove(cast)
        idx = list(g.node).index(n)  # recompute after removal
        g.node.remove(n)
        g.node.insert(idx, new_cast)
        g.node.insert(idx, not_node)
        changed += 1
        node_by_out = {m.output[0]: m for m in g.node if m.output}
        inits = {i.name: i for i in g.initializer}
    return changed


def p2_where_same_branch(model, shapes):
    g = model.graph
    gouts = _graph_outputs(g)
    changed = 0
    for n in list(g.node):
        if n.op_type != "Where" or len(n.input) != 3 or n.input[1] != n.input[2]:
            continue
        out, x = n.output[0], n.input[1]
        if out in gouts:
            continue
        so, sx = shapes.get(out), shapes.get(x)
        if so is None or sx is None or so != sx:
            continue  # possible broadcast expansion — not value-exact to rewire
        g.node.remove(n)
        _rewire(g, out, x)
        changed += 1
    return changed


def p3_same_dtype_cast(model, shapes):
    g = model.graph
    gouts = _graph_outputs(g)
    changed = 0
    for n in list(g.node):
        if n.op_type != "Cast":
            continue
        out, src = n.output[0], n.input[0]
        if out in gouts:
            continue
        to = next((a.i for a in n.attribute if a.name == "to"), None)
        s_src = shapes.get(src)
        if to is None or s_src is None or s_src[0] != to:
            continue
        g.node.remove(n)
        _rewire(g, out, src)
        changed += 1
    return changed


def p4_cse(model):
    g = model.graph
    gouts = _graph_outputs(g)
    seen = {}
    changed = 0
    for n in list(g.node):
        if n.op_type not in _CSE_OPS or len(n.output) != 1:
            continue
        if n.output[0] in gouts:
            continue
        ins = tuple(sorted(n.input)) if n.op_type in _COMMUTATIVE else tuple(n.input)
        attrs = tuple(sorted(a.SerializeToString() for a in n.attribute))
        key = (n.op_type, ins, attrs)
        if key in seen:
            g.node.remove(n)
            _rewire(g, n.output[0], seen[key])
            changed += 1
        else:
            seen[key] = n.output[0]
    return changed


def p5_init_dedupe(model):
    g = model.graph
    by_bytes = {}
    changed = 0
    for init in list(g.initializer):
        key = (init.data_type, tuple(init.dims), init.SerializeToString())
        # SerializeToString includes the name — key on raw payload instead
        key = (init.data_type, tuple(init.dims),
               numpy_helper.to_array(init).tobytes())
        if key in by_bytes and by_bytes[key] != init.name:
            _rewire(g, init.name, by_bytes[key])
            g.initializer.remove(init)
            changed += 1
        else:
            by_bytes.setdefault(key, init.name)
    return changed


def p6_dead_elim(model):
    g = model.graph
    changed = 0
    while True:
        used = set(_graph_outputs(g))
        for n in g.node:
            used.update(n.input)
        removed = 0
        for n in list(g.node):
            if n.output and not any(o in used for o in n.output):
                g.node.remove(n)
                removed += 1
        for init in list(g.initializer):
            if init.name not in used:
                g.initializer.remove(init)
                removed += 1
        changed += removed
        if not removed:
            return changed


def rewrite(model_bytes):
    """Run all passes to fixpoint. Returns (new_bytes, pass_counts) or (None, counts)
    when nothing changed."""
    model = onnx.load_from_string(model_bytes)
    counts = defaultdict(int)
    for _ in range(8):
        shapes = _shapes(model)
        round_changed = 0
        for name, fn in (("p1_sub_one_cast", lambda: p1_sub_one_cast(model)),
                         ("p2_where_same", lambda: p2_where_same_branch(model, shapes)),
                         ("p3_cast_same", lambda: p3_same_dtype_cast(model, shapes)),
                         ("p4_cse", lambda: p4_cse(model)),
                         ("p5_init_dedupe", lambda: p5_init_dedupe(model)),
                         ("p6_dead", lambda: p6_dead_elim(model))):
            c = fn()
            counts[name] += c
            round_changed += c
        if not round_changed:
            break
    if not sum(counts.values()):
        return None, counts
    return model.SerializeToString(), counts


def razor_regression(parent_bytes, cand_bytes, name):
    before, after = [], []
    razor_scan(name, parent_bytes, before)
    razor_scan(name, cand_bytes, after)
    b = {(k, d) for _, k, d, _ in before}
    new = [(k, d, s) for _, k, d, s in after if (k, d) not in b]
    return new


def price(task, path):
    v = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                        str(task), path, "--price", "--bar", "-9"],
                       capture_output=True, text=True, timeout=600)
    import re
    m = re.search(r"cost=(\d+).*?Δ=([+-]?[\d.]+)", v.stdout)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)


def pending_sources():
    import csv
    out = {}
    try:
        for r in csv.DictReader(open(AUDIT, errors="replace")):
            if r.get("verdict") == "REAL_WIN":
                t = int(r["task"][4:])
                p = os.path.join(ROOT, "results", r["task"], r["file"])
                if os.path.exists(p):
                    out[t] = p
    except OSError:
        pass
    return out


def _parent_bytes(task, origin):
    if origin == "pending":
        return open(pending_sources()[task], "rb").read()
    with zipfile.ZipFile(ZIP_PATH) as z:
        return z.read(f"task{task:03d}.onnx")


def process_one(t, origin):
    """Rewrite+verify+emit ONE member. Prints exactly one status line (or nothing when
    no pass fires). Runs inside a child process: ORT verification of certain members
    dies with SIGFPE (the scan_razors zeros-probe crash class — a hardware FPE Python
    cannot catch), which must kill only this member, not the sweep."""
    parent = _parent_bytes(t, origin)
    try:
        cand, counts = rewrite(parent)
    except Exception as e:
        print(f"task{t:03d} [{origin}] SCAN_FAIL {type(e).__name__}: {e}", flush=True)
        return
    if cand is None:
        return
    tag = " ".join(f"{k}={v}" for k, v in sorted(counts.items()) if v)
    try:
        rep = verify_identical(parent, cand, t)
    except RefuseError as e:
        print(f"task{t:03d} [{origin}] REFUSE ({tag}): {e}", flush=True)
        return
    except Exception as e:
        print(f"task{t:03d} [{origin}] VERIFY_ERR ({tag}): "
              f"{type(e).__name__}: {str(e)[:100]}", flush=True)
        return
    new_razors = razor_regression(parent, cand, f"task{t:03d}.onnx")
    if any(s == "HARD" for _, _, s in new_razors):
        print(f"task{t:03d} [{origin}] RAZOR-REGRESSION ({tag}): {new_razors}", flush=True)
        return
    od = os.path.join(OUT_ROOT, f"task{t:03d}")
    os.makedirs(od, exist_ok=True)
    op = os.path.join(od, f"task{t:03d}.onnx")
    with open(op, "wb") as f:
        f.write(cand)
    cost, delta = price(t, op)
    json.dump({"task": t, "origin": origin, "parent_sha8": sha8(parent),
               "cand_sha8": sha8(cand), "passes": dict(counts), "verify": rep,
               "new_razor_warns": new_razors, "cost": cost, "delta_vs_pin": delta},
              open(os.path.join(od, "VERIFY.json"), "w"), indent=1)
    print(f"task{t:03d} [{origin}] OK ({tag}) cost={cost} Δvs_pin={delta} "
          f"bytes {len(parent)}→{len(cand)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", help="comma-separated task numbers (default: all)")
    ap.add_argument("--pending", action="store_true",
                    help="also sweep REAL_WIN pending artifacts (parent = the artifact)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SWEEP_JOBS", "8")),
                    help="concurrent verify children (each ~2-5 threads under the "
                         "sitecustomize caps; cgroup CPU quota is 46 cores)")
    ap.add_argument("--_child", help="internal: process one '<task>:<origin>'")
    a = ap.parse_args()
    if a._child:
        t, origin = a._child.split(":")
        process_one(int(t), origin)
        return
    want = {int(x) for x in a.tasks.split(",")} if a.tasks else None

    sources = []  # (task, origin)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for nm in sorted(z.namelist()):
            if nm.startswith("task") and nm.endswith(".onnx"):
                t = int(nm[4:7])
                if not want or t in want:
                    sources.append((t, "pin"))
    if a.pending:
        for t in sorted(pending_sources()):
            if not want or t in want:
                sources.append((t, "pending"))
    if a.limit:
        sources = sources[:a.limit]

    os.makedirs(OUT_ROOT, exist_ok=True)

    def run_child(src):
        t, origin = src
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--_child", f"{t}:{origin}"],
                               capture_output=True, text=True, timeout=900)
            line = (r.stdout or "").strip()
            if r.returncode < 0:
                line = (line + f"\ntask{t:03d} [{origin}] CRASH signal {-r.returncode} "
                        f"during verify (SIGFPE class — member is a zeros-probe crash "
                        f"suspect; skipped)").strip()
            elif r.returncode != 0:
                line = (line + f"\ntask{t:03d} [{origin}] child rc={r.returncode}: "
                        f"{(r.stderr or '')[-150:].strip()}").strip()
        except subprocess.TimeoutExpired:
            line = f"task{t:03d} [{origin}] TIMEOUT (900s) — skipped"
        return line

    # Parallel children: verify is the CPU cost and children are isolated anyway (the
    # SIGFPE containment), so a pool is free safety-wise. Each child holds ~2-5 threads
    # under the sitecustomize caps → jobs=8 ≈ 30-40 threads ≈ well inside the 46-core
    # cgroup quota and a rounding error against pids.max.
    from concurrent.futures import ThreadPoolExecutor
    summary, wrote = [], 0
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        for line in pool.map(run_child, sources):
            if not line:
                continue
            for l in line.splitlines():
                summary.append(l)
                if "] OK (" in l:
                    wrote += 1
                print(l, flush=True)
    with open(os.path.join(OUT_ROOT, "SUMMARY.md"), "w") as f:
        f.write(f"# bool_collapse_sweep — {wrote} candidates written "
                f"(pin {open(os.path.join(ROOT,'data','PIN_SHA.txt')).read().strip()})\n\n"
                + "\n".join("- " + s for s in summary) + "\n")
    print(f"\n{wrote} candidate(s) written; {len(summary) - wrote} skipped/refused. "
          f"See {OUT_ROOT}/SUMMARY.md")
    print("Bank path (lane F discipline): fast_verify NNN <cand> --bar 0.0 --full, then "
          "submit_result.py NNN <cand> --gated --lane F")


if __name__ == "__main__":
    main()
