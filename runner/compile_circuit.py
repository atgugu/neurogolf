#!/usr/bin/env python3
"""compile_circuit — Engine C: tensor-network Boolean/finite-domain circuit compiler.

Compiles a small CIRCUIT SPEC (JSON or python dict — writable by an LLM straight
from a rule's source, no protobuf knowledge needed) into ONE terminal Einsum via
ngolf_einsum.build_einsum_model. Encoding (proven by the canary study in
offline_golfers/ruleforge_golfer/work/tensor_network_logic_compiler/):

  * each logical wire            = one einsum contraction index (a letter)
  * input literal                = (10, d) unary factor: one-hot color -> wire state
  * gate                         = tiny dense truth tensor ((2,2,2) AND/OR/XOR, (2,2) NOT, ...)
  * output map                   = (d..., 10) factor: wire state(s) -> color planes
  * every non-output wire        = summed index (existential choice, charged memory 0)

All intermediate state lives inside the single node; only the (deduped) truth-table
factors are billed as params. Dense tensors only — SparseTensorProto initializers
fail ORT/onnx Einsum type inference (see build_canaries.py).

CIRCUIT SPEC v1 — broadcast "cellwise": output[i,j] depends only on input[i,j].
{
  "task": 214,                      # optional; used by --parity
  "broadcast": "cellwise",          # v1: the only supported value (default)
  "wires": {
    "a": {"kind": "literal", "colors": [1,2]},        # Bool wire: color in set -> (10,2)
    "s": {"kind": "literal", "table": [[...d...] x10]},# finite-domain wire: (10,d), one-hot rows
    "z": {"kind": "gate", "op": "and", "in": ["a","s2"]},
    #      op in {and, or, xor} (2 Bool ins), not (1 Bool in), eq (2 same-domain ins),
    #      table (any ins; "table" nested list of shape (d_in1,...,d_ink,d_out),
    #             0/1 entries, one-hot along the LAST axis)
  },
  "output": {"wires": ["color","z"], "map": [[[...10...] x d_z] x10]}
  #   map shape = (domain(w) for w in wires..., 10); raw[o] = map[states..., o];
  #   grader decode is (raw > 0). "wire": "z" is sugar for "wires": ["z"].
  #   The reserved wire name "color" is the raw input color (domain 10) — this is
  #   how "else keep the original color" is expressed.
}
Literals may carry "cell": "same" (the only v1 value; anything else raises).

SCHEDULING (the compiler OWNS this — reorder_task350.py lesson, 4.1 s -> 0.008 s):
ORT's Einsum contracts operands pairwise LEFT TO RIGHT with no path search, so
operand order IS the contraction path. compile() emits operands topologically
INTERLEAVED — input first, then for each gate in topo order its not-yet-emitted
literal inputs immediately followed by the gate factor — so each wire index is
eliminated as soon as possible and the running tensor never carries more than a
couple of live wire labels. Never emit all literals first: that path is 2^wires.

EXTENSION POINTS (documented TODOs — not built; build_canaries.py does not
demonstrate them, so v1 does not claim them):
  * broadcast "neighborhood": literals reading offset cells need shifted copies of
    the input (extra free "input" operands with shift factors on h/w labels).
  * coordinate factors: (30,) / (30,r) factors sharing the h/w labels (the shape
    reorder_task350's spatial maps use) for position-dependent rules.
  * global aggregation (mostcolor/leastcolor-style) is NOT expressible cellwise.

Pricing: params_of() here is builder-side budgeting only. The AUTHORITATIVE price
is `python3 runner/fast_verify.py NNN out.onnx --price` (read-only subprocess;
see --price-authoritative).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ngolf_einsum import (  # noqa: E402
    INPUT_NAME, build_einsum_model, onehot, params_of, parity,
)

HERE = Path(__file__).resolve().parent
# build_einsum_model validates every equation through np.einsum(optimize=False),
# whose C kernel imposes three caps (all probed empirically on this venv's numpy):
#   1. NAIVE COST — c_einsum iterates the FULL label space (no pairwise path), so
#      validation costs prod(all label dims) = 90000 * prod(wire domains) iters
#      (~2.5e7/s: 9 Bool wires ~2 s, 11 ~9 s, 13 ~37 s, beyond that it hangs).
#      This binds FIRST and is a property of the emitter's validation only — the
#      ORT contraction itself stays microseconds thanks to interleaved scheduling.
#   2. distinct subscripts: 32           -> MAX_DISTINCT_LABELS (wires <= 27)
#   3. NPY_MAXARGS: 31 operands + output -> MAX_OPERANDS
# Wider circuits: fuse groups of Bool literals into one multi-input 'table' gate
# (fewer, fatter factors). ONNX's own a-zA-Z limit (52) never binds.
MAX_OPERANDS = 31
MAX_DISTINCT_LABELS = 32
NAIVE_VALIDATION_BUDGET = 3.0e8   # ~12 Bool wires; override via compile(max_naive_cost=)
RESERVED = "nchwo"  # batch/color/spatial/output labels — never wire labels
LABELS = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
          if c not in RESERVED]
COLOR_WIRE = "color"   # reserved: the raw input color index (domain 10)
NUM_COLORS = 10
GATE2 = ("and", "or", "xor")


# ---------- factor constructors (proven shapes from build_canaries.py) ----------

def unary_from_colors(colors) -> np.ndarray:
    """(10,2) literal: every valid one-hot color activates exactly one Bool state."""
    a = np.zeros((NUM_COLORS, 2), np.float32)
    for c in range(NUM_COLORS):
        a[c, int(c in set(colors))] = 1.0
    return a


def gate2_table(op: str) -> np.ndarray:
    g = np.zeros((2, 2, 2), np.float32)
    for a in range(2):
        for b in range(2):
            r = {"and": a & b, "or": a | b, "xor": a ^ b}[op]
            g[a, b, r] = 1.0
    return g


def not_table() -> np.ndarray:
    g = np.zeros((2, 2), np.float32)
    g[0, 1] = g[1, 0] = 1.0
    return g


def eq_table(d: int) -> np.ndarray:
    g = np.zeros((d, d, 2), np.float32)
    for a in range(d):
        for b in range(d):
            g[a, b, int(a == b)] = 1.0
    return g


def _check_onehot_last_axis(a: np.ndarray, what: str) -> None:
    """Deterministic truth tensor: 0/1 entries, exactly one 1 along the last axis."""
    if not np.isin(a, (0.0, 1.0)).all():
        raise ValueError(f"{what}: entries must be 0/1, got values outside {{0,1}}")
    if not (a.sum(axis=-1) == 1.0).all():
        raise ValueError(f"{what}: rows must be one-hot along the last axis "
                         f"(exactly one 1 per input combination)")


# ---------- spec loading / validation ----------

def _as_spec(spec):
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a dict or a JSON file path, got {type(spec)}")
    return spec


class Circuit:
    """Validated circuit: wire domains, factors, topo order — shared by the
    compiler and the independent numpy evaluator."""

    def __init__(self, spec):
        spec = _as_spec(spec)
        self.spec = spec
        bc = spec.get("broadcast", "cellwise")
        if bc != "cellwise":
            raise NotImplementedError(
                f"broadcast={bc!r}: v1 supports only 'cellwise' "
                f"(neighborhood/coordinate factors are documented TODO extension "
                f"points in the module docstring)")
        wires = spec.get("wires", {}) or {}
        if not isinstance(wires, dict):
            raise ValueError("'wires' must be a dict of name -> wire spec")
        if COLOR_WIRE in wires:
            raise ValueError(f"wire name {COLOR_WIRE!r} is reserved (the raw input color)")

        self.domain: dict[str, int] = {}      # wire -> state count
        self.factor: dict[str, np.ndarray] = {}
        self.kind: dict[str, str] = {}
        self.gate_in: dict[str, list[str]] = {}

        # literals first (their domains gate the gates)
        for name, w in wires.items():
            k = w.get("kind")
            if k not in ("literal", "gate"):
                raise ValueError(f"wire {name!r}: kind must be 'literal' or 'gate', got {k!r}")
            self.kind[name] = k
            if k != "literal":
                continue
            cell = w.get("cell", "same")
            if cell != "same":
                raise NotImplementedError(
                    f"wire {name!r}: cell={cell!r}; v1 is cellwise (cell='same' only). "
                    f"Neighborhood offsets are a documented TODO extension point.")
            if ("colors" in w) == ("table" in w):
                raise ValueError(f"literal {name!r}: give exactly one of 'colors' or 'table'")
            if "colors" in w:
                colors = w["colors"]
                if not colors or not all(isinstance(c, int) and 0 <= c <= 9 for c in colors):
                    raise ValueError(f"literal {name!r}: 'colors' must be non-empty ints in 0..9")
                t = unary_from_colors(colors)
            else:
                t = np.asarray(w["table"], np.float32)
                if t.ndim != 2 or t.shape[0] != NUM_COLORS or t.shape[1] < 2:
                    raise ValueError(f"literal {name!r}: table must be (10, d>=2), got {t.shape}")
                _check_onehot_last_axis(t, f"literal {name!r} table")
            self.factor[name] = t
            self.domain[name] = t.shape[1]

        # gates: resolve in topological order (Kahn over spec insertion order)
        pending = [n for n in wires if self.kind[n] == "gate"]
        self.gate_order: list[str] = []
        while pending:
            progress = False
            for name in list(pending):
                ins = wires[name].get("in", [])
                if not isinstance(ins, list) or not ins:
                    raise ValueError(f"gate {name!r}: 'in' must be a non-empty list of wires")
                for i in ins:
                    if i == COLOR_WIRE:
                        raise ValueError(
                            f"gate {name!r}: {COLOR_WIRE!r} may only feed the output map in v1")
                    if i not in wires:
                        raise ValueError(f"gate {name!r}: unknown input wire {i!r}")
                if any(i not in self.domain for i in ins):
                    continue  # an input gate not resolved yet
                self._resolve_gate(name, wires[name], ins)
                self.gate_order.append(name)
                pending.remove(name)
                progress = True
            if not progress:
                raise ValueError(f"gate cycle involving {sorted(pending)}")

        # output
        out = spec.get("output")
        if not isinstance(out, dict):
            raise ValueError("spec needs an 'output': {'wire(s)': ..., 'map': ...}")
        if ("wire" in out) == ("wires" in out):
            raise ValueError("output: give exactly one of 'wire' (str) or 'wires' (list)")
        self.out_wires = [out["wire"]] if "wire" in out else list(out["wires"])
        for w in self.out_wires:
            if w != COLOR_WIRE and w not in wires:
                raise ValueError(f"output: unknown wire {w!r}")
        dims = tuple(NUM_COLORS if w == COLOR_WIRE else self.domain[w]
                     for w in self.out_wires)
        m = np.asarray(out["map"], np.float32)
        if m.shape != dims + (NUM_COLORS,):
            raise ValueError(f"output map shape {m.shape} != expected {dims + (NUM_COLORS,)} "
                             f"for output wires {self.out_wires}")
        self.out_map = m

        # every declared wire must reach the output (unused wires are silent
        # param waste — the encoding sums them out harmlessly, so error loudly)
        used, stack = set(), [w for w in self.out_wires if w != COLOR_WIRE]
        while stack:
            w = stack.pop()
            if w in used:
                continue
            used.add(w)
            stack.extend(self.gate_in.get(w, []))
        unused = sorted(set(wires) - used)
        if unused:
            raise ValueError(f"wires never reach the output: {unused} — remove them "
                             f"(they would be billed as params for nothing)")

    def _resolve_gate(self, name, w, ins):
        op = w.get("op")
        doms = [self.domain[i] for i in ins]
        if op in GATE2:
            if doms != [2, 2]:
                raise ValueError(f"gate {name!r}: op {op!r} needs exactly 2 Bool inputs, "
                                 f"got domains {doms}")
            t = gate2_table(op)
        elif op == "not":
            if doms != [2]:
                raise ValueError(f"gate {name!r}: 'not' needs exactly 1 Bool input, got {doms}")
            t = not_table()
        elif op == "eq":
            if len(doms) != 2 or doms[0] != doms[1]:
                raise ValueError(f"gate {name!r}: 'eq' needs 2 same-domain inputs, got {doms}")
            t = eq_table(doms[0])
        elif op == "table":
            t = np.asarray(w.get("table"), np.float32)
            if t.ndim != len(ins) + 1 or list(t.shape[:-1]) != doms or t.shape[-1] < 2:
                raise ValueError(f"gate {name!r}: table shape {t.shape} != expected "
                                 f"{tuple(doms)} + (d_out>=2,)")
            _check_onehot_last_axis(t, f"gate {name!r} table")
        else:
            raise ValueError(f"gate {name!r}: op must be one of "
                             f"{GATE2 + ('not', 'eq', 'table')}, got {op!r}")
        self.factor[name] = t
        self.domain[name] = t.shape[-1]
        self.gate_in[name] = list(ins)

    # ---------- independent reference evaluation (integer LUTs, NO einsum) ----------

    def eval_numpy(self, grid) -> np.ndarray:
        """raw color planes (10, H, W) the single Einsum must produce on the valid
        cells — computed by per-cell truth-table propagation, deliberately NOT via
        np.einsum, so it is an independent check of the compiled contraction."""
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or g.min() < 0 or g.max() > 9:
            raise ValueError("grid must be HxW ints in 0..9")
        state: dict[str, np.ndarray] = {COLOR_WIRE: g}
        for name in self.kind:
            if self.kind[name] == "literal":
                lut = self.factor[name].argmax(axis=1)      # rows validated one-hot
                state[name] = lut[g]
        for name in self.gate_order:
            lut = self.factor[name].argmax(axis=-1)
            state[name] = lut[tuple(state[i] for i in self.gate_in[name])]
        raw = self.out_map[tuple(state[w] for w in self.out_wires)]   # (H, W, 10)
        return np.moveaxis(raw, -1, 0).astype(np.float32)

    @property
    def deterministic_output(self) -> bool:
        """True iff the output map itself is a one-hot color decision, i.e. the
        circuit denotes a grid->grid function under the grader's (raw>0) decode."""
        m = self.out_map
        return bool(np.isin(m, (0.0, 1.0)).all() and (m.sum(axis=-1) == 1.0).all())

    def ref_fn(self):
        """grid -> grid reference (for ngolf_einsum.parity). Deterministic maps only."""
        if not self.deterministic_output:
            raise ValueError("output map is not one-hot per state: the circuit has raw "
                             "semantics only; compare raw arrays via verify() instead")
        return lambda grid: self.eval_numpy(grid).argmax(axis=0)


# ---------- compile / build ----------

def compile(spec, max_naive_cost=NAIVE_VALIDATION_BUDGET):
    """Circuit spec -> (equation, operands, meta).

    Allocates one einsum letter per wire (n/c/h/w/o reserved) with loud guards at
    the emitter's numpy caps — naive-validation cost first, then 32 distinct
    subscripts, then 31 operands (see NAIVE_VALIDATION_BUDGET et al. above) — and
    SCHEDULES operands topologically interleaved: input, then per gate (topo order)
    its unseen literal inputs immediately followed by the gate factor, then any
    output-only literals, then the output map. Identical factor arrays are passed
    as equal arrays so ngolf_einsum dedupes them into one billed initializer.
    """
    ckt = Circuit(spec)
    names = list(ckt.kind)
    naive_cost = 90000.0                       # n*c*h*w*o = 1*10*30*30*10
    for w in names:
        naive_cost *= ckt.domain[w]
    if naive_cost > max_naive_cost:
        raise ValueError(
            f"naive-validation overflow: prod(label dims) = {naive_cost:.2e} > "
            f"{max_naive_cost:.1e}. build_einsum_model validates through "
            f"np.einsum(optimize=False), which iterates the FULL label space "
            f"(~2.5e7 iters/s) — this would hang the build. The ORT contraction "
            f"itself would stay fast (interleaved schedule); the cap is the "
            f"emitter's validation only. Fuse literal groups into multi-input "
            f"'table' gates, or raise max_naive_cost knowingly.")
    if len(names) + len(RESERVED) > MAX_DISTINCT_LABELS:
        raise ValueError(
            f"label overflow: {len(names)} wires + {len(RESERVED)} reserved labels "
            f"({RESERVED!r}) = {len(names) + len(RESERVED)} distinct einsum subscripts "
            f"> {MAX_DISTINCT_LABELS} (numpy's einsum cap, which build_einsum_model "
            f"validates through; ONNX's 52-letter limit never binds first). "
            f"Fuse literal groups into multi-input 'table' gates.")
    lab = {w: LABELS[i] for i, w in enumerate(names)}
    lab[COLOR_WIRE] = "c"

    terms, operands, schedule = ["nchw"], [INPUT_NAME], ["input"]
    emitted = set()

    def emit_literal(w):
        terms.append("c" + lab[w])
        operands.append(ckt.factor[w])
        schedule.append(f"lit:{w}")
        emitted.add(w)

    for gname in ckt.gate_order:
        for i in ckt.gate_in[gname]:
            if ckt.kind.get(i) == "literal" and i not in emitted:
                emit_literal(i)
        terms.append("".join(lab[i] for i in ckt.gate_in[gname]) + lab[gname])
        operands.append(ckt.factor[gname])
        schedule.append(f"gate:{gname}")
        emitted.add(gname)
    for w in ckt.out_wires:                       # literals feeding only the output
        if ckt.kind.get(w) == "literal" and w not in emitted:
            emit_literal(w)
    terms.append("".join(lab[w] for w in ckt.out_wires) + "o")
    operands.append(ckt.out_map)
    schedule.append("outmap")

    if len(operands) > MAX_OPERANDS:
        # Unreachable in v1 (labels bind first at wires+5) — defense in depth for the
        # neighborhood extension, whose shifted-input copies add operands, not wires.
        raise ValueError(
            f"operand overflow: {len(operands)} einsum operands > {MAX_OPERANDS} "
            f"(numpy's einsum cap, hit inside build_einsum_model shape validation). "
            f"Fuse literal groups into multi-input 'table' gates to shrink the count.")

    equation = ",".join(terms) + "->nohw"
    uniq, params = set(), 0
    for op in operands:
        if isinstance(op, str):
            continue
        key = (op.shape, op.tobytes())
        if key not in uniq:
            uniq.add(key)
            params += op.size
    meta = {
        "equation": equation,
        "schedule": schedule,
        "terms": list(terms),
        "labels": {w: lab[w] for w in names},
        "domains": dict(ckt.domain),
        "label_count": len(set("".join(terms))),
        "n_operands": len(operands),
        "params_estimate": int(params),   # matches params_of after dedupe
        "deterministic_output": ckt.deterministic_output,
    }
    return equation, operands, meta


def build(spec, max_naive_cost=NAIVE_VALIDATION_BUDGET):
    """Circuit spec -> single-node onnx.ModelProto (via ngolf_einsum, never raw protobuf)."""
    equation, operands, _ = compile(spec, max_naive_cost=max_naive_cost)
    return build_einsum_model(equation, operands, output_shape=[1, 10, 30, 30])


# ---------- verify ----------

def verify(spec, model=None, draws=40, seed=0, runs=30, task=None, parity_draws=40):
    """(a) independent numpy circuit evaluation vs ORT raw output on random grids
    (+ real task grids via ngolf_einsum.parity when task is given);
    (b) wall-clock per-run timing — must be < 0.1 s/run (the scheduling proof)."""
    import onnxruntime as ort
    ckt = Circuit(spec)
    if model is None:
        model = build(spec)
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # grader
    o.log_severity_level = 4
    sess = ort.InferenceSession(model.SerializeToString(), o,
                                providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(seed)
    ok, n, first_diff = True, 0, None
    for i in range(draws):
        h, w = int(rng.integers(1, 31)), int(rng.integers(1, 31))
        g = rng.integers(0, 10, size=(h, w))
        x = onehot(g)
        raw = sess.run(None, {INPUT_NAME: x})[0]
        want = np.zeros_like(raw)
        want[0, :, :h, :w] = ckt.eval_numpy(g)
        n += 1
        if not np.allclose(raw, want, rtol=0.0, atol=1e-5):
            ok, first_diff = False, {
                "case": f"rand#{i}", "grid_hw": (h, w),
                "max_abs": float(np.max(np.abs(raw - want))),
                "cells_differ": int(((raw > 0) != (want > 0)).sum()),
            }
            break

    x_full = onehot(rng.integers(0, 10, size=(30, 30)))
    sess.run(None, {INPUT_NAME: x_full})                    # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {INPUT_NAME: x_full})
        times.append(time.perf_counter() - t0)

    par = None
    if task is not None:
        p_ok, p_n, p_diff = parity(model, ckt.ref_fn(), task=task, draws=parity_draws)
        par = {"task": task, "ok": bool(p_ok), "n": p_n, "first_diff": p_diff}

    return {
        "numpy_vs_ort_ok": ok, "draws_checked": n, "first_diff": first_diff,
        "run_s_mean": float(np.mean(times)), "run_s_max": float(np.max(times)),
        "timing_ok": bool(np.max(times) < 0.1),
        "params": params_of(model),
        "parity": par,
        "pricing_note": ("params_of is builder-side budgeting; authoritative price is "
                         "`python3 runner/fast_verify.py NNN out.onnx --price`"),
    }


def price_authoritative(task, onnx_path):
    """Read-only subprocess to the project's authoritative pricer."""
    cmd = [sys.executable, str(HERE / "fast_verify.py"), str(int(task)),
           str(onnx_path), "--price"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return r.returncode, (r.stdout + r.stderr).strip()


# ---------- CLI ----------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compile a cellwise circuit spec to one terminal Einsum ONNX.")
    ap.add_argument("spec", help="circuit spec JSON file")
    ap.add_argument("--out", help="write the .onnx here")
    ap.add_argument("--task", type=int, help="task number (default: spec['task'])")
    ap.add_argument("--parity", action="store_true",
                    help="run ngolf_einsum.parity on the real graded grids of --task")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--price-authoritative", action="store_true",
                    help="also price via fast_verify.py --price (read-only subprocess)")
    args = ap.parse_args(argv)

    spec = _as_spec(args.spec)
    task = args.task if args.task is not None else spec.get("task")
    equation, operands, meta = compile(spec)
    model = build(spec)
    print(f"equation : {equation}")
    print(f"schedule : {' '.join(meta['schedule'])}")
    print(f"labels   : {meta['labels']}  (distinct in equation: {meta['label_count']})")
    print(f"params   : {params_of(model)} (estimate {meta['params_estimate']}; "
          f"memory 0 by construction — single node)")

    rep = verify(spec, model, draws=args.draws, runs=args.runs,
                 task=task if args.parity else None)
    print(f"numpy-vs-ORT : {'OK' if rep['numpy_vs_ort_ok'] else 'FAIL'} "
          f"on {rep['draws_checked']} random grids"
          + ("" if rep["first_diff"] is None else f"  first_diff={rep['first_diff']}"))
    print(f"timing       : mean {rep['run_s_mean'] * 1e3:.3f} ms  "
          f"max {rep['run_s_max'] * 1e3:.3f} ms per run  "
          f"({'OK' if rep['timing_ok'] else 'FAIL'} vs 0.1 s bar)")
    if rep["parity"] is not None:
        p = rep["parity"]
        print(f"parity       : task {p['task']} "
              f"{'OK' if p['ok'] else 'FAIL'} on {p['n']} real graded grids"
              + ("" if p["first_diff"] is None else f"  {p['first_diff']}"))
    print(f"note         : {rep['pricing_note']}")

    if args.out:
        import onnx
        onnx.save(model, args.out)
        print(f"saved        : {args.out}")
        if args.price_authoritative:
            if task is None:
                print("price-authoritative: skipped (no task number)")
            else:
                rc, out = price_authoritative(task, args.out)
                print(f"fast_verify --price (rc={rc}):\n{out}")
    elif args.price_authoritative:
        print("price-authoritative: needs --out (fast_verify prices a file)")

    ok = rep["numpy_vs_ort_ok"] and rep["timing_ok"] and (
        rep["parity"] is None or rep["parity"]["ok"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
