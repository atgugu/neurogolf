#!/usr/bin/env python3
"""scan_razors — static fleet scan for undefined-behavior members (Kaggle coin-flips).

Finds: (1) Conv/QLinearConv biases shorter than out_channels (the −18.88 class),
(2) constant Gather/GatherElements indices out of range for the indexed dim,
(3) negative constant indices (legal but flag), (4) members whose zeros-probe crashes
ORT locally (SIGFPE class — data-dependent indexing; not automatically wrong, but the
first suspects for the unexplained −18.19 'retry-clears' short class),
(5) non-finite (inf/NaN) values in any initializer or Constant tensor — WARN, not HARD:
environment-sensitive, not proven-UB. Evidence (2026-07-12): pin b3407fa4 ships four
such members (122/144/171/258) that scored FULL on the Kaggle LB (7435.7 vs CSV 7435.35)
and pass the minimalist gate env, while an independent local env scored the same bytes
0/266. Policy: never HARD-reject what the LB proves valid; tag it so bank policy routes
non-finite members SOLO-CANARY (they cannot be trusted from local evidence alone).

Severity (audit 2026-07-08 / T5):
  HARD — proven UB, initializer-delivered (Conv/QLinearConv short bias, Gather OOB, load
         failure). These `sys.exit(1)` so bank.py drops the chunk — behavior UNCHANGED.
  WARN — extended, op-correct checks that are advisory only (ConvTranspose short bias with
         out_channels=weight.dims[1]*group, Scatter*/ScatterND OOB, Constant-node-delivered
         bias, unknown-shape OOB-uncheckable, zeros-probe crash). Printed, NEVER blocking,
         so the extension can never reject a valid member. Run with --strict (after
         confirming 0 false-positives over the live pin's members) to gate on WARN too.

Usage: python3 scan_razors.py [--base /tmp/pv2_70_263.zip] [--zeros-probe] [--strict]
Run on every new wave zip BEFORE banking.
"""
import argparse, io, subprocess, sys, tempfile, zipfile, os
import onnx
import numpy as np
from onnx import numpy_helper

def shape_of(g, name, inits):
    if name in inits:
        return list(inits[name].dims)
    for t in list(g.value_info) + list(g.input) + list(g.output):
        if t.name == name and t.type.HasField("tensor_type"):
            return [d.dim_value for d in t.type.tensor_type.shape.dim]
    return None

def scan(name, data, findings, pin_output_dtype=None):
    try:
        m = onnx.load(io.BytesIO(data))
        g = onnx.shape_inference.infer_shapes(m, strict_mode=False).graph
    except Exception as e:
        findings.append((name, "LOAD/INFER", str(e)[:80], "HARD")); return
    inits = {i.name: i for i in g.initializer}
    # resolve Constant-node outputs too: a weight/bias/index delivered via a Constant node is
    # invisible to an initializer-only scan (T5). Used only for the WARN-level extended checks.
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = a.t
    wt = {**inits, **consts}
    # non-finite weights: environment-sensitive coin-flip class (see module docstring) —
    # WARN + canary tag; local validity evidence (either way) does not transfer reliably.
    for tname, t in wt.items():
        try:
            arr = numpy_helper.to_array(t)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.floating) and arr.size and not np.isfinite(arr).all():
            findings.append((name, "NONFINITE_WEIGHT",
                             f"{tname}: {int((~np.isfinite(arr)).sum())}/{arr.size} non-finite", "WARN"))
    # output-dtype drift vs the shipped pin member is advisory: legal, but the #1 tell for an
    # accidental Cast-tail change; the grader decodes (raw>0) so dtype flips can move answers.
    if pin_output_dtype is not None and m.graph.output:
        ot = m.graph.output[0].type.tensor_type.elem_type
        if ot != pin_output_dtype:
            findings.append((name, "OUTPUT_DTYPE",
                             f"output elem_type {ot} != pin's {pin_output_dtype}", "WARN"))
    for n in g.node:
        # ===== proven HARD classes — initializer-delivered, behavior UNCHANGED =====
        if n.op_type == "Conv" and len(n.input) >= 3 and n.input[1] in inits and n.input[2] in inits:
            oc, bl = inits[n.input[1]].dims[0], (inits[n.input[2]].dims[0] if inits[n.input[2]].dims else 1)
            if bl < oc: findings.append((name, "SHORT_BIAS", f"Conv bias {bl}<{oc}", "HARD"))
        if n.op_type == "QLinearConv" and len(n.input) >= 9 and n.input[3] in inits and n.input[8] in inits:
            oc, bl = inits[n.input[3]].dims[0], (inits[n.input[8]].dims[0] if inits[n.input[8]].dims else 1)
            if bl < oc: findings.append((name, "SHORT_BIAS", f"QLinearConv bias {bl}<{oc}", "HARD"))
        if n.op_type in ("Gather", "GatherElements") and len(n.input) >= 2 and n.input[1] in inits:
            idx = numpy_helper.to_array(inits[n.input[1]])
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            dshape = shape_of(g, n.input[0], inits)
            if dshape and 0 <= axis < len(dshape) and dshape[axis] > 0 and idx.size:
                if int(idx.max()) >= dshape[axis] or int(idx.min()) < -dshape[axis]:
                    findings.append((name, "OOB_INDEX",
                                     f"{n.op_type} idx[{int(idx.min())},{int(idx.max())}] vs dim {dshape[axis]}", "HARD"))

        # ===== extended WARN classes — op-correct, advisory only (never block a bank) =====
        # ConvTranspose short bias: out_channels = weight.dims[1] * group  (NOT dims[0] — the
        # ConvTranspose weight layout is [C_in, C_out/group, kH, kW], unlike Conv).
        if n.op_type == "ConvTranspose" and len(n.input) >= 3 and n.input[1] in wt and n.input[2] in wt:
            w = wt[n.input[1]]
            if len(w.dims) >= 2:
                group = next((a.i for a in n.attribute if a.name == "group"), 1)
                oc = w.dims[1] * group
                bl = wt[n.input[2]].dims[0] if wt[n.input[2]].dims else 1
                if bl < oc:
                    findings.append((name, "SHORT_BIAS_CONVT", f"ConvTranspose bias {bl}<{oc}", "WARN"))
        # Conv/QLinearConv short bias delivered via a Constant node (initializer scan misses it)
        if n.op_type == "Conv" and len(n.input) >= 3 and n.input[1] in wt and n.input[2] in wt \
                and not (n.input[1] in inits and n.input[2] in inits) and wt[n.input[1]].dims:
            oc = wt[n.input[1]].dims[0]
            bl = wt[n.input[2]].dims[0] if wt[n.input[2]].dims else 1
            if bl < oc:
                findings.append((name, "SHORT_BIAS_CONST", f"Conv(const-node) bias {bl}<{oc}", "WARN"))
        # ScatterElements: indexes along `axis` (same shape rule as GatherElements)
        if n.op_type == "ScatterElements" and len(n.input) >= 2 and n.input[1] in wt:
            idx = numpy_helper.to_array(wt[n.input[1]])
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            dshape = shape_of(g, n.input[0], inits)
            if dshape and 0 <= axis < len(dshape) and dshape[axis] > 0 and idx.size:
                if int(idx.max()) >= dshape[axis] or int(idx.min()) < -dshape[axis]:
                    findings.append((name, "OOB_SCATTER",
                                     f"ScatterElements idx[{int(idx.min())},{int(idx.max())}] vs dim {dshape[axis]}", "WARN"))
            elif dshape is None and idx.size:
                findings.append((name, "UNKNOWN_SHAPE", "ScatterElements data shape unknown — OOB uncheckable", "WARN"))
        # ScatterND: indices' LAST dim = coordinate depth selecting the leading dims of data
        if n.op_type == "ScatterND" and len(n.input) >= 2 and n.input[1] in wt:
            idx = numpy_helper.to_array(wt[n.input[1]])
            dshape = shape_of(g, n.input[0], inits)
            if dshape and idx.size and idx.ndim >= 1:
                depth = idx.shape[-1]
                if depth <= len(dshape):
                    flat = idx.reshape(-1, depth)
                    bad = any(dshape[k] > 0 and (int(flat[:, k].max()) >= dshape[k]
                                                 or int(flat[:, k].min()) < -dshape[k])
                              for k in range(depth))
                    if bad:
                        findings.append((name, "OOB_SCATTER", f"ScatterND idx out of range vs {dshape[:depth]}", "WARN"))
            elif dshape is None and idx.size:
                findings.append((name, "UNKNOWN_SHAPE", "ScatterND data shape unknown — OOB uncheckable", "WARN"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/tmp/pv2_70_263.zip")
    ap.add_argument("--zeros-probe", action="store_true", help="also run each member on the zeros input (slow)")
    ap.add_argument("--strict", action="store_true",
                    help="gate (exit 1) on WARN findings too — only after confirming 0 "
                         "false-positives over the live pin's members")
    a = ap.parse_args()
    z = zipfile.ZipFile(a.base)
    findings = []
    for name in z.namelist():
        scan(name, z.read(name), findings)
    if a.zeros_probe:
        # ZEROS_CRASH is a HEURISTIC (a member that SIGFPEs on the degenerate all-zeros grid
        # is NOT automatically wrong) → WARN only; never wire this into a hard-fail (T5).
        demo = np.zeros((1, 10, 30, 30), np.float32); demo[0, 0, 0, 0] = 1.0
        for name in z.namelist():
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                f.write(z.read(name)); p = f.name
            r = subprocess.run([sys.executable, "-c",
                                "import sys,numpy as np,onnxruntime as ort;"
                                "o=ort.SessionOptions();o.log_severity_level=4;"
                                f"s=ort.InferenceSession('{p}',o);"
                                "d=np.zeros((1,10,30,30),np.float32);d[0,0,0,0]=1;"
                                "s.run(None,{'input':d})"], capture_output=True, timeout=60)
            if r.returncode != 0:
                findings.append((name, "ZEROS_CRASH", f"rc={r.returncode}", "WARN"))
            os.unlink(p)
    hard = [f for f in findings if f[3] == "HARD"]
    warn = [f for f in findings if f[3] == "WARN"]
    if findings:
        print(f"{len(hard)} HARD + {len(warn)} WARN findings:")
        for n, kind, det, sev in findings:
            print(f"  [{sev}] {n}  {kind}  {det}")
    if hard or (a.strict and warn):
        sys.exit(1)
    if warn:
        print(f"clean of HARD razors — {len(warn)} WARN advisory(ies) (run --strict to gate on them)")
    else:
        print("clean — no static razors found")

if __name__ == "__main__":
    main()
