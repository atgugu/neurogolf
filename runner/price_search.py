#!/usr/bin/env python3
"""price_search — breadth-first design search: an admin model PROPOSES, the harness PRICES.

The fleet's #1 measured failure (40% of real failures; 65% of those had a VERIFIED rule)
is "budget never closed": the worker knows the rule but hand-prices only 2-4 designs.
This tool asks the fast model for 15-25 design budgets across ≥5 families, then
RECOMPUTES every table arithmetically (params=elements; bytes by dtype; output tensor
free) and returns only designs that CLOSE vs your bar — ranked, with margins. The model's
numbers are never trusted: a hallucinated budget either reprices correctly or is dropped.

Usage (from your pack dir, ideally after your rule is hypo-verified):
  python3 ../../runner/price_search.py NNN "one-sentence rule statement"
Output: a ranked table of PAPER-FEASIBLE designs + near-misses. These are budget
hypotheses — the rule must still FIT the design; verify with g.budget() as you build.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BYTES = {"f32": 4, "float32": 4, "f16": 2, "float16": 2, "u8": 1, "uint8": 1,
         "i8": 1, "int8": 1, "b": 1, "bool": 1, "i32": 4, "int32": 4,
         "u32": 4, "uint32": 4, "i64": 8, "int64": 8}


def price(design):
    p = int(design.get("params", 0))
    mem = 0
    for t in design.get("tensors", []):
        dt = str(t.get("dtype", "")).lower()
        if dt not in BYTES:
            return None
        n = 1
        for d in t.get("shape", []):
            n *= int(d)
        mem += n * BYTES[dt] * int(t.get("count", 1))
    return p + mem


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tid = int(sys.argv[1])
    rule = sys.argv[2] if len(sys.argv) > 2 else "(see pack)"
    import csv, math
    row = None
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv"))):
        if r["task"] == f"task{tid:03d}":
            row = r
    cost, pts = int(row["cost"]), float(row["score"])
    lane = ""
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "lanes", "QUEUE.csv"))):
        if r["task"] == f"task{tid:03d}":
            lane = r["lane"]
    bar = int(cost * math.exp(-0.15))
    card = json.load(open(os.path.join(ROOT, "data", "strategies.json"))).get(f"{tid:03d}", {})
    cert = json.load(open(os.path.join(ROOT, "data", "source_certs.json"))).get(f"{tid:03d}", {})
    rp = os.path.join(ROOT, "packs", f"task{tid:03d}", "rule.py")
    rule_src = open(rp, errors="replace").read()[:2500] if os.path.exists(rp) else ""

    prompt = f"""SINGLE-RESPONSE TASK. You have NO tools; your FIRST message IS the final answer.
You are a design-budget enumerator for ONNX graph golf. Grader cost = params (initializer
ELEMENT count, dtype-free) + memory (every NODE OUTPUT: elements × dtype-bytes {BYTES});
the GRAPH OUTPUT tensor is FREE (terminal renderer); tensor reuse is free.
Task {tid}: rule = {rule}
{('Verified numpy rule:\n' + rule_src) if rule_src else ''}
Certified bounds: M_proof={cert.get('M_proof')} out_law={(cert.get('facts') or {}).get('output_shape_law')}
Strategy card: {json.dumps(card)[:900]}
Pin cost {cost} ({pts:.2f} pts). REGISTER TARGET: total cost ≤ {bar}.
Propose 18 GENUINELY DIFFERENT design budgets spanning ≥5 structural families (terminal
Einsum, Gather-LUT, ConvInteger stencil, crop-native, packed bitset, row/col projections,
grouped conv, scatter-stamp…). Vary crop size (use the certificate!), dtypes, packing.
Reply with ONLY a JSON array, each element:
{{"family":"...","sketch":"≤25 words how the rule maps onto it",
 "tensors":[{{"shape":[1,1,10,10],"dtype":"u8","count":2}},...],  // every charged NODE OUTPUT, output excluded
 "params":123}}"""
    cmd = [sys.executable, os.path.join(HERE, "admin_model.py"), "--prompt", prompt,
           "--effort", os.environ.get("ADMIN_PS_EFFORT", "low"), "--timeout", "300"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    m = re.search(r"\[.*\]", p.stdout, re.S)
    if not m:
        sys.exit("price_search: no JSON in model output — rerun or design by hand")
    try:
        designs = json.loads(m.group(0))
    except ValueError:
        sys.exit("price_search: unparseable JSON — rerun or design by hand")
    priced = []
    for d in designs:
        c = price(d)
        if c is not None:
            priced.append((c, d))
    priced.sort(key=lambda x: x[0])
    import math as _m
    feas = [(c, d) for c, d in priced if c <= bar]
    near = [(c, d) for c, d in priced if bar < c <= bar * 1.2]
    print(f"task{tid:03d}: pin {cost} → register at ≤ {bar}.  "
          f"{len(priced)} designs repriced, {len(feas)} CLOSE on paper:\n")
    for c, d in feas:
        print(f"  {c:>6} B → {25 - _m.log(max(1, c)):.2f} pts  [{d.get('family', '?')}] "
              f"{d.get('sketch', '')[:90]}")
    if near:
        print("\n  near-misses (within 20% — one deleted tensor away):")
        for c, d in near:
            print(f"  {c:>6} B  [{d.get('family', '?')}] {d.get('sketch', '')[:80]}")
    print("\n[prices recomputed by this harness — trustworthy. Sketches are HYPOTHESES: "
          "the rule must still fit; confirm with g.budget() as you build.]")


if __name__ == "__main__":
    main()
