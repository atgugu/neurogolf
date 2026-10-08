#!/usr/bin/env python3
"""compile_strategies — a tailored, evidence-based strategy card for EVERY queued task.

Fuses per task: pin anatomy (op patterns, dtype mix, penultimate-tensor size, oversized
planes vs certificate) + Hodel verdict + source certificate + fleet record + headroom →
primary/secondary attack family with NAMED kill-targets. Deterministic; no LLM.

Usage: python3 compile_strategies.py [--pin-dir <extracted pin dir>]
Writes: data/strategies.json + lanes/STRATEGY_INDEX.md
make_pack.py injects the card at the top of each ATTACK.md.
"""
import argparse, collections, csv, io, json, math, os, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN
from config import ARCHIVE
TRACES = os.path.join(ARCHIVE, "reasoning_traces")


def anatomy(data):
    import onnx, numpy as np
    m = onnx.load(io.BytesIO(data))
    g = onnx.shape_inference.infer_shapes(m, strict_mode=True).graph
    ops = collections.Counter(n.op_type for n in g.node)
    outname = g.output[0].name
    final = next((n for n in g.node if outname in n.output), None)
    tmap = {t.name: t for t in list(g.value_info) + list(g.output)}
    feats = {"n_nodes": len(g.node), "ops": dict(ops.most_common(15)),
             "final_op": final.op_type if final else "?"}
    tensors, penult = [], 0
    for n in g.node:
        for o in n.output:
            if not o or o == outname:
                continue
            it = tmap.get(o)
            if it is None or not it.type.HasField("tensor_type"):
                continue
            tt = it.type.tensor_type
            dims = [d.dim_value for d in tt.shape.dim if d.HasField("dim_value")]
            if not dims or any(d <= 0 for d in dims):
                continue
            npd = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type))
            b = int(np.prod(dims)) * npd.itemsize
            tensors.append((b, n.op_type, str(npd), dims, o))
            if final and o in final.input:
                penult = max(penult, b)
    tensors.sort(reverse=True)
    feats["penult_bytes"] = penult
    feats["top"] = [(b, op, dt, sh) for b, op, dt, sh, _ in tensors[:6]]
    feats["wide_bytes"] = sum(b for b, _, dt, _, _ in tensors if dt in ("float32", "float16", "int64"))
    feats["mem_sum"] = sum(b for b, *_ in tensors)
    return feats


def oversized(feats, M):
    if not M:
        return 0
    tot = 0
    for b, _, _, sh in feats["top"]:
        sp = [d for d in sh[-2:] if d >= 8]
        if sp and max(sp) > M:
            f = 1.0
            for d in sp:
                if d > M:
                    f *= M / d
            tot += b * (1 - f)
    return int(tot)


def fleet(t):
    import glob
    fs = [os.path.basename(f) for f in glob.glob(f"{TRACES}/task{t}__*")]
    done = [f for f in fs if "__DONE" in f]
    return len(fs), len(done)


def decide(t, lane, cost, params, memory, score, feats, cert, verdict):
    ops = feats["ops"]; M = cert.get("M_proof")
    conf = cert.get("confidence") == "high"
    ov = oversized(feats, M) if conf else 0
    p_share = params / max(1, cost)
    strat, kills = [], []
    for b, op, dt, sh in feats["top"][:4]:
        if b >= 800:
            kills.append(f"{b}B {dt}{sh} from {op}")
    # ordered heuristics — first match = primary, next = secondary
    if feats["penult_bytes"] > 1200:
        strat.append(("TERMINAL-RENDERER FIX",
                      f"the final {feats['final_op']}'s input is {feats['penult_bytes']}B but the "
                      f"output tensor is FREE — restructure so the LAST op does the full expansion "
                      f"(Equal-vs-arange / Gather / ConvTranspose) from a tiny state"))
    if ov > max(500, 0.25 * memory):
        strat.append(("CROP-NATIVE (certified)",
                      f"~{ov}B of planes exceed the source-certified bound M={M}; "
                      f"Slice→compute at native size→Pad"))
    if feats["wide_bytes"] > 0.35 * max(1, feats["mem_sum"]):
        strat.append(("PLANE-KILL + u8 RELOWER",
                      f"{feats['wide_bytes']}B ({100*feats['wide_bytes']//max(1,feats['mem_sum'])}%) "
                      f"sits in fp32/f16/i64 tensors — value-exact relower to uint8/bool, compose in u8"))
    mp = ops.get("MaxPool", 0) + ops.get("Conv", 0) * (ops.get("Clip", 0) > 2)
    if mp >= 6:
        strat.append(("MORPHOLOGY ROUND-CAP",
                      f"{ops.get('MaxPool',0)}×MaxPool/{ops.get('Conv',0)}×Conv iteration chain — cap "
                      f"rounds via certified diameter (M={M}), or one big-kernel dilation where unconstrained"))
    bitn = sum(v for k, v in ops.items() if k.startswith("Bit"))
    if bitn >= 20:
        strat.append(("COMPRESSED SCHEDULE",
                      f"{bitn} bitwise nodes — fewer propagation rounds / packed state; NEVER more unroll"))
    if p_share > 0.4:
        strat.append(("INIT-PACK",
                      f"params are {100*p_share:.0f}% of cost ({params}) — pack tables into int64 "
                      f"(64 bits/unit), factorize/dedupe initializers"))
    if ops.get("Einsum", 0) >= 2:
        strat.append(("EINSUM-COMPRESS", f"{ops['Einsum']}×Einsum — merge contractions, shrink operand tables"))
    if ops.get("Gather", 0) + ops.get("GatherElements", 0) >= 8:
        strat.append(("LUT-REFINE", f"{ops.get('Gather',0)}×Gather — fewer/smaller tables, index arithmetic"))
    if not strat:
        strat.append(("REBUILD-FROM-RULE", "no dominant structural weakness — rebuild from the rule "
                                           "with a terminal renderer and u8 working set"))
    if isinstance(verdict, dict):
        v = verdict.get("verdict", "?")
        ag = verdict.get("splits", {}).get("arc-gen", {})
        hodel_note = ("Hodel TRUE-RULE: the rule is exactly right — implement it CHEAPLY "
                      "(never transcribe DSL→ONNX op-by-op; that channel scored −339)"
                      if v == "TRUE-RULE" else
                      f"Hodel {v}: diverges on {ag.get('bad',0)}/{ag.get('ok',0)+ag.get('bad',0)} "
                      f"arc-gen draws — the pack renders the failing examples; decode the variant first")
    else:
        hodel_note = f"Hodel {verdict}"
    n_att, n_done = fleet(t)
    return {"primary": strat[0], "secondary": strat[1] if len(strat) > 1 else None,
            "kill_targets": kills[:4], "hodel": hodel_note,
            "fleet": f"{n_att} past attempts, {n_done} DONE",
            "target_cost": int(cost * math.exp(-(0.05 if lane == 'A' else 0.15)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-dir", default=None, help="extracted pin dir (else reads submission.zip)")
    a = ap.parse_args()
    verd = {f"{r['task']:03d}": r
            for r in map(json.loads, open(f"{CLEAN}/candidates/hodel_verdicts.jsonl"))}
    certs = json.load(open(f"{ROOT}/data/source_certs.json"))
    rows = {}
    with open(f"{ROOT}/data/pin_pv2_70_263_pertask.csv") as f:
        for r in csv.DictReader(f):
            if r["cost"]:
                rows[r["task"][4:]] = (int(r["cost"]), int(r["params"]), int(r["memory"]), float(r["score"]))
    queue = list(csv.DictReader(open(f"{ROOT}/data/lanes/QUEUE.csv")))
    zin = None if a.pin_dir else zipfile.ZipFile(f"{CLEAN}/submission.zip")

    out, idx = {}, ["# STRATEGY INDEX — one tailored line per queued task\n"]
    for q in queue:
        t = q["task"][4:]
        data = (open(os.path.join(a.pin_dir, f"task{t}.onnx"), "rb").read() if a.pin_dir
                else zin.read(f"task{t}.onnx"))
        try:
            feats = anatomy(data)
        except Exception as e:
            out[t] = {"error": str(e)[:100]}; continue
        c, p, m, s = rows[t]
        card = decide(t, q["lane"], c, p, m, s, feats, certs.get(t, {}), verd.get(t, {}))
        out[t] = card
        idx.append(f"- task{t} [{q['lane']}] {s:.2f}→cost≤{card['target_cost']}: "
                   f"**{card['primary'][0]}** — {card['primary'][1][:110]}")
    json.dump(out, open(f"{ROOT}/data/strategies.json.tmp", "w"), indent=1)
    os.replace(f"{ROOT}/data/strategies.json.tmp", f"{ROOT}/data/strategies.json")
    open(f"{ROOT}/data/lanes/STRATEGY_INDEX.md.tmp", "w").write("\n".join(idx))
    os.replace(f"{ROOT}/data/lanes/STRATEGY_INDEX.md.tmp", f"{ROOT}/data/lanes/STRATEGY_INDEX.md")
    prim = collections.Counter(v["primary"][0] for v in out.values() if "primary" in v)
    print(f"compiled {len(out)} strategy cards → data/strategies.json")
    for k, v in prim.most_common():
        print(f"  {v:>3} × {k}")


if __name__ == "__main__":
    main()
