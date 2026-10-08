#!/usr/bin/env python3
"""make_pack — assemble packs/taskNNN/ATTACK.md: everything a worker model needs.

Usage: python3 make_pack.py 133 [--no-context]

Pulls: fresh pin numbers (CSV) · source certificate · generator SOURCE CODE (the
competition oracle) · pin-member anatomy (op histogram + top charged tensors) ·
dossier + graphs-analysis excerpts · arcgen_context live digest (optional, ~30s).
"""
import argparse, collections, csv, io, json, math, os, re, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PACKS = os.environ.get("PACKS_ROOT", os.path.join(HERE, "..", "packs"))
from config import CLEAN, ARCHIVE, RESOURCES
DOCS = os.path.join(ARCHIVE, "docs")
EXCERPT_LINES = 130

sys.path.insert(0, HERE)
from proven_corpus import render as proven_corpus_render
from hidden_oracle import render as hidden_oracle_render
from evidence import file_is_current
from rank_difficulty import target_cost   # rule-class compile targets (single source)

# THE GRAPH TO BEAT is shown only where reading the graph IS the strategy; for
# fresh-design families the old graph (and its kill-target enumeration) is an ANCHOR
# toward micro-shaving — the measured winning move is re-deriving the rule as a
# contraction, not deleting the named tensors (study7300, 30+ instances).
READ_GRAPH = {"TERMINAL-RENDERER FIX", "EINSUM-COMPRESS", "COMPRESSED SCHEDULE",
              "INIT-PACK", "LUT-REFINE"}

# strategy family → preferred compile-exemplar mechanisms (intel/compile_exemplars/)
_FAM_MECH = {
    "REBUILD-FROM-RULE": ["compiled-terminal-einsum", "selector-palette-recolor",
                          "boolean-selectors-sentinels"],
    "EINSUM-COMPRESS": ["compiled-terminal-einsum", "mode-dim-packing", "factor-reuse"],
    "LUT-REFINE": ["selector-palette-recolor", "rank-factor-color",
                   "arithmetic-replaces-table"],
    "TERMINAL-RENDERER FIX": ["selector-palette-recolor", "compiled-terminal-einsum"],
    "COMPRESSED SCHEDULE": ["mode-dim-packing", "factor-reuse", "compiled-terminal-einsum"],
    "INIT-PACK": ["arithmetic-replaces-table", "radix-position-factors",
                  "trig/fourier-color"],
    "CROP-NATIVE (certified)": ["crop-placement-attrs", "conv-attr-geometry"],
    "PLANE-KILL + u8 RELOWER": ["counting-histogram", "compiled-terminal-einsum",
                                "relational/pairwise-input²"],
    "MORPHOLOGY ROUND-CAP": ["boolean-selectors-sentinels", "relational/pairwise-input²"],
}


def compile_section(tid, primary, target):
    """COMPILE THE RULE — the measured #1 family (study7300: 30+ verified instances on
    this exact base, mean win +0.44 vs the wave's +0.17 median). Injected AHEAD of the
    strategy card so builders write the contraction first."""
    exdir = os.path.join(HERE, "..", "data", "intel", "compile_exemplars")
    ex_txt = ""
    try:
        raw = json.load(open(os.path.join(exdir, "INDEX.json")))
        entries = raw.get("exemplars", []) if isinstance(raw, dict) else raw
        idx = [e for e in entries if isinstance(e, dict) and not e.get("skipped")]
        prefs = _FAM_MECH.get(primary, ["compiled-terminal-einsum",
                                        "selector-palette-recolor", "factor-reuse"])
        def rank(e):
            mechs = e.get("mechanisms", [])
            p = min((prefs.index(m) for m in mechs if m in prefs), default=len(prefs))
            close = abs(math.log(max(1, e.get("cost") or target) / max(1, target)))
            return (p, close)
        for e in sorted(idx, key=rank)[:3]:
            body = read_excerpt(os.path.join(exdir, e["file"]), 999)[:3600]
            ex_txt += (f"\n### Exemplar task{int(e['task']):03d} — "
                       f"{'/'.join(e.get('mechanisms', []))} (cost {e.get('cost')}, "
                       f"{float(e.get('points') or 0):.2f} pts)\n{body}\n")
    except (OSError, ValueError, KeyError):
        ex_txt = "\n(compile-exemplar library not built yet — data/intel/compile_exemplars/)\n"
    return f"""
## COMPILE THE RULE — the measured #1 family (try this FIRST)
The verified study7300 corpus (all 111 improved pairs re-scored vs this base) shows the
dominant win engine is: compile the WHOLE rule into ONE terminal multi-operand Einsum
(single node ⇒ charged memory 0 — the node's internal workspace is free), built from:
- the FREE input repeated 2–10× as operands — products of the input with itself are
  pairwise/relational/quadratic features (same-color tests, counts, coincidences);
- 2–5 TINY factor matrices reused across index roles (each charged ONCE — dedupe!):
  10×10 color relations, rank-2/3 palette factors, trig/radix position codes;
- boolean branches as scalar SELECTOR operands (0/1 factors), never graph branches;
- integer-exact sentinels that force decisions through the (raw>0) decode: ±1, 5000,
  −1000, −0.001 ε-suppression — values exact in fp32;
- operand ORDER = ORT speed: interleave each input copy with the factor(s) that
  eliminate its indices (measured 4.1 s → 0.008 s for the same equation).
**HARD BUDGET: this task's rule class ({primary or 'unclassified'}) compiles at
≤{target} elements. WRITE THE EQUATION AND THE FACTOR TABLE BEFORE ANY OPS** — if the
factor table alone exceeds the budget, the representation is wrong, not the budget.
Builder support: `sys.path.insert(0,"../../runner"); from ngolf_einsum import
build_einsum_model, parity` — `parity()` checks your equation against rule.py on real
draws BEFORE you spend a gate round. A build that verifies CORRECT but misses the class
budget is NOT a NO — register it as a factorization ticket (contract step 5).
{ex_txt}"""


def read_excerpt(path, n=EXCERPT_LINES):
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
        head = "".join(lines[:n])
        more = f"\n…[{len(lines)-n} more lines in {path}]" if len(lines) > n else ""
        return head + more
    except OSError:
        return f"(missing: {path})"


def read_current_excerpt(path, n=EXCERPT_LINES):
    if not file_is_current(path):
        return "(pre-fix historical dossier withheld by data/EVIDENCE_ERA.json)"
    return read_excerpt(path, n)


def anatomy(tid):
    try:
        return _anatomy(tid)
    except Exception:
        return ("(pin anatomy unavailable — set NEUROGOLF_CLEAN to a neurogolf_clean "
                "checkout containing submission.zip; see docs/SETUP.md)")


def _anatomy(tid):
    import onnx, numpy as np
    zf = zipfile.ZipFile(os.path.join(CLEAN, "submission.zip"))
    data = zf.read(f"task{tid:03d}.onnx")
    m = onnx.load(io.BytesIO(data))
    g = onnx.shape_inference.infer_shapes(m, strict_mode=True).graph
    ops = collections.Counter(n.op_type for n in g.node)
    outname = g.output[0].name
    tmap = {t.name: t for t in list(g.value_info) + list(g.output)}
    big = []
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
            npd = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
            big.append((int(np.prod(dims)) * np.dtype(npd).itemsize, n.op_type, str(npd), dims))
    big.sort(reverse=True)
    lines = [f"- nodes: {len(g.node)} · op histogram: "
             + " ".join(f"{k}:{v}" for k, v in ops.most_common(12))]
    lines.append("- top charged tensors (bytes · producing-op · dtype · shape):")
    for b, op, dt, sh in big[:12]:
        lines.append(f"    {b:>6} B  {op:<14} {dt:<8} {sh}")
    out_dt = onnx.helper.tensor_dtype_to_np_dtype(g.output[0].type.tensor_type.elem_type)
    lines.append(f"- OUTPUT dtype of pin member: {out_dt} — KEEP IT unless the brief says "
                 f"otherwise (bool-rename on task080 cost −12.5 on Kaggle).")
    return "\n".join(lines)


def graph_dump(tid, max_nodes=120):
    """Node-by-node disassembly of the current best (pin) member — THE GRAPH TO BEAT.
    Full listing for graphs ≤ max_nodes; for bigger ones, the nodes producing the
    largest charged tensors (the cost lives there) plus the terminal chain."""
    try:
        return _graph_dump(tid, max_nodes)
    except Exception:
        return ("(pin disassembly unavailable — set NEUROGOLF_CLEAN to a neurogolf_clean "
                "checkout containing submission.zip; see docs/SETUP.md)")


def _graph_dump(tid, max_nodes=120):
    import onnx, numpy as np
    zf = zipfile.ZipFile(os.path.join(CLEAN, "submission.zip"))
    m = onnx.load(io.BytesIO(zf.read(f"task{tid:03d}.onnx")))
    g = onnx.shape_inference.infer_shapes(m, strict_mode=True).graph
    vi = {}
    for t in list(g.value_info) + list(g.output) + list(g.input):
        if t.type.HasField("tensor_type"):
            tt = t.type.tensor_type
            dims = [d.dim_value for d in tt.shape.dim if d.HasField("dim_value")]
            vi[t.name] = (onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type).name, dims)
    inits = {}
    for i in g.initializer:
        arr = onnx.numpy_helper.to_array(i)
        inits[i.name] = (f"init[{arr.dtype.name}{list(arr.shape)}]"
                         + (f"={arr.ravel().tolist()}" if arr.size <= 24 else ""))

    def fmt(n):
        ins = []
        for x in n.input:
            if x in inits:
                ins.append(inits[x])
            elif x in vi:
                ins.append(f"{x}:{vi[x][0]}{vi[x][1]}")
            else:
                ins.append(x or "''")
        outs = [f"{o}:{vi[o][0]}{vi[o][1]}" if o in vi else o for o in n.output]
        attrs = {}
        for a in n.attribute:
            try:
                v = onnx.helper.get_attribute_value(a)
                attrs[a.name] = "<tensor>" if a.type == onnx.AttributeProto.TENSOR else v
            except Exception:
                attrs[a.name] = "?"
        at = f" {attrs}" if attrs else ""
        return f"{n.op_type:<16} {', '.join(ins)} -> {', '.join(outs)}{at}"

    nodes = list(g.node)
    if len(nodes) <= max_nodes:
        body = "\n".join(fmt(n) for n in nodes)
        note = f"(complete graph, {len(nodes)} nodes)"
    else:
        def outbytes(n):
            b = 0
            for o in n.output:
                if o in vi and vi[o][1]:
                    import numpy as _np
                    b = max(b, int(_np.prod(vi[o][1])) * _np.dtype(vi[o][0]).itemsize)
            return b
        ranked = sorted(nodes, key=outbytes, reverse=True)[:60]
        keep = {id(n) for n in ranked} | {id(n) for n in nodes[-12:]}
        body = "\n".join(fmt(n) for n in nodes if id(n) in keep)
        note = (f"({len(nodes)} nodes — showing the 60 with the LARGEST outputs, i.e. where "
                f"the cost lives, plus the final 12 (the renderer chain))")
    return f"{note}\n```\n{body[:14000]}\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int)
    ap.add_argument("--no-context", action="store_true", help="skip arcgen_context (slow)")
    a = ap.parse_args()
    tid = a.task

    hexid = json.load(open(os.path.join(DATA, "task_map.json")))[f"task{tid:03d}"]
    cert = json.load(open(os.path.join(DATA, "source_certs.json"))).get(f"{tid:03d}", {})
    row = None
    with open(os.path.join(DATA, "pin_pv2_70_263_pertask.csv")) as f:
        for r in csv.DictReader(f):
            if r["task"] == f"task{tid:03d}":
                row = r
    cost, pts = int(row["cost"]), float(row["score"])
    bar = int(cost * math.exp(-0.15))
    local_note = ""
    _can = os.path.join(HERE, "..", "results", f"task{tid:03d}", f"task{tid:03d}.onnx")
    if os.path.exists(_can):
        try:
            import numpy as _np
            sys.path.insert(0, os.path.join(RESOURCES, "audit_2026-07-06"))
            from score_bundle import score_onnx as _so
            try:
                _j = json.load(open(f"{CLEAN}/extracted/task{tid:03d}.json"))
                _g = _np.array(_j["train"][0]["input"], _np.int8)
                _a = _np.zeros((30, 30), _np.int8); _a[:_g.shape[0], :_g.shape[1]] = _g
                _d = (_np.arange(10)[:, None, None] == _a[None]).astype(_np.float32)[None]
            except Exception:
                _d = _np.zeros((1, 10, 30, 30), _np.float32); _d[0, 0, 0, 0] = 1
            _r = _so(_can, _d)
            if _r.get("cost") and _r["points"] > pts:
                local_note = (f"\n## ⚠ LOCAL FRONTIER (unbanked win already exists!)\n"
                              f"results/task{tid:03d}/task{tid:03d}.onnx is ALREADY at cost "
                              f"{_r['cost']} → {_r['points']:.4f} pts (Δ{_r['points']-pts:+.3f} "
                              f"over the pin). THE GRAPH TO BEAT IS THAT ARTIFACT, not the "
                              f"pin — anything between the pin and it is worthless.\n")
        except Exception:
            pass
    gen_path = f"{CLEAN}/external/ARC-GEN/tasks/task_{hexid}.py"
    try:
        gen_src = open(gen_path, errors="replace").read()
    except OSError:
        gen_src = ("# (generator source unavailable — needs NEUROGOLF_CLEAN + external/ARC-GEN;\n"
                   "#  see docs/SETUP.md. The generator IS the task rule, verbatim.)")

    ctx = ""
    if not a.no_context:
        try:
            p = subprocess.run(["python3", "candidates/arcgen_context.py", str(tid)],
                               cwd=CLEAN, capture_output=True, text=True, timeout=120)
            ctx = p.stdout[-6000:]
        except Exception as e:
            ctx = f"(arcgen_context failed: {e})"

    # Hodel rule-as-code + pre-computed TRUE-RULE/PARTIAL verdict (instant, cached)
    try:
        p = subprocess.run(["python3", "candidates/hodel.py", str(tid)],
                           cwd=CLEAN, capture_output=True, text=True, timeout=60)
        hodel = p.stdout[:5000] or p.stderr[:500]
    except Exception as e:
        hodel = f"(hodel.py failed: {e})"
    # PARTIAL ⇒ append divergence stats + rendered failing examples (the variant delta)
    try:
        vrec = next(r for r in map(json.loads, open(f"{CLEAN}/candidates/hodel_verdicts.jsonl"))
                    if r["task"] == tid)
        if vrec["verdict"] != "TRUE-RULE":
            ag = vrec["splits"].get("arc-gen", {})
            fi = vrec.get("fail_idx", {}).get("arc-gen", [])
            hodel += (f"\n\nDIVERGENCE DATA: Hodel fails {ag.get('bad',0)}/{ag.get('ok',0)+ag.get('bad',0)}"
                      f" arc-gen draws (failing idx: {fi[:20]}). The TRUE rule = Hodel's rule "
                      f"MODIFIED in exactly the way these failing examples show — study them:")
            d = subprocess.run(["python3", "candidates/hodel.py", str(tid), "--diff"],
                               cwd=CLEAN, capture_output=True, text=True, timeout=120)
            hodel += "\n" + d.stdout[-4500:]
    except Exception:
        pass

    # Human rule descriptions (LARC/H-ARC) + human error modes
    try:
        p = subprocess.run(["python3", "candidates/human_annotations.py", str(tid)],
                           cwd=CLEAN, capture_output=True, text=True, timeout=30)
        human = p.stdout[:2500].strip() or "(none)"
    except Exception as e:
        human = f"(human_annotations failed: {e})"

    # Past fleet record: outcome-labeled trace filenames (June-July campaigns)
    import glob
    traces = sorted(os.path.basename(f) for f in
                    glob.glob(os.path.join(ARCHIVE, f"reasoning_traces/task{tid:03d}__*"))
                    if file_is_current(f))
    if traces:
        done = [t for t in traces if "__DONE" in t]
        rec = [f"{len(traces)} past fleet attempts on this task: {len(done)} DONE, "
               f"{sum('__NO__' in t for t in traces)} NO."]
        rec += [f"  {t.rsplit('.',1)[0]}" for t in traces[-15:]]
        fleet_record = "\n".join(rec)
    else:
        fleet_record = "(no past fleet attempts recorded)"

    # Past build scripts for THIS task (fleet code — best few-shot material)
    past = sorted(glob.glob(f"{CLEAN}/candidates/*/build_task{tid:03d}*.py")
                  + glob.glob(f"{CLEAN}/candidates/*/build_task{tid}*.py"),
                  key=os.path.getmtime)
    past = [p for p in past if file_is_current(p)]
    past_src = ""
    if past:
        past_src = (f"# newest of {len(past)} fleet build scripts: {past[-1]}\n"
                    + open(past[-1], errors="replace").read()[:6000])
    proven_txt = proven_corpus_render(tid)
    oracle_txt = hidden_oracle_render(tid)

    # donors + catalog references
    donor_txt = "(none found)"
    dp = os.path.join(DATA, "donors.json")
    if os.path.exists(dp):
        dl = json.load(open(dp)).get(f"{tid:03d}", [])
        if dl:
            # tier the framing by similarity — donors are injected down to sim 0.12 but only
            # sim ≥ 0.30 warrants "migrate the design"; below that it's a loose lead the builder
            # must verify, not an authoritative recipe (T11 — 51 tasks had weak donors framed
            # as authoritative → wrong-family attempts on a proven channel).
            donor_txt = ("HISTORICAL donor prices are pre-scorer-fix context. Verify every donor with "
                         "fast_verify; do not treat a historical score as a free or native-priced tensor.\n"
                         + "\n".join(
                f"- **{d['donor']}** (generator similarity {d['sim']}, scores {d['donor_pts']:.2f}) — "
                f"**{'MIGRATE: near-identical — port its design directly' if float(d['sim']) >= 0.30 else 'LOOSELY related — verify the mechanism transfers before relying on it'}**\n"
                f"  pin ops: {d['donor_ops']}\n"
                f"  extract: `python3 -c \"import zipfile;zipfile.ZipFile('{CLEAN}/submission.zip')"
                f".extract('{d['donor']}.onnx','.')\"` and study how it stays cheap"
                for d in dl))
    cat_txt = "(no catalog rows)"
    try:
        import sqlite3, glob as _g
        db_path = os.path.join(ARCHIVE, "task_candidates_gate_pass/onnx_catalog.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(db_path)
        db = sqlite3.connect(db_path)
        rows_db = db.execute("SELECT sha256,score,variant_name,source_tag FROM onnx_graphs "
                             "WHERE task_id=? AND score>0 ORDER BY score DESC LIMIT 3",
                             (tid,)).fetchall()
        lines = []
        for sha, sc, var, src in rows_db:
            fs = _g.glob(os.path.join(ARCHIVE, f"task_candidates_gate_pass/task{tid:03d}/{sha[:8]}*.onnx"))
            lines.append(f"- {sc:.3f} pts · {var} · {src}" + (f"\n  file: {fs[0]}" if fs else ""))
        if lines:
            cat_txt = ("HISTORICAL catalog prices are pre-scorer-fix context. Verify with fast_verify; "
                       "catalog rows are not proven Kaggle payments.\n" + "\n".join(lines))
    except Exception as e:
        cat_txt = f"(catalog query failed: {e})"

    # Lane E sweep certificate (generator-source extraction), if the sweep produced one
    sweep_txt = ""
    sw = os.path.join(HERE, "..", "results", "sweep", f"task{tid:03d}.json")
    if os.path.exists(sw):
        sweep_txt = ("\n## Lane E sweep certificate (source-proven; quotes generator lines)\n"
                     "```json\n" + open(sw, errors="replace").read()[:4000] + "\n```\n")

    # Transfer ticket: a taxonomy-sibling task won recently; sidecar-generated adaptation
    # hypotheses (advisory — a cost-budget table is still required before any code)
    tk = os.path.join(HERE, "..", "packs", f"task{tid:03d}", "TRANSFER_TICKET.md")
    if os.path.exists(tk):
        sweep_txt += ("\n## Transfer hypothesis from a sibling win (verify everything)\n"
                      + open(tk, errors="replace").read()[:2500] + "\n")

    # LUT blueprint: machine-harvested template table + (possibly verified) key fn
    lb = os.path.join(HERE, "..", "packs", f"task{tid:03d}", "LUT_BLUEPRINT.md")
    if os.path.exists(lb):
        sweep_txt += ("\n## Gather-LUT blueprint (empirically harvested)\n"
                      + open(lb, errors="replace").read()[:2000] + "\n")

    # Pin pseudocode: annotated semantic map of the graph this build must beat
    pp = os.path.join(HERE, "..", "packs", f"task{tid:03d}", "PIN_PSEUDOCODE.md")
    if os.path.exists(pp):
        sweep_txt += ("\n## Pin-graph pseudocode (what you must beat — advisory)\n"
                      + open(pp, errors="replace").read()[:3500] + "\n")

    # ORT-1.24 measured legality sheet (fleet evidence — ops that died, with task ids)
    lg = os.path.join(DATA, "ORT124_LEGALITY.md")
    if os.path.exists(lg):
        sweep_txt += "\n" + open(lg, errors="replace").read()[:1400] + "\n"

    # Standing stress counterexamples: rule.py failed machine-executed generator draws
    # (explicit kwargs — possibly outside the random-path distribution; still, check
    # the named branch before trusting the rule blindly)
    sx = os.path.join(HERE, "..", "results", "stress", f"task{tid:03d}.json")
    if os.path.exists(sx):
        try:
            sv = json.load(open(sx))
            cxs = sv.get("counterexamples") or []
            if cxs and sv.get("repair") != "REPAIRED_AND_VERIFIED":
                sweep_txt += ("\n## STRESS WARNING — rule.py has machine-confirmed "
                              "counterexamples\n"
                              f"{len(cxs)} adversarial generator draw(s) broke rule.py "
                              "(explicit kwargs; may be outside the gate's random-path "
                              "distribution — verify the branch, don't panic): "
                              + json.dumps(cxs[0].get("kwargs", {}))[:300]
                              + f" ({cxs[0].get('why', '')[:80]})\n")
        except (ValueError, OSError):
            pass

    facts = cert.get("facts", {})
    strat_txt = ""
    card = None
    sp = os.path.join(DATA, "strategies.json")
    if os.path.exists(sp):
        card = json.load(open(sp)).get(f"{tid:03d}")
        if card and "primary" in card:
            sec = f"\n- secondary: **{card['secondary'][0]}** — {card['secondary'][1]}" if card.get("secondary") else ""
            kt = card.get("kill_targets", [])
            # coverage verdict: do the named tensors actually DOMINATE the cost? A diffuse
            # graph mis-framed as concentrated makes builders micro-shave for +0.2 and stop
            # (T14/E-F1: e.g. task366 top-4 = 19% of cost). State the fraction + the verdict.
            _mem, _kb = int(row.get("memory") or 0), 0
            for _k in kt:
                _mm = re.match(r"(\d+)B", _k)
                if _mm:
                    _kb += int(_mm.group(1))
            _cov = (_kb / _mem) if (kt and _mem) else None
            # kill-target ENUMERATION only for READ_GRAPH families: for fresh-design
            # strategies the list anchors builders to the pin's representation, and the
            # measured winning move there is a replacement contraction (study7300 M2).
            if card["primary"][0] in READ_GRAPH:
                kills = "".join(f"\n  - {k}" for k in kt)
                if _cov is not None:
                    kills += (f"\n  → these ≈{_cov * 100:.0f}% of the {_mem} B memory — "
                              + ("REPLACE the structure that carries them (don't shave "
                                 "them in place); a ≥2x cut is in reach."
                                 if _cov >= 0.55 else
                                 "cost is DIFFUSE, so deleting them caps low — a durable win needs a "
                                 "SMALLER GRAPH (fewer nodes / cheaper family), not these deletions."))
            else:
                kills = ("\n  - (enumeration withheld for fresh-design families"
                         + (f" — the named tensors cover ≈{_cov * 100:.0f}% of the {_mem} B memory"
                            if _cov is not None else "")
                         + "; deleting them is the micro-shave anchor. Re-derive the rule "
                           "as a terminal contraction — see COMPILE THE RULE above.)")
            strat_txt = f"""
## TAILORED STRATEGY (compiled from anatomy + certificate + Hodel + fleet history)
- primary: **{card['primary'][0]}** — {card['primary'][1]}{sec}
- kill targets (the tensors that ARE the cost):{kills if kills else ' (diffuse — see anatomy)'}
- {card['hodel']}
- fixed-scorer register target: cost ≤ {card['target_cost']}
Start from the primary. If it dead-ends structurally, the secondary is pre-approved as
your one family pivot.
"""
    # THE GRAPH TO BEAT — strategy-conditional: show the full disassembly only where
    # reading the graph IS the strategy (READ_GRAPH, module level) or in Lane D
    # (briefs reference it; anti-anchor framing applies). For fresh-design strategies
    # (plane-kill / crop-native / rebuild-from-rule / morphology) the old graph is an
    # ANCHOR toward micro-shaving — withhold the listing, keep only the cost line-items.
    lane = ""
    try:
        for r in csv.DictReader(open(os.path.join(HERE, "..", "data", "lanes", "QUEUE.csv"))):
            if r["task"] == f"task{tid:03d}":
                lane = r["lane"]; break
    except OSError:
        pass
    primary = card["primary"][0] if card and "primary" in card else ""
    compile_txt = compile_section(tid, primary, target_cost(primary or "?", cost))
    if lane == "D" or primary in READ_GRAPH:
        beat_section = f"""## THE GRAPH TO BEAT — disassembly of the current best member for this task
This is the design to REPLACE, not to polish. It is here so you can (a) read the RULE off
it, and (b) see which tensor families make up the {cost}-byte cost — then DELETE those
families with a different representation. If your plan keeps most of these nodes and
tweaks dtypes/shapes, you are micro-shaving (−10% = +0.11 = below the bar); price a
replacement family instead.
{graph_dump(tid)}"""
    else:
        beat_section = f"""## THE GRAPH TO BEAT — withheld on purpose
Your strategy is a REPLACEMENT design ({primary or 'fresh build'}): design from the RULE
(generator source + Hodel + human descriptions + examples), NOT from the old graph — the
old design is an anchor toward micro-shaving. Its cost line-items are already listed in
the anatomy/kill-targets above; that is all you need to beat. If you genuinely need the
old member (e.g. as a behavioral oracle for distillation), extract it:
`python3 -c "import zipfile;zipfile.ZipFile('{CLEAN}/submission.zip').extract('task{tid:03d}.onnx','.')"`"""
    # REBUILD-FROM-RULE = anatomy routing is weakest — add the geometry→idiom router
    geo_section = ""
    if primary == "REBUILD-FROM-RULE":
        gp = os.path.join(DATA, "GEOMETRY_IDIOMS.md")
        if os.path.exists(gp):
            geo_section = ("\n## Family-selection aid — geometry → terminal idiom\n"
                           "Match this task's rule geometry to a row, open the anchor tasks' "
                           "pin members as worked examples, then price that family first.\n"
                           + open(gp, errors="replace").read())


    doc = f"""# ATTACK PACK — task{tid:03d}  (generator {hexid})

READ FIRST: ../../data/RULES.md (cost model, banned surface, ORT traps, acceptance contract).
Your lane brief (thesis + bar) is in ../../data/lanes/ — find task{tid:03d} there.
{local_note}
{compile_txt}
{strat_txt}

## Numbers (fresh — priced against the LIVE pin {open(os.path.join(HERE,'..','data','PIN_SHA.txt')).read().strip()}, server-exact replica)
- pin: cost **{cost}** (params {row['params']}, memory {row['memory']}) → **{pts:.4f} pts**
- to register you need (Lane {lane or '?'}): cost ≤ **{bar}** (Δ ≥ +0.15 — this is only the floor; aim ≥2× for a durable win)
- rule-class compile target: cost ≤ **{target_cost(primary or '?', cost)}** — the bar above registers; THIS is the win condition (a verified-correct build over it becomes a ticket, not a NO)
- targets for THIS pin (cost {cost}): **2× cut ⇒ ≤{cost // 2} B (Δ≈+0.69 — a durable win)** · next 0.5-pt band ⇒ ≤{int(cost * math.exp(-0.5))} B · absolute band ceilings 17⇒≤2981 · 18⇒≤1097 · 19⇒≤403 B

## Source certificate (from generator SOURCE — the only trustworthy invariants)
- M_proof (max grid dim): **{cert.get('M_proof')}** · confidence: {cert.get('confidence')}
- output_shape_law: {facts.get('output_shape_law')} · out_square: {facts.get('out_square')}
- palettes: in={facts.get('palette_in')} out={facts.get('palette_out')} · binary_in/out: {facts.get('binary_in')}/{facts.get('binary_out')}
- max colors/example in/out: {facts.get('max_colors_per_example_in')}/{facts.get('max_colors_per_example_out')}
{sweep_txt}
## Pin-member anatomy (what you must beat — these tensors are the cost)
{anatomy(tid)}

{beat_section}
{geo_section}

## Generator source (the competition oracle — the rule, verbatim)
```python
{gen_src}
```

<!-- ═══ DETAIL SECTIONS — inlined only on disk; the prompt points here ═══ -->

## Hodel reference solver (rule-as-code + verdict — TRUST ONLY IF "TRUE-RULE")
HISTORY WARNING: transcribing Hodel DSL ops one-to-one into ONNX was this project's FIRST
strategy and its worst failure (0 wins, −339 net). Hodel tells you WHAT the rule computes,
NEVER how to compute it cheaply — derive the rule, then design the cheapest ONNX
representation from the idiom menu.
POLICY: TRUE-RULE ⇒ the rule is exactly right; implement it (cheaply). PARTIAL ⇒ the
arc-gen rule DIVERGES — the divergence data below localizes the variant; decode it first.
{hodel}

## Human rule descriptions (LARC/H-ARC — HINTS, not authority: the generator source decides; "common wrong outputs" = the ambiguity traps humans hit)
{human}

## Clean-era fleet record on this task
{fleet_record}

## Newest clean-era build script for this task (fleet code — reuse idioms, beat its cost)
CAVEAT: "newest" ≠ "best" — this is the most recently touched script, possibly a rejected
attempt. Cross-check its family against the fleet record above before adopting it.
```python
  {past_src if past_src else '(none — you are first; see ../../data/exemplars/ for winning-style build.py files)'}
```

## Live-solver + generator digest
{ctx if ctx else '(run: cd ' + CLEAN + ' && python3 candidates/arcgen_context.py ' + str(tid) + ')'}

## DONOR TASKS (similar generator, already solved cheaply — MIGRATE the strong ones, treat weak ones as leads; see per-donor strength)
{donor_txt}

{proven_txt}

{oracle_txt}

## Best gate-passing catalog designs for THIS task (study the mechanism, then beat it — never copy bytes)
{cat_txt}

## Clean-era rule knowledge and failure history
Pre-fix dossiers are withheld rather than summarized: their scorer-era floors, prices, and
actions are not evidence. Generator/source evidence and exact Kaggle-paying corpus designs
remain available above.
### deep dossier (excerpt)
{read_current_excerpt(f'{DOCS}/deep_dossiers/task{tid:03d}_deep_dossier.md')}

### graphs analysis (excerpt)
{read_current_excerpt(f'{DOCS}/tasks_analysis/task{tid:03d}_graphs_analysis.md')}

## ⚠️ COST-MODEL TRUTH (scorer hack fixed 2026-07-10 — overrides ALL older advice)
The real Kaggle grader charges EVERY node-output tensor except the graph output. There
are NO "tagged scratch", "native-scored-away", or "skip_outputs" exemptions — those
existed only in a locally-hacked scorer, and models tuned to them SHORTED on the real
LB (probes #p807…: e.g. a "19.66-pt" relower design truly pays 12.36). Consequences:
- Only three things are free: the INPUT tensor, the graph OUTPUT tensor, and a single
  node's internal ORT workspace (e.g. one terminal Einsum). Everything else is charged.
- Do NOT copy `skip_outputs` / doc_string-tag tricks from older exemplars or build.py
  files — they buy nothing and hide the real price.
- Price every transient: a Slice→Cast relower pays the f32 transient (crop×4 B) PLUS
  the u8 result. It can still win, but only if the small plane is reused enough.
- Believe only `fast_verify.py` (grader-true since 2026-07-10) and `g.budget()`/`save()`
  (auto-charges all tmps). Any "free tensor" claim needs a Kaggle probe as proof.

## Your contract
1. Write `build.py` (in this folder) that emits `task{tid:03d}.onnx` — **use the ngolf
   builder library** (`import sys; sys.path.insert(0,"../../runner"); from ngolf import G`)
   — it emits valid ONNX, auto-clamps indices, tracks the exact grader cost as you build
   (`g.budget()`), and its terminal `out_*()` renderers exploit the free output tensor.
   For the COMPILE-THE-RULE family use `from ngolf_einsum import build_einsum_model,
   parity` (single-node terminal Einsum, repeated-input operands, dedup'd factors).
   Read `../../runner/ngolf.py` (short) and `../../data/DSL_TO_ONNX.md` (the combinator→
   cheap-idiom table for the Hodel program below). Raw onnx.helper is allowed but you
   own every trap yourself.
2. Check semantics BEFORE the gate: `parity(model, rule_fn, task={tid})` (ngolf_einsum) on
   the real graded grids, then iterate:
   `python3 ../../runner/fast_verify.py {tid} task{tid:03d}.onnx` — fix what it says. ≤6 rounds.
3. On PASS: `python3 ../../runner/fast_verify.py {tid} task{tid:03d}.onnx --full`
4. On FULL PASS: copy .onnx + build.py to ../../results/task{tid:03d}/ and append to
   ../../results/LEDGER.csv: `task{tid:03d},<cost>,<pts>,<Δ>,<lane>,<sha8>,<notes>`
5. Stop condition: 6 failed verify rounds on the same structural family ⇒ PIVOT family
   once; a second family failure ⇒ write ../../results/task{tid:03d}/NO.md with what you
   learned and move to the next task in your lane. EXCEPTION — a build that gates
   CORRECT and only misses the bar / class budget is NOT a NO: submit_result.py exits 3
   ("TICKET") and registers your graph as a factorization-ticket donor for the
   mechanical sweep (data/tickets/). Never delete a correct artifact.
"""
    outdir = os.path.join(PACKS, f"task{tid:03d}")
    os.makedirs(outdir, exist_ok=True)
    final = os.path.join(outdir, "ATTACK.md")
    with open(final + ".tmp", "w") as f:
        f.write(doc)
    os.replace(final + ".tmp", final)  # atomic: workers may read packs mid-write
    print(f"wrote {outdir}/ATTACK.md")


if __name__ == "__main__":
    main()
