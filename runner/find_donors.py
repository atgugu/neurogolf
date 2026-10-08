#!/usr/bin/env python3
"""find_donors — for each queued task, the most similar WELL-SOLVED tasks (donors).

Similarity = TF-IDF cosine over ARC-GEN generator source code. A donor is a task whose
pin member scores ≥ 18.3 (proven-cheap representation). The 133←132 sibling migration was
the audit's single best named bet — this scales the pattern fleet-wide.

Usage: python3 find_donors.py [--pin-dir <extracted pin>]
Writes data/donors.json; make_pack embeds top donors with their op recipes.
"""
import argparse, collections, csv, io, json, math, os, re, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN


def tokens(src):
    toks = re.findall(r"[a-z_]{3,}", src.lower())
    stop = {"the", "and", "for", "that", "with", "this", "task", "def", "return",
            "import", "from", "common", "generate", "args", "int", "list", "str"}
    return [t for t in toks if t not in stop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-dir", default=None)
    a = ap.parse_args()
    hexmap = json.load(open(f"{ROOT}/data/task_map.json"))
    rows = {}
    with open(f"{ROOT}/data/pin_pv2_70_263_pertask.csv") as f:
        for r in csv.DictReader(f):
            if r["cost"]:
                rows[r["task"][4:]] = float(r["score"])
    # corpus
    docs = {}
    for t, hx in ((k[4:], v) for k, v in hexmap.items()):
        p = f"{CLEAN}/external/ARC-GEN/tasks/task_{hx}.py"
        if os.path.exists(p):
            docs[t] = collections.Counter(tokens(open(p, errors="replace").read()))
    df = collections.Counter()
    for d in docs.values():
        df.update(set(d))
    N = len(docs)

    def vec(d):
        return {w: c * math.log(N / df[w]) for w, c in d.items()}

    def cos(a, b):
        num = sum(v * b.get(w, 0) for w, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0

    vecs = {t: vec(d) for t, d in docs.items()}
    donors = [t for t, s in rows.items() if s >= 18.3 and t in vecs]

    def ophist(t):
        import onnx
        if a.pin_dir:
            data = open(os.path.join(a.pin_dir, f"task{t}.onnx"), "rb").read()
        else:
            data = zipfile.ZipFile(f"{CLEAN}/submission.zip").read(f"task{t}.onnx")
        m = onnx.load(io.BytesIO(data))
        return " ".join(f"{k}:{v}" for k, v in
                        collections.Counter(n.op_type for n in m.graph.node).most_common(8))

    queue = sorted({r["task"][4:] for r in csv.DictReader(open(f"{ROOT}/data/lanes/QUEUE.csv"))})
    out, cache = {}, {}
    for t in queue:
        if t not in vecs:
            continue
        sims = sorted(((cos(vecs[t], vecs[d]), d) for d in donors if d != t), reverse=True)[:3]
        entry = []
        for s, d in sims:
            if s < 0.12:
                continue
            if d not in cache:
                cache[d] = ophist(d)
            entry.append({"donor": f"task{d}", "sim": round(s, 3),
                          "donor_pts": rows[d], "donor_ops": cache[d]})
        if entry:
            out[t] = entry
    json.dump(out, open(f"{ROOT}/data/donors.json.tmp", "w"), indent=1)
    os.replace(f"{ROOT}/data/donors.json.tmp", f"{ROOT}/data/donors.json")
    strong = sum(1 for v in out.values() if v and v[0]["sim"] >= 0.3)
    print(f"donors for {len(out)}/{len(queue)} queued tasks ({strong} with sim ≥ 0.3)")
    for t, v in sorted(out.items(), key=lambda kv: -kv[1][0]["sim"])[:10]:
        print(f"  task{t} ← {v[0]['donor']} sim={v[0]['sim']} ({v[0]['donor_pts']:.1f} pts)")


if __name__ == "__main__":
    main()
