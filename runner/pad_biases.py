#!/usr/bin/env python3
"""pad_biases — the L0 keystone fix, automated.

Pads short Conv biases (task144: 4→10, task230: 5→10 — and any other short-bias member
found) with zeros, writes hardened members + a hardened base zip.

Usage: python3 pad_biases.py [--base /tmp/pv2_70_263.zip]
Then:  gate both members (commands printed), solo-probe the hardened base against the current pin.
"""
import argparse, io, os, shutil, tempfile, zipfile
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLEAN
import onnx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "hardened"))


def pad_model(data: bytes):
    m = onnx.load(io.BytesIO(data))
    inits = {i.name: i for i in m.graph.initializer}
    fixed = []
    for n in m.graph.node:
        w = b = None
        if n.op_type == "Conv" and len(n.input) >= 3:
            w, b = inits.get(n.input[1]), inits.get(n.input[2])
        elif n.op_type == "QLinearConv" and len(n.input) >= 9:
            w, b = inits.get(n.input[3]), inits.get(n.input[8])
        if w is None or b is None:
            continue
        oc = w.dims[0]
        bl = b.dims[0] if b.dims else 1
        if bl < oc:
            arr = onnx.numpy_helper.to_array(b)
            padded = np.zeros(oc, dtype=arr.dtype); padded[:bl] = arr
            nb = onnx.numpy_helper.from_array(padded, b.name)
            b.CopyFrom(nb)
            fixed.append((n.op_type, bl, oc))
    return m, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/tmp/pv2_70_263.zip")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    zin = zipfile.ZipFile(a.base)
    hardened = {}
    for name in zin.namelist():
        m, fixed = pad_model(zin.read(name))
        if fixed:
            p = os.path.join(OUT, name)
            onnx.save(m, p)
            hardened[name] = fixed
            print(f"{name}: padded {fixed}  -> {p}")
    outzip = os.path.join(OUT, "hardened_base.zip")
    if not hardened:
        # A repin may already contain the padding.  The previous behavior returned while
        # leaving an older hardened_base.zip in place; repin.py then stamped that stale zip
        # as belonging to the new pin, silently reverting newly gained members.  Publish an
        # exact snapshot of the supplied base instead.
        fd, tmp = tempfile.mkstemp(prefix="hardened_base.", suffix=".zip", dir=OUT)
        os.close(fd)
        try:
            shutil.copy2(a.base, tmp)
            os.replace(tmp, outzip)
        finally:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
        print(f"no short biases found — wrote exact hardened snapshot {outzip}")
        return
    fd, tmp = tempfile.mkstemp(prefix="hardened_base.", suffix=".zip", dir=OUT)
    os.close(fd)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in zin.namelist():
            if name in hardened:
                z.write(os.path.join(OUT, name), name)
            else:
                z.writestr(name, zin.read(name))
    os.replace(tmp, outzip)
    print(f"\nwrote {outzip}")
    print("\nNEXT (mandatory before banking):")
    for name in hardened:
        t = int("".join(c for c in name if c.isdigit())[:3])
        print(f"  python3 {HERE}/fast_verify.py {t} {OUT}/{name} --bar -1 --full")
    print(f"  cd {CLEAN} && "
          f"python3 candidates/probe_submit.py submit {outzip} --lane harden "
          f"--note 'bias-pad t144 t230 against current pin'")


if __name__ == "__main__":
    main()
