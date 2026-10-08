#!/usr/bin/env python3
"""task110 attempt #13: selected residue-id basis terminal.

True rule: restore the pristine 29x29 periodic background by selecting the
smallest row period p in 4..9, using exact absolute-column colors when visible,
and otherwise backfilling from columns with the same c mod p.

Priced designs before build:
  A. Compact residue-id selected basis (this file, next 17-point band):
     Params:
       detector tables: sample_rows [4,30]=120, shifted_rows [6,4,30]=720,
         col_mask [30]=30, nonzero color mask [10]=10, periods [6]=6,
         period false branch [6]=6
       residue ids [6,30]=180, arange9 [1,9]=9, reshape [2]=2,
         gather clamp scalars=2, p-offset scalar=1, signed color gate [10,10]=100
       total params = 1186
     Memory:
       same [6] f32=24, both [6] f32=24, valid [6] bool=6,
       period_or_99 [6] i32=24, selected p scalar i32=4,
       p index scalar i32=4, clipped p index scalar i32=4,
       selected residues [30] u8=30, reshaped residues [30,1] u8=30,
       basis equality [30,9] bool=270, selected basis [30,9] f32=1080
       total memory = 1500
     Paper total = 2686, below the 2981 next score band and the 3274 register
     floor. It is not the 1902 two-times target, but it attacks the large
     initializer family instead of rebuilding the banked [6,30,9] basis.

  B. Packed bitset / Gather-LUT target (different family, >=2x target):
     Price-search paper totals were 483-1441 B by packing cmod/backfill tables
     and decoding the active period. Those budgets do target the <=1902 2x
     lane, but they rely on a nonexistent cheap primitive: selecting arbitrary
     runtime colors from FLOAT [1,10,30,30] one-hot into compact u8/int bitsets
     without either a full color-id collapse or the terminal one-hot product.
     ONNX has no PopCount/select primitive, and legal ScatterND/GatherND routes
     still require charged f32 color updates from one-hot input. Rejected before
     build as a paper-only family.

  C. Fully factorized no-selected-basis terminal (different family, lower paper):
     Params: detector 892 + residue basis U [6,30,9]=1620 + gate 100 = 2612.
     Memory: detector scalars/vectors only, 112 B.
     Paper total = 2724. Prior attempt #12 found this statically cheap but
     runtime-dead in scorer/full gate, so it is not rebuilt here.
"""

import sys

import numpy as np

sys.path.insert(0, "../../runner")
from ngolf import G


R = 4
COLS = tuple(range(29))


def residue_basis():
    u = np.zeros((6, 30, 9), np.float32)
    for ip, p in enumerate(range(4, 10)):
        for i in range(29):
            u[ip, i, i % p] = 1.0
    return u


def residue_ids():
    r = np.zeros((6, 30), np.uint8)
    for ip, p in enumerate(range(4, 10)):
        for i in range(30):
            r[ip, i] = i % p if i < 29 else 9
    return r


def build(out="task110.onnx", ort_check=True):
    g = G(task=110)

    sample_rows = np.zeros((R, 30), np.float32)
    shifted_rows = np.zeros((6, R, 30), np.float32)
    for n in range(R):
        sample_rows[n, n] = 1.0
        for ip, p in enumerate(range(4, 10)):
            shifted_rows[ip, n, n + p] = 1.0
    col_mask = np.zeros((30,), np.float32)
    col_mask[list(COLS)] = 1.0
    color_mask = np.ones((10,), np.float32)
    color_mask[0] = 0.0

    row0 = g.init(sample_rows)
    rowp = g.init(shifted_rows)
    colmask = g.init(col_mask)
    cmask = g.init(color_mask)

    same = g.n(
        "Einsum",
        [g.input, g.input, row0, rowp, colmask, cmask],
        [6],
        "f32",
        equation="bchw,bciw,nh,pni,w,c->p",
    )
    both = g.n(
        "Einsum",
        [g.input, g.input, row0, rowp, colmask, cmask, cmask],
        [6],
        "f32",
        equation="bchw,bdiw,nh,pni,w,c,d->p",
    )
    valid = g.eq(same, both)
    periods = g.init(np.arange(4, 10, dtype=np.int32))
    period_or_99 = g.where(valid, periods, g.init(np.full((6,), 99, dtype=np.int32)))
    p_scalar = g.reduce(period_or_99, "Min", [0], keepdims=0)
    p_idx = g.sub(p_scalar, np.array(4, np.int32))
    rids = g.gather(g.init(residue_ids()), p_idx, axis=0)
    rids_col = g.reshape(rids, [30, 1])
    resid_eq = g.eq(rids_col, g.init(np.arange(9, dtype=np.uint8).reshape(1, 9)))
    u = g.cast(resid_eq, "f32")

    # For output color c and exact-column observed color d:
    # d=0 means the absolute column is hidden, so pass fallback through.
    # d=c selects the exact visible color; any other nonblack d suppresses it.
    gate = np.zeros((10, 10), np.float32)
    for c in range(1, 10):
        gate[c, 0] = 1.0
        gate[c, 1:] = -9.0
        gate[c, c] = 9.0
    gate_t = g.init(gate)

    g.n(
        "Einsum",
        [g.input, u, u, u, u, g.input, u, u, gate_t],
        [1, 10, 30, 30],
        "f32",
        is_output=True,
        equation="bchu,rk,hk,wl,ul,bdyw,rm,ym,cd->bcrw",
    )

    print(g.budget())
    g.save(out, ort_check=ort_check)


if __name__ == "__main__":
    build()
