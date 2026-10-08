#!/usr/bin/env python3
"""Task202: pin sentinel u8profile + direct Cauchy Equal (no Sub)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G

OUT = Path(__file__).with_name("task202.onnx")

g = G(task=202)

colors = g.init(np.arange(10, dtype=np.float32))
colors_sq = g.init(np.arange(10, dtype=np.float32) ** 2)
nonzero = g.init(np.array([0.0] + [1.0] * 9, np.float32))
e0 = g.init(np.array([1.0] + [0.0] * 9, np.float32))
e0_chan = g.init(np.array([1.0] + [0.0] * 9, np.float32).reshape(1, 10, 1, 1))
zero = g.init(np.array([0.0], np.float32))
sent = g.init(np.array([-1.0], np.float32))

num_r = g.einsum("nkhw,k->hn", [g.input, colors], [30, 1])
den_r = g.einsum("nkhw,k->hn", [g.input, nonzero], [30, 1])
num_c = g.einsum("nkhw,k->nw", [g.input, colors], [1, 30])
den_c = g.einsum("nkhw,k->nw", [g.input, nonzero], [1, 30])

rc_raw = g.n("Div", [num_r, den_r], [30, 1], "f32")
cc_raw = g.n("Div", [num_c, den_c], [1, 30], "f32")

rvalid = g.greater(den_r, zero)
cvalid = g.greater(den_c, zero)

rowcolor = g.where(rvalid, rc_raw, sent)
colcolor = g.where(cvalid, cc_raw, sent)

markbandH = g.einsum("nkhw,k,hn->nw", [g.input, e0, rowcolor], [1, 30])
markbandV = g.einsum("nkhw,k,nw->hn", [g.input, e0, colcolor], [30, 1])

cauchy_a = g.einsum("nkhw,k,hn->n", [g.input, colors_sq, den_r], [1])
cauchy_b = g.einsum("hn,hn->n", [num_r, num_r], [1])
use_horiz = g.eq(cauchy_a, cauchy_b)

rowcolor_u8 = g.cast(rowcolor, "u8")
colcolor_u8 = g.cast(colcolor, "u8")
markbandH_u8 = g.cast(markbandH, "u8")
markbandV_u8 = g.cast(markbandV, "u8")

A = g.where(use_horiz, rowcolor_u8, markbandV_u8)
B = g.where(use_horiz, markbandH_u8, colcolor_u8)

O = g.eq(A, B)
g.n("Where", [O, e0_chan, g.input], [1, 10, 30, 30], "f32", is_output=True)

print(g.budget())
g.save(OUT)