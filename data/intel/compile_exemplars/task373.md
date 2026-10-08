# task373 — compile exemplar

- **Mechanism:** factor-reuse
- **True new cost (7431, SHA-verified rescorer):** **30** (params 30 + memory 0) -> **21.5988** pts
- **Delta vs old:** **+0.6931 pts** (old 7384 member: cost 60 -> 20.9057 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 30

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task373.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 334 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, input, parity, parity, parity] -> [output]   <== TERMINAL (graph output)
      attr equation='bchw,bdrv,h,r,s->bcrs'

      **EINSUM EQUATION: `bchw,bdrv,h,r,s->bcrs`**


Initializers (1):
- `parity`  float[30]  (30 elems)
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## MECHANISM

Exact params halving inside the single-node terminal-Einsum family. Old: `ncab,a,h,w->nchw` with
TWO sign vectors — row_sign=[1,-1,0...] and col_sign=[1,-1,1,-1,1,-1,0...] — 60 els total, heavily
zero-padded. New: `bchw,bdrv,h,r,s->bcrs` feeds the input TWICE (the second feed `bdrv` supplies
the row/column structure the deleted vector used to provide) and one `parity`(30) =
[1,-1,1,-1,1,-1,0,...] vector THREE times in roles h, r, s. parity is byte-identical to the old
col_sign prefix — the row_sign content was recoverable from it, so the duplicate vector was pure
waste. 30 els, memory=0, delta exactly ln(60/30) = +0.6931 = ln 2. Grader only tests sign, so the
changed raw support/scale is irrelevant.

## REUSABLE MOVE

Before paying for two selector vectors, check whether ONE serves all roles under index renaming:
feeding the same initializer in multiple slots (h,r,s) is free, and an extra feed of the input can
replace a whole stored factor. Halving a factor bank is worth exactly ln 2 = +0.69 pts.
