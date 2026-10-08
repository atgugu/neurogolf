# task139 — compile exemplar

- **Mechanism:** compiled-terminal-einsum
- **True new cost (7431, SHA-verified rescorer):** **389** (params 373 + memory 16) -> **19.0364** pts
- **Delta vs old:** **+0.7396 pts** (old 7384 member: cost 815 -> 18.2968 pts)
- **Terminal op:** Einsum · **nodes:** 2 · **initializer elements:** 373

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task139.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 2098 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (2, topological order):
 0. GatherND     inputs=[input, xidx] -> [edge_vals]
      attr batch_dims=1
 1. Einsum       inputs=[input, edge_vals, l_chan, cat_src, src_vecs, out_vecs, edge2, row_sel2, row_masks, col_sel2, col_masks] -> [output]   <== TERMINAL (graph output)
      attr equation='narc,nxq,lk,ks,sa,ko,bq,blu,ur,blv,vc->norc'

      **EINSUM EQUATION: `narc,nxq,lk,ks,sa,ko,bq,blu,ur,blv,vc->norc`**


Initializers (10):
- `xidx`  int64[1, 1, 4, 3]  (12 elems)
    shape [1, 1, 4, 3] flat=[4, 1, 0, 4, 2, 0, 4, 3, 0, 0, 0, 0]
- `l_chan`  float[3, 3]  (9 elems)
    [[1.0, 1.0, 0.0],
       [0.0, 0.0, 1.0],
       [0.0, 0.0, 1.0]]
- `cat_src`  float[3, 2]  (6 elems)
    [[1.0, 0.0],
       [0.0, 1.0],
       [1.0, 0.0]]
- `src_vecs`  float[2, 10]  (20 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- `out_vecs`  float[3, 10]  (30 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
- `edge2`  float[2, 4]  (8 elems)
    [[1.0, 1.0, 1.0, 0.0],
       [-1.0, -1.0, -1.0, 1.0]]
- `row_sel2`  float[2, 3, 4]  (24 elems)
    shape [2, 3, 4] flat=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
- `row_masks`  float[4, 30]  (120 elems)
    min=0.0 max=1.0 nnz=39/120 first16=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...]
- `col_sel2`  float[2, 3, 4]  (24 elems)
    shape [2, 3, 4] flat=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
- `col_masks`  float[4, 30]  (120 elems)
    min=0.0 max=1.0 nnz=39/120 first16=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...]
```

## MECHANISM

"Read the legend, paint the regions" in 2 nodes. GatherND(batch_dims=1) with xidx(1,1,4,3) =
rows (4,1,0),(4,2,0),(4,3,0),(0,0,0) pulls exactly the legend cells — channel 4, rows 1..3 of
column 0 — into a tiny edge_vals vector (the ONLY charged tensor besides its 16 B). The terminal
Einsum `narc,nxq,lk,ks,sa,ko,bq,blu,ur,blv,vc->norc` treats edge_vals as a ROUTING OPERAND:
edge2(2,4) = [[1,1,1,0],[-1,-1,-1,1]] classifies the legend's presence pattern into branch b;
row_masks/col_masks(4,30) are reusable 0/1 region bands selected per (branch, region) by
row_sel2/col_sel2(2,3,4); l_chan/cat_src/src_vecs/out_vecs route source colors to output colors,
with out_vecs row 2 = [-10,0,...,0,+1(ch7),...] using a -10 suppression sentinel to clear
background where color 7 is written. Old paid 765 B building region features procedurally
(Slice/ReduceMax/Mul/BitwiseAnd); new pays 16 B and 373 params (net -426).

## REUSABLE MOVE

GatherND is a cheap "read the legend" op: pull the few key cells as a small vector and feed it into
the terminal einsum as a routing operand; keep region geometry in reusable 0/1 row/col mask banks
selected by tiny per-branch selector tensors, with negative sentinels clearing overwritten
channels.
