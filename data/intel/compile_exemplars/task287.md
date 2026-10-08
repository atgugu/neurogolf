# task287 — compile exemplar

- **Mechanism:** factor-dedup
- **True new cost (7431, SHA-verified rescorer):** **190** (params 190 + memory 0) -> **19.7530** pts
- **Delta vs old:** **+0.2744 pts** (old 7384 member: cost 250 -> 19.4785 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 190

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task287.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 1022 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, hi, hi, lo, lo, hi, hi, lo, lo, keep] -> [output]   <== TERMINAL (graph output)
      attr equation='nkhw,uh,ur,vh,vr,xw,xs,yw,ys,k->nkrs'

      **EINSUM EQUATION: `nkhw,uh,ur,vh,vr,xw,xs,yw,ys,k->nkrs`**


Initializers (3):
- `hi`  float[2, 30]  (60 elems)
    [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- `lo`  float[4, 30]  (120 elems)
    min=0.0 max=1.0 nnz=16/120 first16=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, ...]
- `keep`  float[10]  (10 elems)
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

## MECHANISM

Single-node terminal Einsum on both sides; the win deduplicates a monolithic position selector.
Old: sym_selector(30,8), 240 els, fed 4x. New equation `nkhw,uh,ur,vh,vr,xw,xs,yw,ys,k->nkrs`
factors the 30x30 position copy/reflection map as (coarse) x (fine): M[h,r] =
(sum_u hi[u,h]*hi[u,r]) * (sum_v lo[v,h]*lo[v,r]) with hi(2,30) marking coarse 4-blocks (positions
0-3 and 12-15 vs 4-11) and lo(4,30) marking fine offsets (16 scattered ones). Correction to the
MD's wiring note: the bytes show hi fed FOUR times and lo FOUR times — the identical (hi,hi,lo,lo)
pair is applied once to the (h,r) row axes and once to the (w,s) column axes, so the same factored
map is reused across both dimensions. keep(10) = [1,1,1,1,0,1,1,1,1,1] excludes color 4. All values
exact 0/1. 190 vs 250 els; the factorization stores 180 els where the dense selector stored 240.

## REUSABLE MOVE

A position permutation/copy matrix factors as coarse x fine indicators:
M[h,r] = sum_u hi[u,h]hi[u,r] * sum_v lo[v,h]lo[v,r]. Store hi(2x30)+lo(4x30) instead of the dense
map, and reuse the identical factor pair on the row axes AND the column axes.
