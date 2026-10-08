# task001 — compile exemplar

- **Mechanism:** mode-dim-packing
- **True new cost (7431, SHA-verified rescorer):** **190** (params 190 + memory 0) -> **19.7530** pts
- **Delta vs old:** **+0.1911 pts** (old 7384 member: cost 230 -> 19.5619 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 190

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task001.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 977 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, input, v, v, m, m, m, m] -> [output]   <== TERMINAL (graph output)
      attr equation='ncab,ndpq,c,d,uri,vpi,usj,wqj->ncrs'

      **EINSUM EQUATION: `ncab,ndpq,c,d,uri,vpi,usj,wqj->ncrs`**


Initializers (2):
- `v`  float[10]  (10 elems)
    [2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
- `m`  float[2, 30, 3]  (180 elems)
    min=-0.70710677 max=1.41421354 nnz=27/180 first16=[1.41421354, 0.0, 0.0, 0.70710677, 0.70710677, 0.0, 0.70710677, 0.0, 0.70710677, 0.70710677, 0.70710677, 0.0, 0.0, 1.41421354, 0.0, 0.0, ...]
```

## MECHANISM

Both old and new are single-node terminal Einsums (memory=0); the +0.19 is pure factor packing.
Old paid 230 els across FIVE tensors: fine_sel(30,3) + coarse_sel(30,3) (both zero after row 9) +
fine_factor(10,2) + coarse_factor(10,2) + channel_sign(10). New packs the two selector banks into
ONE tensor m(2,30,3) — the leading dim is a MODE axis — plus v(10)=[2,-1,...,-1]. The equation
`ncab,ndpq,c,d,uri,vpi,usj,wqj->ncrs` feeds m four times, binding a DIFFERENT letter (u,v,u,w) to
the mode dim per slot, so the contraction mixes mode sheets pairwise without any new parameters;
v is fed twice for the channel-sign quadratic. m's sheets hold sqrt(2)=1.41421354 / 0.70710677
scaled selectors and signed difference rows (27 nonzero of 180). MD: "the leading dimension ...
subsume[s] what previously required separate fine/coarse 30x3 + 10x2 matrices" — 40 els saved,
ln(230/190) = +0.191.

## REUSABLE MOVE

Merge sibling factor matrices into one tensor with an extra leading mode dimension and bind a
different equation letter to that dim in each slot (uri, vpi, usj, wqj): shared content is stored
once and the EQUATION routes which sheet each slot sees — packing costs zero extra ops.
