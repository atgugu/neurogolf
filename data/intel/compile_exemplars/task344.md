# task344 — compile exemplar

- **Mechanism:** factor-geometry
- **True new cost (7431, SHA-verified rescorer):** **692** (params 692 + memory 0) -> **18.4604** pts
- **Delta vs old:** **+0.2739 pts** (old 7384 member: cost 910 -> 18.1866 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 692

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 3059 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, E, E, M, M, E, E, U, G, A] -> [output]   <== TERMINAL (graph output)
      attr equation='bnij,iv,jw,pvl,qwl,rp,sq,ok,nk,kl->bors'

      **EINSUM EQUATION: `bnij,iv,jw,pvl,qwl,rp,sq,ok,nk,kl->bors`**


Initializers (5):
- `E`  float[30, 10]  (300 elems)
    min=0.0 max=1.0 nnz=10/300 first16=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ...]
- `M`  float[10, 10, 3]  (300 elems)
    min=0.0 max=12.96333122 nnz=108/300 first16=[12.92554474, 12.96333122, 0.89023536, 0.01313966, 0.00825038, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
- `U`  float[10, 4]  (40 elems)
    [[2.60233426, 0.3496893, 1.63178647, 2.2867384],
       [0.22243252, -0.01146791, 0.88471901, 0.10089929],
       [0.73637331, -0.3920607, 1.31720138, -2.4697206],
       [-0.75251317, -2.17426443, 1.00528967, -0.03971795],
       [0.22243252, -0.01146791, 0.88471901, 0.10089929],
       [-1.63852775, -0.73398542, 0.18106902, -1.42346978],
       [0.22243252, -0.01146791, 0.88471901, 0.10089929],
       [0.22243252, -0.01146791, 0.88471901, 0.10089929],
       [-1.03088093, 1.80808914, 2.69373202, 0.64247495],
       [0.22243252, -0.01146791, 0.88471901, 0.10089929]]
- `G`  float[10, 4]  (40 elems)
    [[1.39492083, -0.00052947, -1.54348099, -0.00934829],
       [-0.0, 0.0, 0.0, -0.0],
       [1.45217431, -2.08341169, -1.54316711, 0.83559608],
       [-2.37793636, 0.30164438, -1.54358077, -3.20677423],
       [0.0, 0.0, -0.0, 0.0],
       [-1.12028909, -0.00258942, -1.54322648, 0.01812345],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, -0.0, -0.0, 0.0],
       [0.0, 0.0, -0.0, 0.0],
       [-0.0, 0.0, 0.0, -0.0]]
- `A`  float[4, 3]  (12 elems)
    [[-0.06122389, 0.0824578, -0.00011992],
       [-12.10373497, 12.02384472, -0.00549179],
       [-0.00791715, 0.00933577, 4.62605476],
       [-4.52477789, 4.47963524, -0.00110521]]
```

## MECHANISM

Params-only upgrade of a full 3x3 channel-mixing Conv (900+10 els, dense float weights, flagged in
intel section 4 for zero-margin fp accumulation) to a single terminal Einsum
`bnij,iv,jw,pvl,qwl,rp,sq,ok,nk,kl->bors` at 692 els. Geometry is factored ONCE and reused across
axes: E(30,10) — a 0/1 embedding of the first 10 coordinates (nnz 10/300) — appears FOUR times
(`iv`,`jw` on input coords, `rp`,`sq` on output coords), and the geometry core M(10,10,3) appears
TWICE (`pvl` for rows, `qwl` for columns), sharing one latent l coupled by the tiny A(4,3). Colors
run through U/G(10,4) (`nk`,`ok`) with several duplicate rows (a shared default row repeated for
5 colors — dedup inside the factor). A carries near-integer +-12 values with ~1e-3 corrections;
E is exactly binary — the integer-exact re-encoding that fixes the old numeric-margin exposure.

## REUSABLE MOVE

When the same geometric relation applies to rows and columns, pay for it once: put one factor
M[p,v,l] in both the row slot (pvl) and column slot (qwl) and couple them through a tiny latent
matrix; use exact 0/1 coordinate embeddings so decision margins stay integer-exact.
