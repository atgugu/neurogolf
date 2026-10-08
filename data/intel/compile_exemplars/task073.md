# task073 — compile exemplar

- **Mechanism:** rank-factor-color
- **True new cost (7431, SHA-verified rescorer):** **24** (params 24 + memory 0) -> **21.8219** pts
- **Delta vs old:** **+0.5108 pts** (old 7384 member: cost 40 -> 21.3111 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 24

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 321 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, input, input, B, B, B, B, K] -> [output]   <== TERMINAL (graph output)
      attr equation='njab,ncas,ndub,pj,qc,qd,po,pq->noab'

      **EINSUM EQUATION: `njab,ncas,ndub,pj,qc,qd,po,pq->noab`**


Initializers (2):
- `B`  float[2, 10]  (20 elems)
    [[0.0, -6.52067709, 0.0, 0.0, 0.0, 2.66942358, 0.0, 0.0, 0.0, 0.0],
       [-2.90184689, 3.81021261, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- `K`  float[2, 2]  (4 elems)
    [[0.00394402, 0.02058926],
       [-0.26591465, 0.03875801]]
```

## MECHANISM

Old: depthwise Conv (10,1,4,1), 40 els with only 6 nonzero — paying for zero-padded kernel slots.
New: single Einsum `njab,ncas,ndub,pj,qc,qd,po,pq->noab` with input fed THREE times — the cell
itself (`njab`), a same-row context (`ncas`), and a same-column context (`ndub`) — so the
contraction forms cell x row-mate x col-mate color products. All four color slots (`pj`,`qc`,`qd`,
`po`: input j, contexts c,d, output o) go through the SAME rank-2 selector B(2,10) (nonzeros only
on colors 0,1,5), coupled by the dense 2x2 mixer K. Effectively out-color x in-color x context
interaction = B^T-K-B, a low-rank factorization of what the sparse conv kernel spelled out
positionally. 24 els vs 40, both memory=0; delta is pure initializer element count (ln(40/24) =
+0.51).

## REUSABLE MOVE

A sparse cross-color kernel is cheaper low-rank: write the out-color x in-color (x context)
interaction as B^T-K-B with one rank-r B (r x 10) reused in EVERY color slot of the equation;
repeated B slots give row/column context products without new params.
