# task362 — compile exemplar

- **Mechanism:** dtype+contraction
- **True new cost (7431, SHA-verified rescorer):** **121** (params 81 + memory 40) -> **20.2042** pts
- **Delta vs old:** **+1.4047 pts** (old 7384 member: cost 493 -> 18.7995 pts)
- **Terminal op:** Einsum · **nodes:** 9 · **initializer elements:** 81

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task362.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 1160 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (9, topological order):
 0. Einsum       inputs=[input, scale, base] -> [weighted_count]
      attr equation='bkpq,k,xy->bx'

      **EINSUM EQUATION: `bkpq,k,xy->bx`**

 1. Gemm         inputs=[weighted_count, base, base] -> [shift]
      attr alpha=1000.0
      attr beta=-18919.0
 2. Einsum       inputs=[input, scale, basis] -> [row_moment]
      attr equation='bkpq,k,ap->b'

      **EINSUM EQUATION: `bkpq,k,ap->b`**

 3. Einsum       inputs=[input, scale, basis] -> [col_moment]
      attr equation='bkpq,k,aq->b'

      **EINSUM EQUATION: `bkpq,k,aq->b`**

 4. Sum          inputs=[row_moment, shift] -> [row_target_f]
 5. Sub          inputs=[col_moment, shift] -> [col_target_f]
 6. Concat       inputs=[row_target_f, base] -> [row_base]
      attr axis=1
 7. Concat       inputs=[col_target_f, base] -> [col_base]
      attr axis=1
 8. Einsum       inputs=[input, row_base, row_base, col_base, col_base, basis, basis, basis, basis, Q, Q, Q, Q, mix, scale] -> [output]   <== TERMINAL (graph output)
      attr equation='bopq,ba,bd,be,bh,ur,vr,wc,xc,aus,dvs,ews,hxs,s,o->borc'

      **EINSUM EQUATION: `bopq,ba,bd,be,bh,ur,vr,wc,xc,aus,dvs,ews,hxs,s,o->borc`**


Initializers (5):
- `scale`  float[10]  (10 elems)
    [-0.001, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
- `base`  float[1, 1]  (1 elems)
    [[1.0]]
- `basis`  float[2, 30]  (60 elems)
    [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-1.26105261, -1.14994156, -1.0388304, -0.9277193, -0.81660819, -0.70549703, -0.59438598, -0.48327482, -0.37216377, -0.26105261, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- `Q`  float[2, 2, 2]  (8 elems)
    shape [2, 2, 2] flat=[-0.11111111, 0.0, 0.0, 0.0, 1.26105261, 1.0, 1.0, 0.0]
- `mix`  float[2]  (2 elems)
    [-1000.0, 0.1]
```

## MECHANISM

Old version punched zeros into fp16 term tables with ScatterElements at COMPUTED indices (crashed
on index-OOB for most inputs) and emitted fp16 output. New replaces all index math with continuous
integer-exact fp32 algebra: `bkpq,k,xy->bx` computes a weighted cell count with
scale=[-0.001, 1,1,1,1, 0, 1,1,1,1] — color 5 (gray) excluded, color 0 weighted by the -0.001
epsilon so its count perturbs nothing at integer scale; Gemm with alpha=1000.0, beta=-18919.0
(bytes; the MD omits the constants) affine-maps the count into the row/col shift; Sum/Sub apply it
to coordinate moments (`bkpq,k,ap->b` over basis=[ones; -1.26105+0.11111*k ramp]); Concat with
base=1.0 forms homogeneous 2-vectors row_base/col_base. The terminal Einsum feeds row_base/col_base
TWICE each (quadratic position forms), basis 4x, Q(2,2,2) 4x, and mix=[-1000.0, 0.1] — the -1000
sentinel suppresses the non-selected branch. Output normalized to fp32. Memory 368->40 B, params
125->81; robust where the old graph raised runtime errors.

## REUSABLE MOVE

Replace computed-index Scatter writes with position algebra: encode the target coordinate as a
scalar moment, affine-map it with an exact-constant Gemm (alpha/beta integers), and evaluate
position-selective quadratics (coordinate ramp basis fed twice + a -1000 branch sentinel) inside
the terminal einsum — no int casts, no out-of-bounds risk, fp32-exact.
