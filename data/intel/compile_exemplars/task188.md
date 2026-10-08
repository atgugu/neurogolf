# task188 — compile exemplar

- **Mechanism:** boolean-selectors-sentinels
- **True new cost (7431, SHA-verified rescorer):** **113** (params 60 + memory 53) -> **20.2726** pts
- **Delta vs old:** **+1.8006 pts** (old 7384 member: cost 684 -> 18.4720 pts)
- **Terminal op:** Einsum · **nodes:** 16 · **initializer elements:** 60

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task188.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 1066 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (16, topological order):
 0. Einsum       inputs=[input] -> [area]
      attr equation='nkhw->n'

      **EINSUM EQUATION: `nkhw->n`**

 1. Einsum       inputs=[input, coef, coef] -> [w_s]
      attr equation='nkhw,pw,pw->n'

      **EINSUM EQUATION: `nkhw,pw,pw->n`**

 2. Einsum       inputs=[input, coef, coef] -> [h_s]
      attr equation='nkhw,ph,ph->n'

      **EINSUM EQUATION: `nkhw,ph,ph->n`**

 3. Greater      inputs=[w_s, h_s] -> [w_gt_h]
 4. Equal        inputs=[w_s, h_s] -> [is_square]
 5. Einsum       inputs=[input, input, coef] -> [v_match]
      attr equation='nkhw,nkzw,ph->n'

      **EINSUM EQUATION: `nkhw,nkzw,ph->n`**

 6. Less         inputs=[v_match, h_s] -> [h_content]
 7. And          inputs=[is_square, h_content] -> [square_h]
 8. Or           inputs=[w_gt_h, square_h] -> [is_h]
 9. Add          inputs=[h_s, h_s] -> [h2_s]
10. Add          inputs=[w_s, w_s] -> [w2_s]
11. Where        inputs=[is_h, h2_s, h_s] -> [row_dim_s]
12. Where        inputs=[is_h, w_s, w2_s] -> [col_dim_s]
13. Concat       inputs=[row_dim_s, area] -> [row_feat]
      attr axis=0
14. Concat       inputs=[col_dim_s, area] -> [col_feat]
      attr axis=0
15. Einsum       inputs=[input, row_feat, coef, col_feat, coef] -> [output]   <== TERMINAL (graph output)
      attr equation='nkhw,p,ph,q,qw->nkhw'

      **EINSUM EQUATION: `nkhw,p,ph,q,qw->nkhw`**


Initializers (1):
- `coef`  float[2, 30]  (60 elems)
    [[0.27939767, 0.48393095, 0.70710677, 0.68763489, -0.838193, -0.9266572, -1.00738263, -1.08210254, -1.15166926, -1.21692061, -1.27851737, -1.33700299, -1.39283264, -1.44639361, -1.49702013, -1.54600239, -1.59359372, -1.639979, -1.68532467, -1.72976422, -1.7734102, -1.81635654, -1.85868192, -1.90045285, -1.94172502, -1.98254454, -2.02295041, -2.06297588, -2.10264969, -2.14199686],
       [0.34919471, 0.60482299, -0.70710677, -0.96289057, -1.04758418, -1.15814781, -1.2590394, -1.35242534, -1.43973207, -1.52206779, -1.60016453, -1.6746304, -1.74592328, -1.8144083, -1.88040829, -1.94416952, -2.00590491, -2.0657959, -2.1239996, -2.18065, -2.23585367, -2.28973794, -2.34238291, -2.39387107, -2.44427347, -2.49365783, -2.54208326, -2.58960271, -2.63626695, -2.68211842]]
```

## MECHANISM

Repeat-collapse rule (detect 2x horizontal or vertical tiling, emit the base motif). Three tiny
Einsums produce SCALARS: `nkhw->n` (area), `nkhw,pw,pw->n` / `nkhw,ph,ph->n` (quadratic width/height
signatures through `coef` fed twice), `nkhw,nkzw,ph->n` (v_match, an input-squared vertical
self-similarity moment). The branchy part of the rule runs on those scalars as one-byte booleans:
Greater/Equal/Less/And/Or compute is_h (horizontal vs vertical repeat), Where picks h_s vs 2*h_s.
Concat packs [dim_s, area] into 2-element feats which enter the terminal Einsum
`nkhw,p,ph,q,qw->nkhw` as SELECTOR OPERANDS: each scalar multiplies a row of the single dual-purpose
basis `coef`(2,30), so the boolean branch outcome selects which coordinate profile scales the
output. Memory is only 53 B (scalars + bools); params one 60-el coef vs the old 223 els + 461 B.
MD: "Direction logic (is_h etc.) stays tiny (scalars + bools) before feeding the terminal
renderer."

## REUSABLE MOVE

Do the rule's if/else on scalars: reduce the grid to a few scalar moments with tiny einsums, run
the boolean logic with Greater/And/Or/Where (bools cost 1 byte), then Concat the scalars into a
short vector that enters the terminal einsum as a selector operand multiplying rows of one reused
coef basis.
