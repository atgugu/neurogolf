# task078 — compile exemplar

- **Mechanism:** relational/pairwise-input2
- **True new cost (7431, SHA-verified rescorer):** **82** (params 82 + memory 0) -> **20.5933** pts
- **Delta vs old:** **+2.2266 pts** (old 7384 member: cost 760 -> 18.3667 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 82

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task078.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 581 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, input, W, W, W, W, W, H, Q] -> [output]   <== TERMINAL (graph output)
      attr equation='nirc,njsc,po,pi,pi,hj,qj,hq,qr->norc'

      **EINSUM EQUATION: `nirc,njsc,po,pi,pi,hj,qj,hq,qr->norc`**


Initializers (3):
- `W`  float[2, 10]  (20 elems)
    [[-4.0, -1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-2.0, 4.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- `H`  float[2, 1]  (2 elems)
    [[-1.07226861],
       [0.35944617]]
- `Q`  float[2, 30]  (60 elems)
    [[-0.00071798, -0.0004773, -0.00053141, -28.01127625, -18.57274818, 197.49641418, 294.27142334, 428.52374268, 0.00135854, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.00168604, 0.00120469, 0.00244284, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.00253711, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
```

## MECHANISM

Old member was a dense terminal Conv with a (10,5,15,1) weight (750 els). New feeds the input
TWICE (`nirc`, `njsc` — colors i and j in the SAME column c, at rows r and s), so the contraction
sees the product input[i,r,c]*input[j,s,c]: a pairwise same-column color predicate. The (i,j) pair
is classified by the rank-2 color matrix `W`(2,10) fed FIVE times (`po`,`pi`,`pi`,`hj`,`qj`) — the
doubled `pi` slot squares the projection — coupled through `H`(2,1) (`hq`) and rendered over rows
by `Q`(2,30) (`qr`). Q's values are the numeric vocabulary in miniature: integer-exact saturation
sentinels +-1000.0, epsilon coefficients ~0.0005-0.0025 that keep irrelevant rows sign-neutral, and
large row weights (197.5 / 294.3 / 428.5) that encode which row fires for which pair class.
Cost is pure params: 82 els vs 760, both graphs memory=0 single-node ("Re-use of the same
initializer multiple times charges only once").

## REUSABLE MOVE

Pairwise color predicate in one node: repeat the input as (`nirc`,`njsc`) sharing the column index
so the einsum multiplies same-column cells; classify the color pair with one rank-2 matrix reused
in every color slot, and render rows with a coefficient vector that uses +-1000 saturation
sentinels and ~1e-3 epsilon suppression for don't-care rows.
