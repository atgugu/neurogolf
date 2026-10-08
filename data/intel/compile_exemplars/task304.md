# task304 — compile exemplar

- **Mechanism:** selector-palette-recolor
- **True new cost (7431, SHA-verified rescorer):** **300** (params 300 + memory 0) -> **19.2962** pts
- **Delta vs old:** **+0.6541 pts** (old 7384 member: cost 577 -> 18.6422 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 300

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task304.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 1593 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, mod, mod, input, input, center, center, sign, floor, floor, input, mod, mod, mod, mod, channel] -> [output]   <== TERMINAL (graph output)
      attr equation='bhmn,mu,nv,bhrs,bhde,tui,tvj,t,oi,pj,bqxy,xg,og,yl,pl,kq->bkop'

      **EINSUM EQUATION: `bhmn,mu,nv,bhrs,bhde,tui,tvj,t,oi,pj,bqxy,xg,og,yl,pl,kq->bkop`**


Initializers (5):
- `floor`  float[30, 3]  (90 elems)
    min=0.0 max=1.0 nnz=9/90 first16=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, ...]
- `mod`  float[30, 3]  (90 elems)
    min=0.0 max=1.0 nnz=9/90 first16=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ...]
- `channel`  float[10, 10]  (100 elems)
    min=-1.0 max=1.0 nnz=18/100 first16=[0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ...]
- `center`  float[2, 3, 3]  (18 elems)
    shape [2, 3, 3] flat=[3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- `sign`  float[2]  (2 elems)
    [1.0, -1.0]
```

## MECHANISM

Old 8-node graph: counts -> mode color (ReduceMax/Equal) -> crop gate -> Where(gate,+1,-1) ->
final Einsum; 175 B memory + 402 params. New folds ALL of it into one terminal Einsum
(`bhmn,mu,nv,bhrs,bhde,tui,tvj,t,oi,pj,bqxy,xg,og,yl,pl,kq->bkop`): input fed 4x (mode statistics,
gate evidence, and the rendered copy), `mod`/`floor` (30,3) are one-hot (position mod 3) and
floor(position/3) bases over the active 9 rows — the 3x3 block-coordinate factorization. The old
Where(gate, one, neg_one) branch is compiled as a 2-slot axis t contracted with sign=[1.0,-1.0]:
each branch owns a factor sheet (center[0]=3*I vs center[1]=all-ones, an 18-el pair) and the
input-derived gate decides which term dominates the sign. `channel`(10,10) with entries {0,-1,+1}
is the palette-recolor matrix — -1 entries clear the old color channel while +1 sets the new one
("Clearing the old channel matters as much as setting the new channel"). params 402->300,
memory -> 0.

## REUSABLE MOVE

A Where(cond,+1,-1) branch compiles into the einsum as a 2-slot axis contracted with
sign=[+1,-1], one factor sheet per branch. Palette recolor is a 10x10 {-1,0,+1} matrix on the
color axis: -1 to clear the source channel, +1 to set the target.
