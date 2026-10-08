# task321 — compile exemplar

- **Mechanism:** radix-position-factors
- **True new cost (7431, SHA-verified rescorer):** **180** (params 180 + memory 0) -> **19.8070** pts
- **Delta vs old:** **+0.5108 pts** (old 7384 member: cost 300 -> 19.2962 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 180

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 952 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, Q, L, Q, L, C] -> [output]   <== TERMINAL (graph output)
      attr equation='nqrs,fq,as,fk,ac,c->nkrc'

      **EINSUM EQUATION: `nqrs,fq,as,fk,ac,c->nkrc`**


Initializers (3):
- `Q`  float[3, 10]  (30 elems)
    [[2.57735372, -13.04148483, 0.0, 0.0, -1.32392859, 0.0, 0.0, 0.0, 0.0, -3.2888],
       [-0.65563452, -10.1550703, 0.0, 0.0, 10.93655491, 0.0, 0.0, 0.0, 0.0, 35.35276794],
       [0.18396705, 1.94094813, 0.0, 0.0, -87.05265045, 0.0, 0.0, 0.0, 0.0, 27.49111938]]
- `L`  float[4, 30]  (120 elems)
    min=0.0 max=1.0 nnz=12/120 first16=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
- `C`  float[30]  (30 elems)
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## MECHANISM

Old: dilated Conv, kernel 1x3 with dilation_w=5, weights (10,10,1,3)=300 els, only 9 nonzero — the
dilation lattice was paid for as dense weight. New: `nqrs,fq,as,fk,ac,c->nkrc` with Q(3,10) fed
twice (input color q -> latent f, latent f -> output color k: a rank-3 color channel map) and
L(4,30) fed twice (`as`,`ac`). L is the radix factor: L[a,s] = 1 exactly at s in {a, a+5, a+10} —
a stride-5 comb per phase a (bytes: row0 hits 0,5,10; row1 hits 1,6,11; ...). Contracting the
source column s through L extracts the phase-a component; contracting back out through the same L
targets column c of the same phase, and C(30) (=1 on columns 0..3) gates writes to the leading
period. The Conv's dilation geometry became a reused 0/1 position factor: 180 els vs 300, memory=0
on both sides.

## REUSABLE MOVE

Periodic column structure (period k) compiles to a comb factor L[a,s]=1 iff s = a (mod k) over the
active band: contract source position through L to a phase index, and back out through the SAME L
(plus a gate vector) to the target — a dilated Conv's lattice as one reused radix matrix.
