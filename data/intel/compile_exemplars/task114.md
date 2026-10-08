# task114 — compile exemplar

- **Mechanism:** compiled-terminal-einsum
- **True new cost (7431, SHA-verified rescorer):** **251** (params 251 + memory 0) -> **19.4745** pts
- **Delta vs old:** **+0.9607 pts** (old 7384 member: cost 656 -> 18.5138 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 251

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task114.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 1316 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, P, P, X, V, X, V, P, P, C, C, A, B] -> [output]   <== TERMINAL (graph output)
      attr equation='nkij,ai,bj,uxp,pa,vyq,qb,xr,yc,fk,go,tgf,tuv->norc'

      **EINSUM EQUATION: `nkij,ai,bj,uxp,pa,vyq,qb,xr,yc,fk,go,tgf,tuv->norc`**


Initializers (6):
- `A`  float[2, 3, 3]  (18 elems)
    shape [2, 3, 3] flat=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.89999998, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.89999998]
- `B`  float[2, 2, 2]  (8 elems)
    shape [2, 2, 2] flat=[5000.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 1.0]
- `X`  float[2, 5, 3]  (30 elems)
    shape [2, 5, 3] flat=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
- `V`  float[3, 5]  (15 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0, 0.0]]
- `C`  float[3, 10]  (30 elems)
    [[1.0, 0.809017, 0.30901697, -0.30901703, -0.809017, -1.0, -0.80901694, -0.30901712, 0.30901712, 0.80901724],
       [0.0, 0.58778524, 0.95105654, 0.95105648, 0.58778524, -9e-08, -0.5877853, -0.95105648, -0.95105648, -0.58778495],
       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
- `P`  float[5, 30]  (150 elems)
    min=0.0 max=1.0 nnz=5/150 first16=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
```

## MECHANISM

32 procedural nodes (Slice+Conv probes, 4 Split, 9 Where, 7 Concat building a 5x5 pattern, Equal +
Pad; 616 B charged) become ONE Einsum, `nkij,ai,bj,uxp,pa,vyq,qb,xr,yc,fk,go,tgf,tuv->norc`.
P(5,30), all one-hot rows, probes the input at 5 fixed coordinates per axis (fed 4x: probe rows,
probe cols, output rows, output cols — the same coordinate dictionary reused for reading AND
writing). The color-equality predicate is trigonometric: C(3,10) rows are exactly cos(2*pi*k/10),
sin(2*pi*k/10), and 1 (bytes: 1.0, 0.809017, 0.309017, ... / 0.0, 0.5878, 0.9511, ...), and
A(2,3,3) sheet 0 = diag(1, 1, -0.9): the contraction cos*cos + sin*sin - 0.9 is positive IFF the
two colors are equal — Equal compiled to a phase dot-product with margin 0.1; sheet 1 supplies the
constant-true term (1 - 0.9). B(2,2,2) = [5000, 50; 50, 0 | 0,0;0,1] mixes the branch outcomes with
a 5000-vs-50 integer dominance hierarchy so "all probes agree" outranks any partial pattern.
Memory 616 -> 0; params 40 -> 251 (net -405 cost).

## REUSABLE MOVE

Colors compare without Equal: embed color c as (cos 2*pi*c/10, sin 2*pi*c/10, 1) and contract two
embeddings through diag(1,1,-0.9) — positive iff equal, margin 0.1. Weight branch outcomes with a
5000/50 sentinel hierarchy so full agreement dominates partial matches inside one terminal einsum.
