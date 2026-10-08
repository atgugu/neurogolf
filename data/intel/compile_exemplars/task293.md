# task293 — compile exemplar

- **Mechanism:** relational/pairwise-input2
- **True new cost (7431, SHA-verified rescorer):** **40** (params 40 + memory 0) -> **21.3111** pts
- **Delta vs old:** **+3.2098 pts** (old 7384 member: cost 991 -> 18.1013 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 40

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task293.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 408 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, input, input, U, input, U, U, U, U] -> [output]   <== TERMINAL (graph output)
      attr equation='bpij,akuv,dmih,mx,enlj,ny,ks,zs,ps->bkij'

      **EINSUM EQUATION: `bpij,akuv,dmih,mx,enlj,ny,ks,zs,ps->bkij`**


Initializers (1):
- `U`  float[10, 4]  (40 elems)
    [[0.0, 0.0, 0.0, 100.0],
       [0.01263341, 4.08017731, -5.09281063, 0.0],
       [-12.53061485, -3.90510988, 15.43572426, 0.0],
       [-0.36675256, 2.6647613, -3.29800868, 0.0],
       [-0.23753047, 3.11517596, -3.87764549, 0.0],
       [-0.64214182, 1.90405726, -2.26191545, 0.0],
       [-0.47121352, 2.33622575, -2.86501217, 0.0],
       [-0.94227523, 1.5472362, -1.60496092, 0.0],
       [-0.84308553, 1.61506391, -1.77197838, 0.0],
       [0.32491425, 5.3984971, -6.72341156, 0.0]]
```

## MECHANISM

The whole 38-node old pipeline (Slice/ArgMin/GatherElements x5/GatherND/ScatterND/Where, 940 B of
charged intermediates) is compiled into ONE terminal Einsum. The free input is fed FIVE times under
different index roles (`bpij`, `akuv`, `dmih`, `enlj`, `ps`): the output cell (i,j) is contracted
against the colors present at its row-mates (`dmih` reads column h of row i) and column-mates
(`enlj` reads row l of column j), so the contraction computes products of grid values — pairwise
relational predicates — without ever materializing them. The single billed tensor is `U` (10,4),
fed FOUR times (`mx`,`ny`,`ks`,`zs`); its rows are per-color 3-vectors of decision-boundary
coefficients plus one dominant integer-exact sentinel `100.0` in the background row that makes the
background term dominate wherever no object logic fires. Per the forensic MD: "All computation
(index selection, logic, recolor, assembly) contracts inside the single terminal Einsum; no named
intermediates" — the only billed state is the reused 40-element factor. Grader decodes (raw > 0),
so only the sign pattern of the huge logits matters.

## REUSABLE MOVE

When the rule relates a cell to other cells in its row/column, feed the input into the same Einsum
several times with shifted index roles (`bpij`·`dmih`·`enlj`) and route every copy's color axis
through the same tiny factor matrix — all pairwise logic contracts inside the terminal node at
memory=0, and one sentinel row (100.0) handles the background default.
