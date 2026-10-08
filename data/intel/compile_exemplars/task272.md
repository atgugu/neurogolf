# task272 — compile exemplar

- **Mechanism:** compiled-terminal-einsum
- **True new cost (7431, SHA-verified rescorer):** **148** (params 148 + memory 0) -> **20.0028** pts
- **Delta vs old:** **+0.9992 pts** (old 7384 member: cost 402 -> 19.0035 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 148

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task272.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 892 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, S, R, S, R, input, S, R, S, R, input, S, S, Q] -> [output]   <== TERMINAL (graph output)
      attr equation='njkc,pj,abk,ag,abr,nmrl,qm,del,dh,dec,nirc,xi,xo,xpq->norc'

      **EINSUM EQUATION: `njkc,pj,abk,ag,abr,nmrl,qm,del,dh,dec,nirc,xi,xo,xpq->norc`**


Initializers (3):
- `Q`  float[2, 2, 2]  (8 elems)
    shape [2, 2, 2] flat=[1.0, -0.5, -0.5, 0.25, -1.0, -1.0, -1.0, 1.25]
- `R`  float[2, 2, 30]  (120 elems)
    min=-0.75983566 max=0.65803701 nnz=20/120 first16=[-0.5, -0.5, -0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
- `S`  float[2, 10]  (20 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
```

## MECHANISM

Old: Slice (channels 0,2 + top-left 5x5 -> charged [1,2,5,5], 200 B) feeding a sparse Conv
(10,2,3,3 + bias, 202 els). New: ONE Einsum `njkc,pj,abk,ag,abr,nmrl,qm,del,dh,dec,nirc,xi,xo,
xpq->norc`, 148 els, memory=0. The Conv is factored into a separable tensor network: S(2,10)
(row0 = e0 color-0 selector, row1 = [0,1,-2,0,...] a signed color-1-minus-2*color-2 detector) is
fed FOUR times to select colors for three separate input reads (`njkc`,`nmrl`,`nirc` — the input
enters three times, providing the row-context x col-context x cell cross terms the 3x3 kernel used
to provide); R(2,2,30) holds separable row/column profiles (row [-0.5,-0.5,0,0.5,0.5,...] —
difference kernels on the 5-cell band, fed twice for rows and twice for cols); Q(2,2,2) =
[[1,-.5],[-.5,.25] | [-1,-1],[-1,1.25]] couples the two branches (its first sheet is the rank-1
outer square of (1,-0.5), the second a correction sheet). The old Slice's selection job became just
more factors — nothing upstream of the terminal node remains.

## REUSABLE MOVE

A small Conv factors into a separable network: color-select with S, row/col difference profiles
with R (feed the input multiple times for the cross terms), couple branches with a rank-2 Q — and
the Slice that fed the Conv disappears, because selection is just another factor in the terminal
einsum.
