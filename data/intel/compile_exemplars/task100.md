# task100 — compile exemplar

- **Mechanism:** selector-palette-recolor
- **True new cost (7431, SHA-verified rescorer):** **183** (params 9 + memory 174) -> **19.7905** pts
- **Delta vs old:** **+0.6738 pts** (old 7384 member: cost 359 -> 19.1167 pts)
- **Terminal op:** Pad · **nodes:** 7 · **initializer elements:** 9

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task100.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:18'], file size 615 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: bool[1, 10, 30, 30]

Nodes (7, topological order):
 0. Einsum       inputs=[input, p] -> [count]
      attr equation='nchw,ab->ncab'

      **EINSUM EQUATION: `nchw,ab->ncab`**

 1. Einsum       inputs=[input, input, input, input, p, p] -> [four]
      attr equation='nchw,nchv,ncuw,ncuv,ab,ab->ncab'

      **EINSUM EQUATION: `nchw,nchv,ncuw,ncuv,ab,ab->ncab`**

 2. Sub          inputs=[count, four] -> [score]
 3. ReduceMax    inputs=[score] -> [mx]
      attr keepdims=1
 4. Equal        inputs=[score, mx] -> [sel]
 5. Expand       inputs=[sel, shape2] -> [block]
 6. Pad          inputs=[block, pads, , pad_axes] -> [output]   <== TERMINAL (graph output)
      attr mode='constant'

Initializers (4):
- `p`  float[1, 1]  (1 elems)
    [[0.02083333]]
- `pads`  int64[4]  (4 elems)
    [0, 0, 28, 28]
- `pad_axes`  int64[2]  (2 elems)
    [2, 3]
- `shape2`  int64[2]  (2 elems)
    [2, 2]
```

## MECHANISM

Winner-take-all 2x2 block painter in 7 nodes, 9 params. Two moment Einsums score each color:
`nchw,ab->ncab` (count, broadcast to 2x2 by the scalar p=1/48 = 0.02083333) and the input-to-the-
4th rectangle moment `nchw,nchv,ncuw,ncuv,ab,ab->ncab` — four input feeds sharing indices so the
contraction computes sum over h,u,w,v of x[h,w]x[h,v]x[u,w]x[u,v] = the squared Gram matrix, i.e. a
rectangle-corner count, with p applied twice (1/48^2). score = count - four separates the target
color. Selection is ArgMax-free: ReduceMax(score) + Equal(score, mx) yields the winning channel
directly as a bool one-hot `sel` (the old graph needed Cast/ArgMax/Equal-vs-chan-ramp, 41 params).
Expand to [.,.,2,2] and Pad([0,0,28,28]) places the 2x2 block at the origin of the free BOOL
output. Only 174 B of charged intermediates; params 41 -> 9.

## REUSABLE MOVE

ArgMax-free winner selection: compute per-channel scalar scores with input-power moment einsums
(x^4 rectangle moments distinguish shapes from counts), then ReduceMax + Equal produces the winning
channel as a bool one-hot with no ArgMax/OneHot/index machinery; Expand+Pad renders the constant
block on the free output.
