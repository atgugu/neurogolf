# task249 — compile exemplar

- **Mechanism:** arithmetic-replaces-table
- **True new cost (7431, SHA-verified rescorer):** **206** (params 23 + memory 183) -> **19.6721** pts
- **Delta vs old:** **+0.2366 pts** (old 7384 member: cost 261 -> 19.4355 pts)
- **Terminal op:** Gather · **nodes:** 8 · **initializer elements:** 23

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task249.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:14'], file size 581 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (8, topological order):
 0. Conv         inputs=[input, count_w, bias] -> [cnt]
      attr kernel_shape=[1, 1]
      attr pads=[0, -3, 0, -25]
      attr strides=[30, 1]
 1. ReduceSum    inputs=[cnt] -> [w_f]
      attr keepdims=0
 2. Cast         inputs=[w_f] -> [w8]
      attr to=2
 3. Sub          inputs=[c10, w8] -> [sub]
 4. Min          inputs=[c10, sub] -> [idx10]
 5. Pad          inputs=[idx10, pads, w8] -> [idx30_8]
      attr mode='constant'
 6. Cast         inputs=[idx30_8] -> [idx30]
      attr to=6
 7. Gather       inputs=[input, idx30] -> [output]   <== TERMINAL (graph output)
      attr axis=3

Initializers (4):
- `count_w`  float[1, 10, 1, 1]  (10 elems)
    shape [1, 10, 1, 1] flat=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- `bias`  float[1]  (1 elems)
    [1.5]
- `c10`  uint8[10]  (10 elems)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- `pads`  int64[2]  (2 elems)
    [0, 20]
```

## MECHANISM

Old stored the answer: position masks e0/m34 (60 els) + a 3x10 index table gathered by a computed
scalar. New SYNTHESIZES the gather indices from a measured width: a 1x1 Conv with strides=[30,1]
and NEGATIVE pads [0,-3,0,-25] probes just 2 cells of row 0 (count_w = all-ones over channels,
bias 1.5), ReduceSum -> scalar w, Cast to UINT8. Then the uint8 trick: sub = c10 - w8 UNDERFLOWS
for i < w (wrapping to ~256), so Min(c10, sub) = i for i < w and i-w for i >= w — i.e. idx = i mod
w over the active range, two elementwise ops and zero stored tables. Pad(value=w8) fills columns
10..29, and the terminal Gather(axis=3) applies the synthesized column permutation to the free
input. Params 93 -> 23 (only c10=[0..9], count_w, bias, pads); memory rises 15 B for the u8 temps —
net -55 cost.

## REUSABLE MOVE

Synthesize gather indices instead of storing them: measure the motif width w as a scalar, then
exploit uint8 underflow — Min(i, i-w) wraps for i < w — to get i mod w in two elementwise ops.
A stored index table becomes a handful of scalar generators.
