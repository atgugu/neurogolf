# task231 — compile exemplar

- **Mechanism:** arithmetic-replaces-table
- **True new cost (7431, SHA-verified rescorer):** **168** (params 12 + memory 156) -> **19.8760** pts
- **Delta vs old:** **+0.1592 pts** (old 7384 member: cost 197 -> 19.7168 pts)
- **Terminal op:** Gather · **nodes:** 14 · **initializer elements:** 12

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task231.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 861 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (14, topological order):
 0. Einsum       inputs=[input] -> [total]
      attr equation='bchw->'

      **EINSUM EQUATION: `bchw->`**

 1. Greater      inputs=[total, t30] -> [g30]
 2. Greater      inputs=[total, t35] -> [g35]
 3. Greater      inputs=[total, t40] -> [g40]
 4. Greater      inputs=[total, t45] -> [g45]
 5. Where        inputs=[g30, i1, i29] -> [d13]
 6. Where        inputs=[g35, i2, i29] -> [d14]
 7. Where        inputs=[g35, i3, i29] -> [d15]
 8. Where        inputs=[g40, i4, i29] -> [d16]
 9. Where        inputs=[g40, i5, i29] -> [d17]
10. Where        inputs=[g45, i0, i29] -> [d18]
11. Where        inputs=[g45, i1, i29] -> [d19]
12. Concat       inputs=[i0, i1, i2, i3, i4, i5, i0, i1, i2, i3, i4, i5, i6, d13, d14, d15, d16, d17, d18, d19, i29, i29, i29, i29, i29, i29, i29, i29, i29, i29] -> [idx]
      attr axis=0
13. Gather       inputs=[input, idx] -> [output]   <== TERMINAL (graph output)
      attr axis=3

Initializers (12):
- `i0`  int32[1]  (1 elems)
    [0]
- `i1`  int32[1]  (1 elems)
    [1]
- `i2`  int32[1]  (1 elems)
    [2]
- `i3`  int32[1]  (1 elems)
    [3]
- `i4`  int32[1]  (1 elems)
    [4]
- `i5`  int32[1]  (1 elems)
    [5]
- `i6`  int32[1]  (1 elems)
    [6]
- `i29`  int32[1]  (1 elems)
    [29]
- `t30`  float[1]  (1 elems)
    [30.0]
- `t35`  float[1]  (1 elems)
    [35.0]
- `t40`  float[1]  (1 elems)
    [40.0]
- `t45`  float[1]  (1 elems)
    [45.0]
```

## MECHANISM

Column-selection rule driven by the global filled-cell count. Old stored vectors: total_floor(7) +
phase_mid(7) + head_idx(13) + tail_idx(10) = 38 els feeding Less/Where/Concat. New moves the
decision TABLE into the graph TOPOLOGY: Einsum `bchw->` gives the scalar total; four Greater ops
against scalar thresholds t30/t35/t40/t45 create a staircase of bool gates; seven Where nodes each
select a single int32 index constant (i0..i5) or the fallback i29; one Concat of THIRTY scalar
wires (constants i0..i6 in a fixed prefix, the seven dynamic d13..d19, then ten i29) assembles the
gather index vector; terminal Gather(axis=3) applies it. Params 38 -> 12 (each i*/t* is a 1-element
init); the "index program" is expressed in wiring, which the scorer does not charge — memory stays
~156 B (dominated by the unavoidable 30xi32 idx vector).

## REUSABLE MOVE

An input-dependent index table can live in graph topology: threshold one scalar statistic with k
Greater ops, Where-in per-slot index constants, and Concat 30 scalar wires into the gather index —
wires and nodes are free, only initializer ELEMENTS are billed.
