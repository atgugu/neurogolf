# task072 — compile exemplar

- **Mechanism:** conv-attr-geometry
- **True new cost (7431, SHA-verified rescorer):** **180** (params 30 + memory 150) -> **19.8070** pts
- **Delta vs old:** **+0.6001 pts** (old 7384 member: cost 328 -> 19.2070 pts)
- **Terminal op:** ConvInteger · **nodes:** 3 · **initializer elements:** 30

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task072.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 423 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: int32[1, 10, 30, 30]

Nodes (3, topological order):
 0. Conv         inputs=[input, diff_w] -> [delta]
      attr dilations=[7, 1]
      attr pads=[0, 0, -17, -25]
 1. Cast         inputs=[delta] -> [delta_i8]
      attr to=3
 2. ConvInteger  inputs=[delta_i8, paint_w] -> [output]   <== TERMINAL (graph output)
      attr pads=[0, 0, 24, 25]

Initializers (2):
- `diff_w`  float[1, 10, 2, 1]  (20 elems)
    shape [1, 10, 2, 1] flat=[0.0, 200.0, 0.0, 0.0, -128.0, 72.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `paint_w`  int8[10, 1, 1, 1]  (10 elems)
    shape [10, 1, 1, 1] flat=[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

## MECHANISM

Two-panel XOR rule. Old materialized both panels (2 Slices -> fp32 copies, 120 B each) then
Equal/Where -> ConvInteger paint (300 B total memory). New computes the panel comparison IN a Conv:
dilations=[7,1] makes the 2-tap vertical kernel compare each cell with its partner 7 rows away
(exactly the panel separation), with integer weights 200 (color-0 bottom tap) / -128, +72 (color-2
top/bottom taps) chosen so every (top,bottom) color combination lands on a distinct signed value;
NEGATIVE pads [0,0,-17,-25] simultaneously crop the result to the 6x5 panel area — compute and crop
in one attr set. Cast to int8 compacts the state (30 B vs 120 B fp32), and the terminal ConvInteger
with pads=[0,0,24,25] places the painted panel on the free canvas via paint_w = [-1,0,0,+1,0,...]
(clear ch0, set ch3). Memory halves 300 -> 150 B; the delta is entirely the killed duplicate
fp32 panel copy.

## REUSABLE MOVE

Geometry via Conv attrs: dilation = panel offset turns a 2-tap kernel into a cross-panel
comparator; negative pads crop the computed state, positive pads on the NEXT conv place it. Cast
the compact state to int8 to halve its memory charge before the ConvInteger paint.
