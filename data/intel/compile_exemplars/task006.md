# task006 — compile exemplar

- **Mechanism:** conv-attr-geometry
- **True new cost (7431, SHA-verified rescorer):** **66** (params 30 + memory 36) -> **20.8103** pts
- **Delta vs old:** **+0.4155 pts** (old 7384 member: cost 100 -> 20.3948 pts)
- **Terminal op:** Conv · **nodes:** 2 · **initializer elements:** 30

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task006.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 465 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (2, topological order):
 0. Conv         inputs=[input, score_w] -> [score]
      attr dilations=[1, 4]
      attr kernel_shape=[1, 2]
      attr pads=[0, 0, -27, -23]
      attr strides=[1, 1]
 1. Conv         inputs=[score, out_w] -> [output]   <== TERMINAL (graph output)
      attr kernel_shape=[1, 1]
      attr pads=[0, 0, 27, 27]

Initializers (2):
- `score_w`  float[1, 10, 1, 2]  (20 elems)
    shape [1, 10, 1, 2] flat=[1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `out_w`  float[10, 1, 1, 1]  (10 elems)
    shape [10, 1, 1, 1] flat=[1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## MECHANISM

Two-panel overlap detector (left/right 3x3 panels around a separator column). Old: one grouped
dilated Conv straight to output, but paying a dense 100-el weight. New splits into score + decode:
Conv #1 (score_w(1,10,1,2), 20 els; per bytes ch0 taps [1,2], ch1 taps [-1,0]) with dilations=[1,4]
bridges the separator (samples col and col+4) and NEGATIVE pads [0,0,-27,-23] crop the result to
exactly the [1,1,3,3] score plane — 36 B, the only charged memory. Sign encodes the rule: score < 0
iff both panels are non-zero at that cell. Conv #2 (out_w(10,1,1,1) = [+1,0,-1,0,...]) is the
sign->palette decoder: ch0 += +score (background wins when positive), ch2 += -score (color 2 wins
when negative); pads=[0,0,27,27] place the 3x3 result at the origin of the free canvas. Verified
byte-identical (raw>0) outputs on all 266 examples. Cost 100 -> 66: -70 params for +36 B memory.

## REUSABLE MOVE

Split "detect" and "paint" into two Convs: the first uses dilation to span the geometry and
negative pads to emit only a tiny score plane (its sign IS the predicate); the second is a 1x1
sign->palette decoder (+1 on the background channel, -1 on the target channel) whose positive pads
place the result. Trading a fat weight for a 36 B score plane wins on log-cost.
