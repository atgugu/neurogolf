# task257 — compile exemplar

- **Mechanism:** trig/fourier-color
- **True new cost (7431, SHA-verified rescorer):** **114** (params 114 + memory 0) -> **20.2638** pts
- **Delta vs old:** **+1.2553 pts** (old 7384 member: cost 400 -> 19.0085 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 114

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 930 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, feat, feat, feat, feat, proj_a, feat, feat, proj_a, feat, feat, feat, feat, proj_a, feat, feat, proj_a, mask, mask, color, color, color, color, proj_a, color, color, proj_a] -> [output]   <== TERMINAL (graph output)
      attr equation='nihw,ph,pr,qh,sr,qs,th,ur,ut,vw,vc,jw,kc,jk,lw,mc,ml,r,c,ao,ai,bo,xi,bx,do,yi,yd->norc'

      **EINSUM EQUATION: `nihw,ph,pr,qh,sr,qs,th,ur,ut,vw,vc,jw,kc,jk,lw,mc,ml,r,c,ao,ai,bo,xi,bx,do,yi,yd->norc`**


Initializers (4):
- `feat`  float[2, 30]  (60 elems)
    [[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- `proj_a`  float[2, 2]  (4 elems)
    [[-1.0, -1.0],
       [1.0, -1.0]]
- `mask`  float[30]  (30 elems)
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `color`  float[2, 10]  (20 elems)
    [[0.29182529, 0.0, 0.0, 0.0, 1.6789062, 0.0, -0.76032484, -5.16125631, 0.47902262, 0.0],
       [0.00244655, 0.0, 0.0, 0.0, -4.65420914, 0.0, -0.66803825, 4.42171431, 1.13145173, 0.0]]
```

## MECHANISM

Dense Conv (10,10,2,2)=400 els with dilations=[5,5] replaced by one 27-operand terminal Einsum,
114 els. The equation `nihw,ph,pr,qh,sr,qs,th,ur,ut,vw,vc,jw,kc,jk,lw,mc,ml,r,c,ao,ai,bo,xi,bx,do,
yi,yd->norc` chains PAIRS of 2-dim slots through `proj_a`(2,2) = [[-1,-1],[1,-1]] — a scaled
rotation matrix (sqrt(2) x rotation by 135 deg), fed SIX times. Repeated application of the same
rotation implements relative-phase arithmetic (the cos/sin recurrence) between row features, column
features, and color features, which is how the periodic dilation-5 structure of the old kernel is
reproduced without position tables. `feat`(2,30) holds two +-1/0 pattern detectors on the first ~9
positions, `mask`(30) gates the leading 4 positions, and `color`(2,10) turns the final 2-dim phase
into per-color amplitudes (mixed-sign floats, e.g. -5.16/+4.42 on color 7). Both members are
single-node memory=0; the entire +1.255 delta is initializer compaction.

## REUSABLE MOVE

Encode 2-state/periodic structure as 2-vectors and ONE reused 2x2 rotation matrix: chaining the
same 4-element factor through multiple einsum slots computes relative phases (cos/sin recurrence),
replacing per-position trig or dilation tables; decode the final phase with a 2 x 10 color matrix.
