# task296 — compile exemplar

- **Mechanism:** factor-geometry
- **True new cost (7431, SHA-verified rescorer):** **90** (params 90 + memory 0) -> **20.5002** pts
- **Delta vs old:** **+1.2496 pts** (old 7384 member: cost 314 -> 19.2506 pts)
- **Terminal op:** Conv · **nodes:** 1 · **initializer elements:** 90

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 622 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Conv         inputs=[input, W] -> [output]   <== TERMINAL (graph output)
      attr dilations=[1, 2]
      attr group=10
      attr kernel_shape=[3, 3]
      attr pads=[2, 4, 58, 116]
      attr strides=[3, 5]

Initializers (1):
- `W`  float[10, 1, 3, 3]  (90 elems)
    min=-1.0 max=1.0 nnz=45/90 first16=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, ...]
```

## MECHANISM

The entire 38-node procedural graph (2 projection Einsums + Div/Cast + 4 Slice crops + 12 Split +
7 Or + 4 Concat + Where/Equal/Pad, 249 B charged) is replaced by ONE Conv whose ATTRIBUTES encode
the geometry: group=10 (per-color independent kernels), kernel 3x3 with dilations=[1,2] (samples a
1x2-spaced lattice), strides=[3,5] (decimates the canvas into the rule's cell grid), and the giant
asymmetric pads=[2,4,58,116] which translate and place the strided result back onto the 30x30
canvas — crop, transform, and paste in a single op. W(10,1,3,3) is ternary {-1,0,+1}: checkerboard
and cross detectors per color channel (e.g. ch0 = [[1,-1,1],[-1,1,-1],[1,-1,1]]). Single node =>
memory=0 exactly; cost 314 -> 90 despite params rising 65 -> 90, because the 249 B of procedural
state vanished.

## REUSABLE MOVE

Conv attributes are free geometry parameters: dilation samples a lattice, stride decimates, and
huge asymmetric pads translate/place the result — a whole crop->transform->paste pipeline in one
params-only node with ternary {-1,0,+1} kernels, memory=0 because it is the terminal single node.
