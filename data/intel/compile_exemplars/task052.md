# task052 — compile exemplar

- **Mechanism:** selector-render
- **True new cost (7431, SHA-verified rescorer):** **30** (params 30 + memory 0) -> **21.5988** pts
- **Delta vs old:** **+1.8667 pts** (old 7384 member: cost 194 -> 19.7321 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 30

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task052.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 355 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, code, input, code, code, input] -> [output]   <== TERMINAL (graph output)
      attr equation='bkhi,sk,blhj,sl,so,bmhc->bohc'

      **EINSUM EQUATION: `bkhi,sk,blhj,sl,so,bmhc->bohc`**


Initializers (1):
- `code`  float[3, 10]  (30 elems)
    [[0.75727922, -1.0, -1.0, -1.0, -1.0, -0.75727922, -1.0, -1.0, -1.0, -1.0],
       [-1.0, 0.89999998, 0.63639611, 0.0, -0.63639611, 1.0, -0.89999998, -0.63639611, -0.0, 0.63639611],
       [-1.0, 0.0, 0.63639611, 0.89999998, 0.63639611, 1.0, 0.0, -0.63639611, -0.89999998, -0.63639611]]
```

## MECHANISM

Six-op graph (Einsum + Greater/Equal thresholds + 2 Concat + Pad, 104 params + ~90 B memory)
becomes one terminal Einsum with a single 30-element `code`(3,10). Algebra (from the MD, verified
against the wiring): with S[k,h] = row-h histogram of color k, U[s,h] = sum_k code[s,k]*S[k,h], and
T[h,c] = sum_m input[m,h,c] (cell-occupancy mask), the equation `bkhi,sk,blhj,sl,so,bmhc->bohc`
computes output[o,h,c] = T[h,c] * sum_s code[s,o] * U[s,h]^2. The paired identical feeds
(`bkhi`,`sk`) and (`blhj`,`sl`) square the projected histogram INSIDE the op — quadratic features
replace the old Greater(E,7)/Equal(E,5) thresholds. code's constants 0 / +-1 / 0.9 / +-0.6363961
(= 0.9/sqrt(2)) / 0.7572792 are tuned so the sign of the quadratic form matches the required output
channel. The third input feed `bmhc` acts as the occupancy mask T so output support follows the
input exactly — the old Concat x3 + Pad spatial construction disappears.

## REUSABLE MOVE

Threshold tests compile to squared projections: feed the input twice with identical row roles so
the einsum squares a histogram projection, pick code values so the quadratic form's SIGN encodes
the class, and multiply by a third input feed (sum over colors) as a free occupancy mask instead of
Pad/Concat rendering.
