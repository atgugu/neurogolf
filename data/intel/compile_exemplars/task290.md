# task290 — compile exemplar

- **Mechanism:** arithmetic-replaces-table
- **True new cost (7431, SHA-verified rescorer):** **283** (params 17 + memory 266) -> **19.3546** pts
- **Delta vs old:** **+0.2267 pts** (old 7384 member: cost 355 -> 19.1279 pts)
- **Terminal op:** Einsum · **nodes:** 17 · **initializer elements:** 17

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task290.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:12'], file size 868 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float16[1, 10, 30, 30]

Nodes (17, topological order):
 0. Einsum       inputs=[input, notbg] -> [cnb32]
      attr equation='bchw,c->bc'

      **EINSUM EQUATION: `bchw,c->bc`**

 1. Cast         inputs=[cnb32] -> [cnb]
      attr to=10
 2. ReduceSum    inputs=[cnb] -> [total]
      attr axes=[1]
      attr keepdims=1
 3. Sqrt         inputs=[total] -> [N]
 4. Shrink       inputs=[cnb] -> [d]
      attr bias=5.0
      attr lambd=0.0
 5. Sub          inputs=[N, c3] -> [a]
 6. Sub          inputs=[a, one] -> [b]
 7. Sub          inputs=[b, one] -> [c]
 8. Sub          inputs=[c, one] -> [e]
 9. Mul          inputs=[a, b] -> [ab]
10. Mul          inputs=[ab, c] -> [abc]
11. Mul          inputs=[c, e] -> [ce]
12. Concat       inputs=[z, ce, a, abc, z, z] -> [cen]
      attr axis=2
13. Concat       inputs=[one, one, one, a, ab, abc] -> [sq]
      attr axis=2
14. Concat       inputs=[cen, sq] -> [r6]
      attr axis=1
15. Conv         inputs=[r6, W] -> [r30]
      attr group=2
      attr kernel_shape=[1]
      attr pads=[0, 24]
16. Einsum       inputs=[d, tcoef, r30, r30] -> [output]   <== TERMINAL (graph output)
      attr equation='bk,t,bti,btj->bkij'

      **EINSUM EQUATION: `bk,t,bti,btj->bkij`**


Initializers (6):
- `notbg`  float[10]  (10 elems)
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- `tcoef`  float16[2]  (2 elems)
    [2.0, -1.0]
- `W`  float16[2, 1, 1]  (2 elems)
    shape [2, 1, 1] flat=[1.0, 1.0]
- `z`  float16[1, 1, 1]  (1 elems)
    shape [1, 1, 1] flat=[0.0]
- `one`  float16[1, 1, 1]  (1 elems)
    shape [1, 1, 1] flat=[1.0]
- `c3`  float16[1, 1, 1]  (1 elems)
    shape [1, 1, 1] flat=[3.0]
```

## MECHANISM

Old: measure counts -> TopK -> Equal against an outer_counts table [8,12,24,32] -> OneHot -> Einsum
against a stored spatial_table(4,2,6) f16 pattern LUT. New evaluates the pattern as a POLYNOMIAL of
the measured size: `bchw,c->bc` (notbg mask) counts non-background cells; ReduceSum + Sqrt gives
N = sqrt(total) (the object is an NxN frame family); Shrink(bias=5, lambd=0) turns per-color counts
into the color gate d = count-5. A Sub/Mul chain builds a=N-3, b=a-1, c=b-1, e=c-1 and products
ab, abc, ce — combinatorial expressions that ARE the old table rows. Concat assembles the 6-vector
pair [0,ce,a,abc,0,0] / [1,1,1,a,ab,abc]; Conv(group=2, kernel=1, pads=[0,24]) right-pads them into
r30[1,2,30]; the terminal Einsum `bk,t,bti,btj->bkij` takes the OUTER PRODUCT r30 x r30 weighted by
tcoef=[2,-1] and gated per color by d — squares/centered boxes emerge from polynomial roots. Params
63 -> 17.

## REUSABLE MOVE

Replace "measure size -> look up pattern" with "measure size -> evaluate polynomial": derive
N = sqrt(count), build (N-3), (N-3)(N-4), ... with a Sub/Mul chain, and let the terminal einsum
outer-product two synthesized coordinate vectors — the pattern table becomes arithmetic on one
scalar.
