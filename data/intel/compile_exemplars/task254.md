# task254 — compile exemplar

- **Mechanism:** crop-placement-attrs
- **True new cost (7431, SHA-verified rescorer):** **254** (params 166 + memory 88) -> **19.4627** pts
- **Delta vs old:** **+0.9473 pts** (old 7384 member: cost 655 -> 18.5154 pts)
- **Terminal op:** Einsum · **nodes:** 4 · **initializer elements:** 166

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task254.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 1314 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (4, topological order):
 0. Einsum       inputs=[input, kc, rowmap, x, x, x, x, l0, l1, l2, l3, x, x, x, x, l0, l1, l2, l3] -> [scores]
      attr equation='nchw,kc,mq,th,uh,vh,sh,qt,qu,qv,qs,aw,bw,dw,ew,ja,jb,jd,je->mj'

      **EINSUM EQUATION: `nchw,kc,mq,th,uh,vh,sh,qt,qu,qv,qs,aw,bw,dw,ew,ja,jb,jd,je->mj`**

 1. ReduceMax    inputs=[scores] -> [mx]
      attr axes=[1]
      attr keepdims=1
 2. Sub          inputs=[mx, scores] -> [colmask]
 3. Einsum       inputs=[input, colmask, colmask, x, x, x, x, l0, l1, l2, l3, kc, quad, po] -> [output]   <== TERMINAL (graph output)
      attr equation='nchw,aj,bj,tw,uw,vw,sw,jt,ju,jv,js,kc,abkp,po->nohw'

      **EINSUM EQUATION: `nchw,aj,bj,tw,uw,vw,sw,jt,ju,jv,js,kc,abkp,po->nohw`**


Initializers (9):
- `x`  float[2, 30]  (60 elems)
    [[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
- `l0`  float[5, 2]  (10 elems)
    [[1.0, -1.0],
       [-4.0, 0.0],
       [6.0, 0.0],
       [-4.0, 0.0],
       [1.0, 0.0]]
- `l1`  float[5, 2]  (10 elems)
    [[1.0, -2.0],
       [1.0, -2.0],
       [1.0, -1.0],
       [1.0, -1.0],
       [1.0, -1.0]]
- `l2`  float[5, 2]  (10 elems)
    [[1.0, -3.0],
       [1.0, -3.0],
       [1.0, -3.0],
       [1.0, -2.0],
       [1.0, -2.0]]
- `l3`  float[5, 2]  (10 elems)
    [[1.0, -4.0],
       [1.0, -4.0],
       [1.0, -4.0],
       [1.0, -4.0],
       [1.0, -3.0]]
- `kc`  float[2, 10]  (20 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
- `rowmap`  float[2, 5]  (10 elems)
    [[1.0, 1.0, 1.0, 1.0, 1.0],
       [-1.0, -1.0, -1.0, -1.0, 9.0]]
- `quad`  float[2, 2, 2, 2]  (16 elems)
    shape [2, 2, 2, 2] flat=[1.0, -1.0, 0.0, -1.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
- `po`  float[2, 10]  (20 elems)
    [[1.0, -100.0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
```

## MECHANISM

Bar-chart rule (find the tallest of 5 bars, repaint). Old: Einsum + 16 procedural nodes slicing a
9-row crop, Max/Min tall/short detection, Where/Concat/Pad paint — 584 B charged. New is 4 nodes:
Einsum #1 `nchw,kc,mq,th,uh,vh,sh,qt,qu,qv,qs,aw,bw,dw,ew,ja,jb,jd,je->mj` computes scores[m,j] per
bar j: x(2,30) row0 = [0,0,1,1,2,2,...] maps paired columns to 5 bar ids (a floor(col/2) radix
factor), and the l0..l3(5,2) stencils are pairwise height-comparison tables (l0 col0 =
[1,-4,6,-4,1], a finite-difference form). The ARGMAX-AS-MASK trick follows: ReduceMax - scores =
`colmask`, exactly 0 at the winning bar. Einsum #2 feeds colmask TWICE (`aj`,`bj`) through
quad(2,2,2,2), a truth-table tensor testing "is zero", routes colors with kc, and renders through
po(2,10) = [[1,-100,-100,...],[0,1,-1,...]] whose -100 integer sentinels crush the losing branches.
Memory 584 -> 88 B (the only charged state is scores/mx/colmask, 2x5 f32).

## REUSABLE MOVE

Argmax as a mask: colmask = max(scores) - scores is exactly 0 at the winner; test it downstream by
feeding colmask twice through a tiny truth-table tensor. Rank items with pairwise-difference
stencils instead of sorting, and crush losing branches with -100 sentinels in the output projector.
