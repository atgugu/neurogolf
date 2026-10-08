# task205 — compile exemplar

- **Mechanism:** counting-histogram
- **True new cost (7431, SHA-verified rescorer):** **2084** (params 34 + memory 2050) -> **17.3580** pts
- **Delta vs old:** **+0.5453 pts** (old 7384 member: cost 3595 -> 16.8127 pts)
- **Terminal op:** Einsum · **nodes:** 73 · **initializer elements:** 34

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task205.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:18'], file size 3604 bytes, ir_version 9

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float16[1, 10, 30, 30]

Nodes (73, topological order):
 0. Einsum       inputs=[input] -> [counts]
      attr equation='bkrc->bk'

      **EINSUM EQUATION: `bkrc->bk`**

 1. ArgMax       inputs=[counts] -> [Bi64]
      attr axis=1
      attr keepdims=0
      attr select_last_index=1
 2. Cast         inputs=[Bi64] -> [Bi]
      attr to=6
 3. Equal        inputs=[ar10, Bi] -> [bEq]
 4. Cast         inputs=[bEq] -> [bF]
      attr to=1
 5. Einsum       inputs=[input, bF] -> [rproj]
      attr equation='bkrc,bxk->bxr'

      **EINSUM EQUATION: `bkrc,bxk->bxr`**

 6. AveragePool  inputs=[rproj] -> [poolr]
      attr kernel_shape=[10]
      attr strides=[1]
 7. ArgMax       inputs=[poolr] -> [sri64]
      attr axis=2
      attr keepdims=0
 8. Cast         inputs=[sri64] -> [sri]
      attr to=6
 9. Reshape      inputs=[sri, one] -> [sr]
10. Einsum       inputs=[input, bF] -> [cproj]
      attr equation='bkrc,bxk->bxc'

      **EINSUM EQUATION: `bkrc,bxk->bxc`**

11. AveragePool  inputs=[cproj] -> [poolc]
      attr kernel_shape=[10]
      attr strides=[1]
12. ArgMax       inputs=[poolc] -> [sci64]
      attr axis=2
      attr keepdims=0
13. Cast         inputs=[sci64] -> [sci]
      attr to=6
14. Reshape      inputs=[sci, one] -> [sc]
15. Concat       inputs=[Bi, sr, sc] -> [fstart]
      attr axis=0
16. Add          inputs=[fstart, sizeF] -> [fend]
17. Slice        inputs=[input, fstart, fend, axF] -> [F]
18. Einsum       inputs=[F] -> [colCnt]
      attr equation='bcij->bj'

      **EINSUM EQUATION: `bcij->bj`**

19. Greater      inputs=[colCnt, thr] -> [colHit]
20. Cast         inputs=[colHit] -> [colHitU]
      attr to=2
21. ArgMax       inputs=[colHitU] -> [cf64]
      attr axis=1
      attr keepdims=0
22. Cast         inputs=[cf64] -> [cf]
      attr to=6
23. ArgMax       inputs=[colHitU] -> [cl64]
      attr axis=1
      attr keepdims=0
      attr select_last_index=1
24. Cast         inputs=[cl64] -> [cl]
      attr to=6
25. GreaterOrEqual inputs=[ar10, cf] -> [cge]
26. LessOrEqual  inputs=[ar10, cl] -> [cle]
27. And          inputs=[cge, cle] -> [cmask]
28. Cast         inputs=[cmask] -> [cmaskF]
      attr to=1
29. Einsum       inputs=[F, cmaskF] -> [rowMr]
      attr equation='bcij,bxj->bi'

      **EINSUM EQUATION: `bcij,bxj->bi`**

30. ReduceMax    inputs=[rowMr, axRed] -> [maxR]
      attr keepdims=1
31. Sub          inputs=[maxR, c15] -> [maxR1]
32. Greater      inputs=[rowMr, maxR1] -> [insideRow]
33. GreaterOrEqual inputs=[rowMr, maxR] -> [cleanRow]
34. Cast         inputs=[insideRow] -> [insRowF]
      attr to=1
35. ArgMax       inputs=[insRowF] -> [rf64]
      attr axis=1
      attr keepdims=0
36. Cast         inputs=[rf64] -> [rf]
      attr to=6
37. Einsum       inputs=[F, insRowF] -> [colMr]
      attr equation='bcij,bi->bcj'

      **EINSUM EQUATION: `bcij,bi->bcj`**

38. ReduceMax    inputs=[colMr, axCpad] -> [maxC]
      attr keepdims=1
39. GreaterOrEqual inputs=[colMr, maxC] -> [cleanCol]
40. Concat       inputs=[insideRow, cleanRow] -> [RiF]
      attr axis=0
41. Pad          inputs=[RiF, pad4, , axRed] -> [RiFp]
42. Add          inputs=[rf, ten] -> [rfE]
43. Slice        inputs=[RiFp, rf, rfE, axSl] -> [RiS]
44. Cast         inputs=[RiS] -> [Rh]
      attr to=10
45. Pad          inputs=[Rh, pad20, , axRed] -> [R]
46. Concat       inputs=[cmask, cleanCol] -> [CjF]
      attr axis=0
47. Pad          inputs=[CjF, pad4, , axCpad] -> [CjFp]
48. Add          inputs=[cf, ten] -> [cfE]
49. Slice        inputs=[CjFp, cf, cfE, axCsl] -> [CjS]
50. Cast         inputs=[CjS] -> [Ch]
      attr to=10
51. Pad          inputs=[Ch, pad20, , axCpad] -> [C]
52. Xor          inputs=[insideRow, cleanRow] -> [mrow]
53. Cast         inputs=[mrow] -> [mrowU]
      attr to=2
54. ArgMax       inputs=[mrowU] -> [mi1_64]
      attr axis=1
      attr keepdims=0
55. Cast         inputs=[mi1_64] -> [mi1]
      attr to=6
56. Gather       inputs=[F, mi1] -> [rowmap]
      attr axis=2
57. Less         inputs=[rowmap, cmaskF] -> [cand]
58. Cast         inputs=[cand] -> [candU]
      attr to=2
59. ArgMax       inputs=[candU] -> [mj1_64]
      attr axis=3
      attr keepdims=0
60. Cast         inputs=[mj1_64] -> [mj1]
      attr to=6
61. Reshape      inputs=[mj1, one] -> [mj]
62. Add          inputs=[sr, mi1] -> [gr]
63. Add          inputs=[sc, mj] -> [gc]
64. Concat       inputs=[gr, gc] -> [pst]
      attr axis=0
65. Add          inputs=[pst, oneone] -> [pen]
66. Slice        inputs=[input, pst, pen, axMp] -> [Mpix]
67. Cast         inputs=[Mpix] -> [MpixH]
      attr to=10
68. Reshape      inputs=[MpixH, s1_1_10] -> [M3]
69. Cast         inputs=[bEq] -> [bF16]
      attr to=10
70. Sub          inputs=[bF16, M3] -> [delta3]
71. Concat       inputs=[M3, delta3] -> [A]
      attr axis=1
72. Einsum       inputs=[A, R, C] -> [output]   <== TERMINAL (graph output)
      attr equation='bsk,sr,stc->bkrc'

      **EINSUM EQUATION: `bsk,sr,stc->bkrc`**


Initializers (16):
- `ar10`  int32[1, 1, 10]  (10 elems)
    shape [1, 1, 10] flat=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- `thr`  float[]  (1 elems)
    [5.5]
- `c15`  float[]  (1 elems)
    [1.5]
- `one`  int64[1]  (1 elems)
    [1]
- `ten`  int32[1]  (1 elems)
    [10]
- `sizeF`  int32[3]  (3 elems)
    [1, 10, 10]
- `axF`  int32[3]  (3 elems)
    [1, 2, 3]
- `axMp`  int32[2]  (2 elems)
    [2, 3]
- `axRed`  int64[1]  (1 elems)
    [1]
- `axSl`  int32[1]  (1 elems)
    [1]
- `pad4`  int64[2]  (2 elems)
    [0, 4]
- `pad20`  int64[2]  (2 elems)
    [0, 20]
- `oneone`  int32[]  (1 elems)
    [1]
- `s1_1_10`  int64[3]  (3 elems)
    [1, 1, 10]
- `axCpad`  int64[1]  (1 elems)
    [2]
- `axCsl`  int32[1]  (1 elems)
    [2]
```

## MECHANISM

Memory-bound task (new cost 2084 = 34 params + 2050 B). The win is compact-record engineering, not
plane-kill: (1) dominant color via `bkrc->bk` counts -> ArgMax -> Equal(ar10) one-hot bF (no
OneHot op); (2) locate the 10x10 subgrid: project rows/cols with `bkrc,bxk->bxr/bxc`, then
AveragePool(kernel=10)+ArgMax finds the densest window origin — pooling as a run-locator; (3) Slice
the [1,1,10,10] crop F (400 B, the biggest charged tensor); (4) per-column counts vs threshold 5.5
-> first/last hit columns via dual ArgMax; (5) build tiny R[2,30]/C[2,1,30] boundary records in
FLOAT16 via Pad(4)/Pad(20)+Slice — f16 halves the memory charge and is integer-exact below 2048;
(6) terminal Einsum `bsk,sr,stc->bkrc` renders the full canvas from the A/R/C records. Old paid
3548 B on repeated full-color fp32 projections; new pays 2050 B, params trimmed 47->34.

## REUSABLE MOVE

When a crop must be found first: pooled-ArgMax on row/col projection vectors gives the window
origin; Slice a small window and reduce it to per-axis boundary records (first/last hit vs a
threshold); render from records with a terminal einsum. Store records in fp16 — integer-exact
< 2048 and half the memory price.
