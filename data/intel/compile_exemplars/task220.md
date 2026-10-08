# task220 — compile exemplar

- **Mechanism:** radix-position-factors
- **True new cost (7431, SHA-verified rescorer):** **437** (params 305 + memory 132) -> **18.9201** pts
- **Delta vs old:** **+0.7225 pts** (old 7384 member: cost 900 -> 18.1976 pts)
- **Terminal op:** Einsum · **nodes:** 19 · **initializer elements:** 305

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 3049 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (19, topological order):
 0. Einsum       inputs=[input, in_basis, b2, coord] -> [row2]
      attr equation='nchw,ic,i,qh->q'

      **EINSUM EQUATION: `nchw,ic,i,qh->q`**

 1. Einsum       inputs=[input, in_basis, b2, coord, col_take] -> [col2]
      attr equation='nchw,ic,i,aw,qa->q'

      **EINSUM EQUATION: `nchw,ic,i,aw,qa->q`**

 2. Einsum       inputs=[row2, row2, one] -> [r2_2]
      attr equation='a,a,k->k'

      **EINSUM EQUATION: `a,a,k->k`**

 3. Mul          inputs=[col2, col2] -> [c2_2]
 4. Add          inputs=[r2_2, c2_2] -> [ss_2]
 5. Concat       inputs=[one, row2, ss_2, col2] -> [feat2]
      attr axis=0
 6. Einsum       inputs=[input, in_basis, b3, coord] -> [row3]
      attr equation='nchw,ic,i,qh->q'

      **EINSUM EQUATION: `nchw,ic,i,qh->q`**

 7. Einsum       inputs=[input, in_basis, b3, coord, col_take] -> [col3]
      attr equation='nchw,ic,i,aw,qa->q'

      **EINSUM EQUATION: `nchw,ic,i,aw,qa->q`**

 8. Einsum       inputs=[row3, row3, one] -> [r2_3]
      attr equation='a,a,k->k'

      **EINSUM EQUATION: `a,a,k->k`**

 9. Mul          inputs=[col3, col3] -> [c2_3]
10. Add          inputs=[r2_3, c2_3] -> [ss_3]
11. Concat       inputs=[one, row3, ss_3, col3] -> [feat3]
      attr axis=0
12. Einsum       inputs=[input, in_basis, b8, coord] -> [row8]
      attr equation='nchw,ic,i,qh->q'

      **EINSUM EQUATION: `nchw,ic,i,qh->q`**

13. Einsum       inputs=[input, in_basis, b8, coord, col_take] -> [col8]
      attr equation='nchw,ic,i,aw,qa->q'

      **EINSUM EQUATION: `nchw,ic,i,aw,qa->q`**

14. Einsum       inputs=[row8, row8, one] -> [r2_8]
      attr equation='a,a,k->k'

      **EINSUM EQUATION: `a,a,k->k`**

15. Mul          inputs=[col8, col8] -> [c2_8]
16. Add          inputs=[r2_8, c2_8] -> [ss_8]
17. Concat       inputs=[one, row8, ss_8, col8] -> [feat8]
      attr axis=0
18. Einsum       inputs=[input, term_in, in_basis, s2, state_l, featcoef, feat2, row_a, coord, row_b, coord, col_a, coord, col_b, coord, s3, state_l, featcoef, feat3, row_a, coord, row_b, coord, col_a, coord, col_b, coord, s8, state_l, featcoef, feat8, row_a, coord, row_b, coord, col_a, coord, col_b, coord] -> [output]   <== TERMINAL (graph output)
      attr equation='nchw,ti,ic,ts,sl,lp,p,la,ah,lb,bh,ld,dw,le,ew,tu,um,mq,q,mf,fh,mg,gh,mj,jw,mk,kw,tv,vr,ry,y,rA,Ah,rB,Bh,rD,Dw,rE,Ew->nthw'

      **EINSUM EQUATION: `nchw,ti,ic,ts,sl,lp,p,la,ah,lb,bh,ld,dw,le,ew,tu,um,mq,q,mf,fh,mg,gh,mj,jw,mk,kw,tv,vr,ry,y,rA,Ah,rB,Bh,rD,Dw,rE,Ew->nthw`**


Initializers (17):
- `coord`  float[2, 30]  (60 elems)
    [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
       [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]]
- `in_basis`  float[4, 10]  (40 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
- `b2`  float[4]  (4 elems)
    [0.0, 1.0, 0.0, 0.0]
- `b3`  float[4]  (4 elems)
    [0.0, 0.0, 1.0, 0.0]
- `b8`  float[4]  (4 elems)
    [0.0, 0.0, 0.0, 1.0]
- `col_take`  float[1, 2]  (2 elems)
    [[0.0, 1.0]]
- `one`  float[1]  (1 elems)
    [1.0]
- `term_in`  float[10, 4]  (40 elems)
    [[-1.0, 0.0, 0.0, 0.0],
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0],
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 1.0],
       [0.0, 0.0, 0.0, 0.0]]
- `s2`  float[10, 2]  (20 elems)
    [[0.0, 1.0],
       [0.0, 1.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [0.0, 0.0],
       [1.0, 0.0],
       [0.0, 0.0],
       [1.0, 0.0],
       [0.0, 0.0]]
- `s3`  float[10, 2]  (20 elems)
    [[0.0, 1.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [0.0, 0.0],
       [0.0, 1.0],
       [0.0, 0.0],
       [1.0, 0.0],
       [0.0, 0.0]]
- `s8`  float[10, 2]  (20 elems)
    [[0.0, 1.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [0.0, 1.0],
       [0.0, 0.0],
       [1.0, 0.0],
       [0.0, 0.0],
       [1.0, 0.0],
       [0.0, 0.0]]
- `state_l`  float[2, 6]  (12 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
- `featcoef`  float[6, 5]  (30 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0],
       [-1000.0, 1003.5, 0.0, -1.0, 0.0],
       [0.0, -1.0, 0.0, 0.0, 0.0],
       [0.0, -1.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 2.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 2.0]]
- `row_a`  float[6, 2]  (12 elems)
    [[1.0, 0.0],
       [1.0, 0.0],
       [0.0, 1.0],
       [1.0, 0.0],
       [0.0, 1.0],
       [1.0, 0.0]]
- `row_b`  float[6, 2]  (12 elems)
    [[1.0, 0.0],
       [1.0, 0.0],
       [0.0, 1.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [1.0, 0.0]]
- `col_a`  float[6, 2]  (12 elems)
    [[1.0, 0.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [0.0, 1.0],
       [1.0, 0.0],
       [0.0, 1.0]]
- `col_b`  float[6, 2]  (12 elems)
    [[1.0, 0.0],
       [1.0, 0.0],
       [1.0, 0.0],
       [0.0, 1.0],
       [1.0, 0.0],
       [1.0, 0.0]]
```

## MECHANISM

Box-around-marker rule for marker colors 2/3/8 -> painted 3x3 squares. Old: dense 900-el 3x3 Conv
stencil. New (19 nodes, 305 params + 132 B): three parallel blocks each compute the marker's
location as coordinate MOMENTS — `nchw,ic,i,qh->q` with coord=[ones; 0..29] returns [count,
sum-of-row-positions], the column version adds `col_take`; `a,a,k->k` squares them, and
feat = [1, row, row^2+col^2, col]. The terminal 39-operand Einsum builds, per marker t, quadratic
polynomials in h and w from the reused coord basis (`la,ah`,`lb,bh` pairs = products of linear
coordinate forms via row_a/row_b/col_a/col_b), which are sign-positive exactly on the 3x3 box at
the marker. featcoef row [-1000.0, 1003.5, 0, -1, 0] is the presence gate: the -1000/+1003.5
integer-exact sentinel pair fires only when the marker exists, so absent markers contribute
nothing. term_in maps each t to its output color set. All geometry lives in one shared coord(2,30)
and 6x2/6x5 factors.

## REUSABLE MOVE

Locate a marker as scalar moments (count, count-weighted position) via a [ones; 0..29] coord basis,
then paint the box with quadratic coordinate polynomials built from products of linear forms in the
SAME basis, gated by a -1000/+1003.5 presence sentinel — a 900-el stencil becomes ~300 els of
reused factors.
