# task310 — compile exemplar

- **Mechanism:** exotic-op-count · TfIdfVectorizer
- **True new cost (7431, SHA-verified rescorer):** **593** (params 363 + memory 230) -> **18.6148** pts
- **Delta vs old:** **+1.6078 pts** (old 7384 member: cost 2960 -> 17.0071 pts)
- **Terminal op:** Einsum · **nodes:** 19 · **initializer elements:** 363

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task310.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 3887 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (19, topological order):
 0. Einsum       inputs=[input] -> [counts]
      attr equation='bchw->bc'

      **EINSUM EQUATION: `bchw->bc`**

 1. Equal        inputs=[counts, zero] -> [eq0]
 2. Where        inputs=[eq0, c9999, counts] -> [big]
 3. ArgMin       inputs=[big] -> [rare_idx]
      attr axis=1
      attr keepdims=1
 4. TfIdfVectorizer inputs=[rare_idx] -> [rare8]
      attr max_gram_length=1
      attr max_skip_count=0
      attr min_gram_length=1
      attr mode='TFIDF'
      attr ngram_counts=[0]
      attr ngram_indexes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      attr pool_int64s=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      attr weights=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
 5. Einsum       inputs=[counts, rare8] -> [halfspan]
      attr equation='bc,bc->b'

      **EINSUM EQUATION: `bc,bc->b`**

 6. Einsum       inputs=[input, rare8, idx30] -> [rs]
      attr equation='bchw,bc,h->b'

      **EINSUM EQUATION: `bchw,bc,h->b`**

 7. Einsum       inputs=[input, rare8, idx30] -> [cs]
      attr equation='bchw,bc,w->b'

      **EINSUM EQUATION: `bchw,bc,w->b`**

 8. Div          inputs=[rs, halfspan] -> [rmean]
 9. Sub          inputs=[rmean, halfspan] -> [rtop]
10. Cast         inputs=[rtop] -> [ri]
      attr to=6
11. Div          inputs=[cs, halfspan] -> [cmean]
12. Sub          inputs=[cmean, halfspan] -> [ctop]
13. Cast         inputs=[ctop] -> [ci]
      attr to=6
14. TfIdfVectorizer inputs=[ri] -> [or3]
      attr max_gram_length=1
      attr max_skip_count=0
      attr min_gram_length=1
      attr mode='TF'
      attr ngram_counts=[0]
      attr ngram_indexes=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
      attr pool_int64s=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
15. TfIdfVectorizer inputs=[ri] -> [or4]
      attr max_gram_length=1
      attr max_skip_count=0
      attr min_gram_length=1
      attr mode='TF'
      attr ngram_counts=[0]
      attr ngram_indexes=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
      attr pool_int64s=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
16. TfIdfVectorizer inputs=[ci] -> [oc3]
      attr max_gram_length=1
      attr max_skip_count=0
      attr min_gram_length=1
      attr mode='TF'
      attr ngram_counts=[0]
      attr ngram_indexes=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
      attr pool_int64s=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
17. TfIdfVectorizer inputs=[ci] -> [oc4]
      attr max_gram_length=1
      attr max_skip_count=0
      attr min_gram_length=1
      attr mode='TF'
      attr ngram_counts=[0]
      attr ngram_indexes=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
      attr pool_int64s=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
18. Einsum       inputs=[E3, E4, E3, E4, HB3, HB4, or3, or4, input, rare8, cap, input, input, rare8, E3, E4, E3, E4, HB3, HB4, oc3, oc4, cap] -> [output]   <== TERMINAL (graph output)
      attr equation='hd,he,RD,RE,Dgd,Eie,g,i,buhx,bu,R,bchw,bvyw,bv,wk,wl,SK,SL,Knk,Lpl,n,p,S->bcRS'

      **EINSUM EQUATION: `hd,he,RD,RE,Dgd,Eie,g,i,buhx,bu,R,bchw,bvyw,bv,wk,wl,SK,SL,Knk,Lpl,n,p,S->bcRS`**


Initializers (8):
- `idx30`  float[30]  (30 elems)
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
- `zero`  float[]  (1 elems)
    [0.0]
- `c9999`  float[]  (1 elems)
    [9999.0]
- `cap`  float[30]  (30 elems)
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `E3`  float[30, 3]  (90 elems)
    min=0.0 max=1.0 nnz=30/90 first16=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ...]
- `E4`  float[30, 4]  (120 elems)
    min=0.0 max=1.0 nnz=30/120 first16=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, ...]
- `HB3`  float[3, 3, 3]  (27 elems)
    shape [3, 3, 3] flat=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
- `HB4`  float[4, 4, 4]  (64 elems)
    shape [4, 4, 4] flat=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
```

## MECHANISM

Rare-color locate + modular re-render. Selector: `bchw->bc` counts, Where(counts==0, 9999) then
ArgMin picks the rarest PRESENT color; TfIdfVectorizer (mode=TFIDF, pool_int64s=0..9,
weights=0.125) converts that integer index into a scaled one-hot `rare8` — an int->one-hot encoder
in a single op. Moments `bchw,bc,h->b` etc. plus Div/Sub/Cast produce the object's corner ints
ri/ci. Then the exotic move (bytes are clearer than the MD here): TF-mode TfIdfVectorizers with
CYCLIC ngram_indexes ([0,1,2,0,1,2,...] and [0,1,2,3,0,...] over pool 0..29) map an integer
position p to one-hot(p mod 3) and one-hot(p mod 4). The terminal Einsum combines E3(30,3)/E4(30,4)
(one-hot h mod 3 / h mod 4), HB3(3,3,3)/HB4(4,4,4) — exact cyclic-shift tensors, verified against
the bytes: HBk[D,g,d]=1 iff d=(g+D) mod k, i.e. D encodes the residue DIFFERENCE d-g — and `cap`
(first-12 mask): the output-row residue is the input-row residue shifted by the corner offset,
simultaneously mod 3 and mod 4: a Chinese-remainder positional translation into the 12x12 area,
entirely by contraction with 0/1 tensors. No Gather/Scatter, no full planes; memory 230 B.

## REUSABLE MOVE

TfIdfVectorizer is a free integer decoder: pool 0..29 gives int->one-hot; cyclic ngram_indexes give
one-hot(p mod k) in ONE op. Pair those residues with tiny mod-k shift tensors
(HBk[D,g,d]=1 iff d=(g+D) mod k) inside the terminal einsum to translate positions arithmetically —
Gather/Scatter-free placement.
