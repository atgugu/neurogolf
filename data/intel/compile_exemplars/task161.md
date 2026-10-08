# task161 — compile exemplar

- **Mechanism:** compiled-terminal-einsum
- **True new cost (7431, SHA-verified rescorer):** **275** (params 85 + memory 190) -> **19.3832** pts
- **Delta vs old:** **+0.9735 pts** (old 7384 member: cost 728 -> 18.4097 pts)
- **Terminal op:** Einsum · **nodes:** 6 · **initializer elements:** 85

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task161.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 774 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (6, topological order):
 0. Einsum       inputs=[input, w, w] -> [gp]
      attr equation='bkrc,r,c->bk'

      **EINSUM EQUATION: `bkrc,r,c->bk`**

 1. Equal        inputs=[gp, two] -> [selC]
 2. Concat       inputs=[ones_b, selC] -> [SSb]
      attr axis=0
 3. Cast         inputs=[SSb] -> [SS]
      attr to=1
 4. Where        inputs=[selC, two, ne0] -> [route]
 5. Einsum       inputs=[route, W, SS, input, e0, SS, input, e0] -> [output]   <== TERMINAL (graph output)
      attr equation='bk,st,sm,bmrx,x,tn,bnyc,y->bkrc'

      **EINSUM EQUATION: `bk,st,sm,bmrx,x,tn,bnyc,y->bkrc`**


Initializers (6):
- `w`  float[30]  (30 elems)
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
- `e0`  float[30]  (30 elems)
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `W`  float[2, 2]  (4 elems)
    [[-1.0, 2.0],
       [2.0, 0.0]]
- `two`  float[]  (1 elems)
    [2.0]
- `ones_b`  bool[1, 10]  (10 elems)
    [[True, True, True, True, True, True, True, True, True, True]]
- `ne0`  float[1, 10]  (10 elems)
    [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
```

## MECHANISM

16-node bool-forest (5 Einsums, 4 Equals, And/Add/Concat; 618 B) recompiled into predicate ->
selector -> terminal render (6 nodes, 190 B). Predicate: `bkrc,r,c->bk` with w = [0,1,1,...,1] on
both axes counts each color's cells OUTSIDE row 0/col 0 (the header); Equal(gp, 2.0) -> selC marks
colors appearing exactly twice. Lift: Concat([ones_b, selC]) -> SS(2,10) selector matrix;
Where(selC, 2.0, ne0) -> route (per-color routing scalars: 2 for selected, -1 for color 0, 0
else). Terminal `bk,st,sm,bmrx,x,tn,bnyc,y->bkrc`: e0 = [1,0,...,0] one-hot position vectors make
the einsum READ the input's header row (bmrx,x at x=0) and header column (bnyc,y at y=0), combining
them through W(2,2) = [[-1,2],[2,0]] and the SS selector — condition, routing, and the
row x column outer-product render all happen inside one contraction. Memory 618 -> 190 B, params
110 -> 85.

## REUSABLE MOVE

Compile predicate->render: compute one boolean predicate vector (count == k), lift it to a selector
matrix by Concat with a ones row, turn it into routing scalars with a single Where, and let the
terminal einsum read specific input rows/columns via one-hot position vectors (e0) — the whole
conditional render is one contraction.
