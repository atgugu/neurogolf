# task099 — compile exemplar

- **Mechanism:** counting-histogram
- **True new cost (7431, SHA-verified rescorer):** **504** (params 504 + memory 0) -> **18.7774** pts
- **Delta vs old:** **+1.0408 pts** (old 7384 member: cost 1427 -> 17.7367 pts)
- **Terminal op:** Einsum · **nodes:** 1 · **initializer elements:** 504

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task099.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:18'], file size 2349 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (1, topological order):
 0. Einsum       inputs=[input, coef, C, cqi, S, csj, S, cpr, S, cqc, S] -> [output]   <== TERMINAL (graph output)
      attr equation='nkij,ts,sok,ta,ai,tb,bj,tp,pr,tq,qc->norc'

      **EINSUM EQUATION: `nkij,ts,sok,ta,ai,tb,bj,tp,pr,tq,qc->norc`**


Initializers (7):
- `coef`  float[7, 2]  (14 elems)
    [[0.18952605, -1.18602455],
       [0.01528719, 1.91639829],
       [0.00179538, -1.00569201],
       [-0.42372221, 1.22071326],
       [0.17214873, 1.66984403],
       [0.08618627, -1.15587366],
       [0.82852745, -0.19480342]]
- `C`  float[2, 10, 10]  (200 elems)
    min=-4.41028023 max=4.81838703 nnz=200/200 first16=[0.29260352, -2.02576852, 4.37926531, 4.38270521, 4.3780117, 4.38456249, 4.37240219, 4.38226604, 4.38085508, 4.37400484, -0.22068651, 0.21612076, -0.96038055, -0.96017343, -0.96052849, -0.96161205, ...]
- `S`  float[5, 30]  (150 elems)
    min=-2.28974628 max=2.19972038 nnz=50/150 first16=[0.54344517, 1.18923616, 1.3436538, 1.14699578, -0.3627317, 0.90861595, -1.62556708, 0.52148014, -0.0111036, 0.9866901, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
- `cqi`  float[7, 5]  (35 elems)
    [[0.68528694, -1.85793638, -0.40262917, -1.29012787, 0.70428789],
       [-0.54994625, 1.27734721, 0.80146062, -1.43177593, 0.80585951],
       [1.40893626, -0.40992615, 0.73236555, 0.74400455, -1.29415619],
       [0.48743951, 0.11624368, -0.02037809, 1.99520266, 0.92931885],
       [-0.01718253, -0.37521285, 0.98649365, 1.33939672, 1.01934922],
       [1.46167231, 0.41686633, -1.95849729, -0.03978804, 1.66808558],
       [0.41954866, 0.5696851, 1.2162149, -0.62205559, 0.03312849]]
- `csj`  float[7, 5]  (35 elems)
    [[1.16748989, -0.60789388, -0.29176414, 0.98993176, -1.72121263],
       [1.88433957, 1.92338264, -0.98633283, 0.15090612, 0.68954313],
       [-1.30345309, 0.52937627, -0.35316053, -1.64944565, -0.81111109],
       [0.26856223, -0.60612434, 0.53896505, 1.26858604, 1.58434844],
       [1.90396249, 0.91139174, -0.84355932, 0.12745909, -1.29717135],
       [0.71711236, -0.84881473, -0.16046382, -0.9022451, -1.59938514],
       [-1.41559434, 1.00250983, 0.20334896, 0.9411639, 0.74261373]]
- `cpr`  float[7, 5]  (35 elems)
    [[0.99054897, -1.97192502, -0.31968689, -0.94738209, 1.56723642],
       [0.21088161, -0.05342231, -2.5362196, 0.62213916, 0.54743433],
       [-0.96775359, 0.3209936, -0.16045648, -1.0413177, 1.29959762],
       [0.17372397, -0.29963359, 0.73102999, 2.04114819, 1.16554725],
       [1.16217363, -2.50982428, -0.25276327, -1.30794466, 1.92154336],
       [0.68478256, -0.86902988, 1.37296772, -0.90480262, 0.81098795],
       [-0.45355007, 0.68897665, 1.53489137, -0.0115337, -0.68411487]]
- `cqc`  float[7, 5]  (35 elems)
    [[-1.24804556, 0.54278028, -0.19104037, -1.31287849, 0.81627804],
       [-1.38904059, -0.49541375, 0.41561913, 1.31172776, -1.00799799],
       [-1.2941972, 1.00447166, 0.66135389, 0.9329856, -1.02226901],
       [1.13345313, -0.91691315, -0.22438367, -0.20963177, 0.84481359],
       [0.58164024, -0.10679126, 2.95931292, -0.56725597, 0.04059995],
       [0.19335532, -0.9257313, -1.67440677, -0.5577746, 0.33973265],
       [-0.75124365, 0.80624306, 1.37286484, 0.6941449, -1.04672456]]
```

## MECHANISM

52-node procedural marker/label constructor (Slices, GatherND gates, ~25 Max/Min ops, a ~100-term
Concat, Pad, Equal against a color bank; 48 params but 1379 B of charged intermediates) collapses
to ONE terminal Einsum: `nkij,ts,sok,ta,ai,tb,bj,tp,pr,tq,qc->norc`. Structure: a bank of t=7
detector->writer terms mixed by `coef`(7,2). Each term reads a weighted spatial moment of the input
— probe factors cqi/csj(7,5) contracted through the shared positional basis S(5,30) (support only
on the first 10 positions) give the (ta,ai)/(tb,bj) row/col probes — and writes to output positions
via cpr/cqc(7,5) through the SAME S (slots `tp,pr`,`tq,qc`). Colors map through the dense
C(2,10,10) input-color->output-color relation table (2 sheets selected by the s latent). Counting
IS the contraction: marker counts appear as index moments, never as charged tensors. Memory
1379->0; params 48->504 — the 456-el param increase buys a 923 cost cut.

## REUSABLE MOVE

Compile "count things, then draw a label that depends on the count" as a bank of T rank-1
detector->writer terms in one einsum: term = (probe basis @ S) x (color relation table) x (target
basis @ S), mixed by a T x 2 coefficient matrix. Share one positional basis S across all probe AND
writer slots.
