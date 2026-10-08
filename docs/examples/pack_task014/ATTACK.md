# ATTACK PACK — task014  (generator 0b148d64)

READ FIRST: ../../data/RULES.md (cost model, banned surface, ORT traps, acceptance contract).
Your lane brief (thesis + bar) is in ../../data/lanes/ — find task014 there.


## COMPILE THE RULE — the measured #1 family (try this FIRST)
The verified study7300 corpus (all 111 improved pairs re-scored vs this base) shows the
dominant win engine is: compile the WHOLE rule into ONE terminal multi-operand Einsum
(single node ⇒ charged memory 0 — the node's internal workspace is free), built from:
- the FREE input repeated 2–10× as operands — products of the input with itself are
  pairwise/relational/quadratic features (same-color tests, counts, coincidences);
- 2–5 TINY factor matrices reused across index roles (each charged ONCE — dedupe!):
  10×10 color relations, rank-2/3 palette factors, trig/radix position codes;
- boolean branches as scalar SELECTOR operands (0/1 factors), never graph branches;
- integer-exact sentinels that force decisions through the (raw>0) decode: ±1, 5000,
  −1000, −0.001 ε-suppression — values exact in fp32;
- operand ORDER = ORT speed: interleave each input copy with the factor(s) that
  eliminate its indices (measured 4.1 s → 0.008 s for the same equation).
**HARD BUDGET: this task's rule class (PLANE-KILL + u8 RELOWER) compiles at
≤700 elements. WRITE THE EQUATION AND THE FACTOR TABLE BEFORE ANY OPS** — if the
factor table alone exceeds the budget, the representation is wrong, not the budget.
Builder support: `sys.path.insert(0,"../../runner"); from ngolf_einsum import
build_einsum_model, parity` — `parity()` checks your equation against rule.py on real
draws BEFORE you spend a gate round. A build that verifies CORRECT but misses the class
budget is NOT a NO — register it as a factorization ticket (contract step 5).

### Exemplar task099 — counting-histogram (cost 504, 18.78 pts)
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
       [1.13345313, -0.91691315, -0

### Exemplar task205 — counting-histogram (cost 2084, 17.36 pts)
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
40. Concat       inputs=

### Exemplar task139 — compiled-terminal-einsum (cost 389, 19.04 pts)
# task139 — compile exemplar

- **Mechanism:** compiled-terminal-einsum
- **True new cost (7431, SHA-verified rescorer):** **389** (params 373 + memory 16) -> **19.0364** pts
- **Delta vs old:** **+0.7396 pts** (old 7384 member: cost 815 -> 18.2968 pts)
- **Terminal op:** Einsum · **nodes:** 2 · **initializer elements:** 373

## THE GRAPH
(dumped from `/workspace/concise_docs/study7300/extracted/7431/task139.onnx`; every Einsum
equation below was verified byte-identical in the model file)

```
Model: opset ['ai.onnx:13'], file size 2098 bytes, ir_version 10

Inputs:  input: float[1, 10, 30, 30]
Outputs: output: float[1, 10, 30, 30]

Nodes (2, topological order):
 0. GatherND     inputs=[input, xidx] -> [edge_vals]
      attr batch_dims=1
 1. Einsum       inputs=[input, edge_vals, l_chan, cat_src, src_vecs, out_vecs, edge2, row_sel2, row_masks, col_sel2, col_masks] -> [output]   <== TERMINAL (graph output)
      attr equation='narc,nxq,lk,ks,sa,ko,bq,blu,ur,blv,vc->norc'

      **EINSUM EQUATION: `narc,nxq,lk,ks,sa,ko,bq,blu,ur,blv,vc->norc`**


Initializers (10):
- `xidx`  int64[1, 1, 4, 3]  (12 elems)
    shape [1, 1, 4, 3] flat=[4, 1, 0, 4, 2, 0, 4, 3, 0, 0, 0, 0]
- `l_chan`  float[3, 3]  (9 elems)
    [[1.0, 1.0, 0.0],
       [0.0, 0.0, 1.0],
       [0.0, 0.0, 1.0]]
- `cat_src`  float[3, 2]  (6 elems)
    [[1.0, 0.0],
       [0.0, 1.0],
       [1.0, 0.0]]
- `src_vecs`  float[2, 10]  (20 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
- `out_vecs`  float[3, 10]  (30 elems)
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
- `edge2`  float[2, 4]  (8 elems)
    [[1.0, 1.0, 1.0, 0.0],
       [-1.0, -1.0, -1.0, 1.0]]
- `row_sel2`  float[2, 3, 4]  (24 elems)
    shape [2, 3, 4] flat=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
- `row_masks`  float[4, 30]  (120 elems)
    min=0.0 max=1.0 nnz=39/120 first16=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...]
- `col_sel2`  float[2, 3, 4]  (24 elems)
    shape [2, 3, 4] flat=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
- `col_masks`  float[4, 30]  (120 elems)
    min=0.0 max=1.0 nnz=39/120 first16=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...]
```

## MECHANISM

"Read the legend, paint the regions" in 2 nodes. GatherND(batch_dims=1) with xidx(1,1,4,3) =
rows (4,1,0),(4,2,0),(4,3,0),(0,0,0) pulls exactly the legend cells — channel 4, rows 1..3 of
column 0 — into a tiny edge_vals vector (the ONLY charged tensor besides its 16 B). The terminal
Einsum `narc,nxq,lk,ks,sa,ko,bq,blu,ur,blv,vc->norc` treats edge_vals as a ROUTING OPERAND:
edge2(2,4) = [[1,1,1,0],[-1,-1,-1,1]] classifies the legend's presence pattern into branch b;
row_masks/col_masks(4,30) are reusable 0/1 region bands selected per (branch, region) by
row_sel2/col_sel2(2,3,4); l_chan/cat_src/src_vecs/out_vecs route source colors to output colors,
with out_vecs row 2 = [-10,0,...,0,+1(ch7),...] using a -10 suppression sentinel to clear
background where color 7 is written. Old paid 765 B building region features procedurally
(Slice/ReduceMax/Mul/BitwiseAnd); new pays 16 B and 373 params (net -426).

## REUSABLE MOVE

GatherND is a cheap "read the legend" op: pull the few key cells as a small vector and feed it into



## TAILORED STRATEGY (compiled from anatomy + certificate + Hodel + fleet history)
- primary: **PLANE-KILL + u8 RELOWER** — 612B (36%) sits in fp32/f16/i64 tensors — value-exact relower to uint8/bool, compose in u8
- secondary: **EINSUM-COMPRESS** — 2×Einsum — merge contractions, shrink operand tables
- kill targets (the tensors that ARE the cost):
  - (enumeration withheld for fresh-design families — the named tensors cover ≈36% of the 2507 B memory; deleting them is the micro-shave anchor. Re-derive the rule as a terminal contraction — see COMPILE THE RULE above.)
- Hodel PARTIAL: diverges on 4/262 arc-gen draws — the pack renders the failing examples; decode the variant first
- fixed-scorer register target: cost ≤ 2187
Start from the primary. If it dead-ends structurally, the secondary is pre-approved as
your one family pivot.


## Numbers (fresh — priced against the LIVE pin 0965029d, server-exact replica)
- pin: cost **2542** (params 35, memory 2507) → **17.1593 pts**
- to register you need (Lane B): cost ≤ **2187** (Δ ≥ +0.15 — this is only the floor; aim ≥2× for a durable win)
- rule-class compile target: cost ≤ **700** — the bar above registers; THIS is the win condition (a verified-correct build over it becomes a ticket, not a NO)
- targets for THIS pin (cost 2542): **2× cut ⇒ ≤1271 B (Δ≈+0.69 — a durable win)** · next 0.5-pt band ⇒ ≤1541 B · absolute band ceilings 17⇒≤2981 · 18⇒≤1097 · 19⇒≤403 B

## Source certificate (from generator SOURCE — the only trustworthy invariants)
- M_proof (max grid dim): **25** · confidence: high
- output_shape_law: le_input · out_square: False
- palettes: in=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] · binary_in/out: False/True
- max colors/example in/out: 3/2

## Lane E sweep certificate (source-proven; quotes generator lines)
```json
{
 "task": "task014",
 "grid_bounds": {
  "in_h": [
   15,
   25
  ],
  "in_w": [
   15,
   25
  ],
  "out_h": null,
  "out_w": null,
  "how_proven": "Random path: `width, height = common.randint(15, 25), common.randint(15, 25)` bounds input to [15,25] per axis. Output: `output = common.grid(max_col - min_col + 1, max_row - min_row + 1)` with min_row\u2208[0,height-1], max_row\u2208[0,height-1] after scanning rarest cells, so out_h=max_row-min_row+1\u2208[1,height]\u2286[1,25]; symmetric for out_w."
 },
 "palette": {
  "in": null,
  "out": null,
  "fixed": false
 },
 "discrete_params": [
  {
   "name": "width",
   "range": "15..25",
   "count": 11
  },
  {
   "name": "height",
   "range": "15..25",
   "count": 11
  },
  {
   "name": "rowthick",
   "range": "2..5",
   "count": 4
  },
  {
   "name": "colthick",
   "range": "2..5",
   "count": 4
  },
  {
   "name": "row",
   "range": "5..(height-5-rowthick)",
   "count": null
  },
  {
   "name": "col",
   "range": "5..(width-5-colthick)",
   "count": null
  },
  {
   "name": "quadrant",
   "range": "0..3",
   "count": 4
  },
  {
   "name": "color_list",
   "range": "common.random_colors(2)",
   "count": null
  },
  {
   "name": "pixel_fill",
   "range": "common.random_pixels(width, height, 0.9)",
   "count": null
  }
 ],
 "output_space": {
  "enumerable": false,
  "size_estimate": null,
  "argument": "Whole output is `output[r-min_row][c-min_col] = grid[r][c]` over the rarest-color bbox; rarest depends on full `colors` (`foreground = [color for color in colors if color]`). Random path also uses `common.random_pixels(width, height, 0.9)` (stochastic per-cell inclusion). Structural params alone already exceed ~500: width(11)*height(11)*rowthick(4)*colthick(4)*quadrant(4)=7744 before row/col/color_list/pixel_fill."
 },
 "invariants": [
  "Output is the axis-aligned bounding box crop of the input over the min/max of rarest cells: `min_row, min_col, max_row, max_col = height, width, -1, -1`; `for r in range(height): for c in range(width): if grid[r][c] != rarest: continue` (update min/max); `output = common.grid(max_col - min_col + 1, ",
  "Rarest is the minimum-frequency non-zero color (ties to min value): `foreground = [color for color in colors if color]` then `rarest = min(set(foreground), key=foreground.count)`. (Computed identically in random path and explicit path.)",
  "Crop rectangle includes every cell inside the bbox of rarest positions (all colors/zeros present there), not only the rarest pixels themselves.",
  "In random path, quadrant gap cells (`quad == -1`) are never written: `if quad == -1: continue`, leaving background 0.",
  "In random path, random-generation loop accepts only when `rarest == color_list[0]`: `if rarest == color_list[0]: break` (resamples otherwise)."
 ],
 "traps": [
  "Docstring says square grid (`the width of the (square) grid`) but random path samples width and height independently (`width, height = common.randint(15, 25), common.randint(15, 25)`).",
  "Rarest tie-breaking is min color digit via `min(set(foreground), key=foreground.count)`, not arbitrary tie resolution.",
  "In random path, samples are accepted only when `rarest == color_list[0]` (break); resampled when `rarest != color_list[0]`.",
  "Cross/gap region (not in any of the four `quad` conditions) stays 0 and is excluded from quadrant coloring.",
  "Explicit `colors=` path (validate/train/test) bypasses random structural params; grid built as `for r in range(height): for c in range(width): grid[r][c] = colors[r * width + c]`; output determined by the full `colors` list, not a small generator tuple."
 ],
 "cheap_form_hint": "Count non-zero color frequencies, select rarest (tie\u2192min color), compute row/col min-max where input equals rarest, then slice/crop that bounding box including all interior pixels.",
 "_audited": "legacy-audit 2026-07-11 17:58",
 "_audit_notes": [
  "grid_bounds.out_h: [1, 25] vs auditor [1, 18] \u2192 null",
  "grid_bounds.out_w: [1, 25] vs auditor [1, 18] \u21
```

## Transfer hypothesis from a sibling win (verify everything)
# TRANSFER HYPOTHESIS from tonight's task049 win (Δ+0.147)
# advisory hints — verify everything; a cost-budget table is still required before code

## Transfer hypothesis: task 049 → task 014 (crop/extract via rarest-color)

- **Hypothesis:** The `w_fg` initializer plus the first `Einsum` (`bchw,cxy->bcxy`) may carry over unchanged if task 014 is fed as `[1,10,H,W]` one-hot (channel 0 = background), because it should reproduce per-color pixel totals over the full grid—the statistic `foreground.count` uses for rarity.

- **Hypothesis:** The `Cast→Where→ReduceMin→Equal` chain (mask nonzero counts, take global minimum, mark matching channels) may transfer as the rarest-color selector, but tie behavior should be checked: 049 keeps *all* channels tied at min count, while 014’s `min(set(foreground), key=count)` breaks ties by smallest color index—verify whether 014 stress pairs ever tie and whether multi-select changes the downstream bbox.

- **Hypothesis:** The second `Einsum` (`bchw,bchj,bcxy->bxy`) plus `Div/Round` width and `area/width_round` height may carry over as the bbox extent estimator for the selected rare color, assuming 014 rare-pixel layouts satisfy the same row-coincidence identity (`width ≈ row_pairs/area`) that held on 049.

- **Hypothesis:** `row_index_i8` / `col_index_i8` coordinate Subtractions and the final three-way `Min` may transfer unchanged to gate pixels inside `[rr.min..rr.max] × [cc.min..cc.max]`—but the gate likely marks *positions in the box*, not “pixels equal to rare”; 014’s crop includes every color inside that rectangle, so the Min output is probably an intermediate mask, not the final rule output.

- **Hypothesis:** The largest re-derivation is output semantics: 014 returns `g[rr.min:rr.max+1, cc.min:cc.max+1]` (a smaller value tensor), whereas 049 emits a fixed `[1,10,H,W]` INT8 mask—verify whether the evaluator accepts mask-and-channel recombination on a fixed canvas, or whether explicit `Slice`/gather sized by derived `height_i8` and `width_round` is required to emit the cropped subgrid.

- **Hypothesis:** The empty-foreground early return (`return g.copy()`) has no analogue in 049 and must be added—e.g., a `ReduceSum` foreground detector on the `w_fg`-weighted input feeding `Where` to bypass rarity/bbox logic and pass input through unchanged when the sum is zero.

- **Hypothesis:** Input layout may need re-derivation if 014’s pack supplies a 2D `[H,W]` integer grid rather than 10-channel one-hot—verify pack I/O first; if 

## Pin-graph pseudocode (what you must beat — advisory)
# Pin-graph pseudocode (advisory — numbers from the machine dump; verify anything load-bearing)

**Detect stage** (covers: ReduceSum, Cast, Slice, LessOrEqual, Where, ArgMin, Add, Equal, Cast, Einsum, Einsum, Greater, Cast, Greater, Cast, ArgMax, ArgMax, ArgMax, ArgMax, Add, Add, Add)

1. counts10_f32 = ReduceSum(input, axes_counts)
2. counts10_f16 = Cast(counts10_f32)
3. counts9_f16 = Slice(counts10_f16, one_i64, thirty_i64)
4. no_color = LessOrEqual(counts9_f16, zero_f16)
5. counts_safe = Where(no_color, big_f16, counts9_f16)
6. tgt_idx0 = ArgMin(counts_safe)
7. tgt_color = Add(tgt_idx0, one_i64)
8. sel_bool = Equal(arange10, tgt_color)
9. sel_f32 = Cast(sel_bool)
10. row_counts = Einsum(input, sel_f32)
11. col_counts = Einsum(input, sel_f32)
12. pres_row = Greater(row_counts, zero_f32)
13. pres_row_u8 = Cast(pres_row)
14. pres_col = Greater(col_counts, zero_f32)
15. pres_col_u8 = Cast(pres_col)
16. first_row = ArgMax(pres_row_u8)
17. last_row = ArgMax(pres_row_u8)
18. first_col = ArgMax(pres_col_u8)
19. last_col = ArgMax(pres_col_u8)
20. tgt_color_end = Add(tgt_color, one_i64)
21. row_end = Add(last_row, one_i64)
22. col_end = Add(last_col, one_i64)

**Transform stage** (covers: Concat, Concat, Slice, Cast, Sub, Sub, Sub, Sub, Concat, Pad)

23. slice_starts = Concat(zero_i64, tgt_color, first_row, first_col)
24. slice_ends = Concat(one_i64, tgt_color_end, row_end, col_end)
25. crop_f32 = Slice(input, slice_starts, slice_ends)
26. crop_u8 = Cast(crop_f32)
27. crop_h = Sub(row_end, first_row)
28. crop_w = Sub(col_end, first_col)
29. pad_h = Sub(thirty_i64, crop_h)
30. pad_w = Sub(thirty_i64, crop_w)
31. pads = Concat(zero_i64, zero_i64, zero_i64, zero_i64, zero_i64, zero_i64, pad_h, pad_w)
32. canvas01 = Pad(crop_u8, pads, pad_value)

**Render stage** (covers: Reshape, Where, Equal)

33. sel4d = Reshape(sel_bool, pal_shape)
34. pal = Where(sel4d, pal_one, pal_base)
35. output = Equal(canvas01, pal)

## Fat tensors
canvas01=900B: computes padded u8 canvas of the sliced target-color crop region; dump produces it via Cast->Pad with no other path shown.  
row_counts=120B: computes row aggregates via Einsum(input, sel_f32); dump produces f32 version that feeds only Greater (pres_row) then u8 cast.  
col_counts=120B: computes col aggregates via Einsum(input, sel_f32); dump produces f32 version that feeds only Greater (pres_col) then u8 cast.


# ORT-1.24 measured legality sheet (fleet evidence, not spec — distilled by the fleet analyst, entries cite the tasks that hit them)

## Dead on ORT 1.24 (measured)
`Einsum (uint8/int8/integer) — NOT_IMPLEMENTED / no path / UNSCORABLE — (tasks: 005,009,017)`
`Scan — in EXCLUDED_OPS / "scan_fused is impossible under this gate" — (tasks: 004,009)`
`bool Where — NOT_IMPLEMENTED — (tasks: 002,014)`
`Pad (bool/broadcast) — ORT Pad/Max broadcast failure — (tasks: 002,008)`
`Gather (uint8 indices) — InferenceError: unsupported type: tensor(uint8) — (tasks: 005) (single report)`
`ConvTranspose (uint8/int8/int32) — ONNX shape inference rejects — (tasks: 009) (single report)`
`ConvInteger (float one-hot) — cannot consume float one-hot input — (tasks: 008) (single report)`
`OneHot (bool) — unsupported → dropped for Equal — (tasks: 004) (single report)`

## Legal but trapped (measured)
`int64 ScatterND — legal but high cost (indices) — (tasks: 005,008)`
`fp32 Conv before uint8 Cast — scorer charges full float32 outputs — (tasks: 008) (single report)`
`DepthToSpace (u8) — legal u8 renderer but exceeds bar — (tasks: 009) (single report)`
`MaxPool/tile — legal but high cost floors — (tasks: 009) (single report)`
`one-hot + channel-0 expansion — must pay full in ONNX — (tasks: 005) (single report)`
`i32 Cast for indices — +3600 B for 30×30 — (tasks: 005) (single report)`
`fp16 Cast on input — +18000 B for 

## Pin-member anatomy (what you must beat — these tensors are the cost)
- nodes: 35 · op histogram: Cast:5 Add:4 ArgMax:4 Sub:4 Concat:3 Slice:2 Where:2 Equal:2 Einsum:2 Greater:2 ReduceSum:1 LessOrEqual:1
- top charged tensors (bytes · producing-op · dtype · shape):
       900 B  Pad            uint8    [1, 1, 30, 30]
       120 B  Einsum         float32  [1, 30]
       120 B  Einsum         float32  [1, 30]
        64 B  Concat         int64    [8]
        40 B  ReduceSum      float32  [10]
        40 B  Cast           float32  [10]
        32 B  Concat         int64    [4]
        32 B  Concat         int64    [4]
        30 B  Greater        bool     [1, 30]
        30 B  Greater        bool     [1, 30]
        30 B  Cast           uint8    [1, 30]
        30 B  Cast           uint8    [1, 30]
- OUTPUT dtype of pin member: bool — KEEP IT unless the brief says otherwise (bool-rename on task080 cost −12.5 on Kaggle).

## THE GRAPH TO BEAT — withheld on purpose
Your strategy is a REPLACEMENT design (PLANE-KILL + u8 RELOWER): design from the RULE
(generator source + Hodel + human descriptions + examples), NOT from the old graph — the
old design is an anchor toward micro-shaving. Its cost line-items are already listed in
the anatomy/kill-targets above; that is all you need to beat. If you genuinely need the
old member (e.g. as a behavioral oracle for distillation), extract it:
`python3 -c "import zipfile;zipfile.ZipFile('$NEUROGOLF_CLEAN/submission.zip').extract('task014.onnx','.')"`


## Generator source (the competition oracle — the rule, verbatim)
```python
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generator."""

import common


def generate(width=None, height=None, colors=None):
  """Returns input and output grids according to the given parameters.

  Args:
    width: the width of the (square) grid
    height: the height of the (square) grid
    colors: a list of digits representing the colors to be used
  """
  if width is None:
    while True:
      width, height = common.randint(15, 25), common.randint(15, 25)
      rowthick, colthick = common.randint(2, 5), common.randint(2, 5)
      row = common.randint(5, height - 5 - rowthick)
      col = common.randint(5, width - 5 - colthick)
      color_list, quadrant = common.random_colors(2), common.randint(0, 3)
      grid = common.grid(width, height)
      for (r, c) in common.random_pixels(width, height, 0.9):
        quad = -1
        quad = 0 if r < row and c < col else quad
        quad = 1 if r < row and c >= col + colthick else quad
        quad = 2 if r >= row + rowthick and c < col else quad
        quad = 3 if r >= row + rowthick and c >= col + colthick else quad
        if quad == -1: continue
        grid[r][c] = color_list[0 if quad == quadrant else 1]
      colors = []
      for r in range(height):
        for c in range(width):
          colors.append(grid[r][c])
      foreground = [color for color in colors if color]
      rarest = min(set(foreground), key=foreground.count)
      if rarest == color_list[0]: break

  grid = common.grid(width, height)
  for r in range(height):
    for c in range(width):
      grid[r][c] = colors[r * width + c]
  foreground = [color for color in colors if color]
  rarest = min(set(foreground), key=foreground.count)
  min_row, min_col, max_row, max_col = height, width, -1, -1
  for r in range(height):
    for c in range(width):
      if grid[r][c] != rarest: continue
      min_row, min_col = min(min_row, r), min(min_col, c)
      max_row, max_col = max(max_row, r), max(max_col, c)
  output = common.grid(max_col - min_col + 1, max_row - min_row + 1)
  for r in range(min_row, max_row + 1):
    for c in range(min_col, max_col + 1):
      output[r - min_row][c - min_col] = grid[r][c]
  return {"input": grid, "output": output}


def validate():
  """Validates the generator."""
  train = [
      generate(width=21, height=21,
               colors=[8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0,
                       8, 8, 8, 0, 0, 8, 0, 8, 0, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8,
                       0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8,
                       8, 0, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0, 0,
                       0, 8, 8, 0, 0, 0, 8, 8, 8, 8, 8, 8, 0, 8, 8, 0, 8, 8, 0,
                       0, 0, 0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 0, 8, 8, 0, 8, 0, 0,
                       8, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 8, 8, 8, 8, 0, 0, 8,
                       0, 8, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8, 8, 0, 0, 8, 0,
                       0, 8, 8, 0, 8, 0, 0, 0, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8,
                       8, 8, 8, 0, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 8, 8,
                       0, 8, 8, 0, 8, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0,
                       8, 8, 8, 8, 0, 8, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0,
                       0, 0, 8, 8, 8, 0, 0, 0, 8, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2,
                       0, 0, 0, 0, 8, 8, 0, 8, 8, 8, 0, 2, 2, 2, 2, 2, 2, 0, 2,
                       0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 2, 2, 2, 2, 2, 0,
                       2, 0, 2, 2, 0, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8, 2, 2, 0, 2,
                       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 8, 0, 2,
                       2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 8, 0, 0, 0, 8, 8, 0,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 8, 8, 0, 0,
                       8, 8, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 8, 8, 8,
                       0, 8, 8, 8]),
      generate(width=19, height=18,
               colors=[2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 0, 2,
                       2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0,
                       0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2,
                       2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0,
                       0, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 2,
                       2, 2, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 2, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 3, 3,
                       0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0, 0, 3, 3, 0,
                       0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 3, 3, 3, 0, 3, 0, 3, 0, 0,
                       2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 3, 3,
                       2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 3,
                       2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 3, 0, 3, 3, 3, 0, 3,
                       0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 3, 0, 3,
                       0]),
      generate(width=17, height=19,
               colors=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
                       1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1,
                       0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
                       1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
                       0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
                       0, 0, 4, 0, 4, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4,
                       4, 0, 4, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 4, 0, 4, 0, 0,
                       4, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 4, 4, 4, 4, 0, 0,
                       0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 4, 4, 4, 0, 4, 4, 0, 0, 1,
                       1, 1, 1, 1, 1, 1, 1, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 1, 0,
                       0, 0, 0, 1, 1, 1, 0, 4, 4, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0,
                       1, 1, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,
                       0, 1, 4, 4, 0, 4, 0, 4, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                       0]),
  ]
  test = [
      generate(width=17, height=15,
               colors=[1, 1, 1, 1, 0, 1, 0, 0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 1, 0,
                       1, 0, 1, 1, 0, 0, 0, 3, 0, 3, 3, 3, 0, 0, 0, 1, 1, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1,
                       0, 0, 3, 3, 0, 3, 3, 0, 3, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                       0, 3, 0, 3, 3, 3, 0, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3,
                       0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
                       0, 0, 0, 0, 3, 0, 0, 3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3,
                       0, 0, 3, 0, 0, 0, 3, 0, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3,
                       0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3, 0, 0,
                       0, 0, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0,
                       0, 3, 3, 3, 0, 3, 3, 0]),
  ]
  return {"train": train, "test": test}

```

<!-- ═══ DETAIL SECTIONS — inlined only on disk; the prompt points here ═══ -->

## Hodel reference solver (rule-as-code + verdict — TRUST ONLY IF "TRUE-RULE")
HISTORY WARNING: transcribing Hodel DSL ops one-to-one into ONNX was this project's FIRST
strategy and its worst failure (0 wins, −339 net). Hodel tells you WHAT the rule computes,
NEVER how to compute it cheaply — derive the rule, then design the cheapest ONNX
representation from the idiom menu.
POLICY: TRUE-RULE ⇒ the rule is exactly right; implement it (cheaply). PARTIAL ⇒ the
arc-gen rule DIVERGES — the divergence data below localizes the variant; decode it first.
═══ Hodel DSL solver · task014 = 0b148d64 · 5 lines ═══
(combinator-style; primitives in quus_purity/vendored/arc-dsl/dsl.py)

def solve_0b148d64(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O

train    3/3 graded
test     1/1 graded
arc-gen  258/262 graded bad4
(cached full-set verdict from candidates/hodel_verdicts.jsonl — use --run to re-execute)

⚠ VERDICT: PARTIAL — diverges on graded examples. ARC-GEN uses a rule VARIANT.
  Do NOT implement as-is. Use as structural hint; decode the variant from the
  failing examples (--diff), confirm in numpy via candidates/hypo.py before ONNX.


DIVERGENCE DATA: Hodel fails 4/262 arc-gen draws (failing idx: [48, 73, 204, 236]). The TRUE rule = Hodel's rule MODIFIED in exactly the way these failing examples show — study them:
═══ Hodel DSL solver · task014 = 0b148d64 · 5 lines ═══
(combinator-style; primitives in quus_purity/vendored/arc-dsl/dsl.py)

def solve_0b148d64(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O

train    3/3 graded
test     1/1 graded
arc-gen  258/262 graded bad4
(cached full-set verdict from candidates/hodel_verdicts.jsonl — use --run to re-execute)

⚠ VERDICT: PARTIAL — diverges on graded examples. ARC-GEN uses a rule VARIANT.
  Do NOT implement as-is. Use as structural hint; decode the variant from the
  failing examples (--diff), confirm in numpy via candidates/hypo.py before ONNX.

─── failing arc-gen example #48 ───
INPUT                EXPECTED     DSL-GOT           
555555000505550550   2222220022   555555000505550550
555505005555555555   2222202222   555505005555555555
555555005555555555   2222222222   555555005555555555
550555005555555555   2222222020   550555005555555555
055555005555555055   2222222022   055555005555555055
555555005555555555   2022222222   555555005555555555
000000000000000000   2222222222   000000000000000000
000000000000000000   0202022020   000000000000000000
055505002222220022   2222222222   055505002222220022
555550002222202222   0200202222   555550002222202222
505055002222222222   2222222222   505055002222222222
555555002222222020   2222222222   555555002222222020
005550002222222022   2222202222   005550002222222022
055505002022222222   2222222222   055505002022222222
555555002222222222   2222222222   555555002222222222
555555000202022020   0222222222   555555000202022020
505555002222222222                505555002222222222
555555000200202222                555555000200202222
555555002222222222                555555002222222222
555505002222222222                555505002222222222
055055002222202222                055055002222202222
555555002222222222                555555002222222222
555555002222222222                555555002222222222
555555000222222222                555555000222222222

─── failing arc-gen example #73 ───
INPUT                EXPECTED     DSL-GOT           
444444404000444444   0777077707   444444404000444444
444440444400444044   7770777707   444440444400444044
444004444400444444   7707007777   444004444400444444
444444444400444444   7777770777   444444444400444444
404444444400444444   7707777777   404444444400444444
444444404400444444   7770770777   444444404400444444
404444004400444044   7777777777   404444004400444044
044444440400404444   7777777777   044444440400404444
000000000000000000   7777777777   000000000000000000
000000000000000000   7777777777   000000000000000000
077707770700444444   7777777770   077707770700444444
777077770700444444   7777777700   777077770700444444
770700777700444444   7777777777   770700777700444444
777777077700044444   7777777777   777777077700044444
770777777700440444                770777777700440444
777077077700444444                777077077700444444
777777777700444444                777777777700444444
777777777700440444                777777777700440444
777777777700444044                777777777700444044
777777777700444444                777777777700444444
777777777000444444                777777777000444444
777777770000444044                777777770000444044
777777777700044444                777777777700044444
777777777700444444                777777777700444444


## Human rule descriptions (LARC/H-ARC — HINTS, not authority: the generator source decides; "common wrong outputs" = the ambiguity traps humans hit)
## Human annotations (LARC / H-ARC)
- ARC hash: `0b148d64`
- LARC: verified ✓, confidence 10, builds 2/3 succeeded
  - **See:** a rectangular pattern in each corner, one a different color than the others.
  - **Size:** becomes the size of the pattern that is a different color.
  - **Rule:** copy the pattern that is a different color
- H-ARC: 6/10 solved (60%), median 54 actions
  - **First solution:** Make the grid the size of the pattern which does not repeat and fill the grid with that pattern (the blue one).
- **Phrase tags:** object_detection, spatial_relation, procedure, loop

## Clean-era fleet record on this task
(no past fleet attempts recorded)

## Newest clean-era build script for this task (fleet code — reuse idioms, beat its cost)
CAVEAT: "newest" ≠ "best" — this is the most recently touched script, possibly a rejected
attempt. Cross-check its family against the fleet record above before adopting it.
```python
  # newest of 6 fleet build scripts: $NEUROGOLF_CLEAN/candidates/build_task146_min.py
#!/usr/bin/env python3
"""task146 min — single 9x3 slice, idx plane, sym on labels only, no blk2 sym, no reshape."""
from __future__ import annotations

import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

try:
    import onnxoptimizer
except ImportError:
    onnxoptimizer = None  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "candidates/task146_min.onnx"
TASK = 146
HEX = "662c240a"
TASK_JSON = ROOT / "extracted" / "task146.json"


def _const(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    nodes: list = []
    inits: list = []

    def add(name: str, arr) -> str:
        inits.append(_const(name, arr))
        return name

    add("cw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    add("axes_hw", np.array([2, 3], dtype=np.int64))
    add("start_grid", np.array([0, 0], dtype=np.int64))
    add("end_grid", np.array([9, 3], dtype=np.int64))
    add("nine_i32", np.array(9, dtype=np.int32))
    add("zero_bool", np.array(False, dtype=bool))
    add("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    add("pads_to_30", np.array([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64))
    for i in range(3):
        add(f"start{i}", np.array([3 * i, 0], dtype=np.int64))
        add(f"end{i}", np.array([3 * i + 3, 3], dtype=np.int64))

    nodes += [
        helper.make_node("Slice", ["input", "start_grid", "end_grid", "axes_hw"], ["grid"]),
        helper.make_node("Conv", ["grid", "cw"], ["idx_f32"]),
        helper.make_node("Cast", ["idx_f32"], ["idx_u8"], to=TensorProto.UINT8),
    ]

    blocks = []
    nonsym = []
    for i in range(2):  # only check blocks 0,1; block 2 is fallback
        nodes += [
            helper.make_node("Slice", ["idx_u8", f"start{i}", f"end{i}", "axes_hw"], [f"blk{i}"]),
            helper.make_node("Transpose", [f"blk{i}"], [f"blk{i}_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Equal", [f"blk{i}", f"blk{i}_t"], [f"sym_eq{i}"]),
            helper.make_node("Cast", [f"sym_eq{i}"], [f"sym_i32{i}"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", [f"sym_i32{i}"], [f"sym_score{i}"], keepdims=0),
            helper.make_node("Less", [f"sym_score{i}", "nine_i32"], [f"nonsym{i}"]),
        ]
        blocks.append(f"blk{i}")
        nonsym.append(f"nonsym{i}")

    nodes.append(
        helper.make_node("Slice", ["idx_u8", "start2", "end2", "axes_hw"], ["blk2"])
    )
    blocks.append("blk2")

    nodes += [
        helper.make_node("Where", [nonsym[1], blocks[1], blocks[2]], ["sel12"]),
        helper.make_node("Where", [nonsym[0], blocks[0], "sel12"], ["selected"]),
        helper.make_node("Equal", ["selected", "arange10"], ["small_bool"]),
        helper.make_node("Pad", ["small_bool", "pads_to_30", "zero_bool"], ["output"], mode="constant"),
    ]

    model = helper.make_model(
        helper.make_graph(
            nodes,
            "task146",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
            [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
            inits,
        ),
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    del model.graph.value_info[:]
    if onnxoptimizer is not None:
        model = onnxoptimizer.optimize(model)
        del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


def verify_all(model: onnx.ModelProto) -> tuple[bool, str]:
    sys.path.insert(0, str(ROOT / "extracted" / "neurogolf_utils"))
    from neurogolf_utils import verify_subset  # noqa: WPS433

    tj = json.load(open(TASK_JSON))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for call in range(2):
        for split in ("train", "test", "arc-gen"):
            _, wrong, _ = verify_subset(sess, tj.get(split, []))
            if wrong:
                return False, f"{split} call{call}: {wrong} wrong"
    sys.path.insert(0, str(ROOT / "external" / "ARC-GEN"))
    mod = importlib.import_module(f"tasks.task_{HEX}")
    batch = []
    for i in range(300):
        random.seed(1_000_146 * i + 17)
        try:
            batch.append(mod.generate())
        except Exception:
            continue
    _, wrong, _ = verify_subset(sess, batch)
    if wrong:
        return False, f"fresh: {wrong}/{len(batch)} wrong"
    return True, "VALID+fresh"


if __name__ == "__main__":
    model = build()
    onnx.save(model, str(OUT))
    ok, msg = verify_all(model)
    print(msg)
    if not ok:
        raise SystemExit(1)
    r = json.loads(
        subprocess.check_output(
            ["python3", "-m", "neurogolf", "score-v4", "--model", str(OUT), "--task", str(TASK)],
            cwd=ROOT,
            text=True,
        )
    )
    mem, par = r["cost"]["memory_bytes"], r["cost"]["params"]
    print(f"cost={mem+par} mem={mem} par={par} pts={r['points']:.4f}")
```

## Live-solver + generator digest
## Live solver validation (framework gate)
- Model: `submission.zip/task014.onnx` · md5 `827aa1e8e550…`
- ✓ **train:** 3/3 pass
- ✓ **test:** 1/1 pass
- ✓ **arc-gen:** 262/262 pass
- ✓ **VALID** on every graded example — hidden-set-safe on visible splits.
- Hodel DSL fails 4 arc-gen graded example(s) but live solver passes — shipped ONNX already encodes the arc-gen variant; treat Hodel as structural hint only.

## ARC-GEN generator (competition oracle)
- Source: `external/ARC-GEN/tasks/task_0b148d64.py` · hash `0b148d64`
- **Signature:** `generate(width=…, height=…, colors=…)` (49 lines)
- **Spec:** Returns input and output grids according to the given parameters. Args: width: the width of the (square) grid height: the height of the (square) grid colors: a list of digits representing the colors to be used
- Fresh examples: `cd external/ARC-GEN && python3 arc_gen.py generate 0b148d64 5`



## DONOR TASKS (similar generator, already solved cheaply — MIGRATE the strong ones, treat weak ones as leads; see per-donor strength)
HISTORICAL donor prices are pre-scorer-fix context. Verify every donor with fast_verify; do not treat a historical score as a free or native-priced tensor.
- **task036** (generator similarity 0.361, scores 18.39) — **MIGRATE: near-identical — port its design directly**
  pin ops: Cast:6 Einsum:3 Equal:2 Log:2 Mul:2 Sub:2 Ceil:2 Where:2
  extract: `python3 -c "import zipfile;zipfile.ZipFile('$NEUROGOLF_CLEAN/submission.zip').extract('task036.onnx','.')"` and study how it stays cheap
- **task031** (generator similarity 0.325, scores 18.63) — **MIGRATE: near-identical — port its design directly**
  pin ops: Cast:9 Einsum:6 Div:5 Log:4 Add:4 Sub:4 Concat:3 Slice:1
  extract: `python3 -c "import zipfile;zipfile.ZipFile('$NEUROGOLF_CLEAN/submission.zip').extract('task031.onnx','.')"` and study how it stays cheap
- **task050** (generator similarity 0.14, scores 18.42) — **LOOSELY related — verify the mechanism transfers before relying on it**
  pin ops: Einsum:1
  extract: `python3 -c "import zipfile;zipfile.ZipFile('$NEUROGOLF_CLEAN/submission.zip').extract('task050.onnx','.')"` and study how it stays cheap

## Proven Kaggle corpus (exact per-member payment proof)
These are safe same-task starting points. Do not widen this list with merely gate-passing artifacts.
- `82122944` → **17.1593 pts**, proof: probe#804 gap+0.00
  iterate from: `$NEUROGOLF_CLEAN/knowledge_proven_paying/task014/82122944.onnx`
- `a390bd44` → **17.1593 pts**, proof: pin fd968f18@7369.66 (reconciled 2026-07-10)
  iterate from: `$NEUROGOLF_CLEAN/knowledge_proven_paying/task014/a390bd44.onnx`
- `a5e06276` → **17.1042 pts**, proof: probe#472 gap+0.00
  iterate from: `$NEUROGOLF_CLEAN/knowledge_proven_paying/task014/a5e06276.onnx`
Worker rule: iterate FROM a proven design of this task; novel op families with no proven ancestor go to solo probe.

## Hidden-set intel (exclusion oracle, Kaggle-proof-backed)
(not mined for this task yet — so treat fuzz verdicts carefully: fuzz-FAIL is NOT Kaggle-fail (036/105 paid after failing 2000-draw fuzz). A fuzz-fail alone must not kill an otherwise strong design; flag it for a solo/mega-batch probe instead.)

## Best gate-passing catalog designs for THIS task (study the mechanism, then beat it — never copy bytes)
(catalog query failed: no such table: onnx_graphs)

## Clean-era rule knowledge and failure history
Pre-fix dossiers are withheld rather than summarized: their scorer-era floors, prices, and
actions are not evidence. Generator/source evidence and exact Kaggle-paying corpus designs
remain available above.
### deep dossier (excerpt)
# task014 — Deep Dossier

> **ARC hash:** `0b148d64` · **Generator:** `external/ARC-GEN/tasks/task_0b148d64.py`  
> **Compiled:** 2026-06-30 · Sources: neurogolf_clean task folder, reasoning_traces (37 logs), reasoning_traces_patterns, neuroGEPA evidence.sqlite, pin_scores

---

## 1. Executive Summary

**Why high historical ROI (realized +0.66):** task014 is a **crop-select minform** on an **injective rarest-color bbox rule**. The original submission paid **8,451 B** (15.96 pts) with a dominant **3,600 B** `tgt_mask_f` fp32 Gather plane plus **2,400 B** all-color `[10,30]` presence profiles. A **June 2026 Einsum-first ladder** (w2290→w3031→w3119) collapsed profiles via `Einsum('nchw,c->nh/nw')`, killed the recolor plane with **dyn-palette**, and landed at **2,542 B** — one of the suite's largest single-task gains (**legacy-builder/max att13/14**, deepmin10).

**Current status (2026-06-30): EARNED FLOOR — FREEZE.** Live pin **L=2,542** (mem 2,507 + params 35) scores **17.1593 pts**. `gate_pair` **PASS** with `freshgen --secret` agree **1.0000**. Registerable +0.05 bar requires **cost ≤ 2,418**; no valid structural family clears it. **≥15 independent floor proofs** (2026-06-22 att16 through 2026-06-30 neurogepa ×4) converge: dominant **`canvas01` uint8 `[1,1,30,30]` = 900 B** is fixed by the `[1,10,30,30]` output contract; alternatives reintroduce **1,680–3,600 B** planes. Roster action: **FLOOR** (mechanical oracle). **Do not allocate build cycles.**

**Tension:** `deepmin13_last2days_patterns_20260630.md` still lists task014 at priority **#8** with **+0.2–0.4** theoretical ROI — **stale** post-w3119. Pattern mining priced headroom before the Einsum-selector breakthrough; post-floor the only remaining shaves are **scalar/palette trims < +0.05** (forbidden). **Trust trace tails:** 19/118 `trace_events` flagged `lazy_floor=1`; several filename `DONE_+0.265`/`+0.286` wins were later **falsified** in `facts` when superseded by w3031/w3119.

| Metric | Value |
|--------|-------|
| Live pin cost | **2,542** (mem 2,507 + params 35) |
| Live pts | **17.1593** |
| Best banked delta (cumulative) | **+0.6573** (w3119, vs 4,905 baseline) |
| Best single structural jump | **+0.6022** (w3031 Einsum-selector) |
| Cumulative pin lineage | **8,451 → 2,542** ≈ **+1.20 pts** realized |
| Theoretical queue ROI (stale) | **+0.2–0.4** (deepmin13 — unrealized) |
| Gate status | **PASS** — gate124 0 wrong; secret agree **1.0000** |
| Registry | 9 wins: **8 BANKED**, **1 REJECTED** |
| Trace index | **118** events; **49 DONE / 69 NO**; lazy-floor **~16%** |

---

## 2. ARC Rule / Generator Description

**Human rule (LARC/H-ARC):** Four rectangular corner patterns appear; three share one color, one differs. Output grid size = the unique pattern's size; fill with that unique pattern. LARC tags: `object_detection`, `spatial_relation`, `procedure`, `loop`. H-ARC median 54 actions.

**Generator `task_0b148d64.py` (verified across multiple fleet passes):**

```
generate(width, height, colors)
```

| Parameter | Role |
|-----------|------|
| `width`, `height` | Square grid dimensions |
| `colors` | Palette digits used in the example |
| Layout | Four quadrant patterns separated by a zero separator band |
| Rare color | **Least frequent nonzero color** in the full input |
| Output | **Tight bounding box crop** of the input around that color's pixels |

**Faithful ONNX inverse (competition variant):**
1. Count pixels per channel 1–9; pick **rarest** (min count; tie → lower color id).
2. Build row/col **presence profiles** for the selected color only.
3. `ArgMax` first/last present index → bbox `(first_row, last_row, first_col, last_col)`.
4. `Slice` the **fp32 one-hot input** by `[channel, bbox]` → dynamic crop.
5. Emit binary/ternary canvas `{0, rare, 255-sentinel}` padded to 30×30.
6. Free final `Equal(canvas, palette)` → `[1,10,30,30]` output.

**Critical structural insight:** The rarest color occupies **exactly one generator quadrant** (~10% hole density inside a near-solid rectangle). Output is therefore **binary in the crop domain** (background vs rare) — enabling the **dyn-palette** trick that bakes color into `pal[1,10,1,1]` instead of a spatial `colidx_crop` plane.

**Injectivity:** Task is **fully injective** — 0/266 graded examples fail injectivity (`facts`, 2026-06-23). No input-ambiguity blocker (unlike task209). Faithfulness gates are satisfiable; cost floor is the binding constraint.

**Hodel DSL:** PARTIAL — fails **4/262** arc-gen graded examples; live ONNX encodes the arc-gen variant (preserves original crop pixels, not LARC "fill with pattern"). Use Hodel for structural hints only.

---

## 3. Timeline of Attempts (Chronological)

| Date | Actor / Lane | Event | Verdict |
|------|--------------|-------|---------|
| **2026-06-05–08** | grind4/6 | uint8 ReduceMax bbox trim on 8,451 B baseline | **WIN** −180 B → 8,271 |
| **2026-06-08** | grind6 W26 | onnxslim graph dedup | **REJECTED** w266 Δ+0.0002 |
| **2026-06-12** | gpt-5.5 att1 | Initial exploration | NO |
| **2026-06-12** | legacy-analyst att2 | Gather-family attempt | NO |
| **2026-06-13** | gpt-5.5 att3 | uint8 bbox index arithmetic | **BANKED** w1319 Δ+0.0024 |
| **2026-06-15** | gpt-5.5-high att4 | tiered-deepmin rarest-color crop | **BANKED** w1884 Δ+0.0523 |
| **2026-06-16** | gpt-5.5-high att5 | 1D bbox profiles + fp16 count slice | **BANKED** w2192 Δ+0.0070 |
| **2026-06-16** | gpt-5.5-xhigh att7 | dynslice annotated exact bbox | **BANKED** w2290 Δ+0.265 |
| **2026-06-16** | legacy-builder att6 | Scout pass | NONE |
| **2026-06-17** | gpt-5.5-xhigh att8 | ReduceMax presence profiles | **BANKED** w2487 Δ+0.286* |
| **2026-06-18** | gpt-5.5-xhigh att9 | Gather re-tries | NO (lazy_floor=1) |
| **2026-06-19** | gpt-5.5-xhigh att10 | dynamic-crop value_info underprofile | **BANKED** w2759 Δ+0.1746 |
| **2026-06-20** | gpt-5.5-xhigh att11–12 | dynplane / deeper gather | NO (lazy_floor) |
| **2026-06-21** | legacy-builder/max att13 | **Einsum channel-selector** kills 2,400 B profiles | **BANKED** w3031 Δ+0.6022 |
| **2026-06-22 01:35** | legacy-builder/max att14 | **dyn-palette recolor** kills colidx_crop plane | **BANKED** w3119 Δ+0.6573 |
| **2026-06-22 01:51** | legacy-builder/max att15 | Premature floor claim (no tried{A,B,C}) | NO_LAZY |
| **2026-06-22 02:21** | legacy-builder/max att16 | Rigorous floor proof — canvas01 900 B fixed | NO (earned floor) |
| **2026-06-22** | deepmin10 #p494 | w3119 transferred on Kaggle bank | BANKED live |
| **2026-06-27–28** | gpt-5.5-xhigh neurogepa ×5 | Re-verify w3119; floor re-proofs | 4× DONE (reconfirm), 5× NO |
| **2026-06-28–29** | codex tiered/autonomous | Byte-identical dynpal rebuild; artifact sweeps | NO (floor) |
| **2026-06-29–30** | gpt-5.5-xhigh neurogepa ×10 | Priced families A–E; donor checks | ALL NO |
| **2026-06-30** | codex-autonomous-build ×3 | Final floor notes appended | NO |

\*w2487 registered Δ+0.0215 vs immediate predecessor; **+0.286** vs 7,776 deepmin era baseline per `wins_facts`.

---

## 4. What Was Attempted (Methods, Models, Families)

### 4.1 Model / family coverage

| Model | Sessions (trace_events) | Outcome |
|-------|-------------------------|---------|
| **gpt-5.5-xhigh** | 82 | 5 DONE (reconfirm w3119), 77 NO (floor era) |
| **legacy-builder/max** | 16 | 2 DONE (w3031/w3119 mega), 2 NO (floor) |
| **gpt-5.5-high** | 8 | 2 DONE (w1884/w2192), 0 NO |
| **gpt-5.5** | 8 | 1 DONE (w1319), 4 NO |
| **legacy-analyst** | 4 | 4 NO |
| **legacy-builder** | 1 | NONE (scout) |

### 4.2 Technique families tried

| Family | Description | Result |
|--------|-------------|--------|
| **A. Gather channel extract** | `Gather(input, tgt_color)` → fp32 `[1,1,30,30]` | **DEAD** — 3,600 B irreducible; grind4/6 era |
| **B. All-color presence profiles** | `[10,30]` fp32 row/col `ReduceMax` tables | **KILLED** by w3031 Einsum-selector (−2,400 B) |
| **C. Per-color ReduceMax profiles** | Selected-channel projections without Einsum | w2290/w2487 intermediate wins; superseded |
| **D. Einsum channel-selector** | `Einsum('nchw,c->nh/nw')` collapse | **w3031 BANKED** Δ+0.6022 — breakthrough |
| **E. Dyn-palette recolor** | `pal=Where(one-hot(tgt),1,base)` binary canvas | **w3119 BANKED** Δ+0.0551 vs w3031; current live |
| **F. One-hot crop → Pad** | 10-channel crop before final Pad | **DEAD** — 1,680–3,322 B; worse than 900 B canvas |
| **G. Selected full-plane** | Materialize `[1,1,30,30]` fp32 selected channel | **DEAD** — 3,600–6,169 B (dynplane/deeper) |
| **H. Deepmin uint8 mask** | Shared ramps, ArgMin count, mask×color | w1884/w1319 early wins; ladder superseded |
| **I. Mechanical** | onnxslim, onnxoptimizer, int64→int32 init | w266 **REJECTED**; 80 params max (< +0.05) |
| **J. Frankenstein donors** | task130 Conv sampler, task399 output-bank, task253 4×4 | **INAPPLICABLE** — fixed construction vs dynamic crop |
| **K. Separator/quadrant shortcut** | Python hypo: detect separator + quadrant | Valid rule check; **no cheaper ONNX** priced |

### 4.3 Validation gates used

- `validate_one.py` / `score-v4`: train 3/3, test 1/1, arc-gen 262/262 — **passes for live pin**
- `gate124.py`: **PASS** 0 wrong (visible ×2 + 300 fresh draws)
- `freshgen --secret`: agree **1.0000**, bad=0/300
- `gate_pair`: **PASS** on `_live_pin.onnx` / `task014_dynpal.onnx`
- Register threshold: **Δ ≥ +0.05** ⇒ cost **≤ 2,418** — **no candidate meets**

…[238 more lines in $NEUROGOLF_ARCHIVE/docs/deep_dossiers/task014_deep_dossier.md]

### graphs analysis (excerpt)
# Task014 ONNX Graph Analysis

**Generated:** 2026-07-04  
**Catalog:** `task_candidates_gate_pass/onnx_catalog.db`  
**Task:** ARC `0b148d64` — rarest-color bbox crop + dyn-palette recolor to `[1,10,30,30]`  
**Graphs analyzed:** 17 gate-pass variants  
**Score range:** 9.135 – 17.120 (spread **7.985**)

---

## Executive Summary

All 17 gate-pass graphs solve the same TRUE-RULE crop-select task (count channels 1–9 → pick rarest → ArgMax bbox → dynamic Slice crop → palette paint). Score improvements are driven by **representation paradigm**: Einsum 1D presence profiles ≫ ReduceMax/Gather profiles ≫ Gather/Reshape deepmin ≫ explicit Slice×62/Pad×58 tiling ≫ fp32-output legacy chains.

| Tier | Score | Representative | Architecture | Params | Ops |
|------|-------|----------------|--------------|--------|-----|
| **S** | 17.10–17.12 | `task014_axes_tail`, `task014_einsum` | Einsum×2 row/col profiles + ArgMax×4 bbox + dyn-palette | 30–35 | 35 |
| **A** | 16.66 | `7186.zip` | Einsum×2 + And/Or/Min bbox guards | 83 | 45 |
| **B** | 16.00–16.50 | `meta_crop1`, `maxprofile`, `dynslice`, `dynplane` | ReduceMax×2 + Gather×2 profile bbox | 19–26 | 31–35 |
| **C** | 16.00–16.04 | `deeper`, `deepmin`, `7167.zip` | Gather×3–4 + Reshape×4 + Cast bloat | 66–89 | 39–43 |
| **D** | 14.49 | `task014_w27` | Reshape×6 + Slice pre-pass legacy | 123 | 46 |
| **E** | 12.30–13.32 | `multisource_golf`, `opt`, `jonathanchan`, `tonylica` | fp32 output, Squeeze/Unsqueeze/Mul chains | 84–150 | 48–118 |
| **F** | 9.14 | `Documents__4.zip` | Slice×62 Pad×58 explicit per-cell tiling | 350 | 230 |

**Best catalog score:** **17.1203** (`task014_axes_tail`, sha `c27cec5b`)  
**Live pin:** **17.1593** (sha `3ebf97e7`, L=2,542 B) — **+0.039 pts** above catalog best  
**Best structural insight:** Einsum row/col presence profiles with dyn-palette `Equal` terminal outscore ReduceMax/Gather profiles by **+0.44–0.82 pts**, legacy Gather/Reshape by **+1.1–1.5 pts**, and explicit Slice/Pad tiling by **+7.98 pts**.

---

## Score Ladder (all 17 graphs)

```
17.120  task014_axes_tail     Einsum×2 ArgMax×4 dyn-palette uint8    35p   35op  2,442 B
17.104  task014_einsum        (identical op topology)                 30p   35op  2,396 B
16.664  7186.zip              Einsum×2 And/Or/Min bbox guards         83p   45op  2,862 B
16.502  task014_meta_crop1    ReduceMax×2 Gather×2 profile bbox       19p   31op  2,264 B
16.327  task014_maxprofile    (identical topology to meta_crop1)      19p   31op  2,264 B
16.306  task014_dynslice      ReduceSum×3 Gather×2 dynslice path      26p   35op  2,608 B
16.273  task014_dynplane      ReduceMax×2 Concat×5 dynplane           19p   32op  2,459 B
16.041  task014_deeper        Gather×3 Cast×9 deep gather             66p   39op  2,547 B
16.034  task014_deepmin       Gather×4 Reshape×4 deepmin              73p   43op  2,741 B
16.003  7167.zip              Gather×3 Reshape×3 submission           89p   39op  4,534 B
14.491  task014_w27           Reshape×6 Slice pre-pass legacy        123p   46op  3,281 B
13.320  multisource_golf      Gather×5 Mul×4 fp32 output             131p   49op  3,586 B
13.053  task014_opt           Mul×25 ReduceSum×10 MatMul×2 grind    150p  118op 16,038 B
12.416  jonathanchan blend    Squeeze×8 Gather×4 fp32 output          84p   48op  3,676 B
12.297  tonylica              Squeeze×8 Unsqueeze×6 fp32              103p   57op  2,662 B
12.297  submission_v32d       (identical topology to tonylica)       103p   57op  2,662 B
 9.135  Documents__4.zip      Slice×62 Pad×58 explicit tiling        350p  230op 25,614 B
```

---

## Architectural Families & What Drives Scores

### 1. Einsum Profile Selectors (Tier S) — +0.44 vs 7186, +7.98 vs worst

**`task014_axes_tail`** (17.120) is catalog best: 35 ops, 35 params, uint8 output, 2,442 B.

```
ReduceSum counts → ArgMin rarest → Einsum×2 (row/col presence) → Greater×2 → ArgMax×4 bbox → Slice crop → Pad → Equal(palette) → uint8 [1,10,30,30]
```

**vs worst (`Documents__4.zip`, 9.135):**
- `op_count_delta`: −195 (230 → 35)
- `size_bytes_delta`: −23,172 B
- Adds: Einsum×2, Equal×2, Where×1, ArgMin×1
- Removes: Slice×60, Pad×57, Unsqueeze×60, Reshape×7, OneHot×1
- `param_delta`: −315

**Near-duplicate pair** (`task014_einsum`, 17.104):
- Identical `op_types` dict (35 ops each)
- `axes_tail` scores +0.016 with +5 params (+46 B) — extra int64 inits (`slice_axes`, `pad_axes`) vs leaner einsum index set
- Both use opset 17/18, uint8 terminal, dyn-palette `pal[1,10,1,1]`

**Step improvement einsum ← 7186 (+0.440):**
- Drops: And×2, GreaterOrEqual×2, LessOrEqual×2, Min×2, Or×1, Reshape×1, Where×2, Neg×1
- Adds: Add×2 (leaner bbox concat path)
- `param_delta`: −53, `op_count_delta`: −10, `size_bytes_delta`: −466

**Lesson:** Collapsing all-color `[10,30]` presence planes into two Einsum contractions (`nchw,c->nh` / `nchw,c->nw`) is the decisive June 2026 breakthrough (w3031/w3119 lineage). Dyn-palette eliminates the 3,600 B `tgt_mask_f` Gather recolor plane.

---

### 2. ReduceMax/Gather Profile Cluster (Tier B) — plateau at ~16.27–16.50

**Core chain (31 ops, 19 params):**
```
ReduceSum → ReduceMax×2 → Cast → Slice → ArgMin → Gather×2 → ArgMax×4 bbox → Pad → Equal
```

**Functional equivalence class** (identical score topology):
- `task014_meta_crop1` (16.502)
- `task014_maxprofile` (16.327) — same 31-op / 19p structure, −0.175 pts (initializer/subtle path diff)

**Variants within cluster:**

| Variant | Score | Δ vs peak | Structural delta |
|---------|-------|-----------|------------------|
| `task014_dynslice` | 16.306 | −0.814 | ReduceSum×3, Gather×2, +7 params — annotated exact bbox |
| `task014_dynplane` | 16.273 | −0.847 | Concat×5, no Gather — spatial plane approach |

**Step improvement meta_crop1 ← maxprofile (+0.175):**
- Identical op topology — score delta from initializer/value_info compression, not op family change.

**Lesson:** ReduceMax 1D profiles are gate-valid but **0.6–0.8 pts below Einsum selectors**. Multiple independent builds (w2487 maxprofile, w2759 dynslice, w2290 dynplane) converge here before Einsum migration.

---

### 3. Gather/Reshape Deepmin Family (Tier C) — 16.00–16.04

Shared pattern: `Gather×3–4 → Reshape×3–4 → Cast×7–9 → Min/And bbox guards`

| Variant | Score | Gather | Reshape | Cast | Size |
|---------|-------|--------|---------|------|------|
| `task014_deeper` | 16.041 | ×3 | ×0 | ×9 | 2,547 B |
| `task014_deepmin` | 16.034 | ×4 | ×4 | ×9 | 2,741 B |
| `7167.zip` | 16.003 | ×3 | ×3 | ×7 | 4,534 B |

**Step improvement dynplane ← deeper (+0.232):**
- Drops Cast×6, Gather×3, And×1, Min×2
- Adds Concat×5, Slice×2 — simpler bbox without deep gather planes

**Lesson:** Pre-Einsum deepmin paths (w1884/w2192 era) carry 66–89 params and Cast bloat. Correct but superseded; 1.1 pts below Tier S.

---

### 4. Legacy W27 / Multisource Paths (Tier D–E) — 12.30–14.49


…[147 more lines in $NEUROGOLF_ARCHIVE/docs/tasks_analysis/task014_graphs_analysis.md]

## ⚠️ COST-MODEL TRUTH (scorer hack fixed 2026-07-10 — overrides ALL older advice)
The real Kaggle grader charges EVERY node-output tensor except the graph output. There
are NO "tagged scratch", "native-scored-away", or "skip_outputs" exemptions — those
existed only in a locally-hacked scorer, and models tuned to them SHORTED on the real
LB (probes #p807…: e.g. a "19.66-pt" relower design truly pays 12.36). Consequences:
- Only three things are free: the INPUT tensor, the graph OUTPUT tensor, and a single
  node's internal ORT workspace (e.g. one terminal Einsum). Everything else is charged.
- Do NOT copy `skip_outputs` / doc_string-tag tricks from older exemplars or build.py
  files — they buy nothing and hide the real price.
- Price every transient: a Slice→Cast relower pays the f32 transient (crop×4 B) PLUS
  the u8 result. It can still win, but only if the small plane is reused enough.
- Believe only `fast_verify.py` (grader-true since 2026-07-10) and `g.budget()`/`save()`
  (auto-charges all tmps). Any "free tensor" claim needs a Kaggle probe as proof.

## Your contract
1. Write `build.py` (in this folder) that emits `task014.onnx` — **use the ngolf
   builder library** (`import sys; sys.path.insert(0,"../../runner"); from ngolf import G`)
   — it emits valid ONNX, auto-clamps indices, tracks the exact grader cost as you build
   (`g.budget()`), and its terminal `out_*()` renderers exploit the free output tensor.
   For the COMPILE-THE-RULE family use `from ngolf_einsum import build_einsum_model,
   parity` (single-node terminal Einsum, repeated-input operands, dedup'd factors).
   Read `../../runner/ngolf.py` (short) and `../../data/DSL_TO_ONNX.md` (the combinator→
   cheap-idiom table for the Hodel program below). Raw onnx.helper is allowed but you
   own every trap yourself.
2. Check semantics BEFORE the gate: `parity(model, rule_fn, task=14)` (ngolf_einsum) on
   the real graded grids, then iterate:
   `python3 ../../runner/fast_verify.py 14 task014.onnx` — fix what it says. ≤6 rounds.
3. On PASS: `python3 ../../runner/fast_verify.py 14 task014.onnx --full`
4. On FULL PASS: copy .onnx + build.py to ../../results/task014/ and append to
   ../../results/LEDGER.csv: `task014,<cost>,<pts>,<Δ>,<lane>,<sha8>,<notes>`
5. Stop condition: 6 failed verify rounds on the same structural family ⇒ PIVOT family
   once; a second family failure ⇒ write ../../results/task014/NO.md with what you
   learned and move to the next task in your lane. EXCEPTION — a build that gates
   CORRECT and only misses the bar / class budget is NOT a NO: submit_result.py exits 3
   ("TICKET") and registers your graph as a factorization-ticket donor for the
   mechanical sweep (tickets/). Never delete a correct artifact.
