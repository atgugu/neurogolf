# compile_exemplars — few-shot corpus for COMPILE-THE-RULE

29 verified winners from the study7300 forensic study (7384 → 7431 member pairs), selected for one
property: **the rule is compiled into one terminal Einsum / factored contraction (or an
attribute-driven Conv equivalent) instead of being executed procedurally.** This library feeds the
COMPILE-THE-RULE section of attack packs and the circuit compiler: each `task{NNN}.md` is a
self-contained exemplar with (1) the SHA-verified true cost/points/delta, (2) the full graph dumped
from the actual 7431 bytes — every node in topological order with all attributes, every initializer
with values, every Einsum equation byte-verified — (3) a MECHANISM section explaining how the rule
is encoded, condensed from the forensic MDs and corrected against the bytes, and (4) a REUSABLE
MOVE a builder can copy.

`INDEX.json` lists per task: mechanisms, cost, points, delta, terminal op, terminal equation,
node count, params/memory. Task 267 is recorded there as skipped (no comparison MD, no rescore row,
no banked fleet win).

Cost model context: `points = max(1, 25 − ln(max(1, params + memory)))`; params = initializer
element count (reuse across operand slots is free); memory = bytes of charged intermediates
(graph input/output free; a single-node graph has memory = 0 exactly). This is why everything below
pushes work *inside* the terminal operator and pays only for tiny reused factors.

## The vocabulary (what these graphs are made of)

- **Input-repetition products** — feed the free input into the same Einsum 2–5× under shifted index
  roles; the contraction then computes products of grid values = relational/pairwise predicates
  (cell × row-mate × col-mate) or squared projections, at zero charged state.
  Exemplars: 293, 078, 052, 073, 100 (x⁴ rectangle moment), 373, 272, 304.
- **Tiny reused factor matrices across index roles** — one small matrix fed in many slots (color-in,
  color-out, row, col); reuse is free, so rank-factor everything: B^T·K·B color maps (073), shared
  geometry factor on rows AND cols (344, 287), one coordinate dictionary for reading and writing
  (114), packed mode dimensions routed purely by equation letters (001), one parity vector in three
  roles (373).
- **Boolean branches as scalar selector operands** — reduce to scalar moments, run
  Greater/And/Or/Where on 1-byte bools, Concat the scalars, and feed the result into the terminal
  Einsum as an operand that scales rows of a shared basis (188, 161, 231, 362); or compile
  Where(cond,+1,−1) directly as a 2-slot axis contracted with sign=[+1,−1] (304, 290's tcoef).
- **Integer-exact sentinels** — fp32 constants chosen so signs are exact and branches dominate:
  ±1 selector/clearing entries everywhere; 100 background default (293); ±1000 saturation (078,
  362 mix, 220's −1000/+1003.5 presence gate); 5000/50 dominance hierarchy (114); −100/−10 output
  suppression (254, 139); −0.001 ε-suppression so a count perturbs nothing at integer scale (362);
  9999 as an ArgMin-excluding infinity (310).
- **Trig/radix factor values** — color c embedded as (cos 2πc/10, sin 2πc/10, 1) with
  diag(1,1,−0.9) giving Equal-as-dot-product (114); a single 2×2 rotation matrix chained for phase
  recurrences (257); mod-k one-hot bases + cyclic-shift tensors HBk[D,g,d]=1 iff d≡g+D (mod k) for
  CRT positional translation (310, 304's mod/floor pair); stride-k comb factors L[a,s]=1 iff s≡a
  (mod k) replacing Conv dilation (321); floor(col/2) bar-binning (254); √2/0.707-scaled selectors
  (001); uint8 underflow computing i mod w (249); polynomial-of-measured-size replacing pattern
  LUTs (290, 231).
- **Factor dedup / mode-packing** — coarse×fine factorization of a dense position map (287),
  merging sibling factor banks into one tensor with an extra mode dim (001), duplicate rows shared
  inside a factor (344), one vector serving row and col roles (373).
- **Attribute geometry (the Conv corner of the family)** — dilation = structural offset,
  strides = decimation, negative pads = compute-and-crop, giant positive pads = placement:
  a crop→transform→paste pipeline with zero nodes of index math (296, 072, 006, 249's negative-pad
  probe, 205's pooled-ArgMax locator).

## Reading order

Big paradigm wins first: 293 (+3.21), 078 (+2.23), 052 (+1.87), 188 (+1.80), 310 (+1.61),
362 (+1.40), 296 (+1.25), 257 (+1.26), 099 (+1.04), 272 (+1.00), 114/161 (+0.96/+0.97) — then the
within-family compression polish (321, 220, 344, 287, 001, 373, 249, 290, 231) and the
memory-engineering pair (205, 072/006).
