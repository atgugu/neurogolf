# Geometry → terminal-idiom routing (extracted from attack_analysis.md 2026-07-05)

POISON LABEL: the SOURCE document's per-task deltas/"Top 30" list are computed against the
stale July-4 pin (7277.65) and are PHANTOM against the current pin — never use it as a
target list. ONLY this routing table was vetted. Rows are priors from a 400-task
synthesis, not guarantees: anchor task IDs point at worked examples (open their pin
members), and some recipes have not yet won on their own anchor. Whatever family you pick,
verify the RULE with hypo.py before building, and price the design before writing ops.

## Terminal Idioms by Task Geometry

Use this table to **select the BUILD family** before touching ops. Wrong geometry → wrong idiom → months of grind.

| Geometry | Terminal idiom | Build recipe | Anchor analyses |
|----------|----------------|--------------|-----------------|
| Pair-product / Kronecker upscale | Rank-2 direct Einsum `naxy,as,xi,yj,...→nkrc` | Precompute pair tensor; single contraction | 001, 040, 011 |
| Periodic / cyclic pattern | 1-op Einsum pattern tensor | `chr,ws→chw` or cyclic embedding | 292, 298, 215 |
| Row/col uniformity | ReduceMax rowmax + Where | Axis reduce → broadcast mask | 003, 004, 052 |
| Bbox / object detect | ConvInteger front + ArgMax peaks | 3×3 Conv color planes → peak NMS | 008, 099, 121 |
| Frame / interior crop | Einsum profile + GridSample | Row/col count profiles → differentiable crop | **029** |
| Motif / template fill | Multi-tensor Einsum `nahw,tca,tcij,...` | Slice motif; contract coefficients | **033** |
| Quadrant mirror / reflect | Gather reverse-index OR ArgMax+Slice(−1) | 3×3 crop → index mirror → Pad | **142** |
| Flood fill / connectivity | BitwiseOr/BitShift schedule OR unrolled Gather×K | Bitset state machine, not MaxPool×40 | **243**, 002 |
| Line intersection / stamp | Axis ReduceMax + MaxPool stamp | Detect crossing → 3×3 yellow Where | **151** |
| Mul-bbox interior | Slice chK → ReduceMax² → Mul → Less → Where | Algebraic bbox; upgrade to Einsum | **166** |
| Rectangle shift / LUT | Einsum×1 shift tensor (480p) | Precompute per-offset LUT | **128** |
| Gravity / rank-sort column | 1-op grouped Conv direct output | Column counts → rank rows | 078 |
| Hollywood panel + upscale | Triple-Einsum panel contract | Sub uniqueness + rank-3 stencil | 011 |
| Component count → identity | Einsum count + Scatter diagonal | Connected components → N×N diag | 325 |
| Kaleidoscope mirror | ArgMax+Gather one-hot mirror | Drop Reshape/GridSample rebuild | **083** |
| Card lattice stamp | QLinearConv assemble OR Einsum plane-kill | Detect centers → stamp template | **080** |
| Dual-mask intersection | Dual-Pad Cast + And/Where | ch0/ch1 threshold masks | **043** |

---
