# Circuit-feature priorities for compile_circuit (Engine C)

> **FORBIDDEN CHANNEL — READ FIRST.** Importing/tracing the **Hodel DSL backend
> into ONNX is the historical −339 channel and is BANNED.** This document maps DSL
> primitives to circuit-feature families for **coverage prioritization only**: the
> compiler re-implements semantics as dense truth-tensor factors inside one Einsum.
> It must **never** import, trace, or transpile arc-dsl code.

Source: `/workspace/repos/NeuroGolf/03/solver_matches.json` — 399 verified
Hodel-solver matches (of 400 tasks; task361 unmatched), 151 distinct DSL
primitives, 3559 total occurrences. Companion data: `circuit_priorities.json`.

**Method.** Every primitive was mapped to one of four circuit-feature families
(ambiguous calls documented in the JSON). `functional_scalar_glue`
(compose/fork/rbind/apply/arithmetic/container ops — 1738 of 3559 occurrences) is
**spec-generation-time** work done by the LLM writing the spec, never a runtime
circuit feature, and is excluded from coverage sets. A task counts as *fully
covered* by a family set when its whole primitive set falls inside those families
plus glue.

## Family coverage (399 matched tasks)

| family                  | occurrences | tasks touching | fully covered alone |
|-------------------------|------------:|---------------:|--------------------:|
| coordinate_geometry     |         693 |            267 |                  38 |
| percell_recolor_palette |         677 |            328 |                  20 |
| object_level            |         400 |            255 |                   6 |
| neighborhood_morphology |          51 |             40 |                   0 |

Greedy full-task unlock order: **coordinate_geometry (+38) → percell (+92, cum 130)
→ object_level (+229, cum 359) → neighborhood_morphology (+40, cum 399).**
Most common family combos: `geometry+object+percell` 105 tasks,
`object+percell` 91, `geometry+percell` 72, `geometry` alone 38.

## Ranked circuit features (support next, in order)

| # | feature | key DSL primitives (freq) | occ | status |
|---|---------|---------------------------|----:|--------|
| 1 | per-cell color masks & fixed recolor | ofcolor 116, replace 54, switch 9 | 184 | **SHIPPED v1** — proven on tasks 337/309 (all graded pairs exact) |
| 2 | masked render + keep-original ("else keep") | fill 166, paint 111, canvas 46, underfill 31, cover 23 | 395 | **SHIPPED v1** for per-cell masks (reserved `color` output wire); object/geometry masks pending |
| 3 | global palette statistics | leastcolor 37, mostcolor 19, palette 16 | 98 | NEXT — global color-count aggregation; unlocks remaining percell-only tasks (e.g. 267, 389) |
| 4 | coordinate factors: mirrors/rotations | vmirror 29, hmirror 24, rot90/180/270 45, dmirror 15 | 118 | NEXT — (30,30) permutation factors on h/w labels; same-shape geometry fits one Einsum directly |
| 5 | crop/concat/scale reshaping | subgrid 54, crop 31, vconcat 30, hconcat 24, upscale 23 | 255 | rectangular position factors; static per-task shapes |
| 6 | position/extent reads | shift 42, width 30, ulcorner 25, height 25, shape 22 | 298 | shift is cheap off-diagonal factors; extents need row/col aggregation |
| 7 | neighborhood adjacency literals | neighbors 14, dneighbors 3 | 17 | the documented v1 extension point: shifted free-`input` operands (input operands are FREE; only shift factors billed) |
| 8 | bbox/outline morphology | box 8, inbox 5, outbox 4, delta 4, backdrop 4 | 28 | needs min/max extent aggregation; family never occurs alone — after 4–7 |
| 9 | connected components | objects 214 (the #1 primitive), fgpartition 16, partition 15 | 245 | RESEARCH — transitive closure is hostile to one Einsum; biggest greedy payoff (+229 tasks) |
| 10 | object selectors & filters | colorfilter 54, sizefilter 28, normalize 20 | 155 | depends on 9 + thermometer-coded counting wires |

## Cellwise-v1 reality check (scan evidence, 2026-07-12)

**None of the 162 hypo-verified `packs/task*/rule.py` files is a pure per-cell
fixed-color-map rule.** All 162 were tested on 20 random grids plus up to 40 real
graded grids: 126 ran clean, of which ~31 degenerate to a fixed map on random
noise (mostly identity — their structural triggers never fire on noise) but every
one diverges on real grids; 36 crash or time out on random grids (structural
assumptions — a true per-cell rule cannot crash on a random grid). The
non-identity fixed-map candidates (tasks 161, 222, 226, 259, 277, 379) were each
checked explicitly against all real graded grids: 100% mismatch.

Pure per-cell recolor tasks DO exist in the solver universe **outside** the
verified-162 set — task016 (multi-pair switch), task276 (switch 2↔6), task309
(replace 7→5), task337 (switch 5↔8); each fixed map fits **all** 265–267 graded
input/output pairs (train+test+arc-gen). 337 and 309 were compiled and verified
end-to-end (100 params, exact on every graded pair); note fast_verify shows both
already hold better pins (cost 10), so they are compiler proofs, not banking
candidates.
