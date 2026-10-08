# DSL → cheap-ONNX mapping (the fixed −339 channel)

Your pack's Hodel program is written in arc-dsl combinators. NEVER transcribe them
op-by-op (that channel scored −339). Instead: read the program to understand the rule,
then implement each combinator with its CHEAP equivalent below (ngolf calls given).
The input is ONE-HOT [1,10,30,30] fp32 and is FREE — exploit that everywhere.

| arc-dsl combinator | cheap ONNX realization (ngolf) | cost notes |
|---|---|---|
| `ofcolor(I,c)` / color mask | `g.color_plane(c, cast="u8")` — channel c IS the mask | 3600B f32 slice + 900B u8; crop first if certified smaller |
| `replace(I,a,b)`, `switch`, any recolor | `g.channel_permute(perm)` — recoloring = permuting one-hot channels | ONE op, 10 params, output can be FREE if terminal |
| `fill(I,c,patch)` / `underfill` | build patch mask (u8), terminal `Where`/compose then `out_equal_arange` | keep composition in u8; only the final op touches 10 channels |
| `paint(I,obj)` | mask-mul + add on u8 code grid, render terminally | never materialize per-object planes |
| `objects(...)` connected components | geodesic flood from seeds: `g.flood(seed, mask, R)` with R = CERTIFIED diameter | 2×900B per round — minimize R; or per-color planes when univalued |
| `mfilter/sfilter` by size/count | count via `ReduceSum` of the mask (i32) or `conv_int` local sums → scalar/vector compare | counts are tiny tensors — [1] or [10] |
| `argmax(objs, size)` "largest object" | per-color: `g.reduce(mask,"Sum",...)` → [10] vector → `ArgMax` → select channel via `gather` | whole selection in ≤44B of tensors |
| `ulcorner/lrcorner/center` | extents via `bbox_mask` internals: ReduceMax over rows/cols gives [H,1]/[1,W] vectors | vectors are 30B, never make planes |
| `backdrop/box/outbox` (bbox) | `g.bbox_mask(m)` — 6 vector ops + broadcast mul | ~1.1KB total incl. one output plane |
| `shift(patch,(di,dj))` | `Pad` then `Slice` (2 ops) or single asymmetric-pads `MaxPool k=1`? use Pad+Slice | 2×plane cost; fuse consecutive shifts |
| `hmirror/vmirror` | `Slice` with `steps=[-1]` on the axis | one op |
| `rot90/rot180/rot270` | `Transpose` + negative-step `Slice` | two ops |
| `dmirror` (transpose) | `g.transpose(x,[0,1,3,2])` | one op |
| `crop/subgrid` | `g.crop(x,h0,h1,w0,w1)` — do this EARLY (certified bounds) so everything downstream is small | biggest single lever |
| `upscale(I,k)` | `Resize` nearest (allowed op) or Gather with precomputed index map [H*k] | index vectors = params, cheap |
| `downscale(I,k)` | `Slice` with `steps=[k,k]` | one op |
| `hconcat/vconcat` | `g.concat([a,b],axis)` | — |
| `connect(a,b)` lines / rays | arange-vs-extent comparisons (like bbox_mask), NOT per-cell loops | vectors only |
| `frontiers` / full uni-color rows | row is uniform ⟺ ReduceMin == ReduceMax along the row → [H,1] vector | vectors only |
| `palette/mostcolor/leastcolor` | `ReduceSum` input over H,W → [1,10,1,1] counts → ArgMax/ArgMin | 40B + 8B |
| `colorcount` | same ReduceSum, pick channel | tiny |
| `gravitate/move-until-hit` | directional flood with obstacle mask, R capped by certificate | see flood |
| `cellwise(a,b,fallback)` | `eq` + `where` in u8 | — |
| `neighbors/dneighbors` counting | `g.conv_int(mask, kernel3x3)` → i32 counts; compare | i32 plane = 4B/cell — crop first, cast down fast |
| template/sprite match | `conv_int(grid, template)` == template_size, or Einsum vs template bank initializer | Einsum output can be a [K] vector — internal scratch is FREE |
| lookup by local pattern | ConvInteger with power-of-2 kernel (bit-encode 3×3 nbhd) → `gather(LUT, code)` | LUT is params; pack into few elements |
| final answer assembly | ALWAYS one of `out_equal_arange` (color codes) / `out_cast` / `channel_permute`-as-final | the output tensor is FREE — end on the expansion |

**Golden rules:** crop to certificate FIRST · compose in u8/bool · statistics as vectors,
never planes · the final op does the [1,10,30,30] expansion · print `g.budget()` BEFORE
gating and compare against your pack's register target.
