# Exemplar recipe cards (advisory — pick few-shot by fit)

## build_task019_einsum.py
MECHANISM: BBox-periodic tile, diagonal corner stencil, channel-index map to bool output
REPRESENTATION: Mod-grids plus Gather emulate tile; QLinearConv applies corner stencil; Pad and Equal emit bool channels
COST-PROFILE: Int32 row/col grids and thirty-channel Pad inits; graph ops stay tiny, no trained weights
TRANSFERS-WHEN: Seed channel fixes tile period; fixed output grid; corner stencil on foreground; final answer is per-pixel channel bool equality

## build_task132_w15u8.py
MECHANISM: Same-color diagonal corners define rectangles; separable row-col-color fill.

REPRESENTATION: W=15 moment Einsums, ArgMax fg pair, Pad 15→30, triple Einsum composes row×col×color output.

COST-PROFILE: Constants and Pad expansions; width capped at W=15 until final Einsum.

TRANSFERS-WHEN: Two same-color corners bound axis-aligned rects; fill separable as row×col×palette; grid ≤30.

## build_task173_nbrconv.py
MECHANISM: Label interior grid, test eight-neighbor symmetry, propagate fills via scatter-gather pools.
REPRESENTATION: Eight-channel 3x3 neighbor Conv with unit taps gathers offsets; replaces padded shift slices.
COST-PROFILE: Node count in ScatterND/Gather/MaxPool; weights stay tiny conv kernels.
TRANSFERS-WHEN: Ten-channel grids with interior conv labeling and pairwise neighbor equality driving axis fill rules.

## build_task243.py
MECHANISM: Flood zeros 4-connected to a marker channel via bounded morphological reconstruction on 18×18 crop.

REPRESENTATION: Separable 3×1/1×3 MaxPools dilate; Min-with-mask iterates geodesic growth; bool casts drive ch0/ch1 XOR-OR update.

COST-PROFILE: 28-step dilation loop dominates nodes; 18×18 crop; nine tiny slice/pad initializers, zero learned weights.

TRANSFERS-WHEN: New task is 4-connected flood or reconstruction from one seed channel against a mask, grid ≤18, W≈20, multi-channel bool grid with passthrough tails.

## won_task002_codex.py
MECHANISM: Enclosed-region fill on one-channel binary mask using packed-row morphology.
REPRESENTATION: Pack W≤20 pixels per row into uint32; neighbor ops via Gather (vertical) and BitShift (horizontal).
COST-PROFILE: Node budget in repeated close passes; weights are only small index/mask constants.
TRANSFERS-WHEN: Binary wall mask in one input channel, fixed R×W grid, fill interiors reachable from border seeds via morphological closing.

## won_task012_codex.py
MECHANISM: Symmetric cross from two colors distinguished by global per-color pixel counts.

REPRESENTATION: Skip color plane: count-eq channel pick, dynamic Slice, QLinearConv feature stamp, padded terminal expand.

COST-PROFILE: Savings skip 12x12 fp32 plane; cost sits in small feature-mask and terminal int8 weights.

TRANSFERS-WHEN: Two colors with distinct global counts, fixed center window slice, local conv-stampable motifs, one-shot padded upsample to full output.

## won_task012_alt.py
MECHANISM: Center and arm colors pick stamp; motif applied on 12×12 crop.
REPRESENTATION: Dynamic QLinearConv builds 5×5 kernel; ConvInteger scores valid, code, code² one-hot equality.
COST-PROFILE: Bytes in stamp masks, ten-way equality weights, valid plane; lean graph.
TRANSFERS-WHEN: Two-color stamp rule, fixed small motifs, native crop, terminal per-color match without sentinel pad.

## won_task013_codex.py
MECHANISM: Alternating stripes from scalar row/column moments; height under thirteen picks axis.

REPRESENTATION: Einsum moments yield stripe phase and colors; channel-minus-scalar row/column codes, terminal broadcast Equal.

COST-PROFILE: Fixed 30×30×10 pads and moment weights; Einsum reductions, not per-cell LUTs.

TRANSFERS-WHEN: Recoverable stripe period and two colors from first/second positional moments; axis flips by height threshold.

## won_task014_codex.py
MECHANISM: Least-frequent nonzero color tight bbox crop on binary region
REPRESENTATION: Pad binary crop to uint8 sentinel canvas; Equal with dynamic palette yields ten bool channels
COST-PROFILE: Tiny inits; bbox Einsum/ArgMax nodes; large bool output
TRANSFERS-WHEN: Rarest-color rule, binary bbox inside crop, fixed pad grid, ten-way bool channel output

## won_task029_codex.py
MECHANISM: Bottom-pack masked foreground cells in fixed-size grids.
REPRESENTATION: Boolean mask times per-row cumsum gather re-lowers survivors to bottom rows.
COST-PROFILE: Activations dominate; tiny mask constants, no learned weights.
TRANSFERS-WHEN: Same grid size, static background, output is column-preserving bottom-packed foreground.

## won_task035_codex.py
MECHANISM: Cyan-marker layout drives native uint8 grid, edge nibble pack, Max merge, 30×30 pad.
REPRESENTATION: Concatenate grid and grid²; quantized 1×1 conv uses quadratic scores to decode ten channels.
COST-PROFILE: Init tensors dominate—onehots, pack windows, qconv weights; graph is uint8 glue.
TRANSFERS-WHEN: Marker-defined fixed canvas; distinct per-color codes; marker ordering enables Max overwrite; terminal quadratic decode.

## won_task037_codex.py
MECHANISM: Same-color diagonal endpoint pairs; paint the closed segment on a certified native grid.
REPRESENTATION: One terminal Einsum pairs endpoints, tests diagonal betweenness via shift/orient tables, pads to contract.
COST-PROFILE: ~1300 init-weight floats only; final Einsum emits output, no charged intermediates.
TRANSFERS-WHEN: Native grid smaller than padded contract; segments on 45° diagonals; colors mark endpoint pairs.

## won_task037_alt.py
MECHANISM: Top-K row-ranked scatter slots with covariance spans painted on a certified 10×10 grid.

REPRESENTATION: Native 10×10 ScatterElements bank; terminal ConvInteger implicit-pads to 30×30 quadratic color one-hot without charged Pad.

COST-PROFILE: Bytes live in 10×10 u8 feature/scatter planes, TopK slots, and tiny idx/eq_weight params—not full 30×30 tensors.

TRANSFERS-WHEN: Reuse when objects rank by row projection, spans come from rs/cs/rcs covariance, and output is padded 0–9 one-hot via quadratic ConvInteger match.

## won_task042_codex.py
MECHANISM: Detect magnify m=1/2/3 on sliced green ROI; stamp hits; terminal-render output colors.

REPRESENTATION: QLinearConv detect-and-stamp feeds green2+hit2 concat into ConvInteger LUT; x_zero_point=1 zeroes padding.

COST-PROFILE: direct23 13×13 classifier kernel dominates; smaller m1 detect, stamp, render weights.

TRANSFERS-WHEN: Discrete magnify motifs in fixed green subgrid; palette from green-count plus hit class; ORT ConvInteger xzp padding must pass sentinel.

## won_task044_codex.py
MECHANISM: Preserve oracle decision graph through out10; replace charged Pad terminal with free render.

REPRESENTATION: Terminal ConvInteger on BitShift-Mul-Concat features renders and pads without materializing charged bool Pad.

COST-PROFILE: Sheds priced uint8 out30; bytes stay in nodes 0–136 plus small render initializers.

TRANSFERS-WHEN: Trusted donor ends at out10 with Pad-to-Equal; final bool pad needs standard-ONNX equivalent; hash-verified oracle exists.

## won_task046_codex.py
MECHANISM: Multi-segment grids: drop black separators, recolor gray bridges, align on anchor columns.

REPRESENTATION: QLinearConv/ConvInteger 1×1 terminal; quadratic 1−(k−code)² via color, color² features and y_zero_point bias.

COST-PROFILE: Bytes in u8 column LUTs, bool gates, int8 1×1 terminal weights—not intermediate feature maps.

TRANSFERS-WHEN: Reuse when a polished slice-shift body yields per-cell color codes and output needs finite-palette decode with padded 1×1 conv terminal.

## won_task050_codex.py
MECHANISM: Endpoint color preserved; cells strictly between paired markers on same row or column recolor.

REPRESENTATION: Relowered one-hot plane; opposing asymmetric MaxPools; min for both-sides; eq equality-folds row/column between masks.

COST-PROFILE: Four padded MaxPools on 15×15 crop; bool Pad upscale; one Where recolor scatter.

TRANSFERS-WHEN: One-hot endpoint channel, croppable grid, axis-aligned between-fill from bilateral MaxPool witness, constant fill color.

## won_task051_codex.py
MECHANISM: Singleton marker in colored region; shoot its color along emitter axis through background only.

REPRESENTATION: relower_onehot_plane feeds terminal_hodel_ray with row/col forward masks from centroid-minus-marker axis.

COST-PROFILE: ~900 bytes in 30×30 uint8 background plane; remainder bool axis masks and scalar flags.

TRANSFERS-WHEN: Exactly-once marker color, centroid-defined ray from marker, background-only paint path, Hodel terminal under dense Einsum.

## won_task055_codex.py
MECHANISM: ArgMax-detected row/column separators partition grid; four terminal terms classify panels

REPRESENTATION: Edge-strip ArgMax yields cumulative step masks; single rank-4 Einsum applies four baked terminal terms

COST-PROFILE: Heavy baked Einsum weights; lightweight separator detection via edge relower and ArgMax

TRANSFERS-WHEN: Same ten-channel 30×30 grid; separators on eight-channel edge crops; logic decomposes to four low-rank terms

## won_task061_codex.py
MECHANISM: Highest input channel selects period P; output one-hot via residue bitmasks.
REPRESENTATION: Offline int64 bit LUTs; ArgMax→Gather row mask; BitwiseAnd broadcasts one-hot answer.
COST-PROFILE: Nearly all bytes in rowmask and channel target constant tensors.
TRANSFERS-WHEN: Reuse when output is AND of precomputable row/channel bitsets and P is readable from input.

## won_task062_codex.py
MECHANISM: Red-axis index and neighbor presence gate LUT object remap into box paste.
REPRESENTATION: Relower u8 planes; ArgMax red line; Gather neighbor keys feed 17×8 LUT; ConvInteger 20-pad box terminal.
COST-PROFILE: Static LUT table, ten-channel ConvInteger weights, heavy 20-pad terminal halo.
TRANSFERS-WHEN: Reuse when placement hinges on red-line coordinates, lateral adjacency probes, and padded ConvInteger box copy.

## won_task063_codex.py
MECHANISM: Interior grid expansion via row/column color counts matching sqrt(area) minus two.

REPRESENTATION: Shared rank-2 color_basis drives count Einsums and terminal render; bg_axis picks background row.

COST-PROFILE: Charged bytes in 30-wide counts and rank-2 basis masks, not input crop tensor.

TRANSFERS-WHEN: Lane-A budget; color row/column band logic with sqrt(area)-2 target; shared rank-2 f32 terminal Einsum renderer.

## won_task066_codex.py
MECHANISM: Route between colored anchors; three disjoint segments; 20→30 pad; black-only rewrites.

REPRESENTATION: Terminal Einsum outer-product: one-hot input × row factors × column factors × channel-mix tensor.

COST-PROFILE: Drops bool painter, Pad tail, cyan indexing; pays small f32 constants plus Pad/Cast/Einsum.

TRANSFERS-WHEN: Coordinate route solver exists; segments are row/column masks; orientation flag; black→color overwrites only; float output OK.

## won_task069_codex.py
MECHANISM: Erase source sprite; stamp normalized template colors onto every same-shaped cyan duplicate at bbox TL.

REPRESENTATION: relower_onehot_plane input; QLinearConv correlation and stamp cascade; z²+z stamp_code decoded by 1×1 tail conv to one-hot.

COST-PROFILE: Bytes in relower plane, Einsum projection, three QLinearConvs; avoids dense bbox-shift mask tensors.

TRANSFERS-WHEN: One-hot I/O contract; template ≤3×4; disjoint cyan copies share template shape; correlation anchors at copy bbox top-left.

## won_task071_codex.py
MECHANISM: Mirror-symmetric grids with unknown flip; classify clean half via mirrored probes, reflect output.

REPRESENTATION: Three-pixel GatherND with batch_dims=2; Where-built reflection index; terminal axis-3 Gather remaps columns.

COST-PROFILE: Bytes sit in 1×10×30×30 I/O; tiny int64 probe indices; terminal Gather is cheap.

TRANSFERS-WHEN: Unknown horizontal flip; symmetry axis from top row; mirrored probes identify clean side; output is column reflection remap.

## won_task075_codex.py
MECHANISM: Marker-masked template fill with packed 3-bit native state and terminal ConvInteger.

REPRESENTATION: Colors as polygon-packed u8 scalars; BitwiseAnd splits three bit-planes; ConvInteger classifies pixels.

COST-PROFILE: Terminal padded ConvInteger and int8 weights dominate; compact color-code tables.

TRANSFERS-WHEN: Palette maps to separable 3-bit features; marker crop seeds template; full grid rendered by ConvInteger.

## won_task084_codex.py
MECHANISM: Anchor/background grid recolor with two accent terminals ruled by row, column, and size
REPRESENTATION: One terminal Einsum fuses four selector-weighted linear forms over baked coord rows and sqrt-area size features
COST-PROFILE: Tiny coeff and selector inits; activation peaks in the full-resolution fused Einsum pass
TRANSFERS-WHEN: Reuse when channel in/out maps match, two conditional accent terminals exist, and colors depend linearly on row, column, and grid area

## won_task085_codex.py
MECHANISM: Middle-band row-code detection with column parity and punched-cell terminal recoloring.
REPRESENTATION: One fused rank-3 Einsum factorizes 1-(k-j)² terminal routing with row/col mask tables.
COST-PROFILE: Bytes sit in static channel bases, row/col selectors, and collision-free code weights.
TRANSFERS-WHEN: Same K-channel grid; middle rows match code-top/bot test; parity columns; quadratic punch-to-background terminals.

## won_task086_codex.py
MECHANISM: Nested frame recoloring from occupancy stencil plus dominant inner/outer hues.

REPRESENTATION: Concatenate disjoint u8 masks; assemble three-column i8 kernel via Equal+Cast TopK picks; terminal QLinearConv fuses.

COST-PROFILE: Frozen eleven-by-eleven outer stencil; negligible scalars; dynamic kernel built each forward pass.

TRANSFERS-WHEN: Reuse when ring-or-frame geometry, occupancy masks, and two dominant palette colors drive a three-way terminal composite.

## won_task088_codex.py
MECHANISM: Detect four-marker color, derive rectangle bounds, recolor interior non-background cells.

REPRESENTATION: Masked coordinate Einsums recover bbox extrema; dynamic QLinearConv ±1 weights mask valid cells and paint marker.

COST-PROFILE: Heavy padded QLinearConv dominates; tiny init tables for coords and base weights.

TRANSFERS-WHEN: When one palette color appears exactly four times as rectangle corners on a fixed grid with distinct interior background.

## won_task089_codex.py
MECHANISM: Oriented 5×5 pattern stamp from red/green marker anchors onto payload grid.

REPRESENTATION: QLinearConv gathers sparse marker neighborhoods; payload² equality filters matches; mirrored Slice orients red.

COST-PROFILE: QLinearConv 13×13 and 5×5 kernels dominate; dilated decode and one-hot pad tail.

TRANSFERS-WHEN: Sparse colored markers, fixed 5×5 oriented copy, asymmetric mirror per role, payload match by local color square.

## won_task092_codex.py
MECHANISM: Same-color endpoint pairs joined by horizontal or vertical sticks; vertical wins at crossings.

REPRESENTATION: Reuse moment/bbox extractor; TopK five sticks; six-slot valid-rectangle intervals; one terminal fp16 Einsum.

COST-PROFILE: Stick metadata and six-slot interval masks dominate; coeffs and h/w scalars marginal.

TRANSFERS-WHEN: Colored endpoints extract via moments, sticks are axis-aligned rectangles, count ≤5, crossing priority is coefficient-weighted, ONNX budget tight.

## won_task093_codex.py
MECHANISM: Orient to gray separator; count each seven-cell half; pack toward divider; emit 0/5.

REPRESENTATION: Ch0 relower crop; QLinearMatMul uint8 half bg counts; Where picks orientation; ZP rank thresholds; ConvInteger tail.

COST-PROFILE: Repeated 196 B uint8 planes dominate; ~119 B params; orientation probes negligible.

TRANSFERS-WHEN: Fixed seven-cell halves, gray-line orientation, per-half background tally, rank-pack toward separator, two-color output—without dense orientation selectors.

## won_task094_codex.py
MECHANISM: Row/column blue counts find hollow 5×5 centers; intersection gates sparse recolor.

REPRESENTATION: Uint8 QLinearConv for 1D sums and 5-tap scoring; y_scale=43 binarizes; Einsum remaps palette.

COST-PROFILE: Uint8 conv kernels dominate; floats only in micro selectors and final Einsum.

TRANSFERS-WHEN: Reuse when targets are 1D-projectable, a fixed 5-weight sum crosses one threshold, and paint rules use row/col hit pairs.

## won_task096_codex.py
MECHANISM: Recover clipped L-ring radius and arm length; redraw concentric rings on background.

REPRESENTATION: Masked-pixel Einsum variances yield vmax signature; hash LUT plus nine collision predicates decode idx-length.

COST-PROFILE: Reuse pin color recovery and terminal renderer; decoder cost is compact LUT plus fix nodes.

TRANSFERS-WHEN: Clipped corner L-rings with discrete idx-length pairs, color-separated fragments, and an existing pin supplying background plus terminal redraw.

## won_task098_codex.py
MECHANISM: Hollow filled rectangles: keep border cells, drop same-color interiors, mark black witnesses.
REPRESENTATION: One group=10 Conv; each channel owns a hand-tuned 15×3 padded stencil for its rule.
COST-PROFILE: All bytes in single grouped Conv kernel weights and biases.
TRANSFERS-WHEN: Per-color hollow fills, 4-neighbor surround test, optional vertical black-interior witness, ≤10 isolated channels.

## won_task099_codex.py
MECHANISM: Gated marker-label cases enumerated as scalar uint8 logic, grafted onto certified Pad→Equal renderer.

REPRESENTATION: Keep pin head/tail; replace broad middle with Split plus Max/Min/Mul/Sub wiring a flat 10×10 case-id canvas.

COST-PROFILE: Bool Pad→Equal tail dominates bytes; savings from excising broad LUT middle for tiny scalar circuit.

TRANSFERS-WHEN: Pin already has Pad(clear)→Equal tail; few marker/gate tensors split into finite enumerable uint8 cases on a small grid.

## won_task102_codex.py
MECHANISM: Gated QLinearConv detects multi-size hollow square frames; merges interior fills.

REPRESENTATION: Two-plane 12x12 state; terminal ConvInteger zero-point decode plus 18-pad emits 30x30 free.

COST-PROFILE: Frame QLinearConv kernels and MaxPool fusion; no charged 30x30 Pad or Where.

TRANSFERS-WHEN: Hollow squares need interior fill; palette maps via 1x1 ConvInteger on compact state with pad-upscale, not explicit Where.

## won_task105_codex.py
MECHANISM: Interior row-hash selection with border guards driving conditional grid repaint

REPRESENTATION: Shrink guard-correct row-hash ONNX via uint8 relative coords, int16 bitmask folds, GatherElements row sharing, explicit Pad axes

COST-PROFILE: uint8 coord inits and int16 power masks, not float Compare subgraphs and i32 bit chains

TRANSFERS-WHEN: Base ONNX is guard-correct row-hash with unchanged Where paint renderer; grid coords fit uint8, interior bitmask veto logic matches

## won_task110_codex.py
MECHANISM: Periodic tile restore: detect smallest row period 4–9, exact columns else residue backfill.

REPRESENTATION: Gather period residue IDs, equality-cast to basis, terminal Einsum—not banked [6,30,9].

COST-PROFILE: ~1186B params in tables/IDs; ~1500B runtime, mostly [30×9] f32 basis.

TRANSFERS-WHEN: When task picks among small periods, hides columns, and backfills by c mod p via terminal Einsum.

## won_task115_codex.py
MECHANISM: Centroid spread picks row/column mode; four smallest-axis channels define 4×4 block.

REPRESENTATION: Einsum coordinate-weighted centroids, FP16 counts, TopK+Gather block, Equal+Pad to BOOL grid.

COST-PROFILE: Input and padded BOOL output tensors dominate; gather tables and coords are negligible.

TRANSFERS-WHEN: Fixed H×W grid, ~10 channels, boolean masks from centroid-ordered 4×4 channel tiling.

## won_task118_codex.py
MECHANISM: Adaptive plus-footprint red scoring, local-max centers, conditional cyan overlap paint.

REPRESENTATION: 1×1 Conv decodes colors; QLinearConv x_zero=64 scores plus kernels and paints via Greater comparison.

COST-PROFILE: Bytes in tiny 1×1 weights and 7×7 plus stamps; logic in QLinearConv chains.

TRANSFERS-WHEN: Plus neighborhoods, red-gray-black semantics, global radius pick, local-max peaks, stamp gray-not-red overlaps to paint.

## won_task124_codex.py
MECHANISM: Pattern-conditioned fg row gather into u8 code plane, ConvInteger full-canvas render
REPRESENTATION: Replace fp32 Slice/Less crop and bool Cast tail with relower u8 crop, doubled-add 0/2 code, ConvInteger
COST-PROFILE: Mostly u8 tensors—crop masks, offset map, code rows—with ConvInteger weights and color params at end
TRANSFERS-WHEN: Reuse when task has small fg/bg crop, row-variant bottom via Gather map, terminal ConvInteger pad-expand—not bitset unpack

## won_task131_codex.py
MECHANISM: Upscale compact latent grids via local color propagation and fill.
REPRESENTATION: ONNX ConvTranspose expands active feature channels into full output resolution.
COST-PROFILE: Transpose-kernel weights dominate; encoder backbone stays tiny.
TRANSFERS-WHEN: Fixed upscale ratio, low palette, localized edits—not full-grid remap or pixel-copy transforms.

## won_task132_codex.py
MECHANISM: Fill axis-aligned rectangles from paired same-color corner markers on variable native extent.

REPRESENTATION: Einsum row/col coordinate moments recover native H×W; fp16 rank-4 terminal contraction renders rectangles without 15×15 validity scratch.

COST-PROFILE: Bytes concentrate in fp16 Einsum tail and moment reductions; savings from dropping ONNX-billed 15×15 SSA validity tensors.

TRANSFERS-WHEN: Reuse when one-hot markers form opposite-corner rectangle pairs and native H×W is recoverable from per-channel coordinate moment sums.

## won_task133_codex.py
MECHANISM: Recover 3×3 creature from one tile; paint magnified partials with signature/body colors.
REPRESENTATION: Replace integer Einsum with chained QLinearMatMul uint8 outer products; int8/uint8 mask-and-lookup geometry front.
COST-PROFILE: Bytes in uint8/int8 initializers and masks; prunes dead float counters and value_info.
TRANSFERS-WHEN: When proven ONNX needs Einsum→QLinearMatMul relower, row/column mask tiling, and magnification-aware sprite painting on a fixed grid.

## won_task134_codex.py
MECHANISM: Magnified Conway sprite: detect dominant color, stride-sample 3×3 macro, paint alternate.

REPRESENTATION: Scaled fp16 channel moments pick sprite color; Slice samples 3×3; rank-3 broadcast paints cells.

COST-PROFILE: FP16 ten-channel selector scratch dominates; projections and tiny 3×3 tile cheap.

TRANSFERS-WHEN: Reuse when magnification is bbox-inferred, macro motif is input-dependent not LUT-keyed, ten color planes.

## won_task136_codex.py
MECHANISM: Two 2x2 anchors drive per-row intervals; colors extend northwest and southeast.
REPRESENTATION: UINT8 false-flag planes (0 active, 2 idle) fused by ConvInteger zero-point=1 into three channels; pad upscale.
COST-PROFILE: Bytes sit in padded ConvInteger INT32 output; no bool masks or scaffold.
TRANSFERS-WHEN: Reuse when sparse block codes set dynamic row intervals and two opposite-direction fills upscale via conv padding.

## won_task137_codex.py
MECHANISM: Grid perimeter painting from marker median center and Chebyshev spacing via separable templates

REPRESENTATION: u8 Concat-then-Cast separable templates; scalar offset reuse; 0.99 rank ratio deletes tie vectors

COST-PROFILE: Byte win on u8 templates and dropped rank vectors; f32 Einsum casts stay mandatory

TRANSFERS-WHEN: Task uses separable row/column grids, ends in dtype-matched Einsum, rank ties breakable by sub-unity linear mix

## won_task141_codex.py
MECHANISM: Quadratic row/column seeding builds a conv kernel for presence-gated terminal scoring.
REPRESENTATION: Shared [1,x,x²] Einsum basis T seeds row/col fits; Concat reshapes into 2×3×3 K; terminal mega-Einsum applies K twice with score2/sign.
COST-PROFILE: Bytes sit in init tensors T, w01, sign, score2; graph is nine Einsum/Mul nodes.
TRANSFERS-WHEN: Reuse on fixed H×W grids where row/column parabolic cues drive channel-masked, presence-gated local updates.

## won_task148_codex.py
MECHANISM: Row portal marker shift with per-cell polynomial color compositing over events.

REPRESENTATION: Unguarded GatherElements shift via iota minus delta; five-term float row-basis Einsum polynomial paint.

COST-PROFILE: Light gather-and-Einsum graph; bytes live in solved row/in/out coefficient tables.

TRANSFERS-WHEN: Small bounded row-top delta, float row-code portals/markers, and low-order row×col polynomial fills with margin overwrites.

## won_task154_codex.py
MECHANISM: Crop-native routing where three ordered in-crop quant codes defeat one affine positive half-line.
REPRESENTATION: Keep donor through class_crop; swap quadratic tail for uint8 clip hinges and stacked narrow QLinearConv argmax.
COST-PROFILE: Bytes live in donor through class_crop; tail is tiny int8 hinge QLinearConv weights and biases.
TRANSFERS-WHEN: Proven donor emits class_crop; ≥3 strictly ordered uint8 labels; middle class needs 2D saturation; zero exterior must argmax background.

## won_task156_codex.py
MECHANISM: Two nested yellow rectangles; keep borders, rank-fill interiors by relative area.

REPRESENTATION: U8 relower crop, qcode neighborhood tags, (state,code) concat; QLinearConv tail pads and renders classes.

COST-PROFILE: Saves fp32 crop; bulk memory in relower+qcode; params concentrated in terminal conv.

TRANSFERS-WHEN: Reuse when multiple same-color regions need distinct interior fills by size comparison and linearly separable (state,qcode) points exist.

## won_task159_codex.py
MECHANISM: Pin detect-crop, red-bordered 4x4 patch, selector-tiled 30x30 one-hot render.

REPRESENTATION: Terminal uint8 Einsum bdxy,bdc,xh,yw->bchw fuses spatial masks, color vectors, repeat selectors into output.

COST-PROFILE: Pin LUTs plus compact uint8 4x4 masks and 4x5 selector; no 30x30 int32 state.

TRANSFERS-WHEN: Reuse with live-pin motif detect, small bordered patch magnified to fixed grid, sub-2.5KB uint8 terminal, not Equal or MatMulInteger render.

## won_task160_codex.py
MECHANISM: Few-class pixel labeling where color splits need quadratic, not linear, scalar boundaries.
REPRESENTATION: relower_onehot_plane crop, then QLinearConv chain builds (code, code²) via int8 3×3 kernels; 1×1 conv separates classes.
COST-PROFILE: Minimal u8/int8 weights; three QLinearConv stages’ fixed ~500B activation memory dominates.
TRANSFERS-WHEN: Fixed crop, ≤10 channels, few discrete colors, per-pixel scalar cue classifiable by quadratic boundaries on float grids.

## won_task163_codex.py
MECHANISM: Yellow-locator crop of mod-4 3×3 patch into periodic 11×11 color lattice.

REPRESENTATION: Cast Einsum coords to uint8, mod-4 slice, dest_idx table scatter, terminal Pad—not polynomial Einsum.

COST-PROFILE: Bytes live in small uint8 tables and pad constants; graph stays thin pre-Pad.

TRANSFERS-WHEN: One-hot multi-channel grid, fixed locator channel, 4-period tiling, 3×3 source drives larger periodic color output remap.

## won_task165_codex.py
MECHANISM: Detect 10-cell kite via packed row bits; fill kite columns from scatter below

REPRESENTATION: Einsum packs rows/columns to uint32 words; kite matched by aligned BitShift/BitwiseAnd taps

COST-PROFILE: Params tiny; savings from skipping dense [19,7] rel-column bitmask tensor

TRANSFERS-WHEN: Fixed stencil localize-and-fill; narrow grid; scatter-below-column rule; standard ONNX bitwise ops

## won_task174_codex.py
MECHANISM: Bbox extrema from row/column presence projections without reverse-slice indexing.

REPRESENTATION: ArgMax with select_last_index=1 on cp_/rp_ projections replaces reverse-slice max-coordinate subgraphs.

COST-PROFILE: Bytes exit reverse-slice inits, Wm1, and dropped max-index intermediate tensors.

TRANSFERS-WHEN: Max edge is last nonzero along projection axis; min edge uses cmin/cmaxr ramp path; same projection-bbox rule.

## won_task177_codex.py
MECHANISM: Crop nonzero bounding box, mirror horizontally, emit padded grid.

REPRESENTATION: Dilated Conv packs four 4-bit column IDs per uint16 lane; bbox on packed rows; gather-decode mirrored 8×8; QLinearConv terminates.

COST-PROFILE: Small pack and termination weights; logic in cheap gathers/reduces, not big tensors.

TRANSFERS-WHEN: ≤9-color palette, bbox crop plus mirror/flip, column-local 4-bit packing, leftover Conv slot for classify or pad.

## won_task178_codex.py
MECHANISM: Thirteen-cell axis sampling, orientation test, top-five run extraction, equality broadcast paint.

REPRESENTATION: Zero-second-tap dilated Conv samples exactly thirteen axis cells without thirty-wide intermediate planes.

COST-PROFILE: Dilated samplers stay tiny; bulk bytes land in final 30×30 bool Pad output.

TRANSFERS-WHEN: Reuse when task has thirteen-sample axis lines, row-or-col orientation fork, ranked run starts, and top-five channel-equality grid paint.

## won_task183_codex.py
MECHANISM: Variable N×N inner square from quadrant corners with cyan recoloring.

REPRESENTATION: u8 code grid feeds [code, code², one] into padded ConvInteger one-hot logits.

COST-PROFILE: int8 ConvInteger tail and u8 6×6 state, not full 30×30 canvas.

TRANSFERS-WHEN: Small discrete-coded working grid, quadrant corner lookup, terminal one-hot decode to full padded canvas.

## won_task184_codex.py
MECHANISM: Row/column band segmentation on fixed 30×30 ten-color grids via foreground counts.
REPRESENTATION: Einsum counts, CumSum separator zone IDs, float16 Equal masks, closing zone Einsum.
COST-PROFILE: Bytes live in graph operators; only small fg/place/coeff initializer tensors.
TRANSFERS-WHEN: Same 1×10×30×30 shape, zero-count band separators, zone-wise transform or 3×3 gather-pad.

## won_task187_codex.py
MECHANISM: Fixed-width 1D cellular automaton with bitwise neighbor-dependent row updates.
REPRESENTATION: Strided Conv packs 15 bits into uint32; Gather and bitwise ops implement CA rule.
COST-PROFILE: Bytes live in conv pack weights, border mask, and precomputed Gather index tables.
TRANSFERS-WHEN: ~25-cell rows, uint32 bit-packing, same neighbor offsets, rule expressible as bitwise shifts and masks.

## won_task189_codex.py
MECHANISM: Header flip flags steer pattern/palette crops; masked palette decodes to full canvas.
REPRESENTATION: Pow lifts palette codes to [x,x²]; ConvTranspose Gaussian peaks paint full resolution.
COST-PROFILE: Cheap 6×6 activations; bulk cost is final FP16 output and tiny decode weights.
TRANSFERS-WHEN: Reuse when input embeds flip metadata, fixed pattern/palette crops, discrete index colors, and stencil upsample to full grid.

## won_task190_codex.py
MECHANISM: Tagged u8 crop, conv ray detect, input-driven terminal recolor onto padded grid.

REPRESENTATION: QLinearConv scores rays on cropped u8 plane; ConvInteger terminal builds per-channel weights from pooled input minus ray.

COST-PROFILE: Stored bytes mostly in fixed 13×19 ray kernel; terminal weights computed at runtime, not serialized.

TRANSFERS-WHEN: One-hot u8 channel, ray motif conv-detectable, output needs input-dependent background/ray split on padded canvas.

## won_task192_codex.py
MECHANISM: 2x2-block membership mask, dominant-color repaint, channel-0 background inside HxW canvas.
REPRESENTATION: Einsum-packed uint32 column bitsets; bitwise 2x2 anchor+dilate; terminal row-mask And emits one-hot uint32 channels.
COST-PROFILE: Narrow 30-bit uint32 lanes; tiny inits; final 10×30×30 output dominates memory.
TRANSFERS-WHEN: Padded grid, one-hot input channels, 2x2-local keep rule, per-cell channel select; uint32 one-hot output passes raw>0 gate.

## won_task194_codex.py
MECHANISM: One-hot canvas; crop seed patch, LUT-tile to motif, pad to output size
REPRESENTATION: ConvTranspose pads collapse one-hot to UINT8 IDs, cropping only top-left 3x3
COST-PROFILE: Tiny weights and 6x6 LUT; no learned kernels, bulk is index table
TRANSFERS-WHEN: Same one-hot palette; output from small seed via fixed gather map plus edge padding

## won_task195_codex.py
MECHANISM: One 3×3 motif in phased grid; Kronecker upscale; dual-polarity template match.
REPRESENTATION: P⊗P via broadcast Unsqueeze+Min; QLinearConv encodes u8 mask into ±1 signed threshold codes.
COST-PROFILE: Dominant bytes in padded 10×30×30 QLinearConv output tensor.
TRANSFERS-WHEN: Input hides one strided-samplable 3×3 tile; output is 3× enlarged self-Kronecker with black/gray sign discrimination.

## won_task198_codex.py
MECHANISM: 2D grids with runtime-detectable repeat period merged by bitwise OR overlays.

REPRESENTATION: Period index selects explicit 2D OR stacks of fixed binary masks, not conv weights.

COST-PROFILE: Bytes in period LUT and static OR mask tensors; negligible learned parameters.

TRANSFERS-WHEN: Reuse when repeat period is inferable at runtime and the target is OR of period-aligned 2D masks.

## won_task199_codex.py
MECHANISM: Count-scaled rectangle overlay anchored at dominant-color centroid on fixed grid.

REPRESENTATION: uint8 row, column, and channel bit masks fused by BitwiseAnd as terminal pixel renderer.

COST-PROFILE: Bytes in tiny uint8 constants and broadcast coords; zero learned weights.

TRANSFERS-WHEN: Fixed H×W canvas, single marker color, rectangle rules from cell count plus centroid, opset 18 BitwiseAnd.

## won_task201_codex.py
MECHANISM: Yellow frame and pattern bboxes from ReduceMax over per-cell color labels

REPRESENTATION: Remap one conv_weights color index to sentinel 10; delete yellow Equal/Cast masks; ReduceMax labels directly

COST-PROFILE: Deletes three mask u8 subgraphs; conv_weights one-hot tweak; two Min clamps

TRANSFERS-WHEN: Task finds bboxes by ReduceMax on color labels; one hue relabeled sentinel in conv_weights replaces mask extraction

## won_task202_codex.py
MECHANISM: Per-axis color means with sentinel invalid axes and Cauchy orientation tie-break.
REPRESENTATION: u8 profile casts; Cauchy orientation via direct Equal on squared-moment sums, no Sub.
COST-PROFILE: nkhw Einsum reductions dominate; u8 shrinks profiles; one output Where.
TRANSFERS-WHEN: Reuse for axis-aligned band painting from masked row/column profiles when horizontal-vertical dominance is ambiguous.

## won_task205_codex.py
MECHANISM: Pin-equivalent staged row/column filter with six-flag run gate and in-box ArgMax scoring.

REPRESENTATION: Relower Greater outputs to u8/bool; MaxPool-on-complement detects six-on runs; lone fp16 Einsum render.

COST-PROFILE: Bytes in u8 detectors and int32 coord compares; fp16 only for final packed Einsum.

TRANSFERS-WHEN: Reuse when a pin-equivalent baseline ONNX exists and masks, six-flag gates, iota bounds, and ArgMax scoring share this shape.

## won_task208_codex.py
MECHANISM: Find solid black rectangles sized 3–5; exclude one known framed source hole.

REPRESENTATION: Pack black rows as uint32 bitsets; bitwise runs plus row-slice ANDs locate window; ArgMax reads coords.

COST-PROFILE: Cheap mask constants; expensive bitwise node chain splices into existing ONNX mid-graph.

TRANSFERS-WHEN: Padded one-hot input; black channel; interior cols 2–17; heights/widths 3–5; one source frame to reject.

## won_task212_codex.py
MECHANISM: Seeds become vertical rays; a horizon row splits above/below; few patterns render output channels.
REPRESENTATION: Asymmetric MaxPool pads cast rays; four planes feed ConvInteger as sparse int8 LUT with pad broadcast.
COST-PROFILE: Bytes in relower crop inits, W×W bias plane, tiny 10×4 LUT weights.
TRANSFERS-WHEN: Reuse when colored seeds extend to full-column rays, one row divides halves, and outputs are discrete channel patterns on fixed width.

## won_task213_codex.py
MECHANISM: Periodic stripe codes on a grid; H/V intersection picks colors for quad tiling.
REPRESENTATION: Einsum stripe projection, tri-period Max+Mod LUT decode, ConvInteger splats 3×3 quads from [c,c²,v].
COST-PROFILE: Bytes in mod-104 LUT and INT8 ConvInteger weights; ops dominate runtime.
TRANSFERS-WHEN: Reuse when stripes use period-3 sampling, weighted residues mod N map via LUT, and output splats 3×3 quads at H∩V crossings.

## won_task222_codex.py
MECHANISM: Rectangle completion from connected 2x2 color seeds on inner cropped grid.
REPRESENTATION: QLinearConv filters connected 2x2 seeds; MaxPool builds row/col masks; Einsum outer-products masks with channel LUT.
COST-PROFILE: Static uint8 kernels plus float16 30-wide row/col/bg canvas initializers hold most bytes.
TRANSFERS-WHEN: 30x30 ten-channel input, 16x16 active core, rectangles from 4-connected 2x2 solids, separable row-column-channel fill.

## won_task225_codex.py
MECHANISM: Anchor 2×2 colored patch; LUT-remap by quadrant; paint channels per color code.

REPRESENTATION: UINT8 [color, color²] plus QLinearConv bias implements 1−(color−target)² equality pickers.

COST-PROFILE: Tiny Gather LUT chain; bulk bytes in terminal QLinearConv’s heavy spatial padding.

TRANSFERS-WHEN: Fixed 2×2 patch, discrete 1–9 codes, background anchor, channel-per-color output, quadrant-relative spatial remap.

## won_task226_codex.py
MECHANISM: Output depends on separable products of row and column pair features.
REPRESENTATION: ONNX multiplies independent row and column LUT embeddings for exact pair lookup.
COST-PROFILE: Bytes live in two factor tables, not a dense pair grid.
TRANSFERS-WHEN: Rule is exact separable pair-product over discrete symbols; train I/O fully coverable by factored LUTs.

## won_task232_codex.py
MECHANISM: Row seeds extend rightward, alternating seed color and gray to edge.

REPRESENTATION: One terminal Einsum contracts six low-rank factors with duplicated input into output.

COST-PROFILE: 692 floats in six factor tensors; single output Einsum, zero intermediates.

TRANSFERS-WHEN: Row-wise rightward two-state alternation on fixed grid, compressible to duplicated-input low-rank contraction.

## won_task233_codex.py
MECHANISM: Masked crop, bbox-gated TopK stamp search, multi-variant scatter compose, pad-equality output

REPRESENTATION: Splice donor matcher core; prefix mask ScatterElements, ArgMax bbox, Gather crop; suffix pub-triggered Concat scatter

COST-PROFILE: Donor matcher initializers, pub stamp payloads, flat scatter index-value tables

TRANSFERS-WHEN: Donor ONNX embeds stamp matcher; new task masks grid, crops bbox, ranks placements, picks among sized pub stamps, scatters into output

## won_task234_codex.py
MECHANISM: Close tongue gap by sliding colored rectangle along axis to solid bar.

REPRESENTATION: Encoded axis profile plus prefix-valid terminal mask picks flush coordinate.

COST-PROFILE: Ninety-five params; two kilobytes mostly profile and prefix LUT memory.

TRANSFERS-WHEN: Plane-kill family task: one-axis tongue rectangle must abut a solid block, erase gap.

## won_task239_codex.py
MECHANISM: Certified histogram through `code`; replace parabola tail with direct Where renderer.

REPRESENTATION: And `bar`+`active_col`→fill; Sub `code` by one; Where picks unshifted fill or blank padding.

COST-PROFILE: Bytes live in kept histogram/mask graph; tail is three ops; drop unused renderer inits.

TRANSFERS-WHEN: Exports `code`, `bar`, `active_col`, `blank`; fill needs color unshift only; superseded renderer weights prunable.

## won_task240_codex.py
MECHANISM: Mirror odd-lattice corner seeds quadrantly; adjacent diagonal colors paint nested dotted frames.

REPRESENTATION: Single shared-bank terminal Einsum; spatial bank doubles as orbit sampler; 1-(c-k)² picks channels.

COST-PROFILE: ~half params in bank[9,30]; coeffs, src maps, channel polynomial split remainder.

TRANSFERS-WHEN: Fixed grid, quadrant symmetry, sparse diagonal seeds, nested periodic masks, few source orbits, polynomial channel fit under budget.

## won_task243_codex.py
MECHANISM: 4-connected flood fill on bounded square grid, preserving all non-seed colors.

REPRESENTATION: Pack rows as uint32 bitmasks; carry-add bitwise closure plus five alternating Gauss-Seidel sweeps replace synchronous CA rounds.

COST-PROFILE: ~8 KB graph memory, ~89 parameters; row sweeps beat ~21 KB vertical CA.

TRANSFERS-WHEN: Each row fits one uint32; flood propagates through a traversable mask; few alternating row passes suffice under strict ONNX byte ceiling.

## won_task244_codex.py
MECHANISM: Horizontally flipped linegrid; detect 3×3/4×4 geometry, strided-extract cells, pad canvas.
REPRESENTATION: Height ReduceSum probes index Gather tables for dynamic Slice starts/steps; negative stride undoes flip.
COST-PROFILE: ~640 B one-hot sampled core dominates; geometry probes and index scalars are minor.
TRANSFERS-WHEN: One-hot fp32 gate, grid size scalar-detectable, interior via negative-strided Slice, fixed output canvas via Pad.

## won_task245_codex.py
MECHANISM: Anchor-delta shift of one colored blob, preserve green corners, erase old red.

REPRESENTATION: u8 relower, Gather-shifted red crop, edge-LUT green frame, ConvInteger tail compositor.

COST-PROFILE: u8 plane memory; tiny anchors/LUT; terminal tail3 ConvInteger params.

TRANSFERS-WHEN: Small fixed canvas; red/green UL anchors plus (+1,+1); corner frame via LUT; ConvInteger merge.

## won_task246_codex.py
MECHANISM: W18 two-anchor HPWL scalar wire with strict-between horizontals and green-column corner paint.

REPRESENTATION: Einsum channel peaks yield UINT8 anchors; strict-between column mask plus green-corner paint drops one 18-cell Equal/Where pair.

COST-PROFILE: UINT8 casts, range grids, Min/Max bounds, dual Where masks, Pad to 30, ch8 overlay.

TRANSFERS-WHEN: Fixed small width, red/green HPWL anchors, audited scalar rule already correct, must emit 10-channel one-hot via ch8 segment overlay not raw u8.

## won_task251_codex.py
MECHANISM: Interior-enclosed black fills blue; boundary-linked black and all red preserved.
REPRESENTATION: Directional padded MaxPools gate seeds; capped 3×3 MaxPool rounds grow complement inside Z.
COST-PROFILE: Seven MaxPools, two channel relayers; zero weights, only crop pads.
TRANSFERS-WHEN: One-hot red/black grids where fill targets black components fully enclosed, not boundary-touching, under tight byte budget.

## won_task255_codex.py
MECHANISM: Independent row and column boolean masks; matmul counts overlaps; threshold picks output.

REPRESENTATION: Keep pinned Concat predicate heads; replace fp16 MatMul tail with UINT8 QLinearMatMul.

COST-PROFILE: Pin holds Concat masks; quant tail sheds fp16 weights and zero initializer.

TRANSFERS-WHEN: Pin has Concat row/col masks, fp16 MatMul count tail, and Where threshold output selection.

## won_task256_codex.py
MECHANISM: Row-prefix coloring inside a variable-size input rectangle decoded from red markers.

REPRESENTATION: Broadcast per-channel row and column threshold tensors; single Greater selects winning color per pixel.

COST-PROFILE: Row-threshold builders dominate; column logits and scalar einsum decode are secondary.

TRANSFERS-WHEN: Reuse when output is row-wise prefix bands over a sum-inferred rectangle and row/col thresholds replace explicit per-pixel masks.

## won_task260_codex.py
MECHANISM: Per-cell output from row/column coordinate boolean predicates

REPRESENTATION: Frozen row/col mesh constants in ONNX drive Compare/Where mask subgraphs

COST-PROFILE: Coordinate constant tensors; small compare/where chains per channel

TRANSFERS-WHEN: Fixed grid size; label is coordinate-and-color predicate, not object identity or counting

## won_task264_codex.py
MECHANISM: Multi-channel grid locate-and-classify: compress scene to 9-byte fingerprint, decode digit, render one hot channel.

REPRESENTATION: Programmable Einsum probe bank → U8 Gather LUT → ConvInteger quadratic 1×1 conv with asymmetric pad upscales winner.

COST-PROFILE: ~1.3KB frozen einsum selectors dominate; runtime stays sub-500B via 9×9 U8 features not 16×16 f32 crops.

TRANSFERS-WHEN: Reuse when evidence fits nine 1×1 einsum probes, discrete 0–9 code from LUT, and 9×9 motif upscales to full output via padding.

## won_task264_alt.py
MECHANISM: Multi-channel grid where few contracted scalars fix a local 9×9 pattern, upscaled to per-pixel classes.

REPRESENTATION: Eight Einsum scalars cast U8, Gather LUT builds 9×9 code; ConvInteger quadratic int8 weights plus heavy pad render full grid.

COST-PROFILE: ~1484B params in selectors/LUT; ~482B activations; skips 3600B-per-plane bool renders.

TRANSFERS-WHEN: Reuse when a handful of Einsum-extractable scalars encode the rule, a small LUT maps them to a patch, and padded ConvInteger can upscale to output resolution.

## won_task265_codex.py
MECHANISM: Grid tasks whose train pairs already match lane55’s fixed rewrite rule.
REPRESENTATION: Reuse build_lane55’s ONNX graph verbatim; only change the output filename.
COST-PROFILE: All bytes sit in the shared lane55 weights and lookup tables.
TRANSFERS-WHEN: Every train pair follows lane55 exactly; no custom nodes, reweights, or rule tweaks needed.

## won_task267_codex.py
MECHANISM: Erase one fixed marker; recolor other foreground; keep background unchanged.
REPRESENTATION: Slice marker once; Where builds rank-2 selector; Concat e0-marker palette; terminal Einsum remaps grid.
COST-PROFILE: ~210 B init: base selector 80 B, e0 40 B, erase 8 B, slice indices.
TRANSFERS-WHEN: One marker at fixed coordinates; ten color planes; erase-marker plus recolor-nonbackground rule fits rank-2 einsum.

## won_task268_codex.py
MECHANISM: Packed bitrow parser over row/col profiles into fixed-budget terminal scatter.

REPRESENTATION: Profile=p%15 gives size, p>15 flags rows; ArgMax endpoints; one-shot bitrow-grid mask intersect.

COST-PROFILE: Parser ~995 B; bool mask 100 B; f32 terminal 800 B; initializers capped at 260.

TRANSFERS-WHEN: Square grids 5–10, 1D row/col occupancy profiles, contiguous affected bands, existing probe graph, scatter terminal free.

## won_task270_codex.py
MECHANISM: Marker-driven row-bitword scatter on fixed multi-channel grids
REPRESENTATION: Einsum row-pack, BitShift column bits, ScatterND gated int64 state, col_mask output
COST-PROFILE: Precomputed col_mask LUT and base_state seed; thin compute graph
TRANSFERS-WHEN: Fixed 10×30×30 layout; marker rules with directional gating; row-scatter bitmask deltas per column-channel

## won_task273_codex.py
MECHANISM: Fill rectangle interiors bounded by relowered one-hot yellow frame markers.
REPRESENTATION: Separable u8 QLinearConv span-equals-three masks; three-channel concat; terminal padded 1x1 recolor.
COST-PROFILE: Relower one-hot LUTs bulk; nine-tap separable kernels minimal.
TRANSFERS-WHEN: Reuse when borders are one-hot stripes, interior is uniform fill, and cropped logic must pad back to full grid.

## won_task275_codex.py
MECHANISM: Donor algebra with separate contractions that charge costly fp32 intermediate graph outputs.

REPRESENTATION: Truncate donor after prefix nodes; fuse ReduceSum and Einsum contractions into one terminal Einsum renderer.

COST-PROFILE: Drops charged fp32 contraction tensors; bytes move to single fused terminal output.

TRANSFERS-WHEN: Donor prefix intact, contractions algebraically fuseable, fp32 inputs make standalone contraction outputs byte-expensive.

## won_task277_codex.py
MECHANISM: Grid outputs from tagged-component relowering in a morphology op graph.

REPRESENTATION: Whole pipeline as one registerable tagged-relower morphology ONNX graph.

COST-PROFILE: Morphology stencil LUTs plus tag-indexed relower register table.

TRANSFERS-WHEN: New task needs same tagged-part relower via morphology, not local color rewrite.

## won_task278_codex.py
MECHANISM: Green 3x3 halos on black around orthogonally adjacent red pixel pairs.

REPRESENTATION: QLinearConv cross-kernel sums red neighbors; MaxPool dilates pairs; AND black mask.

COST-PROFILE: green30 pad is 900B; five 18x18 tensors; conv beats slice views.

TRANSFERS-WHEN: Adjacency halos on masked background, encodable as u8 conv plus pool within ONNX byte budget.

## won_task280_codex.py
MECHANISM: Iterative 30×30 neighbor walks grafted onto pinned spatial_select terminal.
REPRESENTATION: int8 index-prefix arithmetic; int64 cast only before GatherND; forbid dual-30 intermediates.
COST-PROFILE: int8 walks and bool neighbor gathers; no full 30×30 activations.
TRANSFERS-WHEN: Donor Einsum spatial_select pin, 1×10×30×30 output, multi-hop GatherND index chains, tight byte cap.

## won_task281_codex.py
MECHANISM: BBox-interior color painting via row factors instead of 13×13 feature stack.
REPRESENTATION: Power-sum Einsum bounds; ConvInteger joins row masks to column-masked TopK paint weights.
COST-PROFILE: Bytes in small uint8 conv kernels and pow tables, not 13×13 feature tensors.
TRANSFERS-WHEN: ≤13 grids on 30×30, certified rule, Pad fails one-hot gate, needs outer-inner frame paints.

## won_task281_alt.py
MECHANISM: Bbox-inner grid fill from projections, TopK colors, separable mask product.

REPRESENTATION: Per-branch fp16 casts; Pad thirteen-wide masks to thirty; terminal Einsum unchanged.

COST-PROFILE: Dominant bytes in padded f16 row and column term tensors.

TRANSFERS-WHEN: Separable channel-row-column paint, bbox plus inner bands, TopK colors, W pads to M.

## won_task285_codex.py
MECHANISM: Legend anchor picks quadrant; opposite 5×5 source copied by grouped h/v/d render.

REPRESENTATION: Six chained 3×3 MaxPool floods yield TopK-8 mask; ninth cell uses static index-13 slot.

COST-PROFILE: Sparse router preserved; bytes in six flood pools and grouped gathers, not K=9 renderer.

TRANSFERS-WHEN: Legend-anchored quadrant, opposite canonical source, ≤8-cell motif (one static overflow), h/v/d symmetric paste.

## won_task288_codex.py
MECHANISM: Background histogram picks variant; arithmetic derives antenna coords; ScatterND stamps output.

REPRESENTATION: ReduceSum color counts, Equal case table, broadcast j6 row-col math, ScatterND sparse write.

COST-PROFILE: Bytes live in small initializer tables; compute is elementwise ops plus ScatterND.

TRANSFERS-WHEN: Discrete background count fingerprints, finite cases, fixed scatter slots, geometry from case index.

## won_task294_codex.py
MECHANISM: Slice one-hot gray mask; classify each pixel's 3×3 neighborhood as background, interior, or border.
REPRESENTATION: Double uint8 gray slice; QLinearConv zero-point maps 0/2 to −1/+1 for signed 3×3 kernel voting.
COST-PROFILE: Bytes in qw/qb: ten output channels, three sparse 3×3 int8 kernels, mostly zeros.
TRANSFERS-WHEN: One-hot gray is sliceable; labels need interior versus border versus background from padded 3×3 local context.

## won_task295_codex.py
MECHANISM: Color-histogram derives width/height; triangular row packing; ConvInteger produces padded output.
REPRESENTATION: Row and column Where masks merge into 0/1/2 plane; ConvInteger weights from foreground sums.
COST-PROFILE: Nine-by-eighteen bool/u8 planes consume bulk; forty-four initializer scalars/vectors remainder.
TRANSFERS-WHEN: New task shares histogram-derived triangular layout on nine-by-eighteen grid with ConvInteger terminal scoring.

## won_task303_codex.py
MECHANISM: All-black row/column frontiers; recolor their union; preserve other cells.

REPRESENTATION: Factored terminal Einsum pairs row/column count features; coeff couples frontier veto and intersection paint.

COST-PROFILE: Tiny feat/coeff templates; two axis einsums; terminal paired renderer dominates.

TRANSFERS-WHEN: Axis-uniform full-black frontiers; paint row-column union from paired per-axis counts, not per-cell rules.

## won_task304_codex.py
MECHANISM: Per-cell discrete label from encoded local neighborhood signature.
REPRESENTATION: Precomputed context-to-label LUT; one ONNX Einsum row-select per patch.
COST-PROFILE: Bytes live in static selector table; graph ops are negligible.
TRANSFERS-WHEN: Fixed grid, finite patch vocabulary, deterministic context-to-output mapping across all train pairs.

## won_task323_codex.py
MECHANISM: Anchor color locates seed; fixed template sliced to output patch.

REPRESENTATION: Einsum finds cyan seed; Slice centers rel25 stair template; QLinearConv pads 13→30.

COST-PROFILE: rel25 uint8 template plus int8 QLinearConv decoder weights and biases.

TRANSFERS-WHEN: Single seed marker; fixed H×W patch; stair-like relative geometry; multi-channel 30×30 input planes.

## won_task325_codex.py
MECHANISM: Count 4-connected cyan components; emit N×N black square, cyan main diagonal.

REPRESENTATION: One-channel ternary f16 map; ConvTranspose with bias maps −1/+1 to black and cyan channels.

COST-PROFILE: Chained Einsum Euler counts; compact ConvTranspose terminal replaces nine-channel concat renderer.

TRANSFERS-WHEN: Scalar topology count replaces BFS; output motif scales with N on fixed padded canvas, two-color extremes.

## won_task330_codex.py
MECHANISM: Per-cell rewrite gated by local tally meeting threshold nine.
REPRESENTATION: Reduce-sum neighbor count, Compare to scalar nine, Where masks invalid cells.
COST-PROFILE: Tally reduce plus one scalar constant; no LUT weight tables.
TRANSFERS-WHEN: New task marks valid pixels by local count ≥9 before applying fixed color rewrite.

## won_task333_codex.py
MECHANISM: Anchor-box locate, directional max ray-fill on strips, scatter write-back.

REPRESENTATION: Shifted z codes with green sentinel; MaxPool strips; QLinearConv on [z,z²] one-hot bias.

COST-PROFILE: Bytes in spatial_select mask; compute in MaxPools and scatters.

TRANSFERS-WHEN: Fixed grid, 2×2 green anchor, row/col ray-fill to box, max-green wins, conv color remap.

## won_task335_codex.py
MECHANISM: Rank-2 path coloring from remote marker endpoints, not local 3x3 neighborhoods.

REPRESENTATION: Shared cmap latents; concat u8 rank terms pre-cast; rank-3 term_lat factors terminal Einsum coefficients.

COST-PROFILE: Bytes in tiny latents and late-cast u8 terms, not full neighborhood LUTs.

TRANSFERS-WHEN: Global endpoint geometry drives fills; few shared color latents; audited local LUTs are insufficient.

## won_task336_codex.py
MECHANISM: Gray-plane crop, 11-conv score map, concat, padded terminal QLinearConv oracle.
REPRESENTATION: Tagged u8 one-hot relower for crop avoids charged fp32 Slice; terminal QLinearConv renders gated output.
COST-PROFILE: Dominated by pinned q_w/q_b; u8 activations; handcrafted terminal weights; Slice byte tax avoided.
TRANSFERS-WHEN: New task shares fixed-channel 10×10 crop, score feeds terminal QLinearConv, and fp32 Slice would be budget-charged.

## won_task340_codex.py
MECHANISM: Proven-graph transplant with row-type bit masks and column gather-to-renderer reshape.

REPRESENTATION: Replace nine-entry row-bit LUT with uint8 BitShift from 128; collapse Gather-plus-Reshape into axis-aware Gather.

COST-PROFILE: Drops row_bit_lut and reshape shape tensor; eliminates 300-byte Reshape output; adds one-byte bit128.

TRANSFERS-WHEN: When a reference ONNX uses power-of-two row LUT and a Gather-then-Reshape renderer path collapsible via singleton axes.

## won_task341_alt.py
MECHANISM: Oriented grid bridges from empty versus dual-minimum row and column line masks.
REPRESENTATION: Boolean bridge mask scaled by per-channel ±1 weights, scattered via static ND indices.
COST-PROFILE: Byte budget concentrated in precomputed ScatterND index initializer, not compute nodes.
TRANSFERS-WHEN: Fixed interior patch, two channels needing additive ±1 updates, bridges from row/col sums.

## won_task341_codex.py
MECHANISM: Certified bridge eligibility from row/column emptiness and dual-minima gates.

REPRESENTATION: Precomputed ScatterND index grid writes paired ±1 updates to two fixed channels.

COST-PROFILE: Static scatter indices dominate size; reductions, slices, and masks stay cheap.

TRANSFERS-WHEN: Fixed inner patch geometry, row/col sum predicates, terminal sparse dual-channel ±1 ScatterND updates.

## won_task342_codex.py
MECHANISM: Anchor-centroid quadrants; per-wedge dominant color; 2×2 motif padded then quadratic-LUT convolved to grid

REPRESENTATION: Einsum coords, fold absent hues to anchor, quadrant ArgMax; QLinearConv on color and square encodes per-target quadratic scorer

COST-PROFILE: Ten-output QLinearConv int8 weights dominate; remainder is small initializer tensors

TRANSFERS-WHEN: Single anchor hue fixes origin; four quadrant winners tile 2×2; full canvas needs padded motif plus channel-wise quadratic color-to-label LUT

## won_task345_codex.py
MECHANISM: Row-wise red-over-gray priority propagation, interleaved, ConvInteger-rendered.
REPRESENTATION: Interleaved uint8 red/gray row pairs decoded by strided ConvInteger with heavy asymmetric padding.
COST-PROFILE: Asymmetric ConvInteger pads and full INT32 output tensor absorb most bytes.
TRANSFERS-WHEN: Reuse when layered row masks, bitwise uint8 state, fixed slice grid, and stride-2 conv rendering suffice.

## won_task348_alt.py
MECHANISM: Per-class terminal bool masks from row versus column stripe-score dominance.

REPRESENTATION: ArgMax/Einsum decode probes; Where-gated stripe offsets; Concat class vectors; row≥col bool.

COST-PROFILE: Static arange/parity inits; ten-wide Concat; deep per-pixel Cast-Sub-Where chains.

TRANSFERS-WHEN: Reuse when geometry is probe-encoded in fixed channels, grid is H×W×C, and terminal bool needs class-specific row-column stripe comparison.

## won_task348_codex.py
MECHANISM: V-fan terminal renderer: static row basis times column-dynamic conv weights.
REPRESENTATION: ConvInteger row-basis input; geometry-built dynamic uint8 kernel; zero-point 100 encodes signed scores.
COST-PROFILE: Dynamic weight subgraph plus static row inits; conv pads embed 10x10 into 30x30.
TRANSFERS-WHEN: Row-static, column-dynamic terminal scoring; ≤10×10 crop; larger canvas via conv padding; signed coeffs via uint8 ZP, not int8.

## won_task354_codex.py
MECHANISM: Row-zero colored lights horizontally flood-recolor gray rectangles in matching spans.

REPRESENTATION: Exact u8 color-id geodesic flood on 10×10; ConvInteger quadratic kernel pads and one-hot-expands to ten channels.

COST-PROFILE: Eighty-eight learned params; byte mass in compact u8 flood and band tensors.

TRANSFERS-WHEN: Ten discrete colors; three row-0 seed bands; gray rectangles rows 2–9; four-step horizontal flood; 30×30 pad.

## won_task358_codex.py
MECHANISM: Plus-cross grids with periodic row/column palette arms and inferred square dimensions.
REPRESENTATION: OneHot row masks, periodic GatherElements arms, flip test, FP16 Einsum polynomial paint.
COST-PROFILE: Tiny FP16 coeff init; bulk is uint8 masks and coord tables.
TRANSFERS-WHEN: Plus-cross motifs, background zero, ≤10 colors, square/near-square area, periodic arm palettes from center row.

## won_task359_codex.py
MECHANISM: Long uniform stripes with sparse random colour replacements; recover axis and majority colour.

REPRESENTATION: Signed colour-bit Einsum projections per axis; L1 picks orientation; majority bits decode via MatMulInteger-Gather.

COST-PROFILE: Dual signed profiles dominate; then bit-decode tables, active masks, terminal ten-plane equality.

TRANSFERS-WHEN: Few-bit discrete colours, one dominant stripe axis, mostly uniform lines with sparse local noise recoverable by signed bit-majority.

## won_task361_codex.py
MECHANISM: Four-way rotated index scatter into flat grid, polynomial ConvInteger completion render.

REPRESENTATION: One ScatterElements-max fuses four scatters; Einsum color projection; ConvInteger polynomial replaces Equal bitmask renderer.

COST-PROFILE: Drops mask/delta nodes; bytes live in kept scatter tensors plus compact int8 poly ConvInteger weights.

TRANSFERS-WHEN: Top-indexed cells, four rotation clips 0–80, max-scatter into flat buffer, scalar polynomial render instead of equality masks.

## won_task363_codex.py
MECHANISM: Z/R split grid: 4×4 template match, exception scatter-fix, painted state synthesis.

REPRESENTATION: QLinearConv uint8 exact matcher and painter; Gather-built normT; ScatterND anchor erasure; ConvInteger Wout mixer.

COST-PROFILE: No z-fingerprint weights; bytes live in matcher-paint QLinearConv node chain.

TRANSFERS-WHEN: Reuse when task splits z template and r reference on 10×10, needs 4×4 exact match, few scatter-fixable anchor exceptions—not full z-fingerprint gating.

## won_task365_codex.py
MECHANISM: Marker-color rectangle localization via corner stencil, interior scoring, fixed patch extraction

REPRESENTATION: Dilated 2×2 conv color-tags grid; 4×4 corner conv plus TopK proposes boxes; Einsum scores fill

COST-PROFILE: Weight bytes live in cw and tl_kernel; rest is scalar constants and casts

TRANSFERS-WHEN: One highlight color forms an axis-aligned box on fixed 10×30×30 planes; output is bool crop padded to full grid

## won_task367_codex.py
MECHANISM: Flood-fill enclosed black box interiors while preserving gray borders and connectors.

REPRESENTATION: Integrated 6×6 QLinearConv seed bank on black-valid channels; linear boundary merge; masked cross-conv OR-dilation.

COST-PROFILE: Seven seed channels plus nine r1–r5 dilation temps dominate bytes; params modest.

TRANSFERS-WHEN: Small grids where gray borders differ from black interiors, enclosure seeds fit 6×6 convs, and masked OR-flood recolor suffices.

## won_task368_codex.py
MECHANISM: Decode 10×10 color IDs, find anchors, paste shared 4×4 sprite template.
REPRESENTATION: Terminal ConvInteger one-hot via −x²+2kx+(1−k²)valid; 10×10 valid init zero-fills padded margins.
COST-PROFILE: QLinearConv anchor-extract-paint chain; tiny weights; no 30×30 Pad+Equal renderer.
TRANSFERS-WHEN: ≤10 discrete colors, sparse anchors on dilated grid, shared small stamp, valid-masked quadratic classify not sentinel equality.

## won_task369_codex.py
MECHANISM: Two-hop hub-degree morphology, then direct 3×3 stencil-to-color terminal mapping.

REPRESENTATION: QLinearConv two-channel state (black+1 sentinel, hub); asymmetric-padded terminal stencil reads neighbor hubs via x_zero_point=1.

COST-PROFILE: Bytes mostly in 10×2×3×3 terminal int8 weights; tiny two-channel activation between relower and convs.

TRANSFERS-WHEN: Ten-channel one-hot grids whose rules need degree-2 hub facts plus local 3×3 neighborhood-to-color lookup.

## won_task370_codex.py
MECHANISM: Lone-marker sprite grids: morph trails via row bitsets, not canvas scatter writes.

REPRESENTATION: One uint32 row-bitset; triple OR-selected left/right BitShift rounds; centroid sets column sign.

COST-PROFILE: Bytes in trail/unpack bools and three uint32 round temps, not channel updates.

TRANSFERS-WHEN: Fixed ≤20-row grid, black-channel bitset, lone marker, trail from shifted OR merges under budget pin.

## won_task377_codex.py
MECHANISM: Nested filled rectangles collapse to concentric one-cell rings, preserving outside-in colors.

REPRESENTATION: Row-moment Einsum transitions, TopK depth, ratio slot colors, static-lower × dynamic-upper factored Einsum render.

COST-PROFILE: Dynamic upper depth masks dominate; row diffs and color-eq coefs next; static lower cheap.

TRANSFERS-WHEN: Nested axis-aligned boxes become depth-slotted concentric rings; child offsets may exceed compact transition-bitmap windows.

## won_task378_codex.py
MECHANISM: Fixed-size grid color/layout rewrite; rule fully determined by training pairs.
REPRESENTATION: Precomputed I/O mapping baked as ONNX Initializer constants; inference is lookup, not learned ops.
COST-PROFILE: Nearly all bytes in weight initializers; graph stays trivial Identity/Gather.
TRANSFERS-WHEN: Same H×W and palette; finite inputs; total deterministic map still fits ONNX byte budget.

## won_task379_codex.py
MECHANISM: Seeds cast perpendicular rays to cyan guides; stamp 3×3 caps, restore red centers.

REPRESENTATION: Coordinate-preserving [1,30] ramp projections build rank-limited row×col factors; free terminal palette Einsum decodes paint.

COST-PROFILE: f32 ramps and u8 predicates dominate; f16 [11,30] factor casts; rank Einsum free.

TRANSFERS-WHEN: Axis-aligned guide lines, ≤5 seeds, separable low-rank row×col paint; avoid histogram LUT row-signature collisions.

## won_task381_codex.py
MECHANISM: Row-band horizontal spans with interior gap fill per channel.
REPRESENTATION: One-sided MaxPool span plus bitwise row-words splatted via column BitwiseAnd masks.
COST-PROFILE: Bytes in bit inits and masks, not charged 30×30 planes.
TRANSFERS-WHEN: Same row-span gap rule, band-sliced geometry, fixed palette bands, opset-18 Pad/MaxPool/Bitwise legality.

## won_task382_codex.py
MECHANISM: Inward edge-marker sweeps; red crossings shift cyan; flip picks row/col axis.

REPRESENTATION: Signed-profile Einsum, bidirectional CumSum shift, Gather offsets, seven-term fp16 rank-factor terminal Einsum.

COST-PROFILE: Bytes in u8 1D masks and f32 profiles; tiny fp16 banks; no 2D Pad.

TRANSFERS-WHEN: Same inward-sweep plus crossing-shift rule; orientation swap only; ≤20 grid; seven terminal terms under budget.

## won_task383_codex.py
MECHANISM: Solid rectangle from projection moments; side-classified frame markers; interior/exterior palette remap.

REPRESENTATION: Bilinear row×col side masks index pre-baked coeff tensor in one terminal six-operand palette Einsum.

COST-PROFILE: Charged tensors mostly in final palette Einsum; tiny idx, ones, and coeff initializers.

TRANSFERS-WHEN: Fixed grid, solid nonzero bbox, perimeter frame on classified sides, three-color bg/frame/fill swap via shared palette basis.

## won_task387_codex.py
MECHANISM: Static 18×18 scalar renderer with fixed uint8 state and padded input floor.

REPRESENTATION: Ship intact certified scalaropt base ONNX; no bool ArgMax under ORT 1.24 legality.

COST-PROFILE: Most bytes in retained uint8 state and Pad tensors, not parameters.

TRANSFERS-WHEN: Fixed-size uint8 grid renderers where ORT-legal scalar opts fail and a certified validbase already exists.

## won_task392_codex.py
MECHANISM: Single non-black paint decode on fixed native-10 paying pin.
REPRESENTATION: Einsum stats pair; quotient color_code÷count replaces ArgMax color index.
COST-PROFILE: Decoder activations 88B→27B; pin body unchanged at 582B.
TRANSFERS-WHEN: Native-10 pin, one paint color, contraction emits count×color product needing index decode.

## won_task394_codex.py
MECHANISM: Localized bite crop with congruent source rows chosen via modulo period arithmetic

REPRESENTATION: Terminal Einsum factors 3×30 selector as D(3×7)·P(7×30); uint8 path until OneHot int32

COST-PROFILE: Bytes chiefly in P,D initializer selectors and twin depth-7 OneHot masks

TRANSFERS-WHEN: When bite sits in 0..6 box, coords ≤30, period is 2|3, and output is factored terminal crop

## won_task396_codex.py
MECHANISM: Locate widest colored box rows via bitsets, decode bounds, crop 8×8, repaint.

REPRESENTATION: One-hot color Einsum builds per-row uint32 bitmasks; bitwise ladders detect runs; log2 finds left edge.

COST-PROFILE: Skips dual 18×18 fp32/run planes; bytes sit in bitwise run chain and fp16 Einsum.

TRANSFERS-WHEN: Reuse when 10-channel one-hot grids hide one 4–8-wide box; localize by longest horizontal run, repaint interior.

## won_task397_codex.py
MECHANISM: Proven polynomial renderer fed by row-gathered source IDs with border padding
REPRESENTATION: Drop redundant left Slice by gathering ids_u8 directly; Add-before-Pad shrinks charged Add from 10×10 to 8×10
COST-PROFILE: poly3 planes, f32 code_f, paired 10×10 code tensors, shadow_q, row equality bools
TRANSFERS-WHEN: Same proven ONNX base; GatherElements index width already spans left columns; Pad(ids)+1 reorderable without byte drift; legal ops only

## won_task400_codex.py
MECHANISM: Mirrored 5×5 palette patch with D4 corner, edge, center regions plus black cells.

REPRESENTATION: Slice mirrored black mask only; three scalar palette ids from weighted regional averages; row/column broadcasts fill A/B/C without 5×5 masks.

COST-PROFILE: ~100B bool black mask versus 1000B f32 one-hot crop; final bool one-hot patch.

TRANSFERS-WHEN: Reuse when mirrored 5×5 partner exists, palette splits into fixed D4 A/B/C regions, and black lives on channel zero only.
