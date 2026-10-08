# decider for att09 [gpt5.5-xhigh]
REFLECTION:
att1: plan said big-swing colidx30+Equal → result 1660 static, runtime phantom → HALF-RIGHT because rule/contract right, dynamic shapes missed. att2: plan said sub30 crop-native → result fail train#0 → WRONG because output contract is 30x30. att3: plan said coord/LUT Where → result mask floor 900, terminal Pad 798 static/phantom → WRONG because spatial-mask risk. att4: plan said retry LUT → result same proof, kept phantom → WRONG because dead family. Proven: rule verified by hypo all splits + full gate/freshgen; floors: sub30 invalid, LUT mask 900, colidx30 900, native crop 798 static/9148 real. My repeated error: trusting static/zeros pricing and retrying LUT.

DECISION:
pivot-family

FAMILY:
STATIC-U8-COLIDX20: keep verified rare-color bbox rule, but abandon dynamic native one-hot crop. Compose a fixed u8 color-index window, mask locally, static-pad, terminal-render with Equal. Budget 3301 ≤ 5338. Planned charged tensors: [1,1,30,30]·u8·900 count2; [1,1,20,20]·u8·400 count2; [1,1,20,20]·bool·400 count1; [10]·f32·40 count2; [30]·u8·30 count4; [4]·i64·32 count1; params 69.

BUILD-RECIPE:
1. First certify generator bounds before compiling: rare-color bbox must satisfy h≤20, w≤20, r0≤10, c0≤10 on train/eval/arc-gen pack. If false, stop; do not submit this family.
2. In `build_static_u8.py`, use ngolf/ONNX ops `Cast`, `ReduceSum`/`Einsum`, `Where`, `ArgMin`, `ArgMax` to compute rare nonzero color, selected mask, `row_hits [30] u8`, `col_hits [30] u8`, and bbox `[r0,c0,h,w]`.
3. Build `colidx [1,1,30,30] u8` with `Einsum(onehot_u8, color_ids_u8[10])` or equivalent value-exact u8 composition.
4. Static crop only: `Slice(colidx, rows r0:r0+20, cols c0:c0+20)` with fixed output `idx20 [1,1,20,20] u8`; no content-sized Slice output.
5. Make local mask using constant grids `rr20 [1,1,20,1] u8`, `cc20 [1,1,1,20] u8`: `inside20 = Less(rr20,h) And Less(cc20,w)` → `[1,1,20,20] bool`; `Where(inside20, idx20, 0)` → `idx20z [1,1,20,20] u8`.
6. Static `Pad` `idx20z` with pads bottom/right 10,10 to `idx30 [1,1,30,30] u8`.
7. Terminal renderer: `Equal(idx30, chan_ramp [1,10,1,1] u8)` as graph output `[1,10,30,30] bool`.
8. Run real-input `fast_verify`; zeros and real cost must match. Then full gate.

FAMILY-BUDGET-JSON:
{"tensors":[{"shape":[1,1,30,30],"dtype":"u8","count":2},{"shape":[1,1,20,20],"dtype":"u8","count":2},{"shape":[1,1,20,20],"dtype":"b","count":1},{"shape":[10],"dtype":"f32","count":2},{"shape":[30],"dtype":"u8","count":4},{"shape":[4],"dtype":"i64","count":1}],"params":69}

REUSE:
`rule.py`, `hypo.py`, `build.py`, `build_b.py` only as a negative reference, `NOTES.md`, banked `task014.onnx` as oracle, prior gate/freshgen commands.

AVOID:
No sub30 graph output: att2 failed first example. No coordinate/LUT/Where full mask: att3 and att4 re-proved 900B mask floor. No dynamic native one-hot crop: att3/4 798 static became 9148 real. No Pad-axis or fallback micro-shaves from `NOTES.md`.

FIRST-3-ACTIONS:
1. Edit/create `build_static_u8.py` from `build.py`.
2. Add a bound-cert script/assertion for h,w,r0,c0 over the available generator pack.
3. Compile `task014_static_u8.onnx`, then run real-input `fast_verify` comparing zeros-cost vs real-cost.

[harness: decision budget priced at 3301 B vs bar 5338 — CLOSES ✓]