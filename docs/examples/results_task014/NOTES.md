# task014 Notes

## Status
- DONE attempt #9: `task014.onnx` cost 2542, score 17.1593, delta +0.8919, sha8 `82122944`.
- Full gate passed: gate124 visible x2 + 300 fresh draws, freshgen secret agree=1.0000.
- `submit_result.py` recorded the artifact and copied the matching `build.py` into this result folder.

## Rule
Choose the least frequent nonzero input color, take that color's tight bounding box, and
return the original pixels inside that box. `rule.py` passed `hypo.py` on all splits.

## Built Family
Selected-channel sentinel canvas + dynamic palette:
1. Count nonzero colors 1..9 and choose the rarest color with `ArgMin`.
2. Use two `Einsum` projections to get row/column presence for only that selected color.
3. Slice the input over `[N, selected_color, bbox_rows, bbox_cols]`, producing a one-channel crop.
4. Cast that crop to `uint8`, pad it to a `[1,1,30,30]` sentinel canvas with value 255.
5. Final free renderer: `Equal(canvas01, pal)`, where `pal` is 1 on the selected output channel, 0 on channel 0, and unreachable sentinels on all other channels.

## Budget
- Official `score_v2`: cost 2542 = params 35 + memory 2507.
- Largest charged tensors: `canvas01 uint8 [1,1,30,30] = 900 B`, traced `crop_f32 float32 [1,1,12,14] = 672 B`, traced `crop_u8 uint8 [1,1,12,14] = 168 B`, `row_counts`/`col_counts float32 [1,30] = 240 B`.
- FAMILY-BUDGET-JSON: {"tensors":[{"shape":[1,1,30,30],"dtype":"u8","count":1},{"shape":[1,1,12,14],"dtype":"f32","count":1},{"shape":[1,1,12,14],"dtype":"u8","count":1},{"shape":[1,30],"dtype":"f32","count":2},{"shape":[8],"dtype":"i64","count":1},{"shape":[10],"dtype":"f32","count":2},{"shape":[4],"dtype":"i64","count":2},{"shape":[1,30],"dtype":"b","count":2},{"shape":[1,30],"dtype":"u8","count":2},{"shape":[10],"dtype":"f16","count":1},{"shape":[9],"dtype":"f16","count":2},{"shape":[10],"dtype":"b","count":1},{"shape":[1,10,1,1],"dtype":"b","count":1},{"shape":[1,10,1,1],"dtype":"u8","count":1},{"shape":[9],"dtype":"b","count":1},{"shape":[1],"dtype":"i64","count":13}],"params":35}

## Alternatives Repriced
- Native 10-channel crop + terminal `Pad`: `task014.onnx` from prior attempt repriced at cost 9148, delta -0.3887. The data-dependent `[1,10,h,w]` crop is the runtime balloon.
- Coordinate/LUT renderer: charged `inside_mask bool [1,1,30,30] = 900 B` before output and still needed the crop/palette logic; previous proof remains a dead end.
- Axis-tail color-index canvas: fast-passed at cost 2643, but the dynamic-palette sentinel canvas is cheaper at 2542.

## Lesson
For task014, crop only the selected one-hot channel before `Pad`; terminal `Equal` with a
dynamic palette avoids the 10-channel crop runtime balloon while preserving the fixed
`[1,10,30,30]` bool output contract.
