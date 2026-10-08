# Factor sweep summary — Engine A (factor_sweep.py)

Pin: 0965029d, 400 members scanned from `$NEUROGOLF_CLEAN/submission.zip`.
Pricing: `/workspace/minimalist/data/pin_pv2_70_263_pertask.csv`. Points law: score = 25 - ln(cost); est_pts = ln(C/(C-s)) for value-exact saving s.

## Totals per flag class

| flag | tickets | tasks | value-exact (Y) | exact els |
|------|---------|-------|-----------------|-----------|
| SPARSE_INIT | 294 | 166 | 0 | 0 |
| CONV_SHRINK | 5 | 5 | 5 | 122 |
| I64_STATE | 209 | 209 | 0 | 0 |
| STORED_TABLE | 20 | 19 | 0 | 0 |
| FACTOR_PAIR | 156 | 87 | 39 | 98 |
| TAIL_FOLD | 50 | 50 | 0 | 0 |
| DTYPE_NORM | 18 | 18 | 2 | 0 |
| ATTR_MOVE | 196 | 47 | 0 | 0 |
| CONV_GEOMETRY | 8 | 7 | 0 | 0 |
| SCAN_FAIL | 0 | 0 | 0 | 0 |

**Total tickets:** 956 across 328 tasks.
**Sigma est_pts over value-exact tickets (independent sum):** 0.1411 pts.
**Sigma est_pts with per-task joint accounting (savings summed per task before ln):** 0.1411 pts.

## Top 30 tickets by est_pts (value-exact first)

| # | task | flag | est els | est pts | exact | evidence |
|---|------|------|---------|---------|-------|----------|
| 1 | task283 | CONV_SHRINK | 30 | 0.0572 | Y | Conv 'enc_f' w=enc_w[1, 10, 2, 2] -> spatial [1, 1] dil=[1, 1] pads=([0, 0],[-20, -20]) sa |
| 2 | task022 | FACTOR_PAIR | 12 | 0.0279 | Y | exact-dup x3 float32[3, 2] (off1,off2,off3) save=12 |
| 3 | task343 | CONV_SHRINK | 2 | 0.0075 | Y | Conv 'diff4' w=diff4_w[1, 1, 1, 10] -> spatial [1, 8] dil=[1, 1] pads=([0, 0],[0, -2]) sav |
| 4 | task358 | FACTOR_PAIR | 15 | 0.0069 | Y | exact-dup x2 float16[5, 3] (col1_sel,row1_sel) save=15 |
| 5 | task209 | CONV_SHRINK | 30 | 0.0040 | Y | Conv 'color_f' w=cw[1, 10, 2, 2] -> spatial [1, 1] dil=[1, 1] pads=([0, 0],[-10, -10]) sav |
| 6 | task121 | FACTOR_PAIR | 1 | 0.0036 | Y | exact-dup x2 uint8[] (one_u8,w_zp_u8) save=1 |
| 7 | task324 | CONV_SHRINK | 30 | 0.0035 | Y | Conv 'color_f32' w=color_kernel[1, 10, 2, 2] -> spatial [1, 1] dil=[1, 1] pads=([0, 0],[-1 |
| 8 | task055 | FACTOR_PAIR | 4 | 0.0034 | Y | exact-dup x2 int64[4] (h8_edge_starts,v8_edge_starts) save=4 |
| 9 | task117 | FACTOR_PAIR | 8 | 0.0023 | Y | exact-dup x5 int64[2] (t10,t14,t18,t22...) save=8 |
| 10 | task226 | FACTOR_PAIR | 1 | 0.0018 | Y | exact-dup x2 uint8[] (a_col_d_middle,a_row_d_middle) save=1 |
| 11 | task119 | FACTOR_PAIR | 1 | 0.0018 | Y | exact-dup x2 float32[1] (one_f_new,one_row_new) save=1 |
| 12 | task259 | FACTOR_PAIR | 1 | 0.0017 | Y | exact-dup x2 int64[1] (red_axis2_i64,unsq2_i64) save=1 |
| 13 | task259 | FACTOR_PAIR | 1 | 0.0017 | Y | exact-dup x2 int64[1] (red_axis3_i64,unsq3_i64) save=1 |
| 14 | task069 | FACTOR_PAIR | 2 | 0.0015 | Y | exact-dup x2 int64[2] (flip_axes,flip_starts) save=2 |
| 15 | task096 | FACTOR_PAIR | 4 | 0.0013 | Y | exact-dup x5 uint8[] (m_code_2_u8,m_code_4_u8,m_code_5_u8,m_code_6_u8...) save=4 |
| 16 | task319 | FACTOR_PAIR | 5 | 0.0012 | Y | exact-dup x2 int32[5] (att24_range5_i32,safe_name_4) save=5 |
| 17 | task213 | FACTOR_PAIR | 1 | 0.0012 | Y | exact-dup x2 int64[1] (end7,start7) save=1 |
| 18 | task165 | FACTOR_PAIR | 3 | 0.0011 | Y | exact-dup x4 int64[1] (axis_row,one_1,row0_start,shape1) save=3 |
| 19 | task286 | CONV_SHRINK | 30 | 0.0011 | Y | Conv 'd_ci' w=d_Wci[1, 10, 2, 2] -> spatial [1, 1] dil=[1, 1] pads=([0, 0],[-5, -5]) save= |
| 20 | task012 | FACTOR_PAIR | 1 | 0.0011 | Y | exact-dup x2 int32[1] (t16,t19) save=1 |
| 21 | task088 | FACTOR_PAIR | 1 | 0.0009 | Y | exact-dup x2 float32[] (half_f,q_half_f) save=1 |
| 22 | task002 | FACTOR_PAIR | 8 | 0.0008 | Y | exact-dup x2 uint8[1, 8] (t141,t147) save=8 |
| 23 | task112 | FACTOR_PAIR | 1 | 0.0007 | Y | exact-dup x2 int32[1] (slice_end_chan,three_i) save=1 |
| 24 | task377 | FACTOR_PAIR | 1 | 0.0006 | Y | exact-dup x2 int64[1] (ax0,s0) save=1 |
| 25 | task117 | FACTOR_PAIR | 2 | 0.0006 | Y | exact-dup x3 uint8[] (t42,t57,t61) save=2 |
| 26 | task117 | FACTOR_PAIR | 2 | 0.0006 | Y | exact-dup x3 float32[] (t56,t58,t60) save=2 |
| 27 | task396 | FACTOR_PAIR | 1 | 0.0005 | Y | exact-dup x2 uint8[] (q_zp,zero_u8) save=1 |
| 28 | task205 | FACTOR_PAIR | 1 | 0.0005 | Y | exact-dup x2 int64[1] (axRed,one) save=1 |
| 29 | task363 | FACTOR_PAIR | 1 | 0.0005 | Y | exact-dup x2 float32[] (one_f,one_scale) save=1 |
| 30 | task002 | FACTOR_PAIR | 4 | 0.0004 | Y | exact-dup x3 int64[2] (t139,t145,t151) save=4 |

## Sanity anchors

- task138 CONV_SHRINK: NOT FOUND in live pin — pin member differs from expectation.
- Exact-dup CSE (same dtype+shape+bytes, drop-in, Y): 39 groups, 92 tensors involved, 98 redundant els, across 26 tasks. Cross-shape value-aliases (same dtype+bytes, N): 88 groups, up to 1512 more els. Combined: 1610 els across 74 tasks (previous older-pin raw-value measurement: ~184 dup tensors / 2553 els / 108 tasks — same order of magnitude).
- SCAN_FAIL: 0 (none)
- Pin sha256[:8] of scanned zip: 0965029d.
