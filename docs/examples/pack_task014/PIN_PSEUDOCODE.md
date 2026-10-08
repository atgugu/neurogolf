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
