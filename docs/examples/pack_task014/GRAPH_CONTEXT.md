# Trusted graph context — task014

Status: **EXACT PER-MEMBER KAGGLE-PAYMENT PROOF**

- staged read-only graph: `./refs/kaggle_proven_best.onnx`
- full SHA-256: `8212294488a3d661cbfd2c95cd8925f9adfda46b06059b67a695180398a090d8`
- manifest key: `82122944`
- Kaggle-paid points: **17.1593**
- reconciliation proof: probe#804 gap+0.00
- source corpus path: `$NEUROGOLF_CLEAN/knowledge_proven_paying/task014/82122944.onnx`


## How to use this reference

Inspect it before designing the build. It is the trusted correctness oracle and proven
behavioral baseline, but its architecture is **not** assumed optimal. Preserve the exact
reference; write experiments elsewhere. A novel family may beat it, but local gates cannot
upgrade novel bytes to Kaggle-proven status.

## Compact disassembly

```text
nodes=35 initializers=12 initializer_elements=35 file_bytes=2426
ops=Cast:5, Add:4, ArgMax:4, Sub:4, Concat:3, Slice:2, Where:2, Equal:2, Einsum:2, Greater:2, ReduceSum:1, LessOrEqual:1, ArgMin:1, Pad:1, Reshape:1
inputs=input: dtype=1 shape=[1, 10, 30, 30]
outputs=output: dtype=9 shape=[1, 10, 30, 30]
node listing:
  000 ReduceSum input,axes_counts -> counts10_f32
  001 Cast counts10_f32 -> counts10_f16
  002 Slice counts10_f16,one_i64,thirty_i64 -> counts9_f16
  003 LessOrEqual counts9_f16,zero_f16 -> no_color
  004 Where no_color,big_f16,counts9_f16 -> counts_safe
  005 ArgMin counts_safe -> tgt_idx0
  006 Add tgt_idx0,one_i64 -> tgt_color
  007 Equal arange10,tgt_color -> sel_bool
  008 Cast sel_bool -> sel_f32
  009 Einsum input,sel_f32 -> row_counts
  010 Einsum input,sel_f32 -> col_counts
  011 Greater row_counts,zero_f32 -> pres_row
  012 Cast pres_row -> pres_row_u8
  013 Greater col_counts,zero_f32 -> pres_col
  014 Cast pres_col -> pres_col_u8
  015 ArgMax pres_row_u8 -> first_row
  016 ArgMax pres_row_u8 -> last_row
  017 ArgMax pres_col_u8 -> first_col
  018 ArgMax pres_col_u8 -> last_col
  019 Add tgt_color,one_i64 -> tgt_color_end
  020 Add last_row,one_i64 -> row_end
  021 Add last_col,one_i64 -> col_end
  022 Concat zero_i64,tgt_color,first_row,first_col -> slice_starts
  023 Concat one_i64,tgt_color_end,row_end,col_end -> slice_ends
  024 Slice input,slice_starts,slice_ends -> crop_f32
  025 Cast crop_f32 -> crop_u8
  026 Sub row_end,first_row -> crop_h
  027 Sub col_end,first_col -> crop_w
  028 Sub thirty_i64,crop_h -> pad_h
  029 Sub thirty_i64,crop_w -> pad_w
  030 Concat zero_i64,zero_i64,zero_i64,zero_i64,zero_i64,zero_i64,pad_h,pad_w -> pads
  031 Pad crop_u8,pads,pad_value -> canvas01
  032 Reshape sel_bool,pal_shape -> sel4d
  033 Where sel4d,pal_one,pal_base -> pal
  034 Equal canvas01,pal -> output
```
