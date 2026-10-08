# task003 — factorization tickets

Pin: 0965029d (`$NEUROGOLF_CLEAN/submission.zip:task003.onnx`). Pin cost: 139.0 (score 20.0655).

## Ticket 1: I64_STATE [heuristic] — est 8 els

i64 state inventory (heuristic — memory charge depends on scorer's traced peak):
- `slice_en` shape=[4] numel=4 [Slice/Pad bound]
- `slice_st` shape=[4] numel=4 [Slice/Pad bound]


**Recipe** `R3_I64_TO_I32`: i64 is 8B/element in charged memory: recast to i32 where legal / replace TopK ranking with Sign+CumSum+ScatterElements or count arithmetic
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task316.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task393.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task091.onnx`
