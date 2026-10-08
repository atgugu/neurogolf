# task005 — factorization tickets

Pin: 0965029d (`$NEUROGOLF_CLEAN/submission.zip:task005.onnx`). Pin cost: 4157.0 (score 16.6675).

## Ticket 1: SPARSE_INIT [N] — est 27 els

Tensor `bg_u8_1x27` (init) dtype=uint8 shape=[1, 1, 1, 27]: nnz=0/27 (density 0.000). Upper-bound saving numel-nnz = 27 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`

## Ticket 2: I64_STATE [heuristic] — est 8 els

i64 state inventory (heuristic — memory charge depends on scorer's traced peak):
- `pad30` shape=[8] numel=8 [Slice/Pad bound]
- rank-producing ops (i64 outputs): ArgMax:r0_raw_23, ArgMax:c0_raw_25
- inferred i64 intermediates: r0_raw_23, c0_raw_25


**Recipe** `R3_I64_TO_I32`: i64 is 8B/element in charged memory: recast to i32 where legal / replace TopK ranking with Sign+CumSum+ScatterElements or count arithmetic
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task316.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task393.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task091.onnx`
