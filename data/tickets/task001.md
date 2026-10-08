# task001 — factorization tickets

Pin: 0965029d (`$NEUROGOLF_CLEAN/submission.zip:task001.onnx`). Pin cost: 190.0 (score 19.7530).

## Ticket 1: SPARSE_INIT [N] — est 153 els

Tensor `m` (init) dtype=float32 shape=[2, 30, 3]: nnz=27/180 (density 0.150). Upper-bound saving numel-nnz = 153 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`
