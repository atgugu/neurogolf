# task004 — factorization tickets

Pin: 0965029d (`$NEUROGOLF_CLEAN/submission.zip:task004.onnx`). Pin cost: 1024.0 (score 18.0685).

## Ticket 1: SPARSE_INIT [N] — est 172 els

Tensor `base` (init) dtype=float32 shape=[2, 10, 10]: nnz=28/200 (density 0.140). Upper-bound saving numel-nnz = 172 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`

## Ticket 2: SPARSE_INIT [N] — est 43 els

Tensor `col_factor` (init) dtype=float32 shape=[2, 30]: nnz=17/60 (density 0.283). Upper-bound saving numel-nnz = 43 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`

## Ticket 3: SPARSE_INIT [N] — est 96 els

Tensor `eq_l0` (init) dtype=float32 shape=[4, 2, 16]: nnz=32/128 (density 0.250). Upper-bound saving numel-nnz = 96 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`

## Ticket 4: SPARSE_INIT [N] — est 96 els

Tensor `eq_l1` (init) dtype=float32 shape=[4, 2, 16]: nnz=32/128 (density 0.250). Upper-bound saving numel-nnz = 96 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`

## Ticket 5: SPARSE_INIT [N] — est 90 els

Tensor `eq_r0` (init) dtype=float32 shape=[4, 30]: nnz=30/120 (density 0.250). Upper-bound saving numel-nnz = 90 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`

## Ticket 6: SPARSE_INIT [N] — est 90 els

Tensor `eq_r1` (init) dtype=float32 shape=[4, 30]: nnz=30/120 (density 0.250). Upper-bound saving numel-nnz = 90 elements (NOT exact — factorized form has its own operand cost).

**Recipe** `R1_SPARSE_FACTOR`: refactor: rank-k/trig/radix factorization into terminal contraction
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task073.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task321.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task220.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task257.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task344.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task296.onnx`
