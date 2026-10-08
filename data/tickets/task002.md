# task002 — factorization tickets

Pin: 0965029d (`$NEUROGOLF_CLEAN/submission.zip:task002.onnx`). Pin cost: 10543.0 (score 15.7368).

## Ticket 1: I64_STATE [heuristic] — est 16 els

i64 state inventory (heuristic — memory charge depends on scorer's traced peak):
- `t10` shape=[1] numel=1 [Slice/Pad bound]
- `t11` shape=[1] numel=1 [Slice/Pad bound]
- `t139` shape=[2] numel=2
- `t145` shape=[2] numel=2
- `t151` shape=[2] numel=2
- `t157` shape=[4] numel=4 [Slice/Pad bound]
- `t4` shape=[1] numel=1 [Slice/Pad bound]
- `t5` shape=[1] numel=1 [Slice/Pad bound]
- `t6` shape=[1] numel=1 [Slice/Pad bound]
- `t9` shape=[1] numel=1 [Slice/Pad bound]


**Recipe** `R3_I64_TO_I32`: i64 is 8B/element in charged memory: recast to i32 where legal / replace TopK ranking with Sign+CumSum+ScatterElements or count arithmetic
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task316.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task393.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task091.onnx`

## Ticket 2: FACTOR_PAIR [Y] — est 2 els, est 0.0002 pts

EXACT DUPLICATE initializers (3 copies, dtype=int64, shape=[1], numel=1): `t11`, `t6`, `t9`. CSE: keep one, rewire consumers — value-exact saving 2 els.

**Recipe** `R5_OPERAND_REUSE`: mode-dim packing / operand reuse / CSE
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task001.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task373.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task287.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task298.onnx`

## Ticket 3: FACTOR_PAIR [Y] — est 4 els, est 0.0004 pts

EXACT DUPLICATE initializers (3 copies, dtype=int64, shape=[2], numel=2): `t139`, `t145`, `t151`. CSE: keep one, rewire consumers — value-exact saving 4 els.

**Recipe** `R5_OPERAND_REUSE`: mode-dim packing / operand reuse / CSE
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task001.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task373.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task287.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task298.onnx`

## Ticket 4: FACTOR_PAIR [Y] — est 8 els, est 0.0008 pts

EXACT DUPLICATE initializers (2 copies, dtype=uint8, shape=[1, 8], numel=8): `t141`, `t147`. CSE: keep one, rewire consumers — value-exact saving 8 els.

**Recipe** `R5_OPERAND_REUSE`: mode-dim packing / operand reuse / CSE
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task001.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task373.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task287.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task298.onnx`

## Ticket 5: TAIL_FOLD [heuristic]

Final node chain before output `output`: Cast->Pad->Where (ops: t156, t158, output), consuming tiny constants t159[1, 10, 1, 1], t157[4]. Candidate: fold tail into the terminal contraction/renderer.

**Recipe** `R6_TAIL_FOLD`: fold the tail into the terminal contraction/renderer
**Exemplar bytes:** `/workspace/concise_docs/study7300/extracted/7431/task291.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task304.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task332.onnx`, `/workspace/concise_docs/study7300/extracted/7431/task242.onnx`
