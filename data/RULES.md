# RULES — read before building anything (all facts verified against the grader-exact replica, 2026-07-06)

> ## ⚠️ 2026-07-10: SCORER WAS HACKED — READ `SCORER_FIXED_2026-07-10.md` FIRST
> The framework `neurogolf_utils.py` carried local-only `_ngolf_*` memory-skip patches
> (relower / paint-tail / task119/162/224 / reflect / hodel-ray / indexed-u8). Kaggle counts
> those tensors — models tuned to the skips priced up to +7.3 pts too high and SHORTED on the
> LB (probes #p807…). The skips are now neutralized in place. **All prices computed before
> 2026-07-10 are suspect — re-price with `runner/fast_verify.py` before claiming any win.**
> Never trust a "free tensor" without a Kaggle probe. The live pin is whatever
> `data/PIN_SHA.txt` + `data/pin_pv2_70_263_pertask.csv` say (refreshed on every repin) —
> never trust a pin sha/score quoted in prose docs.

> ## INTEGRITY LOCK — automatic NO
> Workers and deciders may NEVER read, edit, or add matchers/tags/hooks to any scorer/pricer
> (`neurogolf_utils.py`, `scoring_v2.py`, `runner/ngolf.py`, `runner/fast_verify.py`). A plan or
> build that does, or that calls a tensor scorer-recognized/native-priced/tagged/skip/free, is
> rejected. Only the graph input, graph output, and one terminal node's internal workspace are
> free; every other node output is charged. `data/SCORER_SHAS.txt` is the wave's authority.

## 1. Cost model (what you are minimizing)
- `score = max(1, 25 − ln(params + memory))` per task. To gain +1 pt, cut cost by e (2.72×). +0.15 ⇒ cost × 0.861.
- **params** = element COUNT of every initializer + Constant-node tensor (dtype irrelevant!
  ⇒ an int64 initializer element carries 64 bits per cost-unit — pack constants).
- **memory** = Σ over every NODE OUTPUT tensor of `elements × dtype_bytes` (static shape
  inference, strict; runtime profile can only raise it). Charged ONCE per tensor name —
  reuse is free.
- **FREE:** the graph input (`input`, [1,10,30,30] fp32, 36 KB of one-hot — mine it) and the
  graph output (`output`) — the FINAL op's output costs 0. Terminal renderers are
  structurally privileged: end with ONE op that expands small state → full output
  (Equal-vs-K broadcast, OneHot-substitute via Equal, Gather, Einsum, ConvTranspose, Pad).
  The SECOND-to-last tensor is charged — keep it small/uint8.
- Dtype bytes: fp32=4, fp16=2, int64=8, int32/uint32=4, uint8/bool/int8=1. Nothing < 1 byte
  (int4 maps to 1 byte — useless). Sub-byte = bit-packing into int32/uint32 lanes only.
- Filesize ≤ 1.44 MB.

## 1b. Think in RATIOS — the log-scale law (why micro-shavings lose)
`score = 25 − ln(cost)` ⇒ points buy cost **ratios**, never byte differences:
−10% = +0.11 (BELOW the +0.15 bar — worthless) · −25% = +0.29 · **−50% = +0.69** ·
−75% = +1.39 · −90% = +2.30.
Evidence: all 74 historical Δ≥0.8 wins were STRUCTURAL replacements (plane-kill, renderer
swap, representation change); **zero** came from shaving an existing design; 21 waves of
micro-golf banked 0. Top teams average ~134 B/task — graphs ~7× cheaper than ours provably
exist for most rules.
Therefore your TARGET is a **≥2× cost cut** or the next score band (17 ⇒ ≤2,981 B ·
18 ⇒ ≤1,097 B · 19 ⇒ ≤403 B). The +0.15 bar is only the REGISTRATION FLOOR, not the goal.
If your plan keeps most of the pin's nodes and tweaks dtypes/shapes, you are
micro-shaving — stop and price a REPLACEMENT family instead. **RECONSTRUCTION IS THE
DEFAULT POSTURE: your first move on every task is to re-derive the rule and design a
from-scratch graph, never to edit the pin.** Small-certain-win mode is
legitimate only as the designated final-attempt fallback (the orchestrator says so
explicitly) and in Lane F (mechanical value-exact scripts).

## 1c. The one-node tier (and why it's already empty)
A graph whose ONLY node output is the graph output has ZERO memory; with ≤1 initializer
element, cost ≤ 1 = **exactly 25.000** (that is all task179/task241 are). The grader
thresholds output as raw>0, so one Conv+bias = a per-cell linear-threshold classifier
(900p → 18.21; 1×3/3×1 → 19.30) with zero intermediates. **Both tiers were exhaustively
swept on 2026-07-07 against every graded example of every task: the exact micro-vocabulary
(identity/transpose/flips/rot180/shifts ±3/dilation/pointwise-recolor) matched 0 remaining
tasks, and single-Conv threshold fits matched 0 of 288 sub-19.15 tasks.** Do NOT spend
attempt time re-deriving these — pursue a one/two-node design only when the decider cites
task-specific evidence. What survives of the idea: before any multi-op design, ask "what
is the FEWEST nodes this rule needs?" — every node above the minimum charges its output.

## 2. Hard bans (unscorable or auto-FAIL — do not emit)
- Ops: `Loop, Scan, If, NonZero, Unique, Compress`, anything `Sequence*`, local functions,
  any domain other than `""`/`ai.onnx` (no com.microsoft).
- Dynamic/symbolic dims anywhere (strict shape inference must resolve every tensor).
- More than 1 input or 1 output; input/output touching initializers.
- Input MUST be named `input` [1,10,30,30] float32; output MUST be named `output`.

## 3. ORT 1.24 trap list (each one cost a past wave — encode, don't rediscover)
- Clamp EVERY computed Gather/GatherElements index (the grader's profiling run uses a
  real grid, but OUR gate probes can hit edge values; OOB = crash = FAIL).
- uint8 is DEAD for: CumSum, ReduceSum, Conv, MatMul. Use int32, or ConvInteger /
  MatMulInteger (they take uint8/int8 in, int32 out).
- Bitwise ops (BitwiseAnd/Or/Xor, BitShift) need **opset 18** — only bump opset if used.
- fp16 `Mod` needs `fmod=1`. fp16 Conv on integer color codes silently diverges (23.0156≠23)
  — never carry color IDs in fp16 through Conv.
- `OneHot` is NOT_IMPLEMENTED in ORT 1.24 for common type combos — use
  `Equal(state[1,1,30,30], arange_K[1,K,1,1])` broadcast as the one-op renderer instead.
- Conv bias length MUST equal out_channels (short bias = OOB read = the −18.88 plague).
- NEVER put inf/NaN in weights: non-finite arithmetic is ENVIRONMENT-SENSITIVE — the
  same bytes scored 0/266 in one local env and full marks on the LB (2026-07-12).
  scan_razors WARN-tags it and bank routes such members solo-canary; local gate evidence
  on them (pass OR fail) proves nothing. Use finite sentinels (±5000 class) instead.
- Keep the pin member's OUTPUT dtype (bool/f32/u8 conventions are load-bearing on several
  tasks; task080's bool→rename lost −12.5 on Kaggle).

## 4. Acceptance contract (what "done" means)
1. `fast_verify.py NNN model.onnx` → PRICE must show Δ ≥ +0.15 vs pin,
   else COST-REJECT — do not gate, do not micro-trim; delete a charged tensor FAMILY.
   The registration bar is the FLOOR; the WIN condition is your pack's rule-class
   compile target ("rule-class compile target: cost ≤ N").
2. fast gate (30 real generator draws) PASS → then `--full`: gate124 300 draws + freshgen
   `--secret` (agree = 1.0000). Exit 0 = registerable.
2b. TICKET, not NO: if `submit_result.py` exits 3, your build gated CORRECT but missed
   the bar — it is registered as a factorization-ticket donor (`data/tickets/`) for the
   mechanical sweep. End your message with `TICKET taskNNN`. Never delete a correct
   artifact and never write NO.md for a ticketed build.
3. Invariants may come ONLY from the generator source in your pack (the "source
   certificate"). Anything inferred from sample grids is the task182 tank class (−14.6).
4. AMBIG-flagged tasks (002, 118, 187 in our lanes): agree=1.0 may be impossible by
   construction — target competition-correct behavior; the secret gate decides. Never
   hand-craft memorizers (secret 0.02 class on task219 history).

## 5. Idiom menu (measured wins — your vocabulary, biggest first)
- **COMPILE THE RULE — terminal tensor-network Einsum** (study7300: 30+ verified
  instances vs this base, mean +0.44; task350 +0.94 banked 2026-07-12; task293 +3.21):
  the WHOLE rule as ONE multi-operand Einsum — free input repeated 2–10× (pairwise/
  relational features), 2–5 tiny reused factor matrices (10×10 relations, rank-2/3
  palettes, trig/radix position codes), boolean branches as scalar selector operands,
  integer-exact sentinels (±1, 5000, −1000, −0.001) through the (raw>0) decode. Operand
  ORDER = ORT speed: interleave each input copy with its eliminating factors (measured
  4.1 s → 0.008 s). Worked exemplars: `data/intel/compile_exemplars/` (29, byte-verified);
  builders: `runner/ngolf_einsum.py` (build_einsum_model/out_einsum/parity) and
  `runner/compile_circuit.py` (circuit-spec JSON → scheduled Einsum). Your pack's
  "COMPILE THE RULE" section carries the class budget — write the equation and factor
  table BEFORE any ops.
- **Plane-kill** (all 74 historical Δ≥0.8 wins share it): never materialize a
  [1,10,30,30] fp32/f16 plane; compute on ONE small uint8/bool plane; emit 10ch at the
  free final node.
- **Terminal Einsum closed form** (task287 +2.08, 253 +1.94, 065 +1.63): rule as one
  contraction of input against small initializer tables; internal scratch is free, only
  the (small or final) output is charged.
- **QLinearConv / ConvInteger minform** (task368 +1.06, 388 +1.27): int8-weight stencils,
  uint8 activations; 40–60% cheaper than fp Conv pipelines.
- **Bbox-crop → compute native → Pad** (task235 +4.02, 183 +3.27): with a SOURCE
  certificate for the bound (M_proof in your pack).
- **Grouped/depthwise & anisotropic Conv** (task130 +4.58: 2925→30 B): 1×K / K×1 kernels,
  group=channels; parity/counting via bias thresholds.
- **Gather LUT** (710 historical wins): precompute a uint8 table initializer, index with a
  small computed key, Gather as (or just before) the terminal op.
- **Bitset schedule** (task243 sched37 +0.63): pack rows into int32/uint32 lanes
  (BitShift+And/Or, opset 18) — but packing cuts BYTES not ROUNDS; if the pin already
  unrolls hundreds of rounds, you need a different schedule, not tighter packing.
- **MaxPool geodesic flood** (mask ∧ dilate, uint8): the cheapest generic constrained
  flood — ~440–900 B/round; use certified diameter to cap rounds.
- **Row/col separable projections + moment coordinates** (task064 MRMC, 066): reduce to
  per-row/col statistics via Einsum/ReduceMax, act on [30]-vectors, re-render terminally.

## 6. Never touch
018 023 209 (proven ambiguous—irrecoverable) · 076 080 350 338 204 (true Kaggle tankers;
350's compiled tensor-network member was banked offline 2026-07-12 as solo-canary —
STILL never touch it in waves) · 054 064 074 077 145 158 324 349 364 (proven floors —
but note task138's FLOOR fell to a value-exact conv-shrink ticket: floors are
family-relative; only tickets/scripted value-exact rewrites may cross them, never wave
attempts) · 128 179 241 (no headroom) · 144 230 (razor members — being fixed separately).

## 7. Discipline that past waves paid for (encode, don't rediscover)
- **Budget FIRST.** Before any code: table every planned tensor (shape·dtype·bytes) +
  params; if the paper total doesn't beat the bar, redesign. 21 waves died to
  "correct-but-expensive": a gate-passing graph that costs more than pin = 0 points.
- **Suppress optional outputs.** EVERY declared node output is charged. Use `""` for
  optional outputs you don't consume (GRU's Y, MaxPool Indices, Dropout mask…) — the
  scorer skips empty names. Free savings.
- **Hodel verdict policy** (in your pack): TRUE-RULE ⇒ implement it. PROVISIONAL/PARTIAL ⇒
  arc-gen diverges from Hodel's program — decode the variant from failing draws
  (`python3 candidates/hodel.py NNN --diff` in neurogolf_clean) before building.
- **Inspect failures, don't guess:** `python3 candidates/diff_one.py <task> <model>` (in
  neurogolf_clean) renders the wrong cells of a failing draw. One look beats three blind
  retries.
- **Steal from winners:** `data/exemplars/` holds real fleet build.py files (Einsum-terminal,
  u8-Einsum, bitset-schedule, neighbor-Conv). Your pack also embeds this task's own past
  fleet build script when one exists — reuse its scaffolding, beat its cost.
- **Determinism:** never emit ops whose result can depend on uninitialized memory (short
  bias, unclamped index). Those become Kaggle coin-flips that poison whole submissions.
- **Proven-corpus first:** when the pack supplies an exact-reconciled Kaggle-paying design for
  this task, iterate from it. A novel op×dtype family with no proven same-task ancestor is a
  solo-probe candidate, never a mega-batch member.

## 8. Safe toolbelt (vetted, high-ROI — everything runs from $NEUROGOLF_CLEAN)
- **`python3 candidates/hypo.py NNN rule.py`** — THE rule verifier: write your decoded rule
  as a numpy function `def rule(grid): ...` and test it against ALL graded examples
  (train/test/arc-gen). MANDATORY before any ONNX when your strategy is rebuild-from-rule
  or distillation (project policy: "hypo.py must pass 100% of every visible example before
  any ONNX"). Iterating a rule in numpy takes seconds; in ONNX it takes minutes.
- **`extracted/taskNNN.json`** — every graded example (input/output grids, all splits) as
  raw JSON. Read the actual grids; don't infer the rule from prose alone.
- **`python3 candidates/anatomy.py MODEL.onnx --task NNN --top 15`** — grader-faithful
  per-tensor memory bill traced over the task's REAL examples. Also
  `anatomy.py budget u8:1,1,30,30 f32:30,30 --params 60` prices a PLANNED design from a
  spec string (use this if you build without ngolf).
- **`python3 candidates/diff_one.py NNN MODEL.onnx [--split arc-gen --max 5 --full]`** —
  renders failing examples INPUT|EXPECTED|GOT|DIFF, cropped to the mismatch; runs each
  example twice to catch state-retention bugs. (fast_verify calls it on FAIL, but call it
  yourself when you want more examples.)
- **`python3 candidates/tricks.py <keywords>`** — prior-art search: how were similar rules
  ALREADY solved? Searches every build-script docstring across all lanes, ranked with that
  task's live points (e.g. `tricks.py flood fill` before designing a flood).
- **`python3 candidates/human_annotations.py NNN`** — human rule descriptions (LARC/H-ARC)
  + human solve-rate + common WRONG outputs (= the ambiguity traps). Also in your pack.
- **`python3 candidates/polish.py in.onnx out.onnx --task NNN`** — safe finisher AFTER your
  win passes the full gate: keeps the optimizer result ONLY if outputs are value-identical
  AND cost strictly drops (exit 0 = improved, 2 = no gain, 1 = rejected). Free bytes;
  re-run fast_verify --full on the polished file before submit_result.
- **`candidates/ort124_matrix.json`** — measured op×dtype aliveness (37 golf-relevant ops
  × 6 dtypes) on the exact grader runtime with min working opsets. An op ABSENT from the
  matrix is UNTESTED, not dead — canary anything untested.
- **`python3 candidates/gate_rule.py …`** — extra independent correctness stream (checks
  vs the decoded TRUE RULE + structured edge grids, not just generator draws). Use on
  canary-class members for a third opinion; gate_pair remains the registration contract.
- `candidates/hodel.py NNN [--diff]`, `candidates/arcgen_context.py NNN`,
  `external/ARC-GEN && python3 arc_gen.py generate <hex> 5` — already in your pack.
- **FORBIDDEN:** `probe_submit.py` (submits to Kaggle — workers NEVER submit),
  `gate083.py`/`gate118.py` (obsolete gates — weaker than gate124; using them = false
  confidence), `compile.py`/Hodel-compiler (the −339 channel), `ingest_*`/`build_ensemble*`
  (repin machinery), `fix_*`/`debug_*` scraps.

## 9. Output & bookkeeping
Deliver `build.py` + `taskNNN.onnx` into `results/taskNNN/`, append to `results/LEDGER.csv`
(`task,cost,pts,delta,lane,sha8,notes`). If you give up: `results/taskNNN/NO.md` with the
floor-reason (that negative is valuable — it prevents the next agent wasting the night).
Claim a task by `touch claims/taskNNN.claim` first; skip tasks already claimed.
