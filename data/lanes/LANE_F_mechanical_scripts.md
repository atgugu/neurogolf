# LANE F — Value-exact mechanical transforms (SCRIPTS, not LLM workers)

Do not spend model tokens here — these are deterministic identities. One engineer-agent
ports the proven experiment scripts from `/tmp/lever_hunt/` (they ran against the shipped
fleet on 2026-07-06) into batch transforms, then gates each output member normally.

1. **Color-ID Conv → Slice collapse** (+2.7 measured on worst-40, 19 tasks):
   proven identity `Equal(Conv_colorID(input), k) == input[:,k]` — port
   `/tmp/lever_hunt/exp_c_colorid_scan.py` (it lists the 19 task IDs) + `exp_b_labelgrid.py`
   (the proof). Replace the Conv+Equal pair with a Slice; value-exact by construction.
2. **bool→float Cast elimination** (+0.71, 33 tasks): `exp_i_cast.py` lists sites where the
   consumer accepts bool — drop the Cast, keep semantics.
3. **Duplicate-initializer CSE** (+1–2, 108 tasks): `exp_j` inline scan found 184 dup
   tensors / 2,553 elements — merge references to one initializer.

**Discipline:** value-exact or nothing (broad dtype sweeps are the historical −16…−31 tank
class). Every transformed member still goes through `fast_verify.py NNN out.onnx --bar 0.0`
+ `--full` before LEDGER (bar 0 is fine here — these are sub-0.15 wins that bank as a
chunk). Bank all Lane F members together in their own ≤12-member chunks.

## 2026-07-12 — Engine A: machine-generated tickets are live
`../tickets/TICKETS.csv` (+ per-task `../tickets/taskNNN.md`) now carries 1004 tickets
from `runner/factor_sweep.py` (deterministic scan of all 400 pin members, pin b3407fa4):
28 value-exact CONV_SHRINK rewrites (Σ≈+0.77, A/B-verified byte-identical under ORT),
36 exact-dup FACTOR_PAIR groups, plus SPARSE_INIT / STORED_TABLE / I64_STATE /
TAIL_FOLD / ATTR_MOVE / CONV_GEOMETRY heuristic leads with exemplar pointers.
Discipline unchanged: value-exact or nothing; every output through
`fast_verify.py NNN out.onnx --bar 0.0 --full` then `submit_result.py NNN out.onnx
--lane F` (Lane F bar is 0.0 in submit_result — byte-identical rewrites carry the pin's
own transfer risk). Worker builds that gate correct but miss their bar also land here
automatically as MISSED_TARGET donor tickets (submit_result exit 3).
