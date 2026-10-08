# LANE D — Named representation builds (strongest model, ONE dedicated agent per task)

These carry the wave's biggest per-task headroom and its hardest failure history. Every
brief below encodes what was already tried — **do not repeat listed dead families**. Pack
first: `python3 runner/make_pack.py NNN`. Bar: Δ ≥ +0.15 (aim much higher). Full gate
mandatory. Members flagged CANARY ship solo at banking, never inside a chunk.

---

## task133 — cost 23,294 → best single bet (+1.4–2.9)
Rule: nested multicolor diagonal/sprite fill. Pin: QLinearConv×6+GatherND compositing.
Dead: unrolled Cast×85 (−0.61), fp grinds (9.6–12.8), w1209 rebuild (rejected).
**Thesis: sibling task132 solves the SAME cluster with an Einsum×5 closed form at cost
1,701 (17.56 pts).** Open task132's pin member (unzip from neurogolf_clean/submission.zip),
study its 5 contractions, re-derive for 133's geometry. Target ≤2,981 (score 17).

## task118 — cost 12,283 → Einsum plane-kill (+1.5)
Rule: hidden-plus recovery. ~2% of draws are provably ambiguous — agree=1.0 may be
impossible; build competition-correct and let the secret gate decide (do NOT memorize;
draw8 was the historical blocker, check it explicitly). Evidence a cheap form exists:
netfit reached 99.57% cell-accuracy at 4.9 KB; bbox analysis showed a 343× working-set
shrink signal. **Thesis: single small plane + terminal Einsum/Equal renderer, ≤4 KB.**

## task286 — cost 27,338 → lattice downsample (+1.5)
Rule: 4-connected flood from a seed pair + checkerboard parity coloring. Pin: 2,395-node
bitset machine. Proven dead: MaxPool flood (true depth K=76), log-depth dilation (jumps
walls), more bit-unrolling, K<19 round caps. **Thesis (dossier P1, never built): the
corridor structure lives on a 2-spacing lattice — downsample to ≤13×13 native grid, flood
THERE (K≤~26 on the small grid), upsample + parity-color terminally.** Certify the lattice
claim from the generator source in your pack before building.

## task366 — cost 36,385 (#2 worst) → first-ever plane-kill attempt (+1–2)
Rule (verbatim in dossier): match dot-patterns to boxes in decreasing dot-count order WITH
claiming (a box once matched is consumed). Pin: foreign 672-node procedural graph — treat
it as a black-box oracle. Dead: no-claim parallel match (fails by construction, 260/266),
broadcast-correlate (185 KB), int32 ScatterND (ORT-invalid). **Thesis: crop to the ≤17×17
slot space; ≤3 boxes ⇒ claiming is a tiny sequential circuit (3 rounds of
argmax-over-remaining); Einsum dot-count signatures; terminal stamp.**

## task187 — cost 16,112 → box-frame discriminator (+1.5)
True rule: paint box interiors + extend lines; shipped member over-approximates with
border-reachability flood. Dead: pure floods (secret agree 0.973–0.993), exact rect
detectors at 209–433 KB. ~0.08% ambiguity — secret gate decides. **Thesis: keep the u8
flood core but add a <5 KB frame discriminator (per-row/col run-length signatures via
ConvInteger + Gather LUT), enabling deletion of the 8–14 KB approximation planes.**

## task219 — cost 9,845 → faithful Einsum row-match (+1)
Rule: band completion per template. GRAVEYARD WARNING: decision-tree "wins" here were
memorizers (secret 0.02–0.03) — anything keyed to visible draws dies. Sibling task176
solves its analogue at 19.90 with an Einsum closed form. **Thesis: template row-match as
one contraction: Einsum(input-rows, template-bank initializer) → argmax → terminal
row-render.** CANARY at banking (tanker-list history).

## task243 — cost 16,054 → a better bitset schedule (+0.5–1.5)
Rule: flood color-1 through 4-connected zero cells; paint reached zeros blue. The catalog's
`sched37` schedule (BitwiseOr×74 + BitShift×36 + Gather×40) demonstrated +0.63 — 516
gate-passers exist, so the task is well-understood. **Thesis: NEW schedule with fewer
propagation rounds via certified diameter (check generator for max region span) + packed
uint32 rows; opset 18.** Beat sched37's cost, don't re-pack the pin.

## task002 — cost 15,090 → border-flood family (+0.6–1)
Rule: enclosed-region yellow fill; generator-ambiguous on "bridge rectangles" (agree≈0.98
cap). A catalog Einsum+BitShift border flood at +0.60 DID transfer on Kaggle once (#p334)
— competition-correct beats generator-faithful here. **Thesis: rebuild that family as a
novel SHA, cheaper: border-seeded packed flood, terminal Equal renderer.**

## task370 — cost 6,989 → compress the einsum_bitset (+0.3–0.6)
Pin already upgraded to the einsum_bitset family (16.148). **Thesis: second-generation
compression of the same family — fuse the Einsum×4 into fewer contractions, shrink the
BitShift state to certified sprite span.** Lower priority; take after your first task.

## task157 — cost 7,243 → pin forensics (+0.5–1)
Rule: guarded exact sprite-to-hole matching with priority rounds. The LIVE pin member
scores +0.45 over every catalog build and is NOT in the catalog — nobody knows why it wins.
**Thesis: disassemble the pin member (op dump in your pack), identify its trick, rebuild
smaller; separately price the historical w2097 dynamic matcher idea (+1.04 once).**

## task101 — cost 14,867 → backbone rebuild (+0.5–1) — CANARY
Rule: scale-m red template match, consume, stamp blue. Dead: ConvStamp (45–200 KB), Einsum
(m is data-dependent — invalid), grind (0/262). **Thesis: ens7215-style ArgMax/
GreaterOrEqual backbone rebuilt as a novel graph, f32 output (load-bearing).**

## task255 — cost 10,809 → DISTILLATION agent (+0.5–1.5) — also L1 zero-suspect
Rule officially "unclear"; only 1–2 gate-passers exist. **Protocol: run the pin member on
2–5k generator draws (arc_gen.py generate a64e4611 N), cluster input→output deltas by
object features (bboxes, colors, counts, symmetry), state the rule, THEN build cheap
(Einsum edge-geometry + QLinearConv dilation was the only survivor family).**

## task285 — cost 19,967 → DISTILLATION (+1–2) — CANARY (tanker list)
Rule (partial): each satellite CC gets a reflected copy (chained reflection). 0/35
sessions ever produced a build. **Thesis: distill exact reflection law from generator
source + oracle runs; then a task176-style closed form (reflection = fixed index Gather).**
Ship ONLY as solo canary.

## task233 — cost 36,393 (#1 worst) → DISTILLATION (+1–3) — CANARY
Rule: rot0 generator-matcher + visible marker overrides; "match logic incomplete". Einsum
proven to have NO ceiling here — do not build pure-Einsum. Dead: Gather/Expand and And×500
grinds (~10). **Thesis: object-substrate match — extract per-object signatures (peel +
ConvInteger moments), match via small LUT, stamp terminally. Use the pin as oracle on 5k
draws to pin down the override rule.**

---

## Secondary briefs (take after a primary completes)

- **task198** (12,298): border classifier (solid→green, broken→yellow), bool output
  load-bearing. Rebuild the 79-op classifier with row/col run-length signatures; keep bool out.
- **task138** (10,514): crop rect + extend interior markers as rays. First VERIFY the
  "gemini att27 +2.0" claim from the GOLD dossier (find the artifact, price it); then
  ConvInteger→dual-Einsum rewrite (+0.27 measured idea).
- **task205** (10,236): isolated dots extend along row+col. Einsum×3 painter exists; build
  a leaner 2-contraction variant (project → paint), terminal Equal.
- **task066** (9,857): U/S connector path. Kill the forced 1.6 KB fp32 plane with a fused
  uint8 cyan-extract; keep the Einsum moment-coordinate localization.
- **task025** (8,159): guide-line slot stamping (injective, agree=1.0 achievable). The
  deepmin13 note claims +4.39 headroom (UNVERIFIED — treat as rumor): sparse slot-index
  Gather stamp instead of Einsum×7 phase-projection.
