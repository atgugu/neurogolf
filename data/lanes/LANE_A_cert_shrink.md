# LANE A — Certified replacement (40 tasks · bar +0.15)

**Thesis (same for every task):** the pin member still computes on planes LARGER than the
generator-source-certified bound (M_proof in each block). The pin's ALGORITHM is your SPEC
(keep its output dtype) — REBUILD the implementation from scratch inside the certified box:
`Slice input to the certified box → same computation at native size → Pad → terminal renderer`,
and take the rebuild as far as the certificate allows (leaner renderer tail, cheaper dtypes,
fewer nodes) — target a ≥2× cost cut, not a trim. No rule discovery needed — the certificate
hands you the spec; the win comes from a smaller GRAPH, not from shaving the existing one.
Bar: `fast_verify.py NNN model --bar 0.15`. Special: task392 = canary-only at banking
(−70 probe history); task173 = FREEZE for representation, cert-crop 30→25 ONLY.

Worker protocol: (1) `touch claims/taskNNN.claim` (skip if exists) → (2) `python3 runner/make_pack.py NNN`
→ (3) read packs/taskNNN/ATTACK.md + RULES.md → (4) build & iterate with runner/fast_verify.py → (5) LEDGER or NO.md.

### task099  (gen 444801d8)
- pin: cost 1502 (params 62 / mem 1440) → 17.685 pts · **register at cost ≤ 1292** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 99` → packs/task099/ATTACK.md

### task348  (gen db3e9e38)
- pin: cost 1705 (params 76 / mem 1629) → 17.559 pts · **register at cost ≤ 1467** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 7, 8]
- pack: `python3 runner/make_pack.py 348` → packs/task348/ATTACK.md

### task341  (gen d6ad076f)
- pin: cost 1429 (params 35 / mem 1394) → 17.735 pts · **register at cost ≤ 1229** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 341` → packs/task341/ATTACK.md

### task345  (gen d9f24cd1)
- pin: cost 1495 (params 64 / mem 1431) → 17.690 pts · **register at cost ≤ 1286** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 2, 5]
- pack: `python3 runner/make_pack.py 345` → packs/task345/ATTACK.md

### task063  (gen 2bee17df)
- pin: cost 1679 (params 58 / mem 1621) → 17.574 pts · **register at cost ≤ 1445** (Δ≥+0.15)
- cert: M_proof=14 (high) · out_law=same_as_input · palette_out=[0, 2, 3, 8]
- pack: `python3 runner/make_pack.py 63` → packs/task063/ATTACK.md

### task131  (gen 56dc2b01)
- pin: cost 3880 (params 675 / mem 3205) → 16.736 pts · **register at cost ≤ 3339** (Δ≥+0.15)
- cert: M_proof=18 (high) · out_law=same_as_input · palette_out=[0, 2, 3, 8]
- pack: `python3 runner/make_pack.py 131` → packs/task131/ATTACK.md

### task075  (gen 363442ee)
- pin: cost 1487 (params 161 / mem 1326) → 17.695 pts · **register at cost ≤ 1279** (Δ≥+0.15)
- cert: M_proof=13 (high) · out_law=same_as_input · palette_out=[0, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 75` → packs/task075/ATTACK.md

### task381  (gen ef135b50)
- pin: cost 1644 (params 24 / mem 1620) → 17.595 pts · **register at cost ≤ 1415** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 2, 9]
- pack: `python3 runner/make_pack.py 381` → packs/task381/ATTACK.md

### task037  (gen 1f876c06)
- pin: cost 2514 (params 262 / mem 2252) → 17.170 pts · **register at cost ≤ 2163** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 37` → packs/task037/ATTACK.md

### task397  (gen fcc82909)
- pin: cost 2363 (params 151 / mem 2212) → 17.232 pts · **register at cost ≤ 2033** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 397` → packs/task397/ATTACK.md

### task281  (gen b548a754)
- pin: cost 1820 (params 78 / mem 1742) → 17.493 pts · **register at cost ≤ 1566** (Δ≥+0.15)
- cert: M_proof=13 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 9]
- pack: `python3 runner/make_pack.py 281` → packs/task281/ATTACK.md

### task336  (gen d4f3cd78)
- pin: cost 1752 (params 152 / mem 1600) → 17.531 pts · **register at cost ≤ 1507** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 5, 8]
- pack: `python3 runner/make_pack.py 336` → packs/task336/ATTACK.md

### task368  (gen e76a88a6)
- pin: cost 1784 (params 68 / mem 1716) → 17.513 pts · **register at cost ≤ 1535** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 368` → packs/task368/ATTACK.md

### task163  (gen 6d0160f0)
- pin: cost 1750 (params 126 / mem 1624) → 17.533 pts · **register at cost ≤ 1506** (Δ≥+0.15)
- cert: M_proof=11 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 163` → packs/task163/ATTACK.md

### task268  (gen aba27056)
- pin: cost 2267 (params 72 / mem 2195) → 17.274 pts · **register at cost ≤ 1951** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 268` → packs/task268/ATTACK.md

### task284  (gen b7249182)
- pin: cost 2901 (params 248 / mem 2653) → 17.027 pts · **register at cost ≤ 2496** (Δ≥+0.15)
- cert: M_proof=24 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 284` → packs/task284/ATTACK.md

### task124  (gen 53b68214)
- pin: cost 1923 (params 71 / mem 1852) → 17.438 pts · **register at cost ≤ 1655** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=fixed:10x10 · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 124` → packs/task124/ATTACK.md

### task387  (gen f35d900a)
- pin: cost 3456 (params 123 / mem 3333) → 16.852 pts · **register at cost ≤ 2974** (Δ≥+0.15)
- cert: M_proof=18 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 387` → packs/task387/ATTACK.md

### task035  (gen 1f642eb9)
- pin: cost 1858 (params 508 / mem 1350) → 17.473 pts · **register at cost ≤ 1599** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 35` → packs/task035/ATTACK.md

### task224  (gen 928ad970)
- pin: cost 1902 (params 57 / mem 1845) → 17.449 pts · **register at cost ≤ 1637** (Δ≥+0.15)
- cert: M_proof=16 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 224` → packs/task224/ATTACK.md

### task273  (gen af902bf9)
- pin: cost 1926 (params 126 / mem 1800) → 17.437 pts · **register at cost ≤ 1657** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 2, 4]
- pack: `python3 runner/make_pack.py 273` → packs/task273/ATTACK.md

### task109  (gen 47c1f68c)
- pin: cost 1779 (params 93 / mem 1686) → 17.516 pts · **register at cost ≤ 1531** (Δ≥+0.15)
- cert: M_proof=13 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 109` → packs/task109/ATTACK.md

### task190  (gen 7ddcd7ec)
- pin: cost 1957 (params 277 / mem 1680) → 17.421 pts · **register at cost ≤ 1684** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 190` → packs/task190/ATTACK.md

### task042  (gen 22233c11)
- pin: cost 1957 (params 239 / mem 1718) → 17.421 pts · **register at cost ≤ 1684** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 3, 8]
- pack: `python3 runner/make_pack.py 42` → packs/task042/ATTACK.md

### task141  (gen 623ea044)
- pin: cost 1382 (params 101 / mem 1281) → 17.769 pts · **register at cost ≤ 1189** (Δ≥+0.15)
- cert: M_proof=21 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 141` → packs/task141/ATTACK.md

### task246  (gen a2fd1cf0)
- pin: cost 1385 (params 123 / mem 1262) → 17.767 pts · **register at cost ≤ 1192** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=same_as_input · palette_out=[0, 2, 3, 8]
- pack: `python3 runner/make_pack.py 246` → packs/task246/ATTACK.md

### task295  (gen bbc9ae5d)
- pin: cost 1566 (params 53 / mem 1513) → 17.644 pts · **register at cost ≤ 1347** (Δ≥+0.15)
- cert: M_proof=18 (high) · out_law=variable · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 295` → packs/task295/ATTACK.md

### task361  (gen e40b9e2f)
- pin: cost 3767 (params 335 / mem 3432) → 16.766 pts · **register at cost ≤ 3242** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 361` → packs/task361/ATTACK.md

### task168  (gen 6e19193c)
- pin: cost 2239 (params 143 / mem 2096) → 17.286 pts · **register at cost ≤ 1927** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 168` → packs/task168/ATTACK.md

### task062  (gen 2bcee788)
- pin: cost 2279 (params 133 / mem 2146) → 17.269 pts · **register at cost ≤ 1961** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[1, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 62` → packs/task062/ATTACK.md

### task061  (gen 29ec7d0e)
- pin: cost 1668 (params 35 / mem 1633) → 17.581 pts · **register at cost ≤ 1435** (Δ≥+0.15)
- cert: M_proof=18 (high) · out_law=same_as_input · palette_out=[1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 61` → packs/task061/ATTACK.md

### task012  (gen 0962bcdd)
- pin: cost 2222 (params 86 / mem 2136) → 17.294 pts · **register at cost ≤ 1912** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 12` → packs/task012/ATTACK.md

### task340  (gen d687bc17)
- pin: cost 3694 (params 209 / mem 3485) → 16.786 pts · **register at cost ≤ 3179** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 340` → packs/task340/ATTACK.md

### task358  (gen e21d9049)
- pin: cost 3033 (params 162 / mem 2871) → 16.983 pts · **register at cost ≤ 2610** (Δ≥+0.15)
- cert: M_proof=21 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 358` → packs/task358/ATTACK.md

### task354  (gen ddf7fa4f)
- pin: cost 2524 (params 66 / mem 2458) → 17.166 pts · **register at cost ≤ 2172** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 354` → packs/task354/ATTACK.md

### task333  (gen d43fd935)
- pin: cost 2558 (params 394 / mem 2164) → 17.153 pts · **register at cost ≤ 2201** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 333` → packs/task333/ATTACK.md

### task355  (gen de1cd16c)
- pin: cost 2656 (params 4 / mem 2652) → 17.115 pts · **register at cost ≤ 2286** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=fixed:1x1 · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 355` → packs/task355/ATTACK.md

### task102  (gen 44d8ac46)
- pin: cost 2527 (params 113 / mem 2414) → 17.165 pts · **register at cost ≤ 2175** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=same_as_input · palette_out=[0, 2, 5]
- pack: `python3 runner/make_pack.py 102` → packs/task102/ATTACK.md

### task392  (gen f8c80d96)
- pin: cost 1854 (params 171 / mem 1683) → 17.475 pts · **register at cost ≤ 1595** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 392` → packs/task392/ATTACK.md

### task173  (gen 72322fa7)
- pin: cost 13633 (params 79 / mem 13554) → 15.480 pts · **register at cost ≤ 11734** (Δ≥+0.15)
- cert: M_proof=25 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 173` → packs/task173/ATTACK.md
