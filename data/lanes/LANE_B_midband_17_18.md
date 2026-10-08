# LANE B — Mid-band rebuilds, 17–18 band (70 tasks · bar +0.15 · model: mid-tier)


**RATIO LAW:** points buy cost ratios (−50% = +0.69; −10% = +0.11 = worthless). Target ≥2× cuts / the next score band — the bar is a floor, not a goal.
**Thesis:** these graphs cost 1.1–3 KB; the rule is understood (see dossier in your pack).
Rebuild with the idiom menu (RULES.md §5): plane-kill + uint8 working set + ONE terminal
renderer. Target cost ≤1097 B (score 18) where the rule allows; minimum acceptable −25%.
Ordered by cost (attack top first). Escalate near-misses (≤20% over target or 1 failing
draw) to the strong-model queue instead of grinding.

Worker protocol: (1) `touch claims/taskNNN.claim` (skip if exists) → (2) `python3 runner/make_pack.py NNN`
→ (3) read packs/taskNNN/ATTACK.md + RULES.md → (4) build & iterate with runner/fast_verify.py → (5) LEDGER or NO.md.

### task363  (gen e5062a87)
- pin: cost 2961 (params 193 / mem 2768) → 17.007 pts · **register at cost ≤ 2548** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 2, 5]
- pack: `python3 runner/make_pack.py 363` → packs/task363/ATTACK.md

### task310  (gen c909285e)
- pin: cost 2960 (params 111 / mem 2849) → 17.007 pts · **register at cost ≤ 2547** (Δ≥+0.15)
- cert: M_proof=30 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 310` → packs/task310/ATTACK.md

### task330  (gen d2abd087)
- pin: cost 2873 (params 249 / mem 2624) → 17.037 pts · **register at cost ≤ 2472** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2]
- pack: `python3 runner/make_pack.py 330` → packs/task330/ATTACK.md

### task090  (gen 3eda0437)
- pin: cost 2846 (params 100 / mem 2746) → 17.046 pts · **register at cost ≤ 2449** (Δ≥+0.15)
- cert: M_proof=30 (high) · out_law=same_as_input · palette_out=[0, 1, 5, 6]
- pack: `python3 runner/make_pack.py 90` → packs/task090/ATTACK.md

### task079  (gen 39a8645d)
- pin: cost 2789 (params 74 / mem 2715) → 17.067 pts · **register at cost ≤ 2400** (Δ≥+0.15)
- cert: M_proof=14 (high) · out_law=fixed:3x3 · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 79` → packs/task079/ATTACK.md

### task091  (gen 3f7978a0)
- pin: cost 2761 (params 122 / mem 2639) → 17.077 pts · **register at cost ≤ 2376** (Δ≥+0.15)
- cert: M_proof=15 (high) · out_law=le_input · palette_out=[0, 5, 8]
- pack: `python3 runner/make_pack.py 91` → packs/task091/ATTACK.md

### task234  (gen 98cf29f8)
- pin: cost 2750 (params 73 / mem 2677) → 17.081 pts · **register at cost ≤ 2366** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 234` → packs/task234/ATTACK.md

### task277  (gen b230c067)
- pin: cost 2742 (params 41 / mem 2701) → 17.084 pts · **register at cost ≤ 2360** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2]
- pack: `python3 runner/make_pack.py 277` → packs/task277/ATTACK.md

### task378  (gen ec883f72)
- pin: cost 2737 (params 417 / mem 2320) → 17.085 pts · **register at cost ≤ 2355** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 378` → packs/task378/ATTACK.md

### task177  (gen 7468f01a)
- pin: cost 2722 (params 118 / mem 2604) → 17.091 pts · **register at cost ≤ 2342** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=le_input · palette_out=[1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 177` → packs/task177/ATTACK.md

### task071  (gen 3345333e)
- pin: cost 2650 (params 56 / mem 2594) → 17.118 pts · **register at cost ≤ 2280** (Δ≥+0.15)
- cert: M_proof=16 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 71` → packs/task071/ATTACK.md

### task154  (gen 6855a6e4)
- pin: cost 2649 (params 99 / mem 2550) → 17.118 pts · **register at cost ≤ 2280** (Δ≥+0.15)
- cert: M_proof=15 (high) · out_law=same_as_input · palette_out=[0, 2, 5]
- pack: `python3 runner/make_pack.py 154` → packs/task154/ATTACK.md

### task125  (gen 543a7ed5)
- pin: cost 2644 (params 447 / mem 2197) → 17.120 pts · **register at cost ≤ 2275** (Δ≥+0.15)
- cert: M_proof=15 (high) · out_law=same_as_input · palette_out=[3, 4, 6, 8]
- pack: `python3 runner/make_pack.py 125` → packs/task125/ATTACK.md

### task174  (gen 72ca375d)
- pin: cost 2596 (params 367 / mem 2229) → 17.138 pts · **register at cost ≤ 2234** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 174` → packs/task174/ATTACK.md

### task250  (gen a48eeaf7)
- pin: cost 2542 (params 54 / mem 2488) → 17.159 pts · **register at cost ≤ 2187** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 2, 5]
- pack: `python3 runner/make_pack.py 250` → packs/task250/ATTACK.md

### task398  (gen feca6190)
- pin: cost 2481 (params 685 / mem 1796) → 17.184 pts · **register at cost ≤ 2135** (Δ≥+0.15)
- cert: M_proof=25 (high) · out_law=variable · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 398` → packs/task398/ATTACK.md

### task105  (gen 4612dd53)
- pin: cost 2466 (params 78 / mem 2388) → 17.190 pts · **register at cost ≤ 2122** (Δ≥+0.15)
- cert: M_proof=14 (high) · out_law=same_as_input · palette_out=[0, 1, 2]
- pack: `python3 runner/make_pack.py 105` → packs/task105/ATTACK.md

### task093  (gen 4093f84a)
- pin: cost 2455 (params 539 / mem 1916) → 17.194 pts · **register at cost ≤ 2113** (Δ≥+0.15)
- cert: M_proof=14 (high) · out_law=same_as_input · palette_out=[0, 5]
- pack: `python3 runner/make_pack.py 93` → packs/task093/ATTACK.md

### task202  (gen 855e0971)
- pin: cost 2405 (params 52 / mem 2353) → 17.215 pts · **register at cost ≤ 2070** (Δ≥+0.15)
- cert: M_proof=None (None) · out_law=None · palette_out=None
- pack: `python3 runner/make_pack.py 202` → packs/task202/ATTACK.md

### task017  (gen 0dfd9992)
- pin: cost 2390 (params 74 / mem 2316) → 17.221 pts · **register at cost ≤ 2057** (Δ≥+0.15)
- cert: M_proof=21 (high) · out_law=same_as_input · palette_out=[1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 17` → packs/task017/ATTACK.md

### task030  (gen 1caeab9d)
- pin: cost 2376 (params 148 / mem 2228) → 17.227 pts · **register at cost ≤ 2045** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 4]
- pack: `python3 runner/make_pack.py 30` → packs/task030/ATTACK.md

### task245  (gen a1570a43)
- pin: cost 2370 (params 70 / mem 2300) → 17.229 pts · **register at cost ≤ 2039** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 2, 3]
- pack: `python3 runner/make_pack.py 245` → packs/task245/ATTACK.md

### task008  (gen 05f2a901)
- pin: cost 2355 (params 73 / mem 2282) → 17.236 pts · **register at cost ≤ 2026** (Δ≥+0.15)
- cert: M_proof=16 (high) · out_law=same_as_input · palette_out=[0, 2, 8]
- pack: `python3 runner/make_pack.py 8` → packs/task008/ATTACK.md

### task046  (gen 234bbc79)
- pin: cost 2285 (params 96 / mem 2189) → 17.266 pts · **register at cost ≤ 1966** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 46` → packs/task046/ATTACK.md

### task069  (gen 321b1fc6)
- pin: cost 2195 (params 469 / mem 1726) → 17.306 pts · **register at cost ≤ 1889** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 9]
- pack: `python3 runner/make_pack.py 69` → packs/task069/ATTACK.md

### task270  (gen ae3edfdc)
- pin: cost 2189 (params 1351 / mem 838) → 17.309 pts · **register at cost ≤ 1884** (Δ≥+0.15)
- cert: M_proof=15 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 7]
- pack: `python3 runner/make_pack.py 270` → packs/task270/ATTACK.md

### task212  (gen 8d510a79)
- pin: cost 2132 (params 62 / mem 2070) → 17.335 pts · **register at cost ≤ 1835** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 5]
- pack: `python3 runner/make_pack.py 212` → packs/task212/ATTACK.md

### task094  (gen 41e4d17e)
- pin: cost 2062 (params 442 / mem 1620) → 17.369 pts · **register at cost ≤ 1774** (Δ≥+0.15)
- cert: M_proof=15 (high) · out_law=same_as_input · palette_out=[1, 6, 8]
- pack: `python3 runner/make_pack.py 94` → packs/task094/ATTACK.md

### task323  (gen d06dbe63)
- pin: cost 2019 (params 667 / mem 1352) → 17.390 pts · **register at cost ≤ 1737** (Δ≥+0.15)
- cert: M_proof=13 (high) · out_law=same_as_input · palette_out=[0, 5, 8]
- pack: `python3 runner/make_pack.py 323` → packs/task323/ATTACK.md

### task036  (gen 1f85a75f)
- pin: cost 1965 (params 43 / mem 1922) → 17.417 pts · **register at cost ≤ 1691** (Δ≥+0.15)
- cert: M_proof=30 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 36` → packs/task036/ATTACK.md

### task084  (gen 3bd67248)
- pin: cost 1955 (params 242 / mem 1713) → 17.422 pts · **register at cost ≤ 1682** (Δ≥+0.15)
- cert: M_proof=21 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 84` → packs/task084/ATTACK.md

### task088  (gen 3de23699)
- pin: cost 1933 (params 81 / mem 1852) → 17.433 pts · **register at cost ≤ 1663** (Δ≥+0.15)
- cert: M_proof=24 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 88` → packs/task088/ATTACK.md

### task055  (gen 272f95fa)
- pin: cost 1908 (params 168 / mem 1740) → 17.446 pts · **register at cost ≤ 1642** (Δ≥+0.15)
- cert: M_proof=None (None) · out_law=None · palette_out=None
- pack: `python3 runner/make_pack.py 55` → packs/task055/ATTACK.md

### task325  (gen d0f5fe59)
- pin: cost 1833 (params 123 / mem 1710) → 17.486 pts · **register at cost ≤ 1577** (Δ≥+0.15)
- cert: M_proof=16 (high) · out_law=le_input · palette_out=[0, 8]
- pack: `python3 runner/make_pack.py 325` → packs/task325/ATTACK.md

### task206  (gen 88a10436)
- pin: cost 1822 (params 112 / mem 1710) → 17.492 pts · **register at cost ≤ 1568** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 6]
- pack: `python3 runner/make_pack.py 206` → packs/task206/ATTACK.md

### task238  (gen 9aec4887)
- pin: cost 1814 (params 353 / mem 1461) → 17.497 pts · **register at cost ≤ 1561** (Δ≥+0.15)
- cert: M_proof=16 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 238` → packs/task238/ATTACK.md

### task184  (gen 780d0b14)
- pin: cost 1786 (params 106 / mem 1680) → 17.512 pts · **register at cost ≤ 1537** (Δ≥+0.15)
- cert: M_proof=None (None) · out_law=None · palette_out=None
- pack: `python3 runner/make_pack.py 184` → packs/task184/ATTACK.md

### task237  (gen 99fa7670)
- pin: cost 1710 (params 278 / mem 1432) → 17.556 pts · **register at cost ≤ 1471** (Δ≥+0.15)
- cert: M_proof=9 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 237` → packs/task237/ATTACK.md

### task132  (gen 56ff96f3)
- pin: cost 1701 (params 137 / mem 1564) → 17.561 pts · **register at cost ≤ 1464** (Δ≥+0.15)
- cert: M_proof=15 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 132` → packs/task132/ATTACK.md

### task170  (gen 6ecd11f4)
- pin: cost 1696 (params 89 / mem 1607) → 17.564 pts · **register at cost ≤ 1459** (Δ≥+0.15)
- cert: M_proof=28 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 170` → packs/task170/ATTACK.md

### task256  (gen a65b410d)
- pin: cost 1665 (params 113 / mem 1552) → 17.582 pts · **register at cost ≤ 1433** (Δ≥+0.15)
- cert: M_proof=13 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3]
- pack: `python3 runner/make_pack.py 256` → packs/task256/ATTACK.md

### task374  (gen ea32f347)
- pin: cost 1617 (params 51 / mem 1566) → 17.612 pts · **register at cost ≤ 1391** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 4]
- pack: `python3 runner/make_pack.py 374` → packs/task374/ATTACK.md

### task034  (gen 1f0c79e5)
- pin: cost 1601 (params 290 / mem 1311) → 17.622 pts · **register at cost ≤ 1377** (Δ≥+0.15)
- cert: M_proof=9 (high) · out_law=same_as_input · palette_out=[0, 1, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 34` → packs/task034/ATTACK.md

### task213  (gen 8e1813be)
- pin: cost 1599 (params 50 / mem 1549) → 17.623 pts · **register at cost ≤ 1376** (Δ≥+0.15)
- cert: M_proof=24 (high) · out_law=le_input · palette_out=[1, 2, 3, 4, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 213` → packs/task213/ATTACK.md

### task134  (gen 5ad4f10b)
- pin: cost 1587 (params 168 / mem 1419) → 17.630 pts · **register at cost ≤ 1365** (Δ≥+0.15)
- cert: M_proof=30 (high) · out_law=fixed:3x3 · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 134` → packs/task134/ATTACK.md

### task019  (gen 10fcaaa3)
- pin: cost 1587 (params 219 / mem 1368) → 17.630 pts · **register at cost ≤ 1365** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=scale:2x2 · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 19` → packs/task019/ATTACK.md

### task051  (gen 25d487eb)
- pin: cost 1582 (params 386 / mem 1196) → 17.634 pts · **register at cost ≤ 1361** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 51` → packs/task051/ATTACK.md

### task302  (gen c0f76784)
- pin: cost 1574 (params 358 / mem 1216) → 17.639 pts · **register at cost ≤ 1354** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=same_as_input · palette_out=[0, 5, 6, 7, 8]
- pack: `python3 runner/make_pack.py 302` → packs/task302/ATTACK.md

### task159  (gen 6b9890af)
- pin: cost 1568 (params 149 / mem 1419) → 17.642 pts · **register at cost ≤ 1349** (Δ≥+0.15)
- cert: M_proof=30 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 159` → packs/task159/ATTACK.md

### task260  (gen a78176bb)
- pin: cost 1536 (params 303 / mem 1233) → 17.663 pts · **register at cost ≤ 1322** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 260` → packs/task260/ATTACK.md

### task185  (gen 7837ac64)
- pin: cost 1517 (params 64 / mem 1453) → 17.676 pts · **register at cost ≤ 1305** (Δ≥+0.15)
- cert: M_proof=29 (high) · out_law=fixed:3x3 · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 185` → packs/task185/ATTACK.md

### task161  (gen 6cdd2623)
- pin: cost 1490 (params 152 / mem 1338) → 17.693 pts · **register at cost ≤ 1282** (Δ≥+0.15)
- cert: M_proof=25 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 161` → packs/task161/ATTACK.md

### task308  (gen c8cbb738)
- pin: cost 1487 (params 102 / mem 1385) → 17.695 pts · **register at cost ≤ 1279** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=le_input · palette_out=[1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 308` → packs/task308/ATTACK.md

### task169  (gen 6e82a1ae)
- pin: cost 1460 (params 60 / mem 1400) → 17.714 pts · **register at cost ≤ 1256** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3]
- pack: `python3 runner/make_pack.py 169` → packs/task169/ATTACK.md

### task112  (gen 4938f0c2)
- pin: cost 1449 (params 84 / mem 1365) → 17.721 pts · **register at cost ≤ 1247** (Δ≥+0.15)
- cert: M_proof=30 (high) · out_law=same_as_input · palette_out=[0, 2, 3]
- pack: `python3 runner/make_pack.py 112` → packs/task112/ATTACK.md

### task156  (gen 694f12f3)
- pin: cost 1388 (params 58 / mem 1330) → 17.764 pts · **register at cost ≤ 1194** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 4]
- pack: `python3 runner/make_pack.py 156` → packs/task156/ATTACK.md

### task275  (gen b190f7f5)
- pin: cost 1356 (params 863 / mem 493) → 17.788 pts · **register at cost ≤ 1167** (Δ≥+0.15)
- cert: M_proof=16 (high) · out_law=variable · palette_out=[0, 1, 2, 3, 4]
- pack: `python3 runner/make_pack.py 275` → packs/task275/ATTACK.md

### task356  (gen ded97339)
- pin: cost 1319 (params 19 / mem 1300) → 17.815 pts · **register at cost ≤ 1135** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 8]
- pack: `python3 runner/make_pack.py 356` → packs/task356/ATTACK.md

### task013  (gen 0a938d79)
- pin: cost 1298 (params 157 / mem 1141) → 17.831 pts · **register at cost ≤ 1117** (Δ≥+0.15)
- cert: M_proof=30 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 13` → packs/task013/ATTACK.md

### task119  (gen 508bd3b6)
- pin: cost 1293 (params 142 / mem 1151) → 17.835 pts · **register at cost ≤ 1112** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=same_as_input · palette_out=[0, 2, 3, 8]
- pack: `python3 runner/make_pack.py 119` → packs/task119/ATTACK.md

### task160  (gen 6c434453)
- pin: cost 1258 (params 94 / mem 1164) → 17.863 pts · **register at cost ≤ 1082** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2]
- pack: `python3 runner/make_pack.py 160` → packs/task160/ATTACK.md

### task240  (gen 9d9215db)
- pin: cost 1248 (params 788 / mem 460) → 17.871 pts · **register at cost ≤ 1074** (Δ≥+0.15)
- cert: M_proof=19 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 240` → packs/task240/ATTACK.md

### task369  (gen e8593010)
- pin: cost 1247 (params 147 / mem 1100) → 17.872 pts · **register at cost ≤ 1073** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[1, 2, 3, 5]
- pack: `python3 runner/make_pack.py 369` → packs/task369/ATTACK.md

### task175  (gen 73251a56)
- pin: cost 1223 (params 937 / mem 286) → 17.891 pts · **register at cost ≤ 1052** (Δ≥+0.15)
- cert: M_proof=21 (high) · out_law=same_as_input · palette_out=[1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 175` → packs/task175/ATTACK.md

### task335  (gen d4a91cb9)
- pin: cost 1201 (params 173 / mem 1028) → 17.909 pts · **register at cost ≤ 1033** (Δ≥+0.15)
- cert: M_proof=20 (high) · out_law=same_as_input · palette_out=[0, 2, 4, 8]
- pack: `python3 runner/make_pack.py 335` → packs/task335/ATTACK.md

### task388  (gen f5b8619d)
- pin: cost 1181 (params 118 / mem 1063) → 17.926 pts · **register at cost ≤ 1016** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=scale:2x2 · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 388` → packs/task388/ATTACK.md

### task244  (gen 9f236235)
- pin: cost 1179 (params 103 / mem 1076) → 17.928 pts · **register at cost ≤ 1014** (Δ≥+0.15)
- cert: M_proof=23 (high) · out_law=le_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 244` → packs/task244/ATTACK.md

### task351  (gen dc0a314f)
- pin: cost 1171 (params 23 / mem 1148) → 17.934 pts · **register at cost ≤ 1007** (Δ≥+0.15)
- cert: M_proof=16 (high) · out_law=fixed:5x5 · palette_out=[1, 2, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 351` → packs/task351/ATTACK.md

### task226  (gen 941d9a10)
- pin: cost 1144 (params 30 / mem 1114) → 17.958 pts · **register at cost ≤ 984** (Δ≥+0.15)
- cert: M_proof=10 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 5]
- pack: `python3 runner/make_pack.py 226` → packs/task226/ATTACK.md

### task301  (gen beb8660c)
- pin: cost 1141 (params 64 / mem 1077) → 17.960 pts · **register at cost ≤ 982** (Δ≥+0.15)
- cert: M_proof=12 (high) · out_law=same_as_input · palette_out=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- pack: `python3 runner/make_pack.py 301` → packs/task301/ATTACK.md
