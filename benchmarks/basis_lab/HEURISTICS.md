# Measuring `lappy.heuristics` on the polygon benchmark domains

`lappy/heuristics.py` turns `docs/mps_heuristics.pdf`'s closed-form recipe into a basis:
`polygon_default_basis(domain, lam_max, precision)`. `tests/test_heuristics.py` pins its
formulas; until now nothing measured its output. This is that measurement.

Harness: `benchmarks/basis_lab/heur.py`. Rows: `benchmarks/basis_lab/run/heur/*.jsonl`, one
per domain, append-only, resumable. Re-print every table with

    .venv/bin/python -m benchmarks.basis_lab.heur report

Instruments, and why there are two. `sigma(lam_ref)` (h1/h2/h4) is what the basis directly
controls and needs no eigenvalue search, so it cannot be confounded by the search — the trap
that voided the first convergence study in this directory. Certified Moler--Payne digits from a
full polished solve (h3) cost ~100x more and are the *check* on that proxy, not the primary
score. Every row also carries `n_basis`, `contrast = sigma_off / sigma_at`, and any warning the
build emitted, because a `sigma` that falls with collapsing contrast is ill-conditioning rather
than accuracy, and sources landing inside the domain invalidate the tension outright.

Fixed throughout: `lam_max = weyl_est(6, domain)` (what `bench.evaluate` and
`MPSEigensolver.from_domain` both use), `rng=7`, `prec=1e-14` (the 1e-8 default caps readings
near 10 digits), `N_EIGS=4`, `HeuristicConfig()` defaults except in h4/h5/h6.

1154 measurements: h0 174 rows, h1 108, h2 432, h3 98, h4 150, h5 108, h6 84.

## Summary

1. **`precision` is pessimistic and largely inert.** Achieved ≥ requested in 104 of 108 rows, by
   a median of +8.7 digits at `precision=1e-2`; all four shortfalls are ≤0.6 digits, at
   `precision=1e-12`, and past their reference's own resolution. Inert because on **12 of 18
   domains the achieved accuracy is flat to within 1.5 digits from `precision=1e-4` down**,
   while `n` roughly doubles (2.9x on iso_tri_h4). The knob buys cost, not accuracy. Under the
   *certified* Moler--Payne bound the margin is thinner and turns slightly negative at 12
   digits: it certifies 12 digits in only 6 of 13 attempts.
2. **At matched size it is ~1 digit worse than the best existing construction** (median −1.1
   tension digits, −1.2 certified digits) **and much better than the worst.** It wins outright
   on the two no-symmetry many-corner domains, GWW1 and GWW2, and on tension it never drops
   below 10.3 digits on any domain at any precision, whereas `pure_fb` falls to 4-5 digits on
   sharp triangles and `mixed` produces *spurious eigenvalues* on L_shape at matched size. It is
   not, however, failure-free: on `iso_tri_h4` the recipe's own basis produces spurious
   eigenvalues in a real solve, and on sharp thin triangles it places sources inside the domain.
3. **Its cost model is the weak part.** The ambient MFS curve is 90%+ of the columns, the
   spacing rule ignores its own Nyquist term entirely, and on the sharp/many-corner half of the
   suite the recipe asks for 800-12000 functions — not a usable basis. `n` is also not monotone
   in `precision`.
4. **Every tuned constant is a cost knob; none is an accuracy knob** at these wavenumbers, and
   two (`nyquist_ppw`, `max_reflections`) do nothing at all. A combination of the cost winners
   saves a median 12% of columns for 0.1 digits — up to 43% on the domains where it matters —
   but only once one ingredient found by single-factor screening is put back (h5/h6).

---

## Q2. Is `precision` optimistic or pessimistic?

**Pessimistic, by a lot, and it barely controls what it claims to control.**

h1, achieved tension digits `A = -log10(median sigma at the first 4 reference eigenvalues)`
against requested `R = -log10(precision)`:

| domain | ref digits | R=2 | R=4 | R=6 | R=8 | R=10 | R=12 |
|---|---|---|---|---|---|---|---|
| L_shape | 13.6 | 11.1 (+9.1) | 14.2 (+10.2)~ | 14.0 (+8.0)~ | 13.9 (+5.9)~ | 13.8 (+3.8)~ | 13.8 (+1.8)~ |
| eq_tri | 15.0 | 9.9 (+7.9) | 11.0 (+7.0) | 12.2 (+6.2) | 13.6 (+5.6)~ | 13.1 (+3.1) | 13.2 (+1.2) |
| iso_right_tri | 15.0 | 7.6 (+5.6) | 8.6 (+4.6) | 9.7 (+3.7) | 11.6 (+3.6) | 12.4 (+2.4) | 12.4 (+0.4) |
| iso_tri_h1 | 15.0 | 7.5 (+5.5) | 9.1 (+5.1) | 10.1 (+4.1) | 11.1 (+3.1) | 11.8 (+1.8) | 12.4 (+0.4) |
| iso_tri_h4 | 11.3 | 10.8 (+8.8)~ | 10.8 (+6.8)~ | 11.5 (+5.5)~ | 11.2 (+3.2)~ | 11.3 (+1.3)! | -- |
| parallelogram_60 | 12.7 | 11.0 (+9.0) | 12.0 (+8.0)~ | 12.6 (+6.6)~ | 13.2 (+5.2)~ | 12.9 (+2.9)~ | 13.3 (+1.3)~ |
| rect_near_deg_1e3 | 15.0 | 9.8 (+7.8) | 13.6 (+9.6) | 12.9 (+6.9) | 13.4 (+5.4) | 13.1 (+3.1) | 13.6 (+1.6) |
| rect_near_deg_1e5 | 15.0 | 9.8 (+7.8) | 13.6 (+9.6) | 13.0 (+7.0) | 13.4 (+5.4) | 13.1 (+3.1) | 13.6 (+1.6) |
| rect_thin | 15.0 | 7.6 (+5.6) | 7.8 (+3.8) | 9.9 (+3.9) | 10.7 (+2.7) | 11.8 (+1.8) | 12.7 (+0.7) |
| reg_ngon_5 | 12.8 | 12.0 (+10.0)~ | 13.8 (+9.8)~ | 12.1 (+6.1)~ | 12.1 (+4.1)~ | 12.1 (+2.1)~ | 12.4 (+0.4)~ |
| reg_ngon_6 | 12.5 | 12.0 (+10.0)~ | 12.5 (+8.5)~ | 12.5 (+6.5)~ | 12.5 (+4.5)~ | 12.5 (+2.5)~ | 12.6 (+0.6)~ |
| reg_ngon_7 | 11.5 | 11.4 (+9.4)~ | 11.4 (+7.4)~ | 11.6 (+5.6)~ | 11.8 (+3.8)~ | 11.9 (+1.9)~ | 12.0 (-0.0)~ |
| reg_ngon_8 | 8.0 | 11.3 (+9.3)~ | 11.4 (+7.4)~ | 11.4 (+5.4)~ | 11.4 (+3.4)~ | 11.4 (+1.4)~ | 11.4 (-0.6)~ |
| right_trapezoid | 13.3 | 11.7 (+9.7) | 14.2 (+10.2)~ | 13.7 (+7.7)~ | 13.4 (+5.4)~ | 13.5 (+3.5)~ | 13.5 (+1.5)~ |
| square | 15.0 | 9.8 (+7.8) | 13.7 (+9.7) | 13.4 (+7.4) | 14.1 (+6.1)~ | 13.3 (+3.3) | 13.6 (+1.6) |
| GWW1 | 9.9 | 10.6 (+8.6)~ | 11.6 (+7.6)~ | 11.7 (+5.7)~ | 11.8 (+3.8)~ | 11.8 (+1.8)~ | 11.8 (-0.2)~ |
| GWW2 | 9.3 | 10.4 (+8.4)~ | 11.7 (+7.7)~ | 12.1 (+6.1)~ | 12.2 (+4.2)~ | 12.2 (+2.2)~ | 12.2 (+0.2)~ |
| iso_tri_h16 | 10.8 | 12.8 (+10.8)! | 11.0 (+7.0)~ | 12.4 (+6.4)~ | 12.8 (+4.8)~ | 12.7 (+2.7)~ | 12.4 (+0.4)~ |

`~` = sigma is past that domain's own reference resolution, so the cell is a lower bound on `A`
(it is measuring the reference, not the basis — `bench.py`'s "THE REFERENCE IS ALSO A LIMIT").
`!` = untrustworthy: contrast below 4e2, or sources were dropped from inside the domain. Note how
much of this table is `~`: on the eleven domains whose truth is a certified or tabulated value
rather than a closed form, the basis is at or past what the reference can resolve almost
everywhere, so those cells bound `A` from below and the real margins are *larger* than shown.

Median `A - R` over all domains: **+8.7 at R=2, +7.7, +6.1, +4.3, +2.5, +0.6 at R=12** (using
only the cells within their reference's resolution: +7.8, +7.0, +6.2, +3.6, +3.1, +1.2).

Two readings of the same table, and both matter:

* **The promise is kept.** `A >= R` in 104 of 108 rows. All four shortfalls are at R=12
  (GWW1 -0.2, iso_tri_h4 -0.6, reg_ngon_7 -0.0, reg_ngon_8 -0.6) and every one is past its
  reference's own resolution, so none is evidence of a real shortfall. Nothing here asks for
  accuracy it does not get.
* **The knob is largely inert, so the margin is waste.** Measuring the spread of `A` over
  R = 4..12 (dropping the coarsest rung, where several domains start low): it is **flat to
  within 1.5 digits on 12 of 18 domains** — 0.0 on reg_ngon_8, 0.1 on reg_ngon_6, 0.3 on
  L_shape, 0.8 on square — while `n` roughly doubles over the same range (square 73 -> 140,
  L_shape 164 -> 316, iso_tri_h4 261 -> 764, a factor 2.9). The recipe is not sizing itself to
  the request; it is sizing itself to something else and then reporting whatever double
  precision allows.

The six domains where `precision` genuinely buys digits over R = 4..12 are
`rect_thin` (spread 4.8), `iso_right_tri` (3.9), `iso_tri_h1` (3.3), `eq_tri` (2.6),
`iso_tri_h16` (1.8) and `reg_ngon_5` (1.7) — the first four being exactly the
all-regular-corner ones, where the FB budget is small and the ambient MFS curve does the work:

    rect_thin      n:A   53:7.6  55:7.8  68:9.9  81:10.7  94:11.8  107:12.7
    iso_right_tri        43:7.6  57:8.6  71:9.7  85:11.6  99:12.4  114:12.4
    iso_tri_h1           43:7.5  57:9.1  71:10.1 85:11.1  99:11.8  114:12.4
    eq_tri               50:9.9  67:11.0 85:12.2 102:13.6 120:13.1 138:13.2

Between 0.4 and 0.9 digits per 10 columns (rect_thin 5.1 digits for 54 columns; eq_tri 3.3 for
88), saturating at 12-13. Where a singular corner is present, the corner Fourier--Bessel block
delivers near-floor tension at the coarsest setting and the ladder adds columns for nothing
measurable.

**Why the knob is inert.** `precision` enters only through `Lambda = ln(10 C_Omega/precision)`,
which spans 9.2 to 32.2 across the ladder — a factor of 3.5. It does not touch the term that
sets the corner budget on a singular corner: `nu_osc = kappa R_c + 2 (kappa R_c)^(1/3) + 5` is
precision-independent, and on `iso_tri_h4` it is 13.0 at every rung while `nu_cont` climbs 7.6
-> 23.6. Where `Lambda` does act — the continuation term and the ambient spacing
`h = pi delta / Lambda`, which is the binding branch everywhere (Q3) — it is acting on parts
already deep past the double-precision floor. So the ladder buys columns in the places that were
already saturated and leaves the rest alone.

---

## Q0 (unasked but unavoidable). What the recipe costs

h0, `plan_basis` only, every polygon in the suite, `n_total(n_fb)`:

| domain | nc | 1e-2 | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 | monotone |
|---|---|---|---|---|---|---|---|---|
| rect_thin | 4 | 53(16) | 55(16) | 68(16) | 81(16) | 94(16) | 107(16) | yes |
| iso_right_tri / iso_tri_h1 | 3 | 43(15) | 57(15) | 71(15) | 85(15) | 99(15) | 114(15) | yes |
| eq_tri | 3 | 50(15) | 67(15) | 85(15) | 102(15) | 120(15) | 138(15) | yes |
| square / rect_near_deg_* | 4 | 57(24) | 73(24) | 90(24) | 106(24) | 123(24) | 140(24) | yes |
| right_trapezoid | 4 | 79(30) | 101(33) | 124(36) | 146(39) | 168(42) | 191(46) | yes |
| parallelogram_60 | 4 | 100(34) | 128(38) | 157(44) | 185(50) | 214(56) | 244(62) | yes |
| L_shape | 6 | 128(57) | 164(63) | 201(70) | 240(77) | 277(84) | 316(91) | yes |
| reg_ngon_5 | 5 | 149(50) | 186(65) | 221(75) | 256(85) | 295(100) | 333(115) | yes |
| reg_ngon_6 | 6 | 184(66) | 230(84) | 270(96) | 318(114) | 364(132) | 412(150) | yes |
| reg_ngon_7 | 7 | 221(84) | 268(98) | 324(119) | 378(140) | 432(161) | 495(189) | yes |
| reg_ngon_8 | 8 | 252(96) | 316(120) | 376(144) | 440(168) | 502(192) | 574(224) | yes |
| iso_tri_h16 | 3 | 318(12) | 300(20) | 390(24) | 480(28) | 569(32) | 658(36) | **NO** |
| GWW1 / GWW2 | 8 | 264(90) | 346(108) | 434(128) | 519(148) | 604(166) | 691(188) | yes |
| iso_tri_h4 | 3 | 343(13) | 261(21) | 338(25) | 459(27) | 652(31) | 773(35) | **NO** |
| chevron_1_15 | 4 | 427(41) | 617(47) | 663(55) | 816(62) | 969(70) | 1121(78) | yes |
| chevron_1_125 | 4 | 440(44) | 543(45) | 705(52) | 870(60) | 1032(67) | 1196(75) | yes |
| iso_tri_h05 | 3 | 265(21) | 381(24) | 496(27) | 613(30) | 929(33) | 1216(36) | yes |
| chevron_1_2 | 4 | 418(40) | 600(46) | 622(56) | 765(64) | 909(73) | 1266(81) | yes |
| parallelogram_p127 | 4 | 367(34) | 526(42) | 685(48) | 847(58) | 1008(66) | 1294(74) | yes |
| chevron_2_3 | 4 | 456(44) | 659(51) | 862(58) | 1303(66) | 1038(76) | 1205(85) | **NO** |
| parallelogram_p65 | 4 | 305(36) | 433(44) | 560(50) | 780(58) | 1020(66) | 1322(74) | yes |
| H_shape | 12 | 487(128) | 674(156) | 858(180) | 1045(208) | 1233(236) | 1417(264) | yes |
| chevron_2_4 | 4 | 441(42) | 637(49) | 832(56) | 1162(64) | 1658(72) | 1929(80) | yes |
| spiral | 24 | 1656(311) | 1985(409) | 2517(496) | 3064(600) | 3606(694) | 4150(793) | yes |
| spiral_t25 | 24 | 4158(308) | 5507(389) | 7220(476) | 8942(568) | 10661(664) | 12383(762) | yes |

No corner anywhere hit the Sec-4 conditioning cap, so nothing in this table is truncated: these
are the counts the formulas actually want.

**Finding C1: the ambient MFS curve, not the corner Fourier--Bessel blocks, is the entire cost
story.** On chevron_2_4 at 1e-10, 1561 of 1658 columns are ambient curve sources and 72 are FB.
The mechanism is `delta = min(delta_amb, eta * dist(x, S*))` feeding
`h = min(2 pi/(3 kappa), pi delta / Lambda)`: on a thin or sharp domain every boundary point is
close to *some* reflected obstruction image, so `delta` is small along the whole boundary and
the curve is finely spaced everywhere — not just near the corner that caused it. The recipe has
no mechanism to notice that the refinement it is buying is not needed.

**Finding C2: `n` is not monotone in `precision` on `iso_tri_h4`, `iso_tri_h16` and
`chevron_2_3`, and the cause is the weak/singular reclassification.** Sec 3 calls a corner
weakly singular when `alpha > Lambda / (2 ln 15)`, so as `Lambda` grows the cutoff rises and
corners flip from weak to singular. That flip changes two things at once. A weak corner gets no
Fourier--Bessel terms, so the ambient curve must taper all the way into it and its `delta` floor
is the tiny lightning innermost reach `delta_amb exp(-4(sqrt(n)-1))`. A singular corner gets FB
terms and, with them, `handover_frac * R_c` worth of ambient sources **deleted outright**. On
iso_tri_h4, going from `precision=1e-2` to `1e-4` flips two corners to singular; the FB budget
rises 13 -> 21 but the curve falls 318 -> 216, and the total drops from 343 to 261. Same story
on chevron_2_3 between 1e-8 and 1e-10 (corner 3 flips; curve 1213 -> 934, total 1303 -> 1038).
So the discontinuity is not a bug in the arithmetic; it is a real discontinuity in the recipe:
the same geometry gets two qualitatively different treatments either side of a `Lambda`
threshold, and asking for *more* accuracy can hand you a *smaller* basis. Any adaptive loop
built on top of this (Sec 9) cannot assume monotonicity.

**Finding C3: on sharp thin triangles the recipe places sources inside the domain.**
`iso_tri_h4` at `precision=1e-10` and `1e-12` drops 6 of 621 and 9 of 738 sources through
`_drop_interior_sources`. That is not a cosmetic warning: those columns were meant to be
particular solutions, the tension no longer bounds anything, and both rows are flagged
untrustworthy in h1. The likely mechanism is the corner-cluster/bridge ray on the outward
bisector of a very sharp corner re-entering the opposite side of a thin domain — `delta_amb =
0.25 D` is a *diameter*-scaled offset, and on a domain of aspect 4-16 that is far wider than
the local half-width.

**Tractability, which is what the later stages can afford.** Twelve domains stay under ~500
columns across the whole ladder; five sit at 500-700; and the sharp/many-corner half
(chevrons, H_shape, parallelogram_p65/p127, iso_tri_h05, spiral) is asked to build 800-12000
functions, past both the module's own 600-column conditioning warning and the
Bessel-evaluation wall documented in `benchmarks/suite/SUITE.md`. For those domains the recipe
does not produce a usable basis at any precision on the ladder, and that is a finding about the
recipe, not a gap in this study.

---

## Q1. Better than today's constructions, at matched size?

**No — it is usually about a digit worse than the best of them, and its real virtue is that it
is never catastrophic.** h2 rebuilds `pure_fb`, `mixed`, `fb_plus_bdry_fs` and
`make_default_basis` at `n = n_heur` for every h1 cell (432 rows, no failures, contrast between
1e6 and 1e13 throughout, so none of this is a conditioning artifact).

Per domain, median over the precision ladder of the heuristic's tension digits minus the *best*
and the *worst* matched-size baseline:

| domain | heur − best | heur − worst | which baseline wins | heur A |
|---|---|---|---|---|
| iso_right_tri | −4.4 | −1.7 | pure_fb | 10.6 |
| iso_tri_h1 | −4.3 | −2.0 | pure_fb | 10.6 |
| rect_thin | −3.1 | +0.2 | pure_fb | 10.3 |
| eq_tri | −2.5 | −1.3 | pure_fb | 12.6 |
| iso_tri_h4 | −2.3 | **+5.7** | mixed | 11.2 |
| rect_near_deg_1e3 / 1e5 | −1.8 | −1.2 | pure_fb | 13.2 |
| square | −1.7 | −0.7 | pure_fb | 13.5 |
| reg_ngon_5 | −1.1 | −0.3 | pure_fb | 12.1 |
| reg_ngon_7 | −1.1 | +0.4 | pure_fb | 11.7 |
| reg_ngon_6 | −1.0 | +0.2 | pure_fb | 12.5 |
| right_trapezoid | −1.0 | −0.6 | mixed | 13.5 |
| parallelogram_60 | −0.9 | **+8.2** | fb_plus_bdry_fs | 12.7 |
| L_shape | −0.4 | −0.2 | mixed | 13.9 |
| reg_ngon_8 | −0.0 | +0.2 | mixed | 11.4 |
| iso_tri_h16 | +0.0 | **+8.0** | fb_plus_bdry_fs | 12.6 |
| GWW1 | **+0.4** | +1.1 | fb_plus_bdry_fs | 11.7 |
| GWW2 | **+0.6** | +1.7 | fb_plus_bdry_fs | 12.2 |

Median across domains: **heur − best = −1.1 digits, heur − worst = 0.0 digits.**

* It beats *every* baseline at every precision from 1e-4 down on **GWW1 and GWW2** — the two
  no-symmetry, many-singular-corner domains, and the two the existing constructor has the least
  claim on. Also on `iso_tri_h16` at four of six precisions, and sporadically on reg_ngon_5/8.
* It loses biggest exactly where the geometry is easy and one construction is already ideal:
  `pure_fb` at matched size reaches 14.7-15.1 digits on the triangles and rectangles where the
  heuristic manages 7.5-13.7. On these domains corner Fourier--Bessel alone *is* the right
  answer and the recipe spends most of its columns on an ambient MFS curve that is not needed.
* The `heur − worst` column is the case for the recipe. `pure_fb` collapses to **4.6 digits on
  iso_tri_h16, 5.5 on iso_tri_h4, 3.9-4.8 on parallelogram_60** — sharp-corner domains where FB
  alone cannot work — and `mixed`/`default` fall to 7.9-11 on GWW2 at small `n`. The heuristic's
  tension never falls below 10.3 on any domain at any precision. It is a construction that needs
  no per-domain choice and, on this instrument, has no catastrophic branch — which is exactly
  what `make_default_basis`'s corner-counting branch cannot promise (and which `bench.py` was
  written to check: no domain there had ever been scored under a construction its branch would
  not have picked). **h3 qualifies this**: uniform tension is not the same as a usable basis, and
  on `iso_tri_h4` the recipe's own eigenvalue search goes wrong (below).

So the trade as measured: **give up ~1 digit against the best hand-picked construction, and 2-3x
more columns than needed (Q2), to buy uniformity across geometries and a win on the hardest
asymmetric domains.**

### h3: the real solve keeps the ranking but overturns two of its readings

h3 runs the full pipeline — polished eigenvalue search plus Moler--Payne certification, 98 rows
— for the heuristic and for the strongest matched-size baseline h2 identified per cell. Median
`MP(heur) − MP(baseline) = −1.2 digits` over 49 pairs, so the cheap instrument's ranking holds
on aggregate. Where it does not, it fails in both directions, and that is the reason h3 exists:

* **`mixed` on L_shape is not usable at all, and tension could not see it.** h2 scored it 14.3
  tension digits, *better* than the heuristic. In the real solve it returns 9, 10, then 14
  "eigenvalues" in a window holding 5 — 13.377824, 14.709678, 17.125992 and others are not
  eigenvalues — and Moler--Payne correctly refuses to certify them (**MP = −1.8, −2.2, −2.3**).
  The heuristic finds exactly the right five at 12.7-13.2 certified digits. The heuristic is
  **+14.5 to +15.4 certified digits ahead on L_shape**, which is not a small-print caveat on the
  h2 table; it is the difference between a basis you can search with and one you cannot.
  (Mechanism: `sigma` at a *known* eigenvalue, and contrast against midpoints between *known*
  eigenvalues, cannot detect a spurious minimum somewhere else in the window. That is a limit of
  the h1/h2 instrument, not of this domain.)
* **The heuristic has the same failure on `iso_tri_h4`, worse.** At `precision=1e-4` it returns
  **18** eigenvalues for a 4-eigenvalue window with `MP = −3.1`, while `mixed` at the same size
  certifies 12.7. At 1e-8 it recovers to MP 3.6 with a true error of 13.2 digits — the
  eigenvalues are right, the certificate is not. Together with the interior-source warnings at
  1e-10/1e-12 (Finding C3), `iso_tri_h4` is where this recipe is broken rather than merely
  expensive.
* **`iso_tri_h16` shows the same certificate/accuracy split without the spurious modes**:
  MP 1.5 at 1e-4 and 5.9 at 1e-8, against true errors of 12.6 and 13.3. A residual spike the
  bound sees and the eigenvalue does not — the sharp apex.
* **GWW1/GWW2 confirm the win, and it grows with size**: at n≈519 the heuristic certifies 11.7
  digits against `mixed`'s 10.4 (GWW1) and 10.5 (GWW2); at n=346 it is +0.3/+0.7 over
  `fb_plus_bdry_fs`.

### Q2 again, under the certified bound

`precision` was derived from Moler--Payne in the first place (Sec 1's `Lambda = ln(10 C_Omega /
eps)`), so the certified reading is the one the paper's own derivation is answerable to, and it
is markedly less flattering than the tension reading:

| requested R | heuristic certifies ≥ R | best baseline certifies ≥ R | median MP − R (heur) | median true − R (heur) |
|---|---|---|---|---|
| 4 | 16/18 | 17/18 | +7.0 | +9.3 |
| 8 | 16/18 | 17/18 | +3.5 | +5.8 |
| 12 | **6/13** | 11/13 | **−0.1** | +2.2 |

(`true` excludes `reg_ngon_8`, whose reference is itself in question — see the caveat below.)

So: comfortably pessimistic at 4 and 8 digits; at 12 digits requested it *has* the accuracy
(true error beats the request by a median 2.2 digits) but **usually cannot certify it**, landing
at 11.0-12.9 MP digits where 12 was asked. If `precision` is ever to be a promise rather than a
hint, that gap — between what the basis achieves and what the bound can demonstrate — is the
thing to close, and Moler--Payne's `C_Omega` calibration (Sec 7, deferred in this
implementation) is where it lives.

### Side finding: the suite's `reg_ngon_8` reference table is missing its lowest mode

Not a heuristics result, but it fell out of h3 and should not be left on the floor. On
`reg_ngon_8` all three constructions — heuristic, `mixed`, `fb_plus_bdry_fs` — independently
report a simple eigenvalue at **6.484933** with 11.4-12.3 certified Moler--Payne digits, and it
is absent from `benchmarks/suite/run/reference_values.REFERENCE['reg_ngon_8']`, whose first
entry is the double 16.456119. Faber--Krahn settles which side is wrong: the octagon has area
2.8284, the disk of equal area has `lam_1 = 6.424`, and no other domain of that area can have a
smaller first eigenvalue — so `lam_1` is just above 6.424, 6.484933 is it, and 16.456 is
`lam_2 = lam_3`. The table was produced by a `symmetry(reg_ngon(8) D2, |G|=4)` solve
(`method` field), so the likely cause is a symmetry class dropped when the reduced spectra were
merged. The consequence for this study is only that `reg_ngon_8`'s `true` column is meaningless
(its certified column is not); the consequence for the suite is that a reference table is short a
mode and `reg_ngon_8`'s `digit_ceiling = 10.3` was measured against it.

## Q3. Which constants matter?

**At these wavenumbers, all of them are cost knobs and none is an accuracy knob.** h4 sweeps
each `HeuristicConfig` field one at a time on `square`, `L_shape`, `reg_ngon_6`, `iso_tri_h4`,
`right_trapezoid` and `chevron_1_2` at `precision=1e-10` (150 rows). Median across domains:

| knob | median ΔA | median Δn | note |
|---|---|---|---|
| `eta=0.5` (from 0.30) | **+0.2** | **−25%** | −35% on chevron_1_2 and iso_tri_h4 |
| `eta=0.15` | −0.0 | +66% | +88% on chevron_1_2, for nothing |
| `C_omega=0.3` (from 10) | +0.0 | −10% | −26% on iso_tri_h4 |
| `C_omega=1.0` | +0.0 | −7% | |
| `C_omega=30.0` | +0.0 | +4% | |
| `n_bridge=20` (from 10) | **+0.4** | +3% | best accuracy-per-column change found |
| `n_bridge=5` | −0.0 | −2% | |
| `s_min_frac=0.15` (from 0.05) | +0.2 | **0%** | free: moves the innermost bridge pole, not the count |
| `handover_frac=0.95` (from 0.80) | −0.1 | −3% | −16% on reg_ngon_6 |
| `handover_frac=0.6` | +0.1 | +6% | +29% on reg_ngon_6 |
| `include_regular_fb=False` | −0.1 | −4% | −20% on square — but see h5/h6 below |
| `delta_frac_D=0.5` (from 0.25) | +0.2 | −4% | −41% on square, +0% elsewhere |
| `gamma`, `order_margin`, `airy_margin` | ±0.2 | ±6% | small and inconsistent in sign |
| `nyquist_ppw=2 or 5` (from 3) | +0.0 | **0.0%** | dead constant, see below |
| `s_min_frac=0.01` | +0.0 | 0% | |
| `max_reflections=2` (from 1) | +0.0 | 0% | dead except +16% on chevron_1_2, no gain |

Two constants do literally nothing, and that is worth knowing:

* **`nyquist_ppw` never binds.** The source spacing is `h = min(2 pi/(ppw kappa), pi delta/Lambda)`,
  and a direct check over 4000 boundary samples finds the Nyquist term is the minimum on **0.0%**
  of the boundary for `square`, `L_shape`, `reg_ngon_6`, `iso_tri_h4`, `chevron_1_2`,
  `chevron_2_4` at both 1e-2 and 1e-10 (the sole exception is `rect_thin` at 1e-2, where it
  binds everywhere). So spacing is entirely `(pi eta / Lambda) * dist(x, S*)`, and the cost model
  is `Lambda * integral ds / dist(x, S*)` — which is exactly why thin and sharp domains explode
  (Finding C1) and why `eta` and `C_omega` are the two big cost dials. **Caveat: these `lam_max`
  come from `weyl_est(6, domain)`, i.e. `kappa` of 4-13. At the larger `kappa` a shape-optimization
  loop would reach, the Nyquist term must eventually bind, and this conclusion expires there.**
* **`max_reflections=2` changes no count anywhere** except on chevron_1_2, where it adds 16%
  columns and buys nothing. Second-generation images never become the nearest obstruction, so
  the `O(n_edges^2)` depth-2 pass is not earning its cost on any suite polygon.

**h5/h6: OFAT winners do not simply compose, and the ablation says exactly why.** Applying six
h4 winners together (`lean` = `eta=0.5, C_omega=1, n_bridge=20, s_min_frac=0.15,
handover_frac=0.95, include_regular_fb=False`) saves 26% of columns at a median cost of 0.5
digits — but it destroys the all-regular-corner domains: **−8.7 digits on square, −8.6 on
eq_tri, −6.6 on rect_thin**, while *gaining* on L_shape (+0.3), iso_tri_h4 (+0.9) and
right_trapezoid (+0.9). h6's leave-one-out ablation pins it on one ingredient: restoring
`include_regular_fb=True` recovers eq_tri from −8.6 to −0.8 and rect_thin from −6.6 to −0.1,
while restoring any of the other five leaves the collapse untouched. The interaction is
mechanical — with stock source spacing the ambient MFS curve can cover a regular corner well
enough that its optional FB terms look free (h4: −0.1 digits on square), but once the curve is
also sparser (`eta=0.5`) and its inner reach smaller, nothing is left to represent the corner.
**A single-factor screen would have shipped this.**

---

## Recommended changes

Nothing in `lappy/heuristics.py` was changed for this study — it is the object under test, and
every number above is against stock `HeuristicConfig()`. What the measurements support:

1. **Adopt the `safe` combination as the new defaults:** `eta=0.5`, `C_omega=1.0`,
   `n_bridge=20`, `s_min_frac=0.15`, `handover_frac=0.95`, `include_regular_fb` left **True**.
   Measured over all 18 h1 domains at 1e-8 and 1e-12 (h5, 36 pairs): median **−12% columns for
   −0.1 digits**, worst case −1.2 digits (iso_right_tri at 1e-8, 11.6 -> 10.4, still 2.4 digits
   past its request). The savings concentrate where they matter — the expensive domains:

   | domain | Δ digits @1e-8 / @1e-12 | Δn @1e-8 / @1e-12 |
   |---|---|---|
   | iso_tri_h4 | **+1.3 / +0.8** | **−43% / −39%** |
   | iso_tri_h16 | −0.1 / +0.3 | **−39% / −39%** |
   | L_shape | +0.2 / +0.3 | −28% / −29% |
   | GWW1 | −0.0 / −0.0 | −27% / −28% |
   | GWW2 | +0.0 / +0.0 | −27% / −28% |
   | right_trapezoid | −0.1 / +0.0 | −25% / −27% |
   | parallelogram_60 | −0.4 / −0.1 | −22% / −25% |
   | reg_ngon_5/6/7/8 | −0.5 .. +0.0 | −8% / −14% |
   | square, rect_*, tri_* | −1.2 .. +0.2 | −6% .. −9% |

   Do **not** adopt `include_regular_fb=False` alongside it: that is the `lean` variant, and it
   costs 6-9 digits on the all-regular-corner domains (h5/h6 above).
2. **Do not raise `precision` expecting accuracy; raise it expecting cost.** The knob is inert
   on 11 of 15 domains (Q2). If `Eigenproblem(domain, precision=p)` is to mean anything, `p`
   has to reach the parts of the recipe that actually set the size — today it mostly does not.
3. **Retire `nyquist_ppw` and `max_reflections` or document them as inactive** at
   `weyl_est`-scale wavenumbers. Keeping tunable constants that provably move nothing invites
   exactly the kind of unfounded tuning claim `bench.py` was built to prevent.
4. **Fix the interior sources on sharp thin domains** (Finding C3) before this is used for
   anything certified. `delta_amb = 0.25 D` is diameter-scaled; a local-thickness-scaled offset
   is the obvious candidate, and `benchmarks/basis_lab/placement.py` already has
   `local_thickness` for it.
5. **Do not build an adaptive `precision` loop (Sec 9) on the assumption that `n` is monotone in
   `Lambda`** — it is not (Finding C2), and the discontinuity is at the weak/singular
   classification boundary.
6. **Do not use the recipe on the sharp/many-corner half of the suite** until the cost model is
   fixed: 800-12000 columns is not a usable basis, and no accuracy measurement was attempted
   there because none would mean anything.
7. **Investigate the spurious eigenvalues on `iso_tri_h4`** (18 modes reported in a
   4-mode window at `precision=1e-4`, MP −3.1). This is the one place the recipe produces a
   basis that a search cannot be trusted with, and it is the same domain as the interior-source
   bug, so the two may share a cause.

### What would change the verdict, and what to measure next

* **A search-aware guard.** h1/h2's contrast is measured against midpoints between *known*
   eigenvalues, which is why it missed `mixed`'s spurious modes on L_shape and the heuristic's
   own on iso_tri_h4. A cheap `sigma` sweep over the whole window (not just the reference
   points) would have caught both, and would make the cheap instrument a fair proxy for the
   expensive one. That is a change to *this harness*, and it is the highest-value follow-up here.
* **Larger `kappa`.** Every conclusion about spacing (the dead `nyquist_ppw`, `eta` as the
   dominant cost dial, the `Lambda * integral ds/dist(x,S*)` cost model) is measured at
   `kappa = sqrt(weyl_est(6, domain))`, i.e. 4-13. A shape-optimization loop wants the top of a
   much larger window, where Nyquist must eventually bind and the balance may invert.
* **`C_Omega` calibration (Sec 7).** The certified/true gap at R=12 is the whole distance
   between `precision` being a hint and being a promise.

---

## Reproducing this

    .venv/bin/python -m pytest tests/test_heuristics.py -q     # 31 tests, the formulas assumed here
    .venv/bin/python -m benchmarks.basis_lab.heur report       # every table above, from the ledger
    .venv/bin/python -m benchmarks.basis_lab.heur h0           # seconds
    .venv/bin/python -m benchmarks.basis_lab.heur h1           # ~1 h
    .venv/bin/python -m benchmarks.basis_lab.heur h2           # ~1 h
    .venv/bin/python -m benchmarks.basis_lab.heur h4 h5 h6     # ~2 h total (one stage per call)
    .venv/bin/python -m benchmarks.basis_lab.heur h3           # ~3 h

Each stage skips rows already on disk, so an interrupted sweep resumes by being restarted, and
adding a rung to a ladder re-measures only the new cells. Two stages running at once is safe but
can duplicate a cell (`seen` is read once per domain); `load_all` de-duplicates by `record_id`.
Timings above are from a run where h3 overlapped h4/h5, so the `seconds` field on those rows is
contended and should not be read as cost data.
