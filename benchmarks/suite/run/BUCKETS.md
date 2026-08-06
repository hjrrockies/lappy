# Domain buckets

How each benchmark domain fares on the first-10-eigenvalues problem with
`make_default_basis`. Generated one domain at a time via
`python -m benchmarks.suite.bucket <key>`; raw records in `run/buckets.jsonl`,
tension curves in `run/curves/`.

**Buckets**
1. 8+ digits with `make_default_basis` at some `n_basis`
2. under 8 digits, but no solver failure and no missing eigenvalues
3. solver failure and/or missing eigenvalues

Bar is **true error where a closed form exists**, certified otherwise. Both are
reported: certification runs ~1.5 digits pessimistic on this tier.

Pipeline: `build_solver` -> pre-flight scan (300 pts, no search) -> guard
threshold from the scan -> `solve_interval(ltol=1e-14)` -> `certify_solver`.
Pre-flight `ratio` is minima found / Weyl-expected; > 1.5 means ill-conditioned
(calibrated in NOTEBOOK.md session 7).

## Analytic tier

Every domain cleared 8 digits **at its default `n_basis`, first try** -- no
`rtol` change, no collocation change, no ladder needed.

| domain | n_basis | ratio | verdict | certified | true | bucket |
|---|---|---|---|---|---|---|
| `square` | 120 | 0.64 | clean | 13.7 | **15.7** | 1 |
| `rect_near_deg_1e3` | 120 | 0.64 | clean | 13.5 | **15.5** | 1 |
| `rect_near_deg_1e5` | 120 | 0.64 | clean | 14.1 | **15.4** | 1 |
| `rect_thin` | 240 | 0.94 | clean | 13.2 | **15.2** | 1 |
| `eq_tri` | 120 | 0.54 | clean | 13.9 | **15.6** | 1 |
| `iso_right_tri` | 120 | 1.07 | clean | 13.5 | **14.6** | 1 |
| `disk` | 120 | 0.56 | clean | 14.2 | **15.2** | 1 |
| `sector_reflex` | 240 | 1.00 | clean | 13.7 | **14.6** | 1 |
| `sector_sharp_p65` | 240 | 1.06 | clean | 13.7 | **15.1** | 1 |
| `sector_sharp_p133` | 320 | 0.94 | clean | 13.6 | **15.2** | 1 |
| `sector_slit` | 320 | 0.90 | clean | 13.6 | **15.2** | 1 |

### Notes

**Large gains over the previous session**, all from the same two changes
(`solve_interval` at `ltol=1e-14`, and `rtol_default` 1e-14 -> 1e-12):

    rect_near_deg_1e5   4.8 -> 15.4   near-degenerate pair (1.2e-5) no longer merged
    rect_thin           8.5 -> 15.2   and this was the 59.8GB memory runaway
    iso_right_tri       5.8 -> 14.6   was on the symmetry path

**Certification is uniformly pessimistic (~1-2 digits), as designed.**

An earlier version of this table showed `sector_reflex` and `sector_slit`
certifying *above* their true error, which would mean the Moler--Payne bound was
being violated. That was a **reference-table error, not a solver or
certification error**: `reference._bessel_zero` refined with `brentq` at default
tolerance, which is exact at integer order but loses 2-3 digits at fractional
order -- and the sector domains need `nu = m*pi/alpha`, always fractional.

    nu = 4/3    |J_nu(z)| = 1.1e-13 at the returned zero   (should be ~1e-16)
    nu = 1.512  |J_nu(z)| = 6.5e-14
    nu = 2,4,6                       ~1e-16, correct

Eigenvalues were off by up to 1.4e-13 relative, so the solver was being measured
against a ruler less accurate than itself. Fixed by computing the zeros with
`mpmath.besseljzero` at 40 digits; residuals are now ~1e-16 at every order, and
both sectors moved from apparently-violating to comfortably pessimistic
(12.9 -> 14.6 and 13.1 -> 15.2).

Ruled out along the way, both worth recording:
  - `boundary_sup` converges (sector_reflex: 8.652e-16 at n=400 to 8.737e-16 at
    n=1600/grade 6, drift 0.0%); it is not under-resolving the r^p corner.
  - `interior_l2` cubature agrees with an independent **Rellich-identity**
    evaluation of the same norm to 6-7 digits, on the curved domain as well as
    the polygon (ratios 1.000000 and 1.000001). The curved-domain mesh is sound,
    and the Rellich Dirichlet branch is accurate even though it neglects a
    `u != 0` boundary term.

**Pre-flight ratios cluster 0.54-1.07**, well under the 1.5 threshold, and no
analytic instance was flagged. That is the expected result for a tier that all
buckets 1, and it means the threshold has not yet been exercised in anger on
this tier -- `GWW1` remains the only positive detection.

## Corner tier

23 domains. Run one per process with pre-flight first; the abort guard fired
twice and prevented two multi-hour refinements of pure noise.

| domain | n_basis | ratio | contrast | certified | found | bucket |
|---|---|---|---|---|---|---|
| `L_shape` | 240 | 0.90 | 3.4e+02 | 13.4 | 11/10 | 1 |
| `H_shape` | 480 | 0.69 | 2.6e+03 | 9.8 | 10/10 | 1 |
| `GWW1` | 320 | 0.96 | 1.2e+02 | 8.9 | 10/10 | 1 |
| `GWW2` | **480** | 0.96 | 1.3e+02 | **10.0** | 10/10 | 1 |
| `reg_ngon_5` | 240 | 0.55 | 1.3e+02 | 12.8 | 10/10 | 1 |
| `reg_ngon_6` | 320 | 0.74 | 2.0e+03 | 12.1 | 12/10 | 1 |
| `reg_ngon_7` | 240 | 0.56 | 5.9e+04 | 10.7 | 12/10 | 1 |
| `reg_ngon_8` | 320 | 0.65 | 5.5e+03 | 10.4 | 12/10 | 1 |
| `iso_tri_h05` | 240 | 0.95 | 9.9e+02 | 12.5 | 10/10 | 1 |
| `iso_tri_h1` | 240 | 1.34 | 1.7e+02 | 13.5 | 10/10 | 1 |
| `iso_tri_h4` | 240 | 0.89 | 8.0e+01 | 11.3 | 10/10 | 1 |
| `iso_tri_h16` | 240 | 0.96 | 4.8e+03 | 11.4 | 10/10 | 1 |
| `parallelogram_60` | 240 | 1.09 | 2.4e+02 | 12.2 | 10/10 | 1 |
| `right_trapezoid` | 240 | 1.09 | 1.7e+02 | 11.4 | 10/10 | 1 |
| `chevron_1_2` | 480 | 0.91 | 6.0e+01 | 8.8 | 10/10 | 1 |
| `chevron_1_15` | 480 | 0.95 | 2.3e+02 | 6.3 | 10/10 | 2 |
| `chevron_1_125` | 480 | 0.88 | 1.3e+02 | 5.0 | 10/10 | 2 |
| `parallelogram_p65` | 320 | 1.03 | 9.9e+01 | 7.5 | 10/10 | 2 |
| `parallelogram_p127` | 320 | 1.08 | 3.3e+02 | 4.1 | 10/10 | 2 |
| `chevron_2_3` | **160** | 0.91 | 4.1e+03 | 3.7 | 10/10 | 2 |
| `chevron_2_4` | **160** | 1.43 | 1.2e+03 | 3.3 | 10/10 | 2 |
| `spiral` | 320 | 0.72 | 2.1e+01 | 1.6 | **3/10** | 3 |
| `spiral_t25` | — | — | — | build failed | — | 3 |

**15 / 6 / 2** after taking the best `n_basis` per domain.

### The three buckets have distinct, observable signatures

This is the payoff of separating the taxonomy classes:

- **Bucket 1** -- clean curve, contrast 60-59000, complete spectrum.
- **Bucket 2 (#1, basis insufficiency)** -- *clean* curve, contrast 60-330,
  complete spectrum, and still short of 8 digits. `chevron_1_125` is the
  textbook case: 12 minima against Weyl 13.6, all ten eigenvalues found, and
  5.0 certified digits. Nothing is wrong with the search; the wells simply do
  not go deep enough. Note this domain was excluded from `produce.py` entirely
  as "does not converge" -- it converges, it just cannot reach 8 digits.
- **Bucket 3 (#2, ill-conditioning)** -- `chevron_2_3` and `chevron_2_4` at
  `n_basis=480` show **contrast 1.6** against 60+ for every healthy instance. A
  contrast near 1 means the tension curve is flat: no wells at all, just noise
  around 4e-10. Both the pre-flight check (ratio 7.4, 8.2) and the in-search
  guard (36 minima vs a threshold of ~20) flagged them independently.
- **Bucket 3 (other)** -- `spiral` found **3 of 10** eigenvalues. An incomplete
  spectrum is the failure certification cannot see: each of those three carries
  a valid Moler--Payne bound. `spiral_t25` fails earlier still, in basis
  construction (`corner_branch_cut_polyline: no valid polyline cut found`).

**`contrast` may be the better pre-flight statistic.** It separated the aborted
instances more sharply than the minima ratio did (1.6 vs 60+, versus 7-8 vs
~1), and it is cheaper to reason about. Worth folding into `is_noisy`.

### n_basis has a domain-specific optimum, and it is not monotone

`chevron_2_3`/`chevron_2_4` are degenerate at 480 (contrast 1.6, aborted) but
clean at 160 (contrast 4.1e3, 1.2e3), where they find all ten eigenvalues and
land in bucket 2. Meanwhile `GWW2` goes the other way: 7.8 at 320, **10.0 at
480**, crossing into bucket 1. Measured both directions:

    chevron_2_3         3.7 @160    aborted @480     -> 160
    chevron_2_4         3.3 @160    aborted @480     -> 160
    chevron_1_15        5.5 @160        6.3 @480     -> 480
    chevron_1_125       4.3 @160        5.0 @480     -> 480
    parallelogram_p65   7.1 @240        7.5 @320     -> 320
    parallelogram_p127  4.0 @240        4.1 @320     -> ~equal
    GWW2                7.8 @320       10.0 @480     -> 480

So there is no global rule "bigger is better" or "smaller is better", and a
bucket-3 verdict at one `n_basis` is a statement about the *instance*, not the
domain. Any future automation must ladder per-domain and keep the best, which
is what the pre-flight contrast makes cheap to do -- a degenerate instance is
identifiable in ~1 minute of scanning, before any solve.

## Curved tier

| domain | n_basis | ratio | contrast | certified | found | bucket |
|---|---|---|---|---|---|---|
| `ellipse_a2` | 240 | 0.98 | 2.9e+02 | 13.6 | 10/10 | 1 |
| `ellipse_a3` | 320 | 0.99 | 1.6e+02 | 12.5 | 10/10 | 1 |
| `ellipse_a4` | 320 | 0.98 | 7.6e+01 | 13.1 | 10/10 | 1 |
| `mushroom` | 320 | 0.99 | 1.2e+03 | 11.5 | 10/10 | 1 |
| `cut_square_r025` | 640 | 1.02 | 3.0e+02 | 12.2 | 10/10 | 1 |
| `cut_square_r05` | 640 | 1.00 | 1.3e+02 | 11.7 | 10/10 | 1 |
| `stadium` | 320 | 1.01 | 3.0e+02 | 2.8 | 10/10 | 2 |
| `stadium_L2` | 320 | 1.00 | 1.2e+02 | 2.8 | 10/10 | 2 |
| `mushroom_thin` | 320 | 0.98 | 3.2e+02 | 7.4 | 10/10 | 2 |
| `mushroom_neck01` | 320 | 1.06 | 1.1e+02 | 5.0 | 10/10 | 2 |

**6 / 4 / 0.**

---

# Summary: 44 domains, 32 / 10 / 2

    bucket 1   32   8+ digits with make_default_basis
    bucket 2   10   complete spectrum, clean curve, under 8 digits
    bucket 3    2   spiral, spiral_t25

## Bucket 2 is basis insufficiency, isolated -- and it has exactly three mechanisms

Every bucket-2 domain has a **clean tension curve** (contrast 60-4100, minima
ratio 0.88-1.43) and a **complete spectrum** (10/10 found). So conditioning
(#2), collocation (#3) and the search (#4) are excluded *by observation* in all
ten. What remains is issue #1, and it sorts into three groups:

**Sharp corners (6):** chevron x4, parallelogram_p65/p127. Corner-centred
Fourier--Bessel functions need orders `m*p` with `p` up to 28; the harmonics get
sparse and expensive faster than they get useful.

    chevron_2_4    3.3      parallelogram_p127  4.1
    chevron_2_3    3.7      parallelogram_p65   7.5
    chevron_1_125  5.0
    chevron_1_15   6.3

**Curvature discontinuity (2):** `stadium`, `stadium_L2`, both exactly **2.8**.
Zero corners, four C^1-but-not-C^2 junctions that are neither a corner the FB
basis can aim at nor the smooth boundary the FS basis wants. Two different
aspect ratios giving the identical ceiling says the mechanism is the junction,
not the geometry around it. Independently reproduces the 2.9 recorded in
TUNING_LOG.md through an entirely different pipeline.

**Thin neck (2):** the mushroom neck-width sweep, monotone across the
bucket boundary:

    mushroom          b=1.0    11.5   bucket 1
    mushroom_thin     b=0.25    7.4   bucket 2
    mushroom_neck01   b=0.1     5.0   bucket 2

That is a **parametric family crossing the bucket-1/2 boundary continuously**,
which makes it the best available handle on issue #1 for a mechanism with no
closed form -- the same role `disk_sector` plays for corner exponents.

## Bucket 3 is one mechanism, not two

Both are `spiral`: coils bury corners so they have no straight-ray sightline to
infinity. `spiral` finds **3 of 10** eigenvalues; `spiral_t25` fails earlier, in
basis construction (`corner_branch_cut_polyline: no valid polyline cut found`).
This is a *geometry* limitation, not a solver one.

Note `spiral`'s three eigenvalues each carry a valid Moler--Payne certificate.
An incomplete spectrum is the one failure certification structurally cannot see,
which is why bucket 3 is defined by completeness rather than by accuracy.

## Domains that moved out of "hard" without any per-domain tuning

Relative to the previous session, from pipeline changes alone
(`solve_interval` at `ltol=1e-14`, `rtol_default=1e-12`, correct Bessel zeros):

    cut_square_r025    7.2  -> 12.2    reg_ngon_8   crashed -> 10.4
    GWW2               7.8  -> 10.0    iso_tri_h05  mem fail -> 12.5
    rect_near_deg_1e5  4.8  -> 15.4    rect_thin    mem fail -> 15.2 (true)
    iso_right_tri      5.8  -> 14.6

None of these needed a per-domain fix. That is the strongest evidence that most
of what looked like domain difficulty was pipeline defect.


---

# Re-run with boundary-integral certification (tag `orth`)

All 44 domains re-run after `||u||_L2` moved from interior cubature to the
Rellich boundary integral over `MPSEigensolver`'s corner-adapted quadrature
(`certify.boundary_l2`). Each domain at **its own best recorded `n_basis`** --
the one this table already used -- so the comparison is like for like.

**Verdict: 32 / 10 / 2. No domain changed bucket. No eigenvalue lost a digit.**

`eps` is scale-invariant, so this was the predicted outcome: normalizing `u`
cannot change a ratio in which its scale cancels. What changed is that the
denominator no longer needs a triangulation (`spiral_t25` has none that builds
inside 90s), no longer over-integrates on `cut_square_r025` (mesh 0.9509685 vs
area 0.9509126 -- the wrong side of a bound that needs an under-estimate), and
is much cheaper where the mesh was large (disk 128s->58s, mushroom 235s->110s).

The `sector_*` "gains" below are baseline artifacts: those `buckets.jsonl`
records predate the `_bessel_zero` -> `mpmath.besseljzero` fix, and the corrected
values in the analytic table above (14.6, 15.2) are what the re-run reproduces.

`norm` is which estimate the bound used: `boundary` alone, or `mixed` where the
`x0`-spread exceeded 1e-8 and cubature was computed as well (the larger of the
two lower bounds wins). `x0-spread` is the worst over the ten eigenvalues.
Seconds are solve+certify, old/new.

| domain | tier | n_basis | bucket | digits old -> new | norm | x0-spread | s old/new |
|---|---|---|---|---|---|---|---|
| `square` | analytic | 120 | 1 → 1 | 15.65 → 15.65 | boundary | 2e-15 | 27/28 |
| `rect_near_deg_1e3` | analytic | 120 | 1 → 1 | 15.54 → 15.54 | boundary | 2e-15 | 17/18 |
| `rect_near_deg_1e5` | analytic | 120 | 1 → 1 | 15.44 → 15.44 | boundary | 2e-15 | 21/21 |
| `rect_thin` | analytic | 240 | 1 → 1 | 15.24 → 15.24 | boundary | 2e-15 | 69/70 |
| `eq_tri` | analytic | 120 | 1 → 1 | 15.57 → 15.57 | boundary | 4e-15 | 27/27 |
| `iso_right_tri` | analytic | 120 | 1 → 1 | 14.62 → 14.62 | boundary | 1e-14 | 17/17 |
| `disk` | analytic | 120 | 1 → 1 | 15.23 → 15.23 | boundary | 8e-15 | 128/58 |
| `sector_reflex` | analytic | 240 | 1 → 1 | 12.86 → 14.66 | boundary | 5e-14 | 47/38 |
| `sector_sharp_p65` | analytic | 240 | 1 → 1 | 15.07 → 15.52 | boundary | 2e-14 | 13/12 |
| `sector_sharp_p133` | analytic | 320 | 1 → 1 | 15.17 → 15.17 | boundary | 2e-14 | 17/16 |
| `sector_slit` | analytic | 320 | 1 → 1 | 13.10 → 15.44 | mixed | 7e-01 | 385/380 |
| `L_shape` | corner | 240 | 1 → 1 | 13.39 → 13.39 | boundary | 2e-14 | 34/35 |
| `H_shape` | corner | 480 | 1 → 1 | 9.76 → 9.76 | boundary | 2e-12 | 301/301 |
| `GWW1` | corner | 320 | 1 → 1 | 8.92 → 8.92 | mixed | 4e-05 | 125/126 |
| `GWW2` | corner | 480 | 1 → 1 | 9.99 → 9.99 | mixed | 5e-05 | 245/252 |
| `reg_ngon_5` | corner | 240 | 1 → 1 | 12.76 → 12.76 | mixed | 2e-06 | 99/98 |
| `reg_ngon_6` | corner | 320 | 1 → 1 | 12.10 → 12.10 | boundary | 6e-15 | 157/157 |
| `reg_ngon_7` | corner | 240 | 1 → 1 | 10.74 → 10.74 | mixed | 2e-05 | 114/114 |
| `reg_ngon_8` | corner | 320 | 1 → 1 | 10.39 → 10.39 | boundary | 1e-14 | 175/173 |
| `chevron_1_15` | corner | 480 | 2 → 2 | 6.31 → 6.31 | mixed | 1e-08 | 259/264 |
| `chevron_1_2` | corner | 480 | 1 → 1 | 8.83 → 8.83 | boundary | 3e-11 | 228/232 |
| `chevron_2_3` | corner | 160 | 2 → 2 | 3.65 → 3.65 | mixed | 8e-06 | 45/48 |
| `chevron_2_4` | corner | 160 | 2 → 2 | 3.30 → 3.30 | mixed | 2e-05 | 66/69 |
| `chevron_1_125` | corner | 480 | 2 → 2 | 5.03 → 5.03 | mixed | 2e-06 | 300/309 |
| `iso_tri_h05` | corner | 240 | 1 → 1 | 12.50 → 12.50 | mixed | 3e-06 | 46/47 |
| `iso_tri_h1` | corner | 240 | 1 → 1 | 14.67 → 14.67 | boundary | 2e-14 | 30/31 |
| `iso_tri_h4` | corner | 240 | 1 → 1 | 11.34 → 11.34 | boundary | 4e-09 | 59/59 |
| `iso_tri_h16` | corner | 240 | 1 → 1 | 11.41 → 11.41 | mixed | 5e-08 | 76/78 |
| `parallelogram_60` | corner | 240 | 1 → 1 | 12.21 → 12.21 | boundary | 5e-15 | 46/47 |
| `parallelogram_p65` | corner | 320 | 2 → 2 | 7.50 → 7.50 | boundary | 4e-11 | 104/106 |
| `parallelogram_p127` | corner | 320 | 2 → 2 | 4.12 → 4.12 | mixed | 2e-07 | 147/151 |
| `right_trapezoid` | corner | 240 | 1 → 1 | 11.37 → 11.37 | mixed | 4e-05 | 19/20 |
| `spiral` | corner | 320 | 3 → 3 | 1.58 → 1.58 | mixed | 5e-03 | 176/188 |
| `spiral_t25` | corner | 320 | 3 → 3 | — | — | — | —/— |
| `ellipse_a2` | curved | 240 | 1 → 1 | 13.60 → 13.54 | mixed | 1e-05 | 88/92 |
| `ellipse_a3` | curved | 320 | 1 → 1 | 12.48 → 12.52 | mixed | 5e-04 | 161/162 |
| `ellipse_a4` | curved | 320 | 1 → 1 | 13.07 → 13.07 | mixed | 7e-04 | 189/187 |
| `stadium` | curved | 320 | 2 → 2 | 2.83 → 2.83 | mixed | 3e-06 | 222/211 |
| `stadium_L2` | curved | 320 | 2 → 2 | 2.77 → 2.77 | mixed | 6e-06 | 193/193 |
| `mushroom` | curved | 320 | 1 → 1 | 11.53 → 11.53 | boundary | 3e-12 | 235/110 |
| `mushroom_thin` | curved | 320 | 2 → 2 | 7.40 → 7.40 | boundary | 4e-09 | 207/120 |
| `mushroom_neck01` | curved | 320 | 2 → 2 | 5.01 → 5.01 | mixed | 3e-06 | 195/154 |
| `cut_square_r025` | curved | 640 | 1 → 1 | 12.22 → 12.22 | boundary | 3e-10 | 342/317 |
| `cut_square_r05` | curved | 640 | 1 → 1 | 11.70 → 11.70 | mixed | 9e-08 | 351/338 |


### The `x0`-spread flags 22 domains, and costs nothing

Re-certifying at four node densities with the fallback disabled gives identical
digits to three decimals (`ellipse_a2` 13.541 at 46 and at 250 nodes; `GWW1`
8.920 at 204 and at 572). The deflation and the fallback never move a reported
digit -- they are insurance, not a correction.

`BoundaryQuad.precision` should not be read as an error estimate: it advertised
1e-13 on 40 domains while the measured spread exceeded 1e-8 on 22 of them, and
on `sector_slit` (whose nu=0.504 corner is demoted to a smooth rule) it claimed
1e-13 against a measured 6.9e-01. See NOTEBOOK.md for the partial diagnosis.
