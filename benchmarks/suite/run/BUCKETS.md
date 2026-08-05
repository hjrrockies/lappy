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
