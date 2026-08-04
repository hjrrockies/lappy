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
