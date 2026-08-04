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
| `sector_reflex` | 240 | 1.00 | clean | 13.7 | **12.9** | 1 |
| `sector_sharp_p65` | 240 | 1.06 | clean | 13.7 | **15.1** | 1 |
| `sector_sharp_p133` | 320 | 0.94 | clean | 13.6 | **15.2** | 1 |
| `sector_slit` | 320 | 0.90 | clean | 13.6 | **13.1** | 1 |

### Notes

**Large gains over the previous session**, all from the same two changes
(`solve_interval` at `ltol=1e-14`, and `rtol_default` 1e-14 -> 1e-12):

    rect_near_deg_1e5   4.8 -> 15.4   near-degenerate pair (1.2e-5) no longer merged
    rect_thin           8.5 -> 15.2   and this was the 59.8GB memory runaway
    iso_right_tri       5.8 -> 14.6   was on the symmetry path

**Certification reads OPTIMISTIC on exactly the two reentrant domains.**
Everywhere else the bound is ~1.5 digits pessimistic, but:

    sector_reflex  (p=2/3)    certified 13.7  >  true 12.9
    sector_slit    (p=0.504)  certified 13.6  >  true 13.1

Both are the reentrant cases; the eight non-reentrant domains are all
pessimistic. `certify.boundary_sup` samples on a mesh graded as `t = s**grade`
with `grade=3.0`, which appears to under-resolve `r^p` singularities for `p < 1`
and so understates the boundary sup. Harmless for bucketing (both clear the bar)
but it matters wherever certification is the *only* evidence -- i.e. every
non-analytic domain, which is most of the suite. Worth checking whether the
reported `bdry_sup_drift` already flags these.

**Pre-flight ratios cluster 0.54-1.07**, well under the 1.5 threshold, and no
analytic instance was flagged. That is the expected result for a tier that all
buckets 1, and it means the threshold has not yet been exercised in anger on
this tier -- `GWW1` remains the only positive detection.
