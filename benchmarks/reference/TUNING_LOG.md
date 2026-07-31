# Tuning log: pushing MPS precision past the first two reference-value passes

Running lab notebook for the deep precision push. Entries in chronological
order per domain. Not a polished report -- records what was tried and what
happened, including dead ends.

## Systemic finding (applies to ALL domains, logged first)

**The old `polish_eigs` bracket width was a silent precision ceiling for
essentially every domain, independent of basis/corner issues.**

`Eigenproblem.solve` → `MPSEigensolver.solve_interval` uses
`ltol_default=1e-8` (relative) for `minimize_on_bracket`'s convergence
tolerance -- i.e. the coarse eigenvalue location it returns is only good to
~1e-8 relative precision. The old `common.polish_eigs(solver, eigs,
ltol=1e-14)` then searched for the true root only within
`eig*(1 ± 1e-14)` -- a window ~1e6x narrower than the coarse estimate's
actual uncertainty. For eigenvalues where the true root wasn't already
sitting inside that absurdly tight window, `golden_search` had no chance of
finding it; polishing did approximately nothing beyond re-evaluating near
the same (already coarse) point.

**Verified fix**: build a solver, use the new `manual_solve` (bypasses
`Eigenproblem`/`solve_interval` entirely, uses `opt.bracket_mins`/
`opt.minimize_on_bracket` directly with a tight `minimize_tol=1e-12` for
the per-bracket minimizer), then polish with `polish_eigs(...,
bracket_rel_width=1e-9)` -- wide enough to contain the (already-good)
coarse estimate's uncertainty, narrow enough to converge tightly from
there. New combined pipeline: `common.solve_domain_v2` /
`escalate_and_solve_v2`.

- `ellipse(2,1)` @ n_basis=240: old result 7.4-11.4 digits -> **new result
  13.3-14.4 digits** (all 11 eigenvalues, tensions 4e-15 to 1.5e-14), with
  literally no change to basis, collocation, or n_basis. Pure pipeline fix.
- `chevron(1,1.5)` @ n_basis=120: old result ~4.5-5.8 digits -> **new
  result: unchanged, 4.5-5.6 digits** (tensions ~3e-6 to 3e-7). Confirms
  this fix is *not* a universal cure -- it only recovers precision that was
  already latent in an already-deep tension well. Chevron's well itself is
  shallow (basis genuinely too poor at the sharp corner), so better
  polishing of a shallow minimum doesn't create resolution that isn't
  there. This is exactly the distinction the diagnostics are meant to
  catch: pipeline-bug-limited vs genuinely resolution-limited.

**Plan going forward**: re-run every domain (not just the previously-flagged
"weak" ones) through `solve_domain_v2`/`escalate_and_solve_v2` first, at
their *already-used* n_basis (no new basis cost) to see how much is free
precision recovery. Only then do domain-specific basis surgery (corner
`f`-weighting, custom `by_corners` C/sigma, denser collocation) for
whatever is still short of 10+ digits.

### Full recheck results (same n_basis as before, `solve_domain_v2` only, no basis changes)

| Domain | Old (digits) | New (digits) | Verdict |
|---|---|---|---|
| ellipse(2,1) @240 | 7.4-11.4 | 13.3-14.4 | **clean win** (pipeline bug) |
| L_shape() @240 | 8.1-10.7 | 12.9-13.3 | **clean win** |
| mushroom @240 | 6.3-11.0 | 11.3-12.5 | **clean win** |
| reg_ngon N=5 @120 | 7.1-11.5 | 11.7-12.3 | **clean win** |
| reg_ngon N=6 @240 | 6.3-8.9 | 11.8-13.0 | **clean win** |
| reg_ngon N=7 @120 | (untried @240 before, hung) | 10.2-11.7 | **clean win**, hang avoided by manual_solve |
| reg_ngon N=8 @120 | 4.0-9.5 (messy 5-cluster) | 8.8-10.3, but **one outlier at 2.8 digits** (lambda=29.5368) | mostly fixed; one mode needs investigation |
| cut_square r=0.25 @240 | 4.7-7.0 | 4.7-6.9 (unchanged) | genuinely resolution-limited -- needs basis surgery |
| cut_square r=0.5 @240 | 3.0-8.3 | 6.6-9.8 (marginal/unchanged, no longer an outlier at 3.0) | mostly resolution-limited |
| chevron(1,1.5) @120 | 4.5-5.8 | 4.5-5.6 (unchanged) | genuinely resolution-limited -- needs basis surgery |
| H_shape() @240 | 6.7-8.5 | 6.7-7.2, but **one mode (lambda=19.739) jumped to 13.1 digits** | mostly resolution-limited, one mode already well-resolved |
| GWW1() @240 | 7-8.5 | 7.4-8.8 (marginal) | mostly resolution-limited |
| GWW2() @240 | 6.5-7.3 | 6.5-7.3, but **one mode (lambda=12.337) jumped to 13.3 digits** | mostly resolution-limited, one mode already well-resolved |

The H_shape/GWW2 "one outlier mode already at 13 digits" pattern is
interesting -- worth a closer look (via `diagnose`) at what's different
about that specific eigenvalue's tension well vs its resolution-limited
neighbors, before starting basis surgery on the rest.

### stadium: n_reg diagnostic confirms the earlier (fs_d/fs_bdry_order-based) diagnosis

`diagnose()` on stadium(1,1) @ n_basis=120: `n_reg=77/124 (62.1%)`,
identical whether collocation is default (bdry_mult=2, int_npts=120) or 6x
denser (bdry_mult=6, int_npts=600) -- `sigma` barely moves (2.15e-04 ->
1.76e-04). Same intrinsic-truncation pattern as chevron/reg_ngon8/
cut_square, quantitatively confirming the original diagnosis from the
earlier session (curvature-discontinuity-driven near-redundancy in the
boundary-source basis, not a sampling problem). Not re-litigated further --
stands as previously documented in stadium.py.

### iso_tri: clean win across every height tried

| height | old (digits) | new (digits) |
|---|---|---|
| h=0.5 | 6.0-9.7 | 10.8-12.3 |
| h=1.0 | 7.4-12.9 | 13.0-13.5 |
| h=2.0 | 7.3-11.5 | 12.0-13.2 |
| h=4.0 | 9.0-10.7 | 11.8-12.9 |
| h=8.0 | 6.8-10.5 | 11.3-12.1 |
| h=16.0 | 7.8-10.1 (old value, kept) | not retried -- see below |

No basis surgery needed for h=0.5..8. `iso_tri.py` and `reference.py` need
updating with these values.

### reg_ngon N=8 and iso_tri h=16 @ n_basis=240: killed after ~10+ min, not hanging but very slow

Both were still burning CPU (not deadlocked) after 10+ minutes at their
next basis-size step (reg_ngon N=8 @240, iso_tri h=16 @120 for the *first*
time, unlike h=0.5..8 which all finished in ~1-2 min each). Consistent with
the known "sharp/thin corner needs high FB order -> slow Bessel evaluation
over the full lambda grid" pattern from chevron -- h=16 is the most
elongated triangle (sharpest apex) in the set, and reg_ngon(N=8)'s crowded
5-eigenvalue region likely needs a much richer basis than 120 to resolve
cleanly, which similarly balloons FB order. Killed rather than chased
further for now; revisit if time allows after the domains that clearly
need basis surgery (chevron, cut_square, H_shape, GWW) are done.

**reg_ngon(N=8) @ n_basis=120 status**: 9 of 10 eigenvalues at 8.8-10.3
digits (great); one outlier at lambda=29.5368 stuck at only 2.8 digits.
`diagnose()`: `n_reg=89/131 (67.9%)` at rtol=1e-14, but denser collocation
(bdry_mult=4, int_npts=300) doesn't change n_reg or sigma at all -- the
truncation is intrinsic to the basis (near-redundant functions at the
default FS placement/order), not a collocation-density problem. Tightening
rtol from 1e-14 to 1e-10 monotonically drops n_reg (72 to 89) but sigma
stays ~4e-4 regardless -- this specific mode's well is genuinely shallow at
n_basis=120, not a regularization artifact. Left as a known gap; the other
9 eigenvalues are reported.

### chevron: basis surgery attempts, mostly dead ends -- genuine slow convergence

chevron(1,1.5) corner structure: `int_angles = [11.3, 270, 11.3, 67.4]`
degrees -- **two** sharp corners (not just the reentrant one), both at the
same 11.3 degree angle. Default `fb_corner_fraction` gives them only 3.1%
of the FB budget each (weight ~ angle magnitude, so the 270-degree corner
dominates at 75%) -> FB orders `[3, 43, 3, 11]` at n_basis=120.

1. **Tried: reweight `f` toward the sharp corners** (`f=[0.35,0.2,0.35,0.1]`
   via `fb_corner_orders`/`fs_corner_orders`/`FundamentalBasis.by_corners`
   called directly, bypassing `make_default_basis`). Result: made things
   *slower*, not better -- giving raw order-count to an already-sharp
   corner produces enormous individual Bessel orders (order=21 terms at an
   11.3-degree corner means exponents up to m*pi/angle ~ 21*15.9 ~ 334!),
   expensive to evaluate and not filling in any "missing" resolution --
   the natural FB harmonics at a fixed sharp angle already start at high
   order (m=1 alone gives exponent ~15.9), so adding more just reaches for
   even higher, sparser harmonics rather than resolving anything at
   moderate order. Dead end; reverted.
2. **Tried: 3x denser collocation with default order allocation**
   (bdry_mult 2->6, int_npts 120->400). Result: no change (sigma
   2.86e-06 -> 3.25e-06, n_reg 97->98 out of 123). Collocation density is
   not the bottleneck here (unlike the hypothesis for other domains).
3. **Tried: moderate n_basis increase, 120->160** (default order
   allocation, ~33% more basis). Result: modest improvement, 4.5-5.6 ->
   5.2-6.3 digits. Real but slow convergence -- consistent with genuinely
   needing much more basis (400+) to reach 10 digits, which was already
   established to be too slow (n_basis=240 didn't finish in 15+ minutes in
   the first pass).

**Conclusion**: chevron's sharp corners are a genuine, not-quickly-fixable
resolution bottleneck given this basis family and the GSVD cost at the
n_basis needed to resolve them. `chevron.py` bumped to n_basis=160 for a
small honest improvement; still short of 10 digits. Not chased further.

### General finding: n_reg truncation to ~60-70% is intrinsic, not a collocation artifact

Checked across chevron (FB+FS mix), reg_ngon N=8 (FB+FS mix), and
cut_square (pure FB, no FS at all) -- in every case, `n_reg/n` sits around
60-70% at the default rtol=1e-14, and a 3-6x increase in collocation
density (`bdry_mult`, `int_npts`) changes `n_reg` by only a percentage
point or two, essentially not moving `sigma` at all. This means the
truncated ~30-40% of nominal basis functions are near-linearly-dependent
*in the basis itself* (before collocation even enters), not under-sampled.
The practical implication: `n_basis` overstates real resolving power by
a fair margin for these mixed/pure-FB bases -- to genuinely gain
resolution you need more raw basis functions (accepting the GSVD cost),
not denser sampling of the ones you already have.

### Moderate basis bumps for the remaining weak domains (given collocation/reweighting didn't help)

Since neither collocation density nor corner reweighting moved the
needle, tried the blunt lever (moderate n_basis increase, staying well
under the n_basis=480 cubic-cost wall) for everything still below target:

| Domain | old n_basis (digits) | new n_basis (digits) |
|---|---|---|
| cut_square r=0.25 | 240 (4.7-6.9) | 320 (6.4-9.0) |
| cut_square r=0.5 | 240 (6.6-9.8) | 320 (**9.1-13.1**) |
| H_shape | 240 (6.7-7.2, one at 13.1) | 320 (7.8-8.2, one at 13.3) |
| GWW1 | 240 (7.4-8.8) | 320 (mostly **9.5-9.9**; 2 outliers at 6.1, 3.9 digits) |
| GWW2 | 240 (6.5-7.3, one at 13.3) | 320 (7.2-8.7, one at 13.2) |
| chevron(1,1.5) | 120 (4.5-5.6) | 160 (5.2-6.3) |
| chevron(1,2) | 120 (~5.2-5.9) | 160 (6.1-7.1) |
| chevron(2,3) | 120 (3.0-3.8) | 160 (3.7-4.6) |
| chevron(2,4) | 120 (3.4-3.9) | 160 (4.2-5.0) |

cut_square r=0.5 essentially reaches the target at n_basis=320 (only a
modest bump from 240, much cheaper than the n_basis=480 wall). cut_square
r=0.25, H_shape, and GWW1 (mostly) get close but not quite there. GWW1 has
two problem eigenvalues near lambda=5.18 and lambda=12.34 -- the second one
(12.337) is suspiciously close to GWW2's "perfect" isolated mode at
12.337005501361730 (13.2 digits there), suggesting GWW1 may have a
genuinely close-but-distinct pair of eigenvalues there that GWW2's geometry
happens not to have -- not investigated further given time. chevron
improves steadily but slowly with n_basis across all 4 params; still well
short of 10 digits even at 160, consistent with genuinely needing much
larger n_basis (400+) which hits the known slow-Bessel-evaluation wall for
its ~11-degree corners.

### Final settled n_basis per domain (used in the scripts/reference.py)

ellipse=240, L_shape=240, mushroom=240, reg_ngon N=5,7=120 N=6=240 N=8=120,
iso_tri h=0.5..8=120 h=16=120(old value kept), cut_square r=0.25,0.5=320,
H_shape=320, GWW1,GWW2=320, chevron all params=160 (h1=1,h2=1.25 still
excluded), stadium=120 (unchanged, genuine floor).
