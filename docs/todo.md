# Overview
This file is a list of to-do items or wishlist items for `lappy`.

Convention: one line per open item, with a pointer to wherever the detail already lives (a
NOTEBOOK entry, a docstring, a commit). Detail does NOT get duplicated here -- this file is for
knowing what is open, not for explaining it. An item leaves when it is done or abandoned; the
finding it produced stays in `benchmarks/suite/run/NOTEBOOK.md`.

## Bases

**Design constraint (Hayden, 2026-08-12):** a SINGULAR corner always needs corner Fourier-Bessel
terms; any basis lacking FB at a singular corner should be expected to fail. So the open question
is never FB-versus-FS, it is how to AUGMENT a corner-FB basis -- with fundamental solutions along
the boundary, or clustered at corners, those being the two methods tried.

**Mostly closed by `lappy/basis_plan.py`.** The predicate hunt that used to fill this section is
over, and not because a predicate was found: `plan_basis` has no branch to predicate. It derives
corner Fourier-Bessel budgets and boundary-offset source arcs from the geometry, `lam_max` and a
target, and `refine_plan`/`check_precision` measure the result rather than predicting it -- which
is the "self-certifying constructor" the sixth failed-predicate entry had already concluded the
evidence favoured. Everything removed from this section is settled in
`benchmarks/basis_lab/PLAN_LAB.md` (the redesign, S0-S5) or `HEURISTICS.md` (1154 measurements of
the recipe it replaced). What is still open on the planner lives in PLAN_LAB.md's own "Open, in
priority order" list; it is not duplicated here.

- [ ] H_shape's reference is only "at least 7.8 digits", which is above where the interesting
      comparisons happen. Needs a better reference before its rankings can be quoted. (The
      planner now certifies 9.5 there, so this bites sooner than it used to.)
- [ ] Find an exact-truth domain that actually exercises a corner singularity. Sectors are
      degenerate (FB about the apex spans the exact eigenfunctions) and polyominoes are
      degenerate the other way (closed-form eigenfunctions are smooth at reentrant corners).
      This is the real obstacle to measuring convergence rates honestly.
- [ ] "Singular corner" means two different things in this codebase: `pi/alpha` non-integer in
      `bases` (counts sharp CONVEX corners -- chevron(1,2) has four) versus the quadrature's
      reentrant/nonintegral notion (same domain, one). Pick distinct names.
- [ ] `PlanConfig.n_cap = 240` is a hard ceiling adopted from S0a's rank saturation, which was
      measured on TWO domains and flagged there as possibly coincidental. Probed since: `n_reg`
      does grow with the wavenumber, but far slower than kappa (L_shape 123 -> 159 as kappa goes
      5.6 -> 26.4; square 84 -> 119 as kappa goes 9.4 -> 45.5). Everything validated is at a
      6-eigenvalue window. A shape-optimization loop wanting lam_1..lam_20 will meet this cap,
      and nothing has measured what happens there.
- [ ] `PlanConfig.rtol` duplicates a value the solver owns; the planner should read it from the
      solver so the two cannot drift. Getting that constant wrong is what made achieved accuracy
      non-monotone in the target (PLAN_LAB S2).
- [ ] Arc-local geometry is sampled at the arc MIDPOINT only. Not the current limiter on the
      suite (thickness varies by under 2x along the arcs measured on iso_tri_h4), but it would
      be on a mushroom or a stadium. Subdividing arcs by thickness variation is the next step.
- [ ] `_apply_cap` can return `n_total > n_cap` when the FB budget alone exceeds it (spiral).
      Reported, but "cap" is then a misleading name.

## MPSEigensolver
- [x] `from_domain()` method with good defaults
- [x] Branch predicate hunt (six dead candidates) -- ABANDONED, and correctly: `basis_plan` has
      no branch. See PLAN_LAB.md.
- [ ] **`solve(k)` can return k accurate eigenvalues that are not the FIRST k.** Measured 9 of 90
      (domain, k) cells over 10 suite polygons: `right_trapezoid` k=10 drops lam_3 = 44.9484877814
      and is correct at k=9 and k=11; `reg_ngon_6` is wrong from index 5 at k=5..10; `eq_tri` at
      k=5. Two causes: `_solve_dir_neu`'s rescue loop is gated on `len(eigs) < k_search` so it
      never runs when the count is right and the set is wrong, and `_find_deficient_gaps`'
      `thresh=1` on the cumulative Weyl deviation cannot see one missing mode (that deviation
      already ranges 0.12-1.5 on right_trapezoid's CORRECT first ten). Fatal for an optimizer,
      which would follow a wrong gradient across a mode swap.
- [ ] No inner-loop entry point: `solve(k)` runs a global Weyl-gridded scan every call, at 2-8 s
      per 4-eigenvalue solve (22 s on H_shape) with solver construction ~10 ms. A shape loop knows
      where lambda was last iterate and wants a local bracket.
- [ ] Test coverage for the weighted-evaluation path (`weights=True` -> `PointSet.sqrt_wts` ->
      `bases` Vandermonde scaling). No current caller, `weights=False` everywhere by default,
      and every consumer sits behind a `hasattr(pts, 'wts')` guard -- which is exactly how
      `kind='mesh'` stayed broken. See `make_default_int_pts` docstring.
- [ ] `kind='mesh'` on a curved domain takes its point count from mesh resolution, not basis
      size: 1.9e4 nodes on a unit disk against ~50 from a random draw. Works, deterministic,
      probably not what an MPS solve wants.
- [ ] `from_domain`'s `prec=` parameter IS `ltol` (passed straight through to the constructor's
      ninth positional). Two names for the lambda-axis tolerance, neither of them obviously that.
      Rename or alias.
- [ ] `detect_floor` computes the censor at `n_max//2`, so L_shape's censor came out at 2.84e-14
      while pure FB at n=128 reaches 9.1e-16 -- thirty times below its own "floor". Derive the
      censor from the reference's documented accuracy, or from the largest size in the ladder.
- [ ] Elongated L-shapes (asym_L 5:1) cannot be certified by ANY basis tried -- two good bases
      disagree by more than their own tension. Either a harder domain than it looks, or a gap
      in the whole approach. Worth a look before it becomes a blind spot.
- [ ] Deliverable tables must be "columns to reach precision p", never "sigma at fixed n" --
      the square comparison inverts across the saturation crossing.

## Reference values
- [ ] **DECIDE: `ellipse_eigs(a=2)[0]` is wrong at the 9th digit.** Table 3.566726599853406;
      three independent solves (n_basis 160/240/320, tension ~4e-15) agree with each other to
      7.5e-16 on 3.566726602928861, a relative difference of 8.62e-10. lambda_2..4 are fine.
      Flagged in the docstring, value NOT changed -- replacing a reference feeds certified
      results elsewhere, so it needs a decision. See NOTEBOOK.
- [ ] Re-verify the other unverified reference entries with the corrected pipeline before using
      them for knob conclusions: ellipse a=3/a=4 (~6.7 digits), chevron pairs other than (1,2),
      cut_square r=0.25, H_shape, iso_tri h=16/20. The ellipse a=2 case shows a documented
      accuracy claim can be four orders optimistic.

## Quadrature / eigenfunction integrals
- [ ] The near-slit ceiling has no lever. `cornerinterp` tops out near 1e-10 for `nu <~ 0.6` on
      a dense corner family, and more order makes it worse past the optimum. Candidates: panel
      subdivision, or a joint node/weight solve (Ma-Rokhlin). See NOTEBOOK, Leg 3b entry.
- [ ] `ellipse_a4` spends 512 nodes at `smooth_safety=1`, driven by
      `geometry_order_for_precision` -- segment 0 resolves only to 1.3e-07 at order 512. It
      certifies, but the parametrization is doing the spending, not the eigenfunction.
- [ ] The corner order model disagrees with truth in both directions (model error falls
      monotonically to 1e-16; true error saturates then degrades). Not a cap -- that was tried
      and retracted, see NOTEBOOK. What replaces it is open.

## Geometry
- [ ] function which computes the maximum distance from a corner of the boundary to another
      point on the boundary
