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
the boundary, or clustered at corners, those being the two methods tried. Read every result
below through that: "FB uniform beat FB-all-on-reentrant" is a BUDGET result (the singular corner
kept 21 terms instead of 128), not evidence against FB.
- [ ] `make_default_basis(domain, lam_max, precision)` -- the current signature asks the caller
      for `n_basis`, the one quantity they have no principled way to choose. `docs/basis_
      heuristics.md` is *a* theory with partial evidence (the FS offset scaling with wavelength
      is the empirically grounded part), not a spec. Approach is open-ended: gather evidence for
      what works, taking the current implementation's ideas as inspiration rather than law.
- [ ] `fb + boundary-offset FS` beats the current `mixed` branch on BOTH multi-singular-corner
      domains (chevron 3.1e-11 vs 2.4e-09; H_shape 5.3e-10 vs 1.2e-07) and has no branch at all.
      Its offset should taper near reentrant corners -- it currently drops 25% of its sources on
      H_shape for landing inside the domain, and wins anyway.
- [ ] `fs_corners` (lightning FS alone) is DEGENERATE on every corner domain tried: contrast
      0.8-1.7, i.e. no eigenvalue signal. It is half of the current multi-corner branch.
- [ ] H_shape's reference is only "at least 7.8 digits", which is above where the interesting
      comparisons happen. Needs a better reference before its rankings can be quoted.
- [ ] Size model must carry geometry: the same target precision costs n~48 on L_shape and
      >192 on chevron/H_shape. No geometry-independent `precision -> n` rule can work.
- [ ] Corner COUNT does not pick the construction: `mixed` beats `pure_fb` on chevron and loses
      to it on H_shape, both with four singular corners. Find what does predict it.
- [ ] Two tension curves stall (chevron `pure_fb` ~2e-08, H_shape `mixed` ~2.4e-07). Check
      `rtol=1e-12` pencil truncation before blaming the basis.
- [ ] Re-run the convergence study. The first one measured `solve_interval`'s `ltol=1e-8`
      bracket tolerance, not the basis, so its "~10 digit plateau" and every fitted rate are
      void (retracted in NOTEBOOK). `bench.py` now uses `manual_solve`+`polish_eigs` and gets
      14.6 digits at n=64 on L_shape where the old path read 10.7.
- [ ] `solve_interval`'s `ltol_default=1e-8` is a relative stopping tolerance on lam, so the
      search gives up after ~8 significant digits, and it is what
      `Eigenproblem.solve` uses -- so the documented headline path is the coarse one while the
      reference tables come from `benchmarks/reference/common.solve_domain_v2`. Either the
      polished pipeline belongs in `lappy`, or the default tolerance is too loose. Also affects
      anything that trusted `solve_interval` accuracy, which included this session's
      `certified_quadrature` measurements (probably benign -- verify_gram compares two
      quadratures at the SAME lam, so the error is common-mode -- but unverified).
- [ ] Find an exact-truth domain that actually exercises a corner singularity. Sectors are
      degenerate (FB about the apex spans the exact eigenfunctions) and polyominoes are
      degenerate the other way (closed-form eigenfunctions are smooth at reentrant corners).
      This is the real obstacle to measuring convergence rates honestly.
- [ ] "Singular corner" means two different things in this codebase: `pi/alpha` non-integer in
      `bases` (counts sharp CONVEX corners -- chevron(1,2) has four) versus the quadrature's
      reentrant/nonintegral notion (same domain, one). Pick distinct names.
- [ ] Decide the objective before optimizing to it: certified digits for `lambda` is what every
      measurement so far scores, but the stated goal is a Hadamard-ready solver and nobody has
      checked whether a basis tuned for `lambda` is also good for `dlambda`.

## MPSEigensolver
- [x] `from_domain()` method with good defaults
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

- [ ] Branch predicate is UNSOLVED; three candidates are dead. Corner count (what the
      constructor uses) separates nothing; sharpest convex corner is refuted (iso_tri h=0.5 and
      h=4 are equally sharp with opposite pure-FB verdicts); bbox aspect is refuted (h=0.5 is
      4:1 and fine, h=2 is 1:1 and fails). See NOTEBOOK retraction.
- [ ] Q3 ANSWERED: L_shape's pure-FB win survives a fair retest (corrected d/h=2 offset,
      budget framing) -- augmentation at m=32 gives 4.2e-10 against pure FB's 9.1e-16. But the
      benefit of augmentation is monotone in ELONGATION: ~1x at aspect 1, 16x at 2:1, 34x at
      3:1. So the design wants a continuous fs_frac driven by an elongation measure, not a
      branch. See NOTEBOOK.
- [ ] `detect_floor` computes the censor at `n_max//2`, so L_shape's censor came out at 2.84e-14
      while pure FB at n=128 reaches 9.1e-16 -- thirty times below its own "floor". Derive the
      censor from the reference's documented accuracy, or from the largest size in the ladder.
- [ ] Predicate hunt: FOUR candidates dead (corner count, sharpest convex corner, bbox aspect,
      FB budget concentration -- the last refuted by intervention, not just correlation). A
      fifth, "non-integer pi/alpha at a convex corner", already fails on iso_tri h=0.5. Consider
      stopping the hunt and shipping the fitted rule below.
- [ ] Elongated L-shapes (asym_L 5:1) cannot be certified by ANY basis tried -- two good bases
      disagree by more than their own tension. Either a harder domain than it looks, or a gap
      in the whole approach. Worth a look before it becomes a blind spot.
- [ ] `fb_corner_fraction`/`fs_corner_fraction` give regular corners weight ZERO, so on a
      domain with one singular corner the default stacks both families there and leaves every
      other corner bare. Measured cost on asym_L: the default is the worst of six constructions
      tried, 16x behind FB@singular + boundary FS. Allocation is the fix, not a new family.
- [ ] Singular corner needs ~a quarter of the budget, not all of it: m=0 fails categorically,
      m=16-64 of 128 is a broad optimum, m=128 (today's default) is ~10x worse than m=64.
- [ ] fs_frac=0.25 as a robust default: never worse than 1.8e-11 across the whole iso_tri
      family where pure FB spans eleven orders. Costs 100x on L_shape though.
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
