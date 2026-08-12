# Overview
This file is a list of to-do items or wishlist items for `lappy`.

Convention: one line per open item, with a pointer to wherever the detail already lives (a
NOTEBOOK entry, a docstring, a commit). Detail does NOT get duplicated here -- this file is for
knowing what is open, not for explaining it. An item leaves when it is done or abandoned; the
finding it produced stays in `benchmarks/suite/run/NOTEBOOK.md`.

## Bases
- [ ] `make_default_basis(domain, lam_max, precision)` -- the current signature asks the caller
      for `n_basis`, the one quantity they have no principled way to choose. `docs/basis_
      heuristics.md` is *a* theory with partial evidence (the FS offset scaling with wavelength
      is the empirically grounded part), not a spec. Approach is open-ended: gather evidence for
      what works, taking the current implementation's ideas as inspiration rather than law.
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
