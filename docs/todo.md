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
- [x] **MEASURED: "a loop wanting lam_1..lam_20 will meet n_cap" -- and k turns out NOT to be the
      limiting variable.** `benchmarks/envelope/k_sweep.py`, 54 cells, ledger in
      `benchmarks/envelope/run/k_sweep.jsonl`, 30 min total. Window scaled as
      `weyl_est(k+4, dom)`, precision 1e-12, digits from `check_precision`.

      | domain | n | capped | k=4 | k=8 | k=12 | k=16 | k=20 | k=24 | drop |
      |---|---|---|---|---|---|---|---|---|---|
      | square | 100 | no | 11.3 | 11.0 | 11.3 | 11.5 | 11.0 | 11.1 | -0.2 |
      | rect_thin | 204 | no | 13.2 | 12.7 | 12.0 | 12.4 | 11.3 | 11.7 | -1.5 |
      | reg_ngon_8 | 232 | no | 11.3 | 11.3 | 11.3 | 10.9 | 10.9 | 10.9 | -0.3 |
      | L_shape | 182 | no | 12.1 | 11.5 | 12.1 | 11.6 | 11.2 | 11.4 | -0.8 |
      | GWW1 | 239 | YES | 9.2 | 9.2 | 9.0 | 8.9 | 8.9 | 8.9 | -0.3 |
      | H_shape | 408 | YES | 9.4 | 9.4 | 9.4 | 9.4 | 9.4 | 9.4 | -0.0 |
      | chevron_1_15 | 238 | YES | 5.9 | 5.9 | 5.3 | 5.3 | 5.3 | 5.2 | -0.6 |
      | iso_tri_h16 | 238 | YES | 6.9 | 5.7 | 4.9 | 4.9 | 4.9 | 4.9 | -1.9 |

      Going from k=4 to k=24 costs 0.2 to 1.9 digits, and on the four uncapped domains accuracy
      is essentially FLAT in k at 11-13 digits. **What binds is `n_cap`, and it binds at every k
      including k=4** -- GWW1 sits at 9 digits whichever eigenvalue you ask for. So the ceiling
      is a property of the GEOMETRY, not of the eigenvalue index, and "lam_20 and beyond
      untested" in the contract was caution rather than a real limit. Cost grows roughly linearly
      in k (square 2.7 s at k=4 -> 20 s at k=24; reg_ngon_8 11.7 -> 101 s).

      `cut_square_r025` is excluded: mixed arc/polygon, so `default_basis_for` refuses. Expected.

- [ ] **`n_cap = 240` is now the measured ceiling on the hard domains and should be revisited.**
      Four of eight suite polygons cap, and all four are the ones stuck below 10 digits. The cap
      still rests on the two-domain rank-saturation measurement S0a called possibly coincidental.
      Raising it is the single change that would move GWW1/H_shape/chevron/iso_tri.

- [x] **`_apply_cap` really does return `n_total > n_cap`, confirmed in the wild:** H_shape
      reports `n_total = 408` against a cap of 240, at every k. Previously known from reading the
      code; now it has a measurement. "Cap" is the wrong name for whatever this is.
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
- [x] `solve(k)` returning k accurate eigenvalues that are not the FIRST k -- FIXED by raising
      the `ppl` default from 5 to 20 (9 of 90 (domain, k) cells wrong -> 2 -> 0). See
      `tests/test_mode_completeness.py` and `Eigenproblem.solve`'s docstring.
- [x] No inner-loop entry point -- ADDED as `Eigenproblem.track`, ~2.5x faster than `solve(1)`
      and immune to the set-selection problem above, since it follows a mode by value.
- [x] **`track` does not scale to a SET, and `Eigenproblem.track_set` is what does.** Following
      `lam_1..lam_K` with K separate `track` calls succeeded ONCE IN 46 STEPS on a rectangle
      family walked down to a relative gap of 6.4e-09 (19 refusals, 25 collapses), each failure
      costing the full `solve` that tracking exists to avoid. And the successes are not all
      right: where the true pair is split by 4.3e-09, the per-value path returned a value wrong
      by 1.1e-08 relative, past every guard -- the tension is small there, and the coincidence
      check only looks within rtol=1e-9. `track_set` does ONE windowed `solve_interval`, so no
      two seeds can converge onto one mode, and reads multiplicity from the tension spectrum, so
      a pair too tight for the grid returns as one eigenvalue of multiplicity 2. Same walk:
      46/46, no refusals, worst error 2.0e-14. In a douse N=4 optimization run it took full
      solves from 43 to 2 and the endpoint error from 5.4e-09 to 7.3e-10.
- [ ] **`solve(k)`'s completeness has no validated detector behind it, only grid resolution.**
      `ppl=20` measured 0 of 90 but a tighter cluster than H_shape's could still defeat it, and
      the audit already in `_solve_dir_neu` provably cannot serve: per-gap Weyl expected counts
      OVERLAP between correct and incorrect results (correct cells reach 2.87 expected modes in
      one gap, incorrect ones span 2.27-2.67), because multiplicity confounds the two-term count
      at these wavenumbers. A detector has to work in the tension's own currency, not Weyl's.
      Low urgency while `track` is the loop's entry point, since tracking has no index to shift.
- [ ] `ppl=20` costs +43% on the initial grid against ppl=10 (12.9 s/cell against 9.0 on the
      90-cell sweep). Scaling `ppl` with the local spectral density, rather than flatly, would
      buy most of it back -- the clusters that need it are local. (Threading has more than repaid
      this in absolute terms, but the waste is still there.)

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

## Speed
Profile first: basis evaluation is ~67% of a solve on planner-built bases and the GSVD stack
~33%. S0a's 52/48 split was measured on `pure_fb` and does not carry over. On a douse-shaped
solve (N=6, n_basis=186, 2026-08-26) the split is more lopsided still: `_bessel` alone is 85.5%.
- [x] Order-0 Bessel: `yv(0,.)`/`yvp(0,.)` -> `y0`/`-y1`, 42x/94x on that call.
- [x] Thread the lambda grid (`mps.n_workers_default`), ~2-3x end to end.
- [ ] **The `jv` ladder in `FourierBesselBasis._bessel` is the largest cost by far, and LOW
      PRIORITY anyway.** 85.5% of a douse-shaped N=6 solve at n_basis=186 (7.31 s of 8.55 s, 400
      calls); 14.5 s of a 36.9 s H_shape solve on the older measurement.
      **Deliberately not being worked on: accurate basis evaluation outranks fast basis
      evaluation, and a hand-written Bessel routine is new code in the one place where a silent
      error is indistinguishable from a converged answer.** Everything below is why it is also
      harder than it looked, recorded so nobody re-derives it.
      - Miller backward recurrence amortises one downward sweep across a chain of orders at a
        FIXED fractional offset, stepping by 1. Our ladder (`bases.py:588-590`) is `{k*alpha}`
        with `alpha = pi/phi`, so it lies on a step-1 lattice only when `alpha` is rational with
        a small denominator -- then `{k*alpha mod 1}` takes `q` values and you need `q` sweeps.
      - Measured 2026-08-26. Regular hexagon: `alpha = 1.5` exactly, K=27, **2** distinct
        `frac(k*alpha)` -- two sweeps cover the whole ladder. A certified douse N=6 iterate:
        `alpha = 1.50003581`, K=27, **27** distinct offsets -- one sweep per order, which is what
        `jv` already does. A 3.6e-5 relative perturbation in `alpha` destroys the lattice.
      - So it applies to `benchmarks/reference` and `benchmarks/suite` (square `alpha=2`, regular
        N-gons `alpha=N/(N-2)`, equilateral/right triangles, rational-angle chevrons) and NOT to
        generic optimiser iterates, which is the entire `douse` workload. See
        `LAPPY_WISHLIST.md` section 8.
      - Symmetry reduction (section 7 there) would pin the angles by construction and hence make
        `alpha` rational again -- so it would restore this lattice as a side effect. That is a
        second argument for it beyond halving `n_p`.
- [ ] **If the generic case is ever worth attacking, these are the two candidates** -- both
      unmeasured, both the same accuracy-risk objection as above, so neither is scheduled.
      - Uniform asymptotics (Olver/Debye): valid for arbitrary real order, no lattice needed, and
        the regime here is `order >~ argument` (orders reach `alpha*K ~ 40` at moderate
        `sqrt(lam)*r`), which is where the expansion is accurate and where `jv` is slowest.
        Current cost is ~293 ns per element.
      - Amortise across `lam`, not across order: the orders are fixed by geometry for a solver's
        life while `lam` moves on every one of ~100 probes per eigenvalue, so any order-dependent
        coefficient could be built once per shape and reused. Needs no rationality, and fits the
        per-instance cache.
- [ ] `solve_interval` uses only `tensions(lam)[:2]` but `gsvdvals` computes every generalized
      singular value. Not obviously recoverable (the GSVD does not truncate the way a symmetric
      eigenproblem does, and `regularize_pencil`'s SVD is needed to form the projection), but it
      is where the other third of the time goes.
- [ ] `track` scans 9 points per iterate; 5 would be ~40% fewer sigmas in the inner loop. Trades
      against the edge guard's reliability, so measure on the L-shape family before adopting.
- [ ] Threading gains flatten between 4 and 8 workers, which smells like the per-sigma LAPACK
      calls competing with a threaded BLAS. Worth checking whether pinning BLAS to 1 thread
      inside the workers beats the current arrangement.
- [ ] `NormalizedBasis._raw_eval` runs at a ~48% hit rate on the douse-shaped loop (578 hits,
      624 misses, `maxsize=8`). `bases.py:280` sizes it for four lambdas of a two-component
      build; whether the gradient path wants more is unmeasured. `lappy.cache.cache_stats()`
      reports the rate now, so this is a query rather than an experiment.
- [ ] `cache.DEFAULT_MAX_BYTES` is 512 MiB per PROCESS, chosen against a measured working set of
      ~28 MB per solver at n_basis=186 -- generous on purpose, and never yet exercised on a real
      N=12/16 grid. Confirm against one before treating it as tuned; remember N workers get N
      budgets.

## Reference values
- [x] `REFERENCE['reg_ngon_6']` and `['reg_ngon_8']` were each short a mode -- CORRECTED from a
      full-domain certified solve. Every polygon table is now cross-checked against one by
      `tests/test_reference_tables.py`, which is the instrument that never existed (both defects
      were found by accident, from outside the codebase).
- [ ] **The per-sector multiplicity estimate in `symsolve.solve_sym` is the actual bug, and it is
      still there.** The registered group is the largest elementary abelian 2-subgroup with real
      characters, which can be a PROPER SUBGROUP of the true symmetry; then degeneracies the full
      group would split survive inside one sector and must be recovered by that sector's own
      multiplicity estimate. It under-counted. The two suite entries that reduce by a proper
      subgroup (reg_ngon(6) D6->D2, reg_ngon(8) D8->D2) are exactly the two that were wrong;
      every |G|=2 entry and `rect D2` (D2 IS the rectangle's full group) are correct. Until this
      is fixed, `benchmarks.suite.emit --write` would reintroduce both defects from the stale
      ledger, which is why the two entries are patched in place rather than regenerated. Curved
      domains are NOT covered by the new cross-check (the full-domain instrument needs a
      hand-built basis there): `disk` is solved with D2 against its true O(2), so it is the
      obvious next suspect.
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
