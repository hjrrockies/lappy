# S0: the two measurements that gate the basis redesign

Harness: `benchmarks/basis_lab/plan_lab.py`. Rows: `run/plan/{s0a,s0b,s0c}.jsonl`.
Re-print with `.venv/bin/python -m benchmarks.basis_lab.plan_lab report`.

The redesign plan makes two claims it had no right to make. S0 tests both before any planner code
is written.

---

## S0a. Where does per-λ time actually go?

`pure_fb` at a ladder of sizes, one fresh λ per timing (every relevant cache memoizes on λ, so a
repeated λ times a dict lookup). Milliseconds:

| domain | n | eval pair | GSVD stack | eval share | σ total | double-eval factor |
|---|---|---|---|---|---|---|
| L_shape | 40 | 1.50 | 1.37 | 52% | 2.84 | 2.25 |
| L_shape | 90 | 5.06 | 6.90 | 42% | 12.02 | 2.20 |
| L_shape | 190 | 18.18 | 37.82 | 32% | 56.39 | 2.07 |
| L_shape | 360 | 54.26 | 94.61 | 36% | 134.98 | 2.04 |
| L_shape | 500 | 90.26 | 85.42 | 51% | 174.60 | 2.05 |
| square | 40 | 3.02 | 1.31 | 70% | 4.33 | 2.22 |
| square | 130 | 15.08 | 10.70 | 58% | 26.09 | 2.16 |
| square | 260 | 43.03 | 39.84 | 52% | 84.07 | 2.09 |
| square | 500 | 121.96 | 110.85 | 52% | 252.57 | 2.08 |

**Finding A1 — the plan's premise was wrong.** It said per-λ cost "appears dominated by
transcendental basis evaluation". It is not: the evaluation share runs 32-70%, median 52%, and on
L_shape the GSVD stack *dominates* through the mid-range. Both terms matter at every size
measured. The accounting closes (σ ≈ eval pair + GSVD stack), so this is a real split, not a
mismeasurement.

*Consequence for the design:* cutting **columns** is the dominant lever, because it hits both
terms and the factorization superlinearly. Cutting **rows** (`m_B ≈ 2n`, `m_I = n`) only helps
evaluation and the QR, so the "reduce collocation rows" idea in the plan is worth roughly half
what I assumed.

**Finding A2 — the double evaluation is real and costs a factor 2.04-2.27 (median 2.10).**
`NormalizedBasis.__call__` calls `norms(lam)`, which evaluates every component point set, and then
evaluates the same points again in the `wts`-falsy branch (`bases.py:323`, `:332`) — the default,
since `bdry_pts`/`int_pts` carry no weights. Removing it takes ~25% off a σ.

**Finding A3 — there is a hard ceiling on useful columns, and the old recipe sailed past it.**
`n_reg` (the rank the SVD truncation keeps) saturates at **235-236 on both domains**:

| domain | n | 40 | 90 | 190 | 260 | 360 | 500 |
|---|---|---|---|---|---|---|---|
| L_shape | n_reg | 40 | 90 | 185 | 216 | 236 | 235 |
| square | n_reg | 39 | 75 | 129 | 161 | 196 | 235 |

Beyond saturation a column costs a full transcendental evaluation and a wider QR and then gets
truncated out of the pencil. `HEURISTICS.md` recorded the old recipe asking for 316 columns on
L_shape and 1929 on chevron_2_4; from n≈360 on L_shape, *none of the extra columns can reach the
answer*. That the two domains saturate at nearly the same number is unexplained and might be
coincidence — the actionable part is per-domain and does not depend on the mechanism.

---

## S0b / S0c. Does the eigenvalue objective agree with the dλ objective?

`docs/scope_and_downstream.md` §7 asks this and records that nobody had checked; every basis study
in this project, including `HEURISTICS.md`'s 1154 rows, scored only the eigenvalue side. Both
objectives are measured **at a known λ**, so no search can confound either:

* eigenvalue objective: Moler--Payne certified digits for `u` at the exact λ;
* dλ objective: relative error of `-∫(∂u/∂n)²(V·n)ds` against truth.

**S0b, rectangles, closed-form dλ** (`dλ/dL = -2π²m²/L³`; dilation is excluded as a check because
the Rellich identity is what `gram` normalizes with, so it is near-tautological). Five
constructions × n ∈ {15…190} × 3 seeds. The harness reproduces
`tests/test_shape_derivative.py`'s case to ~1e-15 where that test asserts 1e-12, which is its
correctness check.

* For n ≥ 60 **every construction saturates at 14-15 digits on both objectives**, so the
  rectangle cannot rank them there. All the information is at n ≤ 40.
* Over 17 unsaturated cells and 60 comparable pairs: **57/60 concordant (95%)**. All 3
  discordances are between constructions sitting at 0.7-3.9 digits — bases nobody would ship.
* dλ digits run about 1 digit *above* certified digits.

**S0c, L_shape, finite-difference reference.** No closed-form dλ exists at a singular corner, and
this is where disagreement was most plausible: `∂u/∂n` blows up like `r^(α-1)`, and a corner
weight of order `r¹` is exactly what cost this project six orders until `weight_family='integer'`
was added. The family translates the `x=-1` edge (moving two *regular* corners, leaving the
reentrant corner fixed, so this measures the basis rather than the Hadamard formula's domain of
validity). Reference: five-point central difference by continuation,
**dλ/dt = -9.0949830984**, stable to ~1.5e-10 relative under step halving.

| family | n | MP digits | dλ digits |
|---|---|---|---|
| pure_fb / default | 30 | 6.7 | 8.9 |
| mixed | 30 | 3.5 | 6.7 |
| fb_plus_bdry_fs | 30 | 3.9 | 5.6 |
| mixed | 50 | 5.0 | 7.7 |
| fb_plus_bdry_fs | 50 | 6.2 | 8.8 |
| mixed | 80 | 6.5 | 9.6 |
| pure_fb / default | 80 | 13.8 | 11.0 (floor) |
| heuristic | 271 | 13.8 | 11.0 (floor) |

dλ digits cap at 11.0, the reference's own resolution. **No cell has MP ≥ 10 with dλ below the
floor** — the failure mode this stage exists to detect (λ accurate, derivative not) does not
occur. Over the unsaturated cells, MP − dλ has median **−2.6 digits**: the certified bound is the
conservative one.

### Answer to the gate

**The objectives agree.** Tune the redesign to certified eigenvalue digits; dλ accuracy follows,
with 1-3 digits of margin. Keep a dλ regression check permanently, both because
`scope_and_downstream.md` §4 requires it and because it is the only instrument sensitive to a
*systematic* error in `‖u‖`, which certification provably cannot see.

Two riders:

* **dλ is more sensitive to the interior draw than λ is.** Seed spreads on the rectangle: MP ≤ 0.5
  digits, dλ up to 1.4. So the determinism fix is worth more to the derivative objective than to
  the eigenvalue one — it is not merely hygiene.
* **Agreement above ~11 digits at a singular corner is untested**, because that is the FD
  reference's floor. Going deeper needs an analytic dλ at a singular corner, which exists only on
  the sector — curved, and already covered at 1e-9 by `tests/test_shape_derivative.py`. Corner-
  *moving* velocities are deliberately out of scope here.

---

## S1. The speed floor (`lappy/bases.py`, `lappy/mps.py`)

Three changes, none of them touching basis *design*:

1. **`NormalizedBasis._raw_eval`**, one cached evaluation per `(lam, pts)`, with `norms`,
   `_eval_pointset` and `__call__` all routed through it. A cache already existed on
   `_weighted_eval`, but the second evaluation never consulted it: `norms(lam)` went through the
   cached path while `__call__`'s default branch called `self.basis._eval_pointset` **directly**,
   bypassing it. Same λ, same points, different code path.
2. The same treatment for the gradient path (`_raw_grad_eval`). A smaller win — `norms` evaluates
   values, not gradients, so this was never a *double* evaluation — but it fixes a comment that
   claimed a cache hit on `_weighted_grad_eval`, which carried no cache at all.
3. **Interior collocation points are reproducible by default** (`make_default_int_pts`,
   `DEFAULT_INT_SEED`). `rng=None` meant the global RNG; pass `np.random.default_rng()` for an
   independent draw.

Measured:

| | before | after |
|---|---|---|
| evaluations reaching the wrapped basis per fresh-λ `A_B`+`A_I` | 4 | **2** (exact count) |
| per-σ time, L_shape n=40 → 500 | 2.84 → 174.60 ms | 2.11 → 129.81 ms |
| per-σ speedup | — | **1.10-1.55x, median 1.36x** |
| full test suite | 186.4 s | **136.9 s** (1.36x) |
| three unseeded builds of the same solver | three different answers | bit-identical |

The suite-wide 1.36x matching the per-σ 1.36x is the corroboration: this is a real cost removal,
not a benchmark artifact. Seeded numerics are **bitwise identical** to pre-S1 on L_shape, square
and eq_tri across five λ each, and `pytest tests/` is 1045 passed / 1 skipped as before.

Note the counter, not the clock, is what pins claim 1. The obvious timing proxy
(`t_pair / (t_eval_bdry + t_eval_int)`) read 2.10 before the fix and 50-210 after — because once
the evaluation is cached the denominator is measuring a cache hit. A count cannot drift like that.

---

## S2. The planner (`lappy/basis_plan.py`)

`plan_basis` / `realize` / `refine_plan`, per the architecture in the plan: ownership as a
partition, one continuous corner rule, per-arc budgets from local geometry, and sizing by measured
refinement. 32 unit tests in `tests/test_basis_plan.py`; `pytest tests/` is 1077 passed / 1 skipped.

**The headline: `target` now does something.** This was the old recipe's central failure — flat to
within 1.5 digits across four decades of requested precision on 12 of 18 domains. Certified digits
against requested target:

| domain | 1e-4 | 1e-7 | 1e-10 |
|---|---|---|---|
| L_shape | 7.3 @ n=88 | 9.4 @ 123 | 12.8 @ 159 |
| right_trapezoid | 7.5 @ 51 | 10.8 @ 73 | 13.8 @ 94 |
| reg_ngon_6 | 8.2 @ 84 | 9.3 @ 114 | 10.3 @ 144 |
| square | 14.6 @ 52 | 14.3 @ 68 | 14.5 @ 88 |

Monotone and responsive where there is room, saturating where there is not, and still pessimistic
by ~2-4 digits — but now by a roughly *constant* margin instead of one collapsing from +10 to 0,
which is the difference between a knob that can be calibrated and one that cannot.

**Against the old recipe, certified digits to certified digits.** New at `target=1e-13`, old at
`precision=1e-8` (its h3 rows):

| domain | new n / MP | old n / MP | size |
|---|---|---|---|
| square | 108 / 14.4 | 106 / 13.1 | 1.02x |
| eq_tri | 78 / 14.5 | 102 / 10.4 | 0.76x |
| L_shape | 199 / 14.0 | 240 / 13.2 | 0.83x |
| right_trapezoid | 114 / 14.0 | 146 / 12.7 | 0.78x |
| reg_ngon_6 | 396 / 13.2 | 318 / 11.2 | 1.25x |
| **iso_tri_h4** | **124 / 9.6** | **459 / 3.6** | **0.27x** |
| iso_tri_h16 | 189 / 4.2 | 480 / 5.9 | 0.39x |

Better certified digits on six of seven, usually smaller. Planned sizes across all 29 polygons
fall by factors of 3-7 on the domains the old recipe blew up on (`chevron_2_3` 212 against 1303,
`iso_tri_h05` 85 against 613, `H_shape` 366 against 1045).

### Two conditioning ceilings, both found by measurement

Neither was in the plan; both are now load-bearing, and both were found because refinement made
things *worse*.

**Sources.** A layer standing off `delta` with spacing `h` delivers about `exp(-2 pi delta / h)`,
which is useless past `ln(1/eps) ~ 36` — beyond that adjacent columns are equal to working
precision, and near-duplicate columns destabilize the rank truncation in `regularize_pencil`. A
refinement loop that grew `n_src` at fixed `delta` on `chevron_1_2`:

    n = 120   worst arc residual 7.3e-07     (just short of target 1e-7)
    n = 172                      6.2e-04
    n = 206                      8.3e-01     certified digits 6.1 -> 1.2

`_src_ceiling` now caps it, and an arc at its ceiling is given a larger offset (which raises the
ceiling) rather than more sources.

**Fourier-Bessel terms.** Term `j` behaves like `r^(j*alpha)`; at a sharp corner `alpha = 9.76`
means the fourth term is `r^39`, numerically zero everywhere but the outer edge of its arc.
`_fb_ceiling` caps `M*alpha`. Its one constant, `fb_inner_frac`, is **fitted, not derived** — swept
against certified digits:

    fb_inner_frac   0.2    0.35   0.5    0.7
    iso_tri_h4      3.9    5.8    7.1    7.1
    iso_tri_h16     3.8    5.4    5.8    7.2

0.2 was tight enough to block the refinement loop entirely. 0.7 is the default.

**And a guard above both.** `refine_plan` returns the best plan it *measured*, never the last one,
so growth can only improve on where it started. With the ceilings in place the loop no longer needs
that guard on the domains tested, which is exactly why it should stay.

### The refinement diagnostic had to be in the objective's own currency

`_residual_by_arc` reports each block's contribution to Moler--Payne's `eps` —
`sqrt(area) * sup|u|` per arc with `orthonorm=True` coefficients, so the denominator is 1 by
construction. It agrees with a full certification to **0.1 digits** on iso_tri_h4, iso_tri_h16,
chevron_1_2, L_shape and square, at roughly 1% of the cost, and it says *which* arc is short.

The first version normalized by the interior-collocation norm and sampled 48 points per arc
uniformly. It read 4e-8 on `chevron_1_2` where the certified bound was 6.1 digits — two digits
optimistic, and missing the residual peak, which sits within a few percent of a corner. That would
have let refinement stop while believing it had converged. Grading toward the corners (as
`certify.boundary_sup` already does) and normalizing the way the bound does fixed both.

### The ceiling constant was wrong, and fixing it fixed the non-monotonicity

The first version of both ceilings used `ln(1/eps_machine) = 36.7`. That is the wrong threshold:
columns stop being distinguishable **to the solver** at its own rank-truncation level, so the
constant is `ln(1/rtol) = 27.6`. Diagnosis on `iso_tri_h16` at `target=1e-13`: corner residuals
improved to 1.0e-06 while *arc* residuals got 14x worse (6.5e-05, from 4.6e-06 at 1e-10) with
`n_reg == n` throughout — the pencil was not truncating, so the extra sources were actively harmful
rather than merely wasted.

Swept over six domains at targets 1e-7 / 1e-10 / 1e-13:

| ceiling constant | 36.7 (eps) | **27.6 (rtol)** | 23.0 | 18.0 |
|---|---|---|---|---|
| iso_tri_h16 | 4.2 / 5.2 / **4.2** | 4.2 / 5.2 / 5.7 | … 5.5 | … 5.3 |
| chevron_1_2 | 6.1 / 6.4 / 6.3 | 6.1 / 6.4 / **8.0** | … 8.8 | … 7.5 |
| L_shape | 9.4 / 12.9 / 14.0 | 9.4 / 12.9 / 13.9 | … 12.1 | … 12.5 |
| square | 14.3 / 14.4 / 14.4 | … 14.2 | … 14.5 | **13.6** (non-mono) |

`ln(1/rtol)` is the only setting monotone on all six without giving up accuracy anywhere. It is
now `_indep_digits(cfg)`, used by both ceilings, with `fb_inner_frac` re-fitted from 0.7 to 0.77 to
hold the FB cap where it had been calibrated (only the ratio matters).

Certified digits after the fix, unrefined plans, `n / MP`:

| domain | 1e-7 | 1e-10 | 1e-13 |
|---|---|---|---|
| square | 68 / 14.3 | 88 / 14.4 | 104 / 14.1 |
| L_shape | 123 / 9.4 | 159 / 12.8 | 191 / 13.9 |
| right_trapezoid | 73 / 10.8 | 94 / 13.8 | 109 / 14.1 |
| reg_ngon_6 | 114 / 9.3 | 144 / 10.3 | 174 / 11.1 |
| iso_tri_h05 | 64 / 8.4 | 85 / 11.0 | 97 / 11.9 |
| chevron_1_2 | 120 / 6.1 | 163 / 6.4 | 195 / 8.0 |
| GWW1 | 193 / 8.8 | 238 / 9.2 | 303 / 10.6 |

**13 of 15 domains monotone.** The two exceptions: `eq_tri` (14.6 / 14.2 / 14.6 — spread 0.4 at the
double-precision floor, i.e. noise, not a defect) and `chevron_2_4` (3.2 / 1.0 / 2.3 — real).

### Where it still falls short, stated plainly

* **Chevrons.** `chevron_1_2` reaches 6.1 certified digits at n=120 (8.0 at 1e-13) and
  `chevron_2_4` 3.2 at 163, both reporting `capped` with a shortfall message rather than silently
  under-serving. The suite's own ceilings are 7.1 and 5.0, so this is a hard family — but the
  planner is not matching even that, and `chevron_2_4` is still non-monotone in target. Deliberately
  **not** chased from this single data point: `chevron_1_15`, `chevron_2_3` and `chevron_1_125` are
  in the S3 holdout, and tuning to one domain is the trap the h4 → h5 → h6 sequence already caught
  once.
* **`rect_thin` costs more than the old recipe** (130 against 81) for 11.7 digits against 10.7.
* **`iso_tri_h4`/`h16` remain weak unrefined** (4.4 / 4.2 at 1e-7); refinement lifts them to
  7.1 / 7.2 but that costs solves, so the one-shot `polygon_default_basis` is not the good path on
  sharp elongated triangles.
* Arc-local geometry is sampled at the arc **midpoint** only. Thickness varies by less than a
  factor 2 along the arcs measured on iso_tri_h4, so it is not the current limiter — but on a
  mushroom or a stadium it would be, and subdividing arcs by thickness variation is the obvious
  next step.

---

## S3. Validation

75 scored cells in `run/plan/s3.jsonl` (25 polygons with reference eigenvalues x 3 targets, plus a
3-seed sweep and a second wavenumber tier), 8 smoothness tests, and a pinned gate in
`run/plan/gate.json`. Score is the triple — certified digits, spurious-minimum count, cost — with
contrast, interior-source and seed-spread guards.

### It generalizes: dev and holdout agree exactly

| | cells meeting target | spurious minima | contrast < 4e2 | median MP − R |
|---|---|---|---|---|
| dev (6 domains) | 11/18 | 0/18 | 0/18 | **+1.0** |
| holdout (19 domains) | 33/57 | 3/57 | 5/57 | **+0.9** |

Identical median margin on held-out domains is the result S3 exists to produce: nothing in the
planner is fitted to the dev set. The three spurious-minimum cells are all `chevron_1_15`; the
contrast collapses are all chevrons plus `reg_ngon_8@1e-7`.

Comfortably clearing target: rectangles, triangles, `reg_ngon_5/7/8`, `parallelogram_60/p65`,
`L_shape`, `right_trapezoid`, `iso_tri_h05`, `GWW1/2`, `H_shape`. Failing: the four chevrons and
`iso_tri_h4`/`h16`.

### The frozen plan holds at a reentrant corner

`smooth` freezes one plan at `t=0` on the L-shape family and realizes it on each member:

| t | n | solved λ | σ | Hadamard dλ |
|---|---|---|---|---|
| −0.030 | 158 | 9.9147306531 | 7.9e-14 | −9.22927786 |
| −0.015 | 158 | 9.7767231280 | 7.1e-14 | −9.16927675 |
| 0.000 | 158 | 9.6397238440 | 6.7e-14 | −9.09498310 |
| +0.015 | 158 | 9.5039405795 | 6.4e-14 | −9.00730522 |
| +0.030 | 158 | 9.3695663833 | 6.7e-14 | −8.90729136 |

One basis size throughout; σ never leaves 6e-14. **A central difference of the solved λ agrees with
the Hadamard dλ to 2.6e-04 — and the difference's own truncation error at h=0.015 is 2.2e-04**, so
they agree as closely as the instrument can resolve. At `t=0`, dλ matches the independent
continuation FD reference to **1.0e-11** (that reference's own resolution is 1.5e-10). This is the
architecture's central claim, on a singular-corner domain, and it holds.

`tests/test_basis_plan_smoothness.py` does the same on the rectangle, where both λ and dλ are
closed form: σ ≈ 4e-16 and dλ relative error 0 to 4e-15 across a 35% change in L with a plan built
for L=2.

### RETRACTED: the first S3 sweep's certification was corrupted

**Two findings were published in this file and are now withdrawn.** The first S3 sweep certified
through `moler_payne(domain, callable, lam)`, which — given only a callable and no solver — cannot
reach the solver's `bdry_quad` and so falls back to `interior_l2`'s Dunavant mesh. That mesh is
cached on **`id(domain)`**, a CPython address that is reused after garbage collection, and the
sweep builds one `Domain` per cell and drops it. Recycled addresses returned a *previous* domain's
cubature points, silently, because they are a valid `PointSet` either way.

Detected by a cross-check that costs nothing and is now permanent: `certified_digits` and
`worst_arc_residual` are computed by different code from different quantities and agree to ~0.1
digits whenever the certification is sound. **10 of 150 cells disagreed by up to 6.7 digits**, in
both directions, so the errors could not be excused as conservative:

    chevron_1_2  high  MP 14.94  own residual  8.23   gap +6.72
    chevron_1_2  low   MP 12.01  own residual  6.41   gap +5.61
    reg_ngon_7   low   MP 17.37  own residual 12.31   gap +5.06
    rect_thin    low   MP 12.88  own residual 13.94   gap -1.06

Fixed in two places. `certify.interior_l2` now keeps a strong reference to the domain in each cache
entry, so the address cannot be recycled under it (`tests/test_certify_mesh_cache.py`; note a
geometric fingerprint would *not* be a safe alternative key — GWW1 and GWW2 are isospectral, with
equal area and perimeter and the same segment count). And this harness no longer uses cubature at
all: `plan_lab.certify` takes `||u||_L2` from the **Rellich boundary identity** via `boundary_l2`,
which is what `eigfun_integrals` was built for, reaches ~1e-13 on most suite domains, costs three
basis evaluations against a mesh build, and never touches the `id()`-keyed cache. After the switch
**all 150 cells agree with their own per-arc residual to within 0.16 digits.**

What survived, what did not, on the re-run:

| claim | first sweep | corrected |
|---|---|---|
| dev / holdout median MP − R | +1.0 / +1.0 | **+1.0 / +0.9** (survives) |
| cells meeting target | 11/18, 33/57 | **11/18, 33/57** (survives) |
| chevron seed spread | **5.6 digits** | **0.0** — worst spread anywhere is 0.5 (retracted) |
| high-κ gain on chevron_1_2 | **+8.5 digits** | **+1.8** (retracted) |
| reg_ngon_7 "at round-off" | 16.3, 17.4 digits | **12.3, 12.4** — never was (retracted) |

So the chevrons are **not** under-determined by the interior draw, and sizing for a larger window
is a moderate help rather than a cure. The generalization result — the whole point of the holdout —
is unaffected.

### What the chevron family actually shows

Corrected numbers, `lam_max` from `weyl_est(6)` against `weyl_est(50)`:

| domain | low: κ / n / MP | high: κ / n / MP | gain |
|---|---|---|---|
| chevron_2_3 | 15.6 / 238 / 1.9 | 31.0 / 239 / 5.9 | +4.0 |
| chevron_1_2 | 13.1 / 163 / 6.4 | 29.0 / 174 / 8.2 | +1.8 |
| chevron_1_15 | 20.3 / 218 / 4.0 | 42.5 / 227 / 5.4 | +1.4 |
| chevron_2_4 | 10.1 / 223 / 1.0 | 21.2 / 231 / 2.3 | +1.3 |
| L_shape | 6.5 / 159 / 12.8 | 15.9 / 181 / 13.1 | +0.3 |

Three separate things are wrong on this family, and they are worth keeping distinct:

1. **Contrast genuinely collapses**: 8 to 20 on `chevron_2_3`/`2_4` against 1e6-1e14 elsewhere.
   Below the 4e2 guard there is barely a tension minimum to find, so those readings are flagged
   untrustworthy in their own right.
2. **The certification is shaky there too.** 28 of 150 cells have an L² `x0` spread above
   `SPREAD_TOL = 1e-8` (`chevron_2_4` reaches 5.1e-05) — `boundary_quadrature` itself warns it
   falls short on near-slit corners. So the low digit counts are partly instrument, not purely
   basis, and a firm number needs `certify_solver`'s cubature cross-check.
3. `chevron_1_15` is the only domain anywhere with **spurious minima** (1, at every target).

`order_margin` was swept on dev as a candidate fix (3 → 12): median MP 11.5 → 12.2, worst 4.9 →
5.9, at +20% columns — a real but modest gain, and *not* a substitute for the κ effect. A
term-by-term ablation on `chevron_1_2` (vary one corner's `M`, freeze everything else) showed the
reentrant corner's count is nearly inert (M 40 → 70 moves MP by 0.5) while the sharp corners and
the α=3.39 corner each buy 1-2 digits. Sizing alone does not fix this family.

### Other corrections made during S3

* Three of the eight smoothness tests failed on first run, and **all three were badly designed
  assertions, not defects**: one differenced the closed-form λ (never touching the basis) at
  h=0.15, where the truncation error alone is 1.1e-2; one asked dλ's second differences to be
  small when dλ has real curvature (−96π²/L⁵·h² ≈ 0.67); one asked re-planning to be size-stable
  when varying is correct and is *the reason freezing matters*. All three now compare against the
  right reference.
* The `smooth` stage's own scan window was too narrow for its step size, and `_lam_near`'s edge
  guard — added after the s0c incident — caught it rather than silently returning a window edge.
  The stage now predicts the shifted eigenvalue from `dλ·t` before searching.

---

## S4. Auto-configuration

**CLAUDE.md principle 1 now runs.** It could not before: `from_domain(basis=None)` raised
`NotImplementedError`, so every caller in the repo hand-built a basis, a collocation set and a
quadrature, and `docs/todo.md` recorded why that persisted — `n_basis` is "the one quantity
[callers] have no principled way to choose".

```python
dom  = geo.L_shape()
evp  = Eigenproblem(dom)
eigs = evp.solve(4)        # 138 columns, 12.7-13.8 true digits, nothing configured
```

* `mps.default_basis_for(domain, lam_max, target, rtol)` is the seam. Polygons go to
  `basis_plan.polygon_default_basis`; a curved boundary **raises and names the alternatives**,
  because inventing a size there would be exactly the unfounded guess this path exists to remove.
* `rtol` is threaded from the solver into `PlanConfig`, so the planner's conditioning ceilings and
  the pencil's rank truncation cannot drift apart. Getting that constant wrong is what made
  achieved accuracy non-monotone in the target (S2).
* `Eigenproblem(domain, precision=p)` is **one dial at both stages**: it sizes the basis and
  becomes the search's `ltol`. Measured on L_shape:

  | precision | n_basis | ltol | true digits |
  |---|---|---|---|
  | 1e-4 | 88 | 1e-4 | 5.6 - 8.6 |
  | 1e-8 | 138 | 1e-8 | 10.4 - 14.0 |
  | 1e-12 | 182 | 1e-12 | 14.2 - 15.4 |

* Solver construction is **lazy and cached**: an `Eigenproblem` stays cheap to make, and two
  `solve` calls share one solver. `_get_evec_solver` falls back to the eval solver rather than
  building a second one.

`tests/test_auto_configuration.py` (9 tests) pins all of the above.

### The POC is retired, not deleted

`lappy/heuristics.py` and its 31 tests were **never committed**, so deleting them would have been
unrecoverable. They are archived at `benchmarks/archive/mps_heuristics_poc/` with a README carrying
the matched-size comparison. `lappy.heuristics` is no longer importable; `heur.py` still prints its
recorded ledger (the measurements are the artifact worth keeping), and only its basis-*building*
stages need the archived file copied back. `bench.paper_heuristic` became `bench.planner`.

## Status

Done: S0 (both gates), S1 (speed floor), S2 (planner), S3 (validation), S4 (auto-configuration).
1071 tests pass; the gate is pinned at 75 cells and clean.

Open, in priority order:

1. **The chevron/sharp-elongated family**, where three causes are tangled and need separating:
   collapsing contrast (real ill-conditioning), a certification whose own L² spread exceeds
   tolerance, and under-provisioned corner blocks. Raising κ buys +1.3 to +4.0 digits. Get firm
   numbers first, via `certify_solver`'s cubature cross-check.
2. **`iso_tri_h4`/`h16`** still miss target at every tier; the high tier barely helps, so this is a
   different mechanism again.
3. Arc geometry is still sampled at the arc midpoint only.
4. `_apply_cap` can return `n_total > n_cap` when the FB budget alone exceeds it (spiral). Reported,
   but "cap" is then a misleading name.

## S5. A sharp corner cannot own an arc

The chevron family's cause, found after the certification was fixed and the numbers were firm.

First, the L² question was **settled and it was not the problem.** For the 28 cells whose boundary
`x0` spread exceeds `SPREAD_TOL`, `boundary_l2` is deliberately conservative
(`sqrt(clip(lo - spread))`), so a large spread should *underestimate* ‖u‖ and understate the digits.
Measured against cubature — exact for polygons, and now correctly cached:

| domain | gram diag | ‖u‖ rellich | ‖u‖ cubature | spread | digits rellich | digits cubature |
|---|---|---|---|---|---|---|
| chevron_2_4 | 1.000000 | 0.992574 | 1.000004 | 1.5e-02 | 0.95 | 0.95 |
| chevron_2_3 | 1.000000 | 0.999845 | 1.000004 | 1.8e-04 | 1.90 | 1.90 |
| L_shape | 1.000000 | 1.000000 | 1.000000 | 6.6e-13 | 12.81 | 12.81 |

Identical to two decimals either way: `eps` is dominated by the boundary `sup|u|`, not the
denominator. So the chevron numbers were real, and the guard flags a hazard that does not bite here.

**The cause is the ownership radius.** A corner owns `R = gamma*clearance`, independent of `alpha`.
For `chevron_1_2`'s sharp corners `alpha = 9.76`, so the block's lowest term is `r^9.76` — below a
tenth of its edge value over ~80% of the arc it owns — while the partition forbids sources from
covering the rest. Shrinking only the over-claiming corners and giving the freed boundary to sources:

    shrink   1.00    0.60    0.40    0.25
    digits   6.40    8.25    9.20    9.76        (contrast 4e7 -> 3e10)

Implemented as `R *= min(1, sharp_ref/alpha)`, a **fitted** functional form. Swept on dev, then
confirmed on the holdout before adoption (`plan_lab sharp`), with `2.0` beating `3.4`:

| domain | disabled | sharp_ref=2.0 |
|---|---|---|
| chevron_2_4 | 0.95 | **6.64** (+5.7) |
| chevron_2_3 | 1.90 | **6.06** (+4.2) |
| chevron_1_2 | 6.40 | **10.29** (+3.9) |
| chevron_1_15 | 4.04 | 6.07 (+2.0) |
| iso_tri_h16 | 5.19 | 7.05 (+1.9) |
| iso_tri_h4 | 4.95 | 6.02 (+1.1) |
| right_trapezoid | 13.83 | 13.33 (−0.5) |

Because `alpha <= 2` (a right angle or blunter) is untouched, **17 of 25 suite polygons do not
change at all** — the intervention is targeted rather than global.

Full gate verdict after adoption, with accuracy and cost now scored separately (conflating them had
made a +1.1-digit gain read as a regression): **19 improved, 3 lost accuracy, 10 costlier at equal
accuracy, 43 unchanged.** Dev goes 11/18 → 13/18 cells meeting target; holdout 33/57 → 32/57 with
the median margin rising +0.9 → +1.1.

The three accuracy losses are `iso_right_tri` and `iso_tri_h1` at 1e-7 (14.3 → 12.1) and
`right_trapezoid` at 1e-10 (13.8 → 13.3). All three still meet their targets with +3.3 to +5.1
digits of margin. **A methodological miss worth recording**: the sweep tested only `target=1e-10`,
so the 2.2-digit loss at 1e-7 was invisible until the gate caught it. A knob sweep should cover the
target ladder, not one rung.

### The planner now exceeds the suite's own best-known values on 11 of 16 domains

`digit_ceiling` is the best certified accuracy previously observed by *any* construction:

| domain | previous ceiling | now | | domain | previous ceiling | now |
|---|---|---|---|---|---|---|
| chevron_1_2 | 7.1 | **10.4** | | GWW2 | 8.7 | **10.7** |
| chevron_2_3 | 4.6 | **6.4** | | GWW1 | 9.9 | **10.9** |
| chevron_2_4 | 5.0 | **6.6** | | H_shape | 8.2 | **9.5** |
| chevron_1_15 | 6.3 | 6.9 | | reg_ngon_8 | 10.3 | **11.6** |
| L_shape | 13.3 | 13.9 | | reg_ngon_7 | 11.7 | 12.4 |

Still short of the ceiling on three: `iso_tri_h4` (12.9 → 7.1), `iso_tri_h16` (10.1 → 7.0) and
`reg_ngon_6` (13.0 → 11.1). Those older numbers came from hand-tuned solves at 240-480 columns; the
planner is using 97-239 and asking for a target it then meets. Worth checking whether it can reach
them when asked for more, which is the clearest remaining question.

## Checking that `precision` was met

Closed the gap that "nothing certifies the request was met". Two methods on `Eigenproblem`, both
measuring rather than predicting:

```python
evp  = Eigenproblem(dom, precision=1e-10)
eigs = evp.solve(3)
rep  = evp.check_precision(eigs)     # {'target', 'achieved', 'digits', 'met', per-block breakdown}
evp.refine_basis(eigs)               # opt-in: grow where short, rebuild the solver
```

`check_precision` reports the worst block's contribution to the Moler--Payne relative bound — the
same quantity `refine_plan` optimizes, agreeing with a full certification to 0.16 digits across 150
cells at roughly 1% of the cost, because `||u||_L2 = 1` for orthonormalized coefficients leaves only
a boundary sup per arc. It returns `None` for a hand-built basis: there is no per-arc structure to
attribute a residual to, and inventing one would be worse than saying so.

| domain | n | achieved | met? |
|---|---|---|---|
| L_shape @ 1e-10 | 159 | 6.6e-11 (10.2 dig) | **yes** |
| chevron_1_2 @ 1e-10 | 163 | 3.9e-07 (6.4 dig) | no |
| iso_tri_h4 @ 1e-10 | 65 | 1.1e-05 (4.9 dig) | no |

`refine_basis` is opt-in because it costs a second solve (the eigenvalues move when the basis
changes), which a shape-optimization loop pays once and amortizes but a one-off solve does not want.
Behaviour on the three cases above:

* `iso_tri_h4`: 65 → 122 columns, 4.9 → 8.1 digits — better, still short, and still reported `met=False`.
* `L_shape`: no growth, no rebuild. Correct no-op.
* `iso_tri_h16`: no growth *possible* — every block is at a conditioning ceiling, so it returns the
  best plan it measured and says so rather than piling on columns that would make things worse.

`residual_by_arc` now takes one λ or several and reports the worst, because the right choice differs
by use: an optimizer tracking a single mode wants that mode alone, while `solve(k)` needs the whole
window (higher modes oscillate faster, so a plan refined only at λ₁ under-serves λ_k). The policy
stays with the caller rather than being baked in.

Open, in priority order:

1. **The chevron/sharp-elongated family**, where three causes are tangled and need separating:
   collapsing contrast (real ill-conditioning), a certification whose own L² spread is above
   tolerance, and under-provisioned corner blocks. Raising κ buys +1.3 to +4.0 digits, which says
   the corner sizing is part of it but not all.
2. **`iso_tri_h4`/`h16`** still miss target at every tier; the high tier barely helps (+1.0, −0.2),
   so these are a different mechanism again.
3. `PlanConfig.rtol` duplicates a value the solver owns; S4 should make the planner read it from
   the solver so the two cannot drift.
4. Arc geometry is still sampled at the arc midpoint only.

---

### A methodological note, recorded because it nearly became a finding

The first S0c run reported **every** construction wrong by an identical `1.35e-01`. An error
independent of the basis is not a basis result, and the cause was in the reference: `_lam_near`
scanned a fixed ±2e-3 relative window around λ(0) for every member of the family, and by t = ±2h
the eigenvalue had left that window, so the minimizer returned the window edge. The three-point
estimate, which used only the in-window points, had been correct all along. The reference now
walks outward by continuation and `_lam_near` **raises** if the discrete minimum lands on the edge
of its scan. This is the same failure mode as the `ltol` trap in `bench.py`'s header — the search,
not the object under study, setting the answer — and it is the second time in this directory that
it has produced a confident wrong number.
