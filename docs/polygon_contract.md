# The polygon contract, v0.9

What `lappy` promises on **polygons**, what has been measured to support each promise, and where
the envelope ends. This is the document a downstream package (`douse`) should build against; if
something is not here, it is not promised.

Version 0.9 is a beta claim about **this path specifically**. Curved domains work and are tested,
but they are not part of it — `mps.default_basis_for` raises for them rather than guessing a
basis size, and `tests/test_reference_tables.py` cannot cross-check their reference values.

---

## 1. The three-line path

```python
dom  = geo.L_shape()
evp  = Eigenproblem(dom, precision=1e-10)
eigs = evp.solve(4)
```

Measured on ten suite polygons at `precision=1e-8` and `1e-10`, nothing else configured:
**8.6 to 13.4 true digits at 60–364 basis columns**, 0.7–16 s per solve. The one sub-8 reading
is `H_shape` at 1e-8 (7.3), which is that domain's own reference ceiling of 7.8, not the solver.

## 2. What each dial promises

### `precision`
**A hint with margin, not a guarantee.** It sizes the basis (through `basis_plan`) *and* becomes
the eigenvalue search's `ltol`, so both stages are solved to matching depth.

Measured over 75 cells (`benchmarks/basis_lab/PLAN_LAB.md` S3): median achieved accuracy runs
**~1 digit better than requested**, on both the development and the held-out domains
(+1.0 / +1.1 — identical margins, which is the result the holdout exists to produce). It is
monotone and responsive where there is room and saturates where there is not.

It is **not** met on the sharp-cornered families (chevrons, thin isoceles triangles). Those fall
short honestly: `plan.capped` and `plan.shortfall` report what the planner knew in advance, and
`Eigenproblem.check_precision(eigs)` **measures** what was actually achieved rather than
predicting it, agreeing with a full Moler–Payne certification to 0.16 digits at ~1% of the cost.
`refine_basis` grows the basis where it is short; it is opt-in because it costs a second solve.

### `solve(k)`
**Returns the first `k` eigenvalues, counting multiplicity** — a stronger claim than `k` of them,
and one that was being broken until the scan grid was fixed. Swept over ten polygons × k ∈ 2…10:

| `ppl` | cells returning the wrong set |
|---|---|
| 5 (old default) | 9 of 90 |
| 10 | 2 of 90 |
| **20 (current)** | **0 of 90** |

**This rests on grid resolution, not on a detector.** A domain with a tighter spectral cluster
than `H_shape`'s could still defeat it, and the Weyl-count audit in `_solve_dir_neu` provably
cannot serve as a backstop: measured per-gap expected counts *overlap* between correct and
incorrect results, because multiplicity confounds the two-term count at these wavenumbers. So
`ppl=20` is **validated, not proven**. For a loop, prefer `track`, which has no index to shift.

### `track(lam_prev, mult=1)`
**Follows one eigenvalue by value from a nearby start**, ~2.7× faster than `solve(1)` and immune
to the set-selection question above. Two refusals, both of which would otherwise return a
plausible wrong number:

- a minimum on the **edge of the scan window** raises — then the window, not the tension, chose
  the answer (this failure produced a 16 %-wrong reference twice in this project's history);
- a minimum where σ exceeds `ttol` raises, checked at `sigma[mult-1]`, so a split cluster is
  caught too.

Degenerate clusters track correctly (`reg_ngon_6`'s doubles to 1e-8, orthonormalizing to 1e-8).

### Eigenfunctions
`eigenfunction_coef(lam, mult=m)` returns an **L²-orthonormal** cluster via the boundary-only
Rellich identity, with no quadrature tuning by the caller. What is *not* promised: **which**
orthonormal basis of a degenerate eigenspace comes back. Löwdin returns one of infinitely many
and it moves under perturbation, so no individual entry of an `m × m` shape-derivative matrix
means anything — only its spectrum does.

### Shape derivatives
`lappy` implements **no** shape-derivative formulas. It provides the pieces:
`weighted_integral(ed, kernel, weight)` for the four bilinear Cauchy-data kernels,
`normal_velocity(bq, dp)` for the `V·n` conversion under lappy's own outward-normal convention,
and per-node `bq.seg_idx` / `bq.tau` so a caller can evaluate a velocity field at the nodes.

**Use `solver.hadamard_quad`, not `solver.bdry_quad`, for these.** The default node set is
`weight_family='even'`, matched to the eigenfunction's exponent family because that is what the
normalization needs; a velocity that moves a **reentrant** corner supplies `V·n ~ r` there, which
it does not integrate exactly. Measured on the L-shape's reentrant vertex against a five-point
central difference: **6.1 digits with `bdry_quad`, 9.3 with `hadamard_quad`**, both answers
looking entirely plausible. `normal_velocity` warns when a velocity moves a ν<1 corner on an
`'even'` set.

---

## 3. The validated envelope

Everything measured above sits inside this box. Outside it, nothing is claimed.

| axis | validated | note |
|---|---|---|
| geometry | **polygons** (`domain.bdry.is_polyline`) | curved works, is tested, is not part of the 0.9 claim |
| spectral window | `lam_max = weyl_est(6, domain)`; `solve(k)` swept to k=10 | λ₂₀ and beyond untested |
| basis size | `n ≤ PlanConfig.n_cap = 240` | a cap adopted from a rank-saturation measurement on **two** domains, which its own notes called possibly coincidental |
| wavenumber | κ ≈ 4–13 | at larger κ the source-spacing conclusions expire and `n_cap` will bind |
| corner exponent | ν < 1 and ν ∈ [1.69, 2.67] for the Hadamard guard | the band **1 < ν < 1.69** is untested in either direction |

`n_reg` (the rank the pencil actually keeps) does grow with κ but far more slowly: L_shape
123 → 159 as κ goes 5.6 → 26.4. A loop wanting λ₁…λ₂₀ will meet `n_cap`, and nothing has measured
what happens there.

## 4. Explicitly out of scope for 0.9

- **Chevrons and thin isoceles triangles.** `chevron_*` reaches 6.4–10.4 certified digits and
  `iso_tri_h4/h16` 7.0–7.1; three causes are tangled there (genuine ill-conditioning with
  collapsing contrast, a certification whose own L² spread exceeds tolerance, and
  under-provisioned corner blocks). `check_precision` reports honestly on all of them.
- **The two spirals.** Pathological by construction.
- **Robin boundary conditions** for orthonormalization (the Rellich identity's Robin form is out
  of scope; `eigenfunction_coef` falls back to raw coefficients with a warning).
- **Mixed Dirichlet/Neumann reentrant corners**, which are out of reach on principle: the Rellich
  boundary integral diverges there for any `x0` off the adjacent edge lines.

## 5. Known defects, open

Recorded here because a downstream package should know them before it starts, not after.

1. **`symsolve`'s per-sector multiplicity estimate under-counts** when a domain is reduced by a
   *proper* subgroup of its true symmetry. This produced two reference tables each short a mode
   (`reg_ngon_6`, `reg_ngon_8`), both corrected. Polygon tables are now cross-checked against an
   independent full-domain solve (`tests/test_reference_tables.py`); **curved tables are not**,
   and `disk` — solved with D2 against a true O(2) — is the obvious next suspect.
2. **No validated completeness detector for `solve(k)`** (see §2).
3. **`PlanConfig.rtol` duplicates a value the solver owns**; the two can drift, and getting that
   constant wrong is what once made achieved accuracy non-monotone in the target.
4. **Arc-local geometry is sampled at the arc midpoint only.** Not the limiter on the suite, but
   it would be on a mushroom or a stadium.

## 6. Cost

Per `solve(4)` with everything auto-configured, after the order-0 Bessel fix and λ-grid
threading: **0.7–4 s** on most suite polygons, 16 s on `H_shape` (364 columns). `track` is
281 ms per iterate on the L-shape at n=159. Solver construction is ~10 ms, so a loop should
build one plan and realize it per iterate rather than re-planning.

Profile, for anyone optimizing further: basis evaluation ~67 %, the GSVD stack ~33 %. The single
largest remaining item is the `jv` ladder in `FourierBesselBasis` (14.5 s of a 36.9 s `H_shape`
solve before threading).
