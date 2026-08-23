# Scope: what `lappy` provides, and what a downstream shape package owns

**Status: §5's two blockers are gone; §3 is built. Updated 2026-08-23.** This began as a record
of a design discussion rather than a commitment, deferring the interface until two moving pieces
settled. Both have (`lappy/basis_plan.py` and the auto-configuration it enabled — see
`benchmarks/basis_lab/PLAN_LAB.md` S2 and S4), and §3's four items are now implemented. What
remains provisional is §1's *division of labour*, not the seam itself.

## 1. The line

`lappy` computes spectra. A separate package — tentatively `douse`, "Domains of Unknown Shape
from Eigenvalues" — wraps it for shape optimization and inverse spectral problems.

| | owns |
|---|---|
| **lappy** | *a* domain and its conventions: geometry, orientation, normals, arclength; eigenvalues; L²-orthonormal eigenfunctions; Cauchy-data boundary integrals (`weighted_integral`) and the quadrature under them |
| **douse** | *families* of domains: `ParametricDomain`, the map from a parameter vector to a `Domain` and its derivative; Hadamard-type shape-derivative formulas; optimizers; inverse problems |

The argument for putting parametrization downstream is that parametrizations are irreducibly
problem-specific — Fourier coefficients of a radial function, spline control points, polygon
vertices, conformal maps. Any single `ParametricDomain` in `lappy` would be over-general or
wrong for most users, and would put a second geometry API beside `Domain` for no benefit to
someone who only wants eigenvalues. It also keeps CLAUDE.md's first principle intact: posing
an eigenproblem stays three lines.

## 2. The seam already exists in the code

This split ratifies a boundary the codebase drew earlier, rather than inventing one:

* `docs/eigfun_integrals.md`: `weighted_integral` keeps the four bilinear Cauchy-data kernels
  and an arbitrary boundary weight as "the extension point for Hadamard-type shape derivatives
  — but only ever at eigenfunction scale. lappy implements no such formulas itself."
* `MPSEigensolver.bdry_quad` is exposed for reuse by code building other boundary functionals
  from the same eigenfunctions. (It used to name Hadamard-type shape derivatives as the example.
  That was wrong for the polygon case and §3 records why; `hadamard_quad` is the one for those.)

The seam is where the mathematics has one: upstream is "solve the PDE, integrate Cauchy data",
downstream is "which functional, which perturbation field, which optimizer".

## 3. What the seam needs before it can be an API

Four items, in rough order of how load-bearing they are.

**Per-node provenance. DONE.** `BoundaryQuad` carries per-node `seg_idx` and `tau`, collected
in `assemble_panels` where they were already being computed and discarded.
`tests/test_shape_derivative.py::test_the_quad_carries_node_provenance` pins them against the
panel-walking reconstruction and against `seg.p(tau)` landing back on the node.

**The orientation convention stays upstream. DONE.** `eigfun_integrals.normal_velocity(bq, dp)`
takes a per-node complex displacement and returns `V·n` under lappy's own outward-normal
convention. A sign or orientation error there yields a *plausible, wrong gradient* that an
optimizer will follow happily, so the sign is pinned by uniform dilation against the Rellich
identity (`test_dilation_through_the_converter_is_the_rellich_identity`), which fixes magnitude
and orientation together, plus a check that a purely tangential velocity returns zero. douse
computes `dp` from its parametrization — using `bq.seg_idx`/`bq.tau` to know where each node
sits — and lappy converts. The convention is enforced in one place rather than agreed in two.

**And it is not only the sign that a downstream package would get wrong.** The node set matters
too, and the default one is the wrong choice for the polygon case. `MPSEigensolver.bdry_quad` is
`weight_family='even'`, matched to the eigenfunction's own exponent family because that is what
the Rellich/Gram normalization needs. A velocity that MOVES A SINGULAR CORNER supplies
`V·n ~ r` there, outside that family. Measured end to end on the L-shape's reentrant vertex
against a five-point central difference over a frozen plan:

| node set | agreement with the FD |
|---|---|
| `solver.bdry_quad` (`'even'`) | **6.1 digits** |
| `solver.hadamard_quad` (`'integer'`) | **9.3 digits** |

Three orders, and both answers look entirely plausible. For a polygon parametrization moving a
vertex *is* the design variable and reentrant corners are the interesting ones, so this is the
ordinary case rather than an edge case — and `bdry_quad`'s docstring used to recommend itself
for exactly this. `MPSEigensolver.hadamard_quad` now builds the right set lazily, and
`normal_velocity` warns when a velocity moves a singular corner on an `'even'` set. The two
node sets are kept separate on purpose: the trade runs the other way for the Rellich weight, and
by as much.

**Degenerate clusters are part of the contract. DOCUMENTED** in
`MPSEigensolver.eigenfunction_coef`, including what is *not* promised: which orthonormal basis of
the eigenspace comes back. A shape derivative of a multiple eigenvalue
is not a derivative — it is a directional derivative of the eigenvalues of the `m x m` matrix
`int (du_i/dn)(du_j/dn) (V.n) ds`. `weighted_integral` already returns `m x m` rather than a
scalar, and `eigenfunction_coef(mult=m)` returns an orthonormal cluster. That pairing is what
makes degenerate shape derivatives possible at all; it looks like an implementation detail and
is not, so it should be documented as intentional. Symmetric domains and most
maximize-a-gap problems live exactly there.

**Corner motion must be detectable.** If a parametrization moves a *reentrant* corner, the
classical Hadamard formula's regularity assumptions weaken and `theta -> lambda` can fail to
be differentiable in the way the formula assumes. Handling that is modeling, so it belongs to
douse — but `corner_specs` already reports which corners are singular and admissible, so douse
can detect "this direction moves a nu<1 corner" and say so instead of returning a confident
number.

## 4. One test stays in `lappy`, permanently

The split creates a specific hazard: with shape derivatives downstream, **lappy retains no
in-repo consumer sensitive to the accuracy of `weighted_integral`.** Its own tests are either
scale-invariant or closed-form norms on easy domains. That is not hypothetical — it is exactly
how the `nu < 1` criterion bug survived: nothing in the suite could feel it, and the
certified-eigenvalue suite provably cannot, since `eps` is scale-invariant (measured:
`right_trapezoid`'s quadrature improved ten orders and its certified digits did not move).

So `lappy` should keep a finite-difference `dlambda` check — rectangle under stretching, disk
under dilation, both with analytic answers — as a permanent acceptance test of the three
things it promises downstream: `lambda` accurate, `u` orthonormal, Cauchy-data integrals
accurate. It is the only test sensitive to a *systematic* error in `||u||`.

Conveniently this needs no `ParametricDomain`: for those two cases `V·n` is three lines
inline. lappy's promise stays tested in lappy, without importing anything from douse.

**This exists: `tests/test_shape_derivative.py`** (26 tests), covering rectangle edge
translation, dilation against the Rellich identity, a degenerate cluster splitting correctly,
the sector radius derivative at a singular corner, the `weight_family='integer'` corner-moving
case, and — through the auto-configured path, on a frozen plan — a polygon VERTEX-moving
derivative at a reentrant corner against a five-point central difference.
`tests/test_basis_plan_smoothness.py` adds the frozen-plan version on the rectangle family,
where both `lambda` and `dlambda` are closed form.

Most of these still build their solvers by hand (`bases.make_default_basis` plus explicit
collocation) rather than through `Eigenproblem(dom, precision=...)`; the vertex-moving test is
the one that goes the whole way, and it is the one that found the node-set defect in §3. More
of the tier-1 cases should follow it.

## 5. Why the API was not committed to (RESOLVED)

Two pieces of `lappy` were still moving, and both could have changed what it is able to promise.
Both have now settled, which is what reopened this document:

1. **Basis-selection heuristics. Settled by `lappy/basis_plan.py`.** Ten benchmark domains sat in
   bucket 2 — complete spectrum, clean tension curve, under 8 digits — diagnosed as basis
   insufficiency. The resolution was not a better branch but *no* branch: `plan_basis` derives the
   construction from geometry, `lam_max` and a target, and `refine_plan`/`check_precision` measure
   what it achieved. Measured black-box on ten polygons, `Eigenproblem(dom, precision=1e-10).solve(4)`
   gives 8.8–13.3 true digits at 60–364 columns with nothing configured.
2. **The `domain -> (Eigenproblem + auto-configured Eigensolver)` pipeline. Settled** (PLAN_LAB S4).
   `mps.default_basis_for` is the seam; `precision` is one dial that sizes the basis and becomes
   the search's `ltol`. The reference work's hand-built solvers are no longer the only path.

So the interface can be fixed, and §3's four items are done. What is left before douse leans on
this is not interface design but two solver-level gaps recorded in `docs/todo.md`: there is no
inner-loop entry point (`solve(k)` rescans globally every call, 2–8 s per solve), and `solve(k)`'s
mode completeness rests on scan-grid resolution rather than on a validated detector.

## 6. Sequencing

The original four steps are done: the §4 contract test exists, the bucket-2 basis research
became `basis_plan`, the auto-configuration pipeline was polished on what it established, and
§3's items are implemented. What replaces them, before douse leans on any of this:

1. **An inner-loop entry point.** `solve(k)` runs a global Weyl-gridded scan every call, 2–8 s
   per 4-eigenvalue solve (22 s on H_shape) against ~10 ms of solver construction. A shape loop
   knows where `lambda` was last iterate and wants a local bracket, not a rescan. This is the
   largest available speedup and it also sidesteps §6's next item entirely, since tracking
   follows one mode by value rather than selecting a set by index.
2. **A validated completeness detector for `solve(k)`.** Grid resolution currently does the work
   (`ppl`), and no cheap audit stands behind it — the Weyl-count test already in
   `_solve_dir_neu` provably cannot serve, because per-gap expected counts overlap between
   correct and incorrect results once multiplicity is in play.
3. **A `dlambda` case through the auto-configured path** (§4's remaining gap).
4. Then fix the douse interface.

The chevrons and thin isoceles triangles stay parked alongside the spirals: `check_precision`
reports honestly on them and the goal is explicitly "non-pathological".

**Measure basis variants carefully.** Interior collocation points used to come from the global
RNG, and the record shows `iso_right_tri` returning 4.9, 4.0 and 2.5 certified digits on three
runs of identical code. That draw is now seeded by default (PLAN_LAB S1), so this is a trap for
anyone who passes `rng=np.random.default_rng()` rather than the default: fix the seed *and*
report spread across several. The spread is itself a signal — a basis whose accuracy depends
strongly on the interior sample is telling you the system is under-determined — and it matters
more to `dlambda` than to `lambda` (S0b: seed spreads up to 1.4 digits against 0.5).

## 7. Open questions

* ~~Does the eigenvalue-digit objective agree with the `dlambda`-accuracy objective?~~
  **ANSWERED: yes.** PLAN_LAB S0b/S0c measured both at a known `lambda`, so no search could
  confound either: 57 of 60 comparable pairs concordant on rectangles, and on the L-shape no cell
  had `MP >= 10` with `dlambda` below its reference floor — the failure mode the stage existed to
  detect. `dlambda` digits run 1–3 *above* certified digits, so tuning to certified eigenvalue
  accuracy is safe and the derivative follows with margin. One rider: `dlambda` is more sensitive
  to the interior collocation draw than `lambda` is (seed spreads up to 1.4 digits against 0.5),
  which is why the S1 determinism fix was worth more than hygiene.
* Where does a `Domain` family's *derivative* geometry live in practice — is
  `ParametricSegment` convenient enough for douse to build `dp/dtheta` on top of, or does it
  end up needing something upstream after all? This is the one place the seam might be in the
  wrong spot.
* If douse hosts both inverse problems and forward optimization, the shape-derivative layer
  should sit at its base rather than inside an `inverse` module.
