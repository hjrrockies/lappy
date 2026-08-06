# Scope: what `lappy` provides, and what a downstream shape package owns

**Status: provisional, 2026-08-06.** This records a design discussion, not a commitment. The
API sketched in §3 is deliberately *not* being built yet — see §5 for why, and what has to
happen first.

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
* `MPSEigensolver.bdry_quad` is exposed precisely "for reuse by code building other boundary
  functionals from the same eigenfunctions (e.g. Hadamard-type shape derivatives)".

The seam is where the mathematics has one: upstream is "solve the PDE, integrate Cauchy data",
downstream is "which functional, which perturbation field, which optimizer".

## 3. What the seam needs before it can be an API

Four items, in rough order of how load-bearing they are.

**Per-node provenance.** To evaluate `V·n` at the quadrature nodes, the downstream package
needs each node's `(seg_idx, tau)`. Today that is only reconstructible by calling the private
`_panel_rule` and redoing the affine map `tau = tau0 + (tau1-tau0)*u`. `BoundaryQuad` should
carry per-node `seg_idx` and `tau` arrays — they are computed anyway inside `assemble_panels`.
A downstream package reaching into private helpers on its first call is the sign of a seam one
field short.

**The orientation convention stays upstream.** A shape velocity enters as `V·n`, and a sign or
orientation error there yields a *plausible, wrong gradient* that an optimizer will follow
happily. `lappy` already owns the normal convention, segment orientation and arclength
parametrization, so it should also own the one converter: given a per-node boundary
displacement `dp` (complex, at the quadrature nodes), return `V·n` using its own normals.
douse computes `dp` from its parametrization; lappy converts. The convention is then enforced
in one place rather than agreed in two.

**Degenerate clusters are part of the contract.** A shape derivative of a multiple eigenvalue
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

## 5. Why the API is not being committed to yet

Two pieces of `lappy` are still moving, and both could change what it is able to promise:

1. **Basis-selection heuristics.** Ten benchmark domains sit in bucket 2 — complete spectrum,
   clean tension curve, under 8 digits — diagnosed as basis insufficiency. Breaking out of
   `make_default_basis` there may change the shape of what a solver needs (column scaling,
   pivoted selection, source placement), and with it what "give me a good eigenfunction" costs
   and how it is configured.
2. **The `domain -> (Eigenproblem + auto-configured Eigensolver)` pipeline.** The reference
   work has been driving solvers by hand (`benchmarks/reference/common.build_solver`
   deliberately bypasses `from_domain`). Folding what was learned back into auto-configuration
   will change the constructor surface that douse would call.

Committing to an interface across that seam now would freeze it against machinery that is
about to change. The four items in §3 are cheap and additive; the *interface* should wait
until the pipeline settles.

## 6. Sequencing

1. Write the §4 contract test. It is a day's work, it is the goal's acceptance criterion, and
   it settles the one open sizing question in the right currency — `smooth_safety` matters for
   `dlambda`, not for `lambda`.
2. Bucket-2 basis research, starting with the six sharp-corner domains (chevrons x4,
   parallelogram_p65/p127) where the mechanism is understood. Treat the four thin-neck domains
   (stadium x2, mushroom_thin, mushroom_neck01) as a separate investigation — likely a
   different mechanism, possibly not a basis problem at all.
3. Polish the auto-configuration pipeline with what (1) and (2) establish.
4. Only then fix the douse interface.

Bucket 3 (the two spirals) stays parked: it is two pathological domains, and the goal is
explicitly "non-pathological".

**Measure basis variants carefully.** Interior collocation points come from the global RNG,
and the record shows `iso_right_tri` returning 4.9, 4.0 and 2.5 certified digits on three runs
of identical code. A trial-and-error basis study must fix the seed *and* report spread across
several, or it will chase draws. The spread is itself a signal: a basis whose accuracy depends
strongly on the interior sample is telling you the system is under-determined.

## 7. Open questions

* Does the eigenvalue-digit objective agree with the `dlambda`-accuracy objective? A basis
  tuned for one may not be best for the other. (1) above is what makes this checkable.
* Where does a `Domain` family's *derivative* geometry live in practice — is
  `ParametricSegment` convenient enough for douse to build `dp/dtheta` on top of, or does it
  end up needing something upstream after all? This is the one place the seam might be in the
  wrong spot.
* If douse hosts both inverse problems and forward optimization, the shape-derivative layer
  should sit at its base rather than inside an `inverse` module.
