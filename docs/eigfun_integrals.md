# `lappy.eigfun_integrals` — boundary integrals of eigenfunctions

Supersedes the basis-level architecture of `rellich_hadamard_mps.pdf` and the
implementation notes in `rellich.md`. The mathematics in `rellich.md` (the master
identity and its Zaremba specialization) is unchanged and still the reference for
the identity itself; what changed is *what the identity is applied to* and *how the
boundary integral is discretized*.

## What this replaces, and why the replacement is narrower

`lappy.rellich` and `lappy.cauchy` are deleted. They computed a **basis-level**
`N × N` Gram matrix and, in the original design, sandwiched it between raw GSVD
coefficient vectors. Both the sandwich and the basis-level Gram are gone.

The narrowing is not tidying — it is what makes the corner quadrature possible.
Near a corner of interior angle `α`, with `ν = π/α`, an exact Dirichlet
eigenfunction's local expansion is *complete* (Kondrat'ev):

```
u = Σ_k c_k J_{kν}(√λ r) sin(kν θ)     ⟹     ∂u/∂n = r^(ν-1) F(r)
```

so `(∂u/∂n)²  = r^(2ν-2) G(r)`, and `G`'s exponents form a known family. That is
exactly the class the corner-adapted rule integrates. A basis-level Gram is **not**
of that form: its columns centred at *other* corners are plain analytic here, with
`O(1)` amplitude at every corner simultaneously. Measured, the corner rule beats the
old graded rule by 3–8 orders on corner-family blocks and *loses* by 2–4 on mixed
ones. There is no single node set that serves both, so the basis-level consumer was
retired rather than adapted.

Two consequences follow for free:

* **The node set needs no basis.** It is a pure function of geometry, `lam_max` and
  a requested accuracy — so it is built once and reused for every `λ` in a search.
* **The seven tuning parameters collapsed to one.** `mult`, `margin`, `q_min`,
  `q_max`, `c_lam`, `beta` and `x0` became `precision` (plus an optional `x0`).

## The API

```python
bq  = boundary_quadrature(domain, lam_max, precision=1e-13)   # geometry only
ed  = eigfun_cauchy_data(basis, lam, coef, bq)                # evaluate first
G   = gram(ed, lam, bq)                                       # (mult, mult)
D   = lowdin_transform(G)                                     # orthonormalizing transform
A   = weighted_integral(ed, kernel, weight)                   # generalized / Hadamard-type
```

`MPSEigensolver.from_domain(domain, basis=basis)` builds `bq` itself, so
`eigenfunction_coef` returns L²-orthonormal coefficients with nothing configured by
the caller. `orthonorm=False` opts out; `orthonorm_precision` and `orthonorm_x0`
are the only knobs.

`weighted_integral` keeps the four bilinear Cauchy-data kernels (`'uv'`, `'NN'`,
`'TT'`, `'cr'`) and an arbitrary boundary weight — the extension point for
Hadamard-type shape derivatives — but only ever at eigenfunction scale. lappy
implements no such formulas itself.

### "Evaluate first, sandwich never"

`eigfun_cauchy_data` contracts the basis with `coef` *before* any integral is
formed. The alternative — assembling `G` over the basis and computing
`coefᵀ G coef` — multiplies `G`'s independently-rounded error through `coef` on both
sides, which is how an ill-conditioned basis produced norms off by `1e-4`.

## Accuracy, and where it runs out

Validated against exact truth in four independent ways
(`tests/test_eigfun_integrals.py`, `tests/test_orthonormalization.py`):

| leg | what it certifies | result |
|---|---|---|
| 1 | single singular corner, end-to-end, exact sector eigenfunctions | ≤1e-13 for α ≤ 1.5π; 5.7e-12 at 1.75π; 2.1e-9 at 1.9π |
| 2 | multi-corner geometry, exact polyomino eigenfunctions — **no singular amplitude** | <1e-12 on 4 reentrant corners |
| 3 | multi-corner **with** singular amplitude, closed-form reference | H_shape's 8 corner panels, worst 1.6e-14 |
| 4 | x0-invariance on real MPS eigenfunctions (reference-free) | 9.3e-13 on L_shape, 2.6e-11 on H_shape |

Node counts at `precision=1e-13`: L_shape 116, H_shape 252, plus_shape 232,
`disk_sector(1.5π)` 70.

**Known limits.** Accuracy degrades as `ν → 1/2` (the slit), where the integrand
stops being integrable at all — expect ~1e-9 at α=1.9π and treat α→2π as out of
reach; place `x0` on such a corner instead, which removes its contribution exactly.
Pure-Neumann singular corners fall back to a smooth rule (their `'uv'` and `'TT'`
kernels need different exponents, and one shared node set cannot match both). Mixed
Dirichlet/Neumann **reentrant** corners are out of reach *on principle*: the
exponents are `(k+½)ν`, so `(∂u/∂n)² ~ r^(ν-2)`, not integrable for ν<1 — the
identity itself diverges there, for any `x0` and any quadrature.

**Curved boundaries** are supported, including two arcs meeting at a singular
corner and a corner where a straight edge meets an arc. One caveat is measured
rather than assumed: the scheme assumes `|dp/dτ| = seg.len` identically, which holds
algebraically for `LineSegment` and, for `ParametricSegment`, via an adaptive
arclength table. A circular arc is machine-exact at any `tol` (its arclength map is
linear), but a curve of varying speed is limited by that table — 6e-4 even at
`tol=1e-7`, with cost rising 10× per decade. That floor sits under *any* boundary
rule, not just this one; `_parametrization_quality` reports it.

## Three things that will look like bugs

1. **`sum(w) ≠ 1`** for a corner panel. The rule is exact on `{γ + jν + m}`, which
   does not contain `t^0`, so it does not integrate constants. Renormalizing would
   destroy exactness on the singular class the rule exists for. Consequently
   `sum(wts) == perimeter` is an invariant only where every corner panel uses the
   substitution rule.
2. **Accuracy is not monotone in order.** Past a `ν`-dependent threshold the
   integrand's dynamic range makes roundoff dominate. Order selection scans a
   calibrated curve (`corner_model_error`); raising the order past the optimum
   returns a *worse* rule.
3. **Panel length is capped geometrically**, at `0.9 ×` the corner's clearance.
   Removing that cap costs up to twelve orders on a domain whose edge is long
   relative to the clearance, and a fixed *fraction* of the edge cannot substitute.

## Choosing an instrument, if you extend this

Three measurement failures cost real time here and are worth inheriting:

* **`mpmath.quad` is not a valid reference** for these integrands. As `ν → 1/2` the
  endpoint behaviour approaches `r^-1` and it errs by 4e-2 even at 40 digits,
  manufacturing convincing but entirely spurious plateaus. Every reference in the
  tests is closed-form.
* **An error indicator that probes only what a rule integrates exactly reports
  machine precision for everything.** The first one read 4.2e-15 for a rule whose
  true error was 1.5e-10.
* **x0-invariance cannot see quadrature error below the eigenfunction's own
  residual.** Its spread came back identical to three significant figures across
  every panel configuration while the true error varied by twelve orders. It is a
  diagnostic for domains with no analytic truth, not a certificate.

And two modelling traps, both of which made a *correct* rule look broken:

* A synthetic corner model must be a genuine member of the class. Superposing two
  corner series (one per edge end) produces cross terms at `γ/2 + m`, outside any
  single-corner class — an integrand no eigenfunction has.
* On an edge singular at both ends, the two representations are asymptotic
  expansions about *different* points; no single closed form is sparse-in-ν about
  both. Score each panel against the expansion valid on it.

A useful sanity check on any such model: at ν=2/3, `kν − 1 + 2q = 0` has no
solution in non-negative integers, so a genuine `∂u/∂n` at a 270° corner has **no
constant term**.

## What `precision` promises, and where it cannot be kept

`precision` is honoured for the *integrand class* the rule is built for, on a boundary whose
parametrization the rule can resolve. Three things break that, and all three are now measured
rather than assumed. Two are fixed; the third is reported.

**1. Corners the smooth rule cannot integrate (fixed, `nonintegral=True`).** The criterion for
"needs the corner-adapted rule" was `nu < 1`, i.e. reentrant only, justified as "a smooth rule
is already exact" otherwise. It is not. The Rellich integrand carries `r^(2ν−2)`, so what
matters is whether that exponent is an *integer*: a 135° corner (`ν = 4/3`) puts `r^(2/3)` on
its edges, which Gauss–Legendre integrates algebraically while the analytic sizing model cannot
see it at all. `quad.smooth_power_error` now asks the question directly instead of guessing —
which also avoids the opposite error, since `ν = 6.78` gives `r^11.55` and is already smooth
enough that forcing a corner rule on it makes things *worse*.

    domain            nodes        before        after
    right_trapezoid    84/212     1.8e-06      2.9e-16
    GWW1              204/440     3.2e-05      5.0e-13
    iso_tri_h05        96/224     1.2e-11      4.8e-16

**2. The parametrization (fixed for mild cases, `resolve_geometry=True`).** See the module
docstring's table: an ellipse fails `∮(r·N)ds = 2|Ω|` — geometry alone, no eigenfunction — at
the node count the oscillation model asks for. Sizing on that closed form as well takes
`ellipse_a2` from 1.0e-06 to 3.9e-16 and `ellipse_a4` from 9.2e-06 to 7.1e-16, and costs
exactly nothing on any exactly-parametrized boundary.

**3. The eigenfunction's own bandwidth (not fixed; use `verify_gram`).** `stadium` is
unaffected by both of the above: its geometry is exact and its corners are integral, yet its
integrals sit at 1.5e-04 because the boundary data has far more harmonics than
`2 sqrt(lam) * arclength` predicts. The model is unreliable in both directions here — a
circle's Cauchy data needs 4 harmonics where the model assumes 64, an ellipse's needs 80 where
it assumes 8 — and no geometry-only rule can know this, because it is a property of the
solution. `smooth_safety` is the blunt lever (stadium 1.5e-04 → 1.0e-08 at 3×); `verify_gram`
is the honest answer, measuring what a node set achieved on the integrand actually in hand.

### Some problems are simply hard, and lappy says so instead of pretending

Near-slit reentrant corners, eccentric ellipses and coiled spirals are not defects to be
engineered away — they are outside what any fixed-order rule delivers, and the design goal is
that they are *visible*:

* `BoundaryQuad.precision` is no longer a claim contradicted by measurement. A singular corner
  demoted to a smooth rule (`sector_slit`, ν = 0.504, measured 6.9e-01 against a reported
  1e-13) sets it to `inf`.
* `BoundaryQuad.shortfalls` names each corner or segment that fell short and why.
* An eccentric ellipse warns that the *parametrization* is the limit, so the reader does not
  go looking for a bug in the eigenfunction.

### Choosing an instrument, revisited

`docs`' earlier advice that x0-invariance "cannot see quadrature error below the
eigenfunction's own residual" was too pessimistic in one direction and too optimistic in the
other. Holding the node set fixed and sweeping `n_basis` 60→480 moves `sup|u|` by six orders
(9.3e-10 → 1.0e-15) and leaves the spread at 1.68e-06, three significant figures identical — so
it is not residual-dominated. But it *is* contaminated by the **eigenvalue's** error, since λ
enters through `1/(2λ)` and the Cauchy data: on `L_shape` the spread reads 3.3e-12 where the
quadrature is 1.4e-13, and on `H_shape` it reads 4.1e-11 where the quadrature is 1.0e-08.

Prefer `verify_gram` (refine the rule, difference the integral). It measures the quadrature and
nothing else. Keep the x0-spread as a free cross-check — it is what caught this in the first
place — but do not read it as a quadrature error bar.
