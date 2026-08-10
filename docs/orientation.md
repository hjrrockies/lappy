# Orientation: boundary integrals in lappy

**Purpose.** A map, not a lab notebook. `benchmarks/suite/run/NOTEBOOK.md` (2700+ lines)
records *how* things were found; this records *what is true now* and *where the difficulty
lives*. Read this first; go there for evidence.

Status: 2026-08-10.

---

## 1. Why boundary integrals at all

Two formulas, one instrument:

```
Rellich    ∫_Ω u² dx  =  (1/2λ) ∮_∂Ω (r·N) (∂u/∂n)² ds        r = x - x0
Hadamard   dλ         =  -      ∮_∂Ω (∂u/∂n)² (V·n)  ds        V = boundary velocity
```

Both turn a statement about the *domain* into an integral over the *boundary*, against the
same Cauchy data `∂u/∂n`, on the same nodes, with the same weights. Only the weight function
differs — `r·N` for the norm, `V·n` for the shape derivative.

This matters because MPS produces `∂u/∂n` on the boundary natively. So:

* **normalizing an eigenfunction** (needed for anything quantitative) and
* **differentiating an eigenvalue with respect to shape** (needed downstream, for optimization
  and inverse problems)

are the *same computation with a different weight*. One quadrature investment buys both. The
alternative — interior cubature for the norm — needs volume points and has measured as the
fragile half every time the two were compared.

The `V = x - x0` case (uniform dilation) makes Hadamard reduce to Rellich exactly, which pins
the sign and scale of the shape derivative for free. `gram()` and the shape derivative are
literally the same call to `weighted_integral(ed, 'NN', ·)` with different weights.

**Consequence:** essentially all of lappy's accuracy risk is concentrated in *one boundary
quadrature*. That is a deliberate bet, and it is working. It also means a defect there is
systemic, which is why so much effort has gone into it.

---

## 2. Where the difficulty actually lives

At a corner of interior angle `α`, with `ν = π/α`, a Dirichlet eigenfunction expands as
`u = Σ c_k J_{kν}(√λ r) sin(kνθ)`, so on an edge leaving the corner

```
∂u/∂n  ~  r^(ν-1) F(r)          (∂u/∂n)²  ~  r^(2ν-2) G(r)
```

The integrand carries a **non-integer power of arclength** at every corner. Gauss–Legendre
converges algebraically on `τ^γ` (as `n^-(2γ+2)`), not spectrally, so an unadapted rule loses
most of the precision. Everything below is machinery for that one fact.

Note carefully: this is **not** only about reentrant corners. `r^(2ν-2)` is non-smooth
whenever `2ν-2` is not a non-negative even integer — which includes plenty of *convex* corners
(`ν = 1.4` gives `r^0.8`). That is why the regular n-gons need corner rules at all.

---

## 3. The case table

This is the thing to keep in your head. `boundary_quadrature()` classifies every corner, then
every panel.

### 3a. Which rule a corner gets

The test for "does this corner need special treatment" is **measured, not assumed**:
`quad.smooth_power_error(2ν-2, order) > precision` asks directly whether plain Gauss can
integrate that power to the target. (`nonintegral=True`, on by default.)

| corner class | edges | rule | `sub` | why |
|---|---|---|---|---|
| smooth rule suffices | either | **Legendre** | — | `smooth_power_error` says plain Gauss reaches the target; nothing to fix |
| `2/ν` an integer | **straight** both sides | **`cornerjac`** | `ν` | family is the sparse `{jν + 2q}`; `t = r^ν` rationalizes it. Among reentrant angles this is **only α = 3π/2** |
| everything else, incl. irrational `ν` | curved, or straight with `2/ν` non-integer | **`cornerinterp`** | `ν` | family is the dense `{jν + m}`; no substitution rationalizes it without crushing nodes, so exactness comes from the *weights* while nodes keep the mild `sub = ν` placement |
| `ν ≤ 1/2 + margin` (near-slit) | either | **demoted to Legendre** | — | the Rellich integrand is barely integrable; no rule recovers it. Reported precision is `inf`. Remedy: put `x0` on that corner (§4) |
| mixed BC (Zaremba corner) | either | **demoted** | — | integrand `~ r^(ν-2)` is not integrable for `ν<1`; the identity itself fails |
| non-Dirichlet `bc_type` | either | **demoted** | — | only Dirichlet corners are wired; Neumann needs different exponents on the `uv`/`TT` kernels |

Demotions are never silent: `shortfalls` names the corner and the reason, `bq.precision`
becomes `inf`, and a warning is raised. `singular_corner_report(domain)` prints the whole
classification and is the first thing to look at when accuracy disappoints.

### 3b. Which weight you are integrating

The corner rule is built for the eigenfunction's own exponent family. A **weight** multiplies
that, and whether the product stays in the exact class depends on the weight's behaviour at
the corner:

| weight | corner behaviour | in the default class? | what to pass |
|---|---|---|---|
| `r·N`, straight edge | exactly **constant** (`m=0`) | yes | default `weight_family='even'` |
| `r·N`, curved edge | analytic series in `s`, integer powers | yes | default |
| `V·n` vanishing to even order | even integer powers | yes | default |
| `V·n` **moving a corner** (`~ r`) | **odd** integer powers | **no** | `weight_family='integer'` |

The last row is the one that bites, and it is exactly what a shape-optimization parametrization
supplies. With the default `sub = ν`, a weight `r^m` becomes `t^(m/ν)` — non-integer with a
*small* exponent, so Gauss decays only as `n^-(2m/ν+2)`. Measured on a 1.5π sector against
40-digit truth:

```
                    p=0       p=1       p=2       p=3
sub = ν,  order 32  2.9e-14   4.6e-07   8.5e-14   4.0e-14
sub = 1/2, order 16 4.7e-15   1.2e-14   1.1e-14   1.0e-14
```

`weight_family='integer'` switches singular corners to `sub = 1/2`, which makes every integer
power the exact polynomial `t^(2m)` while sending the Bessel family to `t^(2jν)` — non-integer,
but with exponents growing by `2ν` per term, which Gauss resolves at once. It assumes nothing
about `ν`, so it covers the generic arc–arc corner where no exact substitution exists.
End-to-end on `dλ/dα` this is 6–8 orders, at fewer nodes in half the cases.

Default stays `'even'`: it is equally accurate and cheaper for everything lappy integrates
itself, and changing it would perturb every recorded reference value.

### 3c. The smooth part of the boundary

Away from corners, panels use plain Gauss–Legendre, with two independent sizing questions:

| question | function | notes |
|---|---|---|
| resolve the *oscillation* `e^{ikτ}` | `smooth_order_for_precision(k, precision)` | `k = √λ_max · panel arclength` |
| resolve the *geometry* | `geometry_order_for_precision(seg, ...)` | curved segments only; asks whether the arclength reparametrization itself is resolved. `resolve_geometry=True` by default |

---

## 4. The methods, one at a time

**`cornerjac`** — Gauss–Jacobi after the substitution `t = r^sub`. Exact when the substitution
maps the corner's exponent family to integers. Cheap (order 8 suffices at α=3π/2 where
`cornerinterp` needs 24). Its limitation is node placement: `τ = t^(1/sub)` crushes the
innermost node as `sub` shrinks, and once `τ_min` falls below `_KRESS_TAU_FLOOR = 1e-9` the
node rounds onto the corner itself under a segment's parametrization, where a basis's `1/r`
terms are fatal. That is why `sub = 1/q` for `ν = p/q` — which would rationalize the dense
family exactly — is *not* used for placement: `q ≥ 4` puts `τ_min` below the floor.

**`cornerinterp`** — nodes from `cornerjac`'s mild `sub = ν` placement, weights solved so the
rule is exact on the corner's actual exponent set. Gets the exactness from the weights instead
of the coordinates, so `τ_min` stays at ~1e-5. The weight solve is deliberately
*under-determined* (`n_exp < order`, minimum-norm least squares), which keeps `Σ|w| = 1`; the
square solve is exact on the span but has `cond(V) ~ 1e19` and weights reaching `-1e4`.

**Known limit, measured this session:** `cornerinterp` cannot be pushed further by raising
`n_exp` — `cond(V)` goes 8e6 → 5.7e15 → 4.4e18 — and it is *not* an arithmetic problem:
rebuilding the same rule at 60 dps makes it worse, with `Σ|w|` exploding to 4e10. The Jacobi
nodes are the wrong nodes for the dense family. Fixing that properly means solving for nodes
*and* weights jointly (true generalized Gaussian quadrature, Ma–Rokhlin). Not attempted, and
`sub = 1/2` (§3b) made it unnecessary for the case that motivated it.

**Panel structure.** Corner panels are anchored at the corner and cover a fraction
`panel_frac` of the segment; that fraction is halved when both ends of a segment are singular.
A *clearance cap* limits the panel to a fraction of the distance to the nearest non-adjacent
boundary piece, because the corner expansion stops being valid beyond it.

**`x0`.** The Rellich identity holds for *every* `x0`, which makes `x0`-invariance a free
diagnostic: any variation across `x0` is pure quadrature error. `default_x0` puts it at the
most singular corner, where `r·N` vanishes identically on both edges — removing that corner
from the integrand rather than resolving it. Free, but it can only ever cover one corner.

---

## 5. The two ways to know it worked — and the open decision

| | mechanism | what it is |
|---|---|---|
| **a priori** | `corner_model_error` → `corner_order_for_precision` → `bq.precision` | *predicts* a rule's error from a model of the integrand class, before computing anything |
| **a posteriori** | `refine_quadrature` → `verify_gram` | *measures*, by recomputing on a refined rule and reporting what moved |

**This is the live architectural question.** `bq.precision` is currently treated as an
advertised precision, but it is a single scalar standing in for a whole class of integrands,
and it is measurably not up to that job:

* Calibrated against closed-form truth over 3822 configurations
  (`benchmarks/eigfun_quad/corner_model_calibration.py`), worst-case **optimism is nine
  orders** — it promises precision it does not deliver. It is also pessimistic by up to 3e5x
  at convex `ν`, which inflates node counts. Two causes pulling opposite ways: unsigned Bessel
  coefficients (pessimistic) and a fixed series depth that cuts at the peak once `k > 8`
  (optimistic). `k` reaches 10.6 across the suite, so both are live.
* Three candidate replacements were built and **all three rejected on measurement** — each
  fixed one regime and broke another. The instrument landed; the fix did not.
* More fundamentally: `chevron_1_2` claims `1e-13` while `verify_gram` measures `4.9e-08`, and
  that gap is *identical* under every corner model tried. The error is not at the corners — it
  comes from the smooth panels, whose requirement is set by the **basis's own singularity
  structure** (where the fundamental solutions put their poles), which no geometry-only model
  can ever see.

So a perfect corner model still would not deliver the advertised number. The open proposal is
to **demote `bq.precision` to a sizing heuristic and let `verify_gram` be what certifies**.
The corner model then only needs to land in the right ballpark, and its calibration leaves the
critical path. Not yet decided.

---

## 6. State of play

**Settled and load-bearing**

* Certification runs on boundary norms (`eps` is scale-invariant, so this is sound and faster).
* `FundamentalBasis` rejects sources inside the domain — those columns are not particular
  solutions, which voids the MPS premise *and* the Moler–Payne hypothesis. This was a real
  correctness bug; its signature is a suspiciously *low* tension background.
* `weight_family='integer'` for corner-moving shape velocities (§3b).
* `tests/test_shape_derivative.py` — the Hadamard contract, 12 tests, ~2s.
* Benchmark tally 39 / 3 / 2 (8+ digits / complete-but-fewer / failure).

**Open**

* Corner order model calibration — instrumented, unsolved (§5).
* Claimed vs measured precision — the architectural decision above.
* Basis-selection heuristics: `make_default_basis(domain, lam_max, precision)` is sketched in
  `docs/basis_heuristics.md`, not built.
* Buckets 2 and 3: `stadium`, `mushroom_neck01`, and the two spirals.
* Still unbuilt from the Hadamard plan: the Tier-3 finite-difference benchmark and the
  objective study (λ digits vs dλ digits across bases).

---

## 7. Where to read further

| topic | file |
|---|---|
| the identity and its derivation | `docs/rellich.md`, `docs/rellich_hadamard_mps.pdf` |
| corner quadrature theory | `docs/corner_quadrature.pdf` |
| gentler introduction, for someone new | `docs/boundary_quadrature_primer.pdf` |
| the integration API | `docs/eigfun_integrals.md`, `docs/boundary_integrals.md` |
| what lappy owns vs a downstream shape package | `docs/scope_and_downstream.md` |
| evidence, chronologically | `benchmarks/suite/run/NOTEBOOK.md`, `FINDINGS.md` |
| per-domain status | `benchmarks/suite/run/BUCKETS.md` |
