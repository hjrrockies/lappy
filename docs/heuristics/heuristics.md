# MPS Settings Strategy: Preliminary Heuristics

This document develops a principled basis for choosing `lappy` solver settings —
primarily `n_basis`, `n_fb`, `n_fs`, `rtol`, `ltol`, and `ppl` — as functions of
the desired accuracy, the number of eigenvalues requested, and the geometry of the
domain. The analysis is preliminary; the benchmark sweeps are designed to refine
the constants and validate the functional forms.

---

## 1. Background: the MPS tension function

The Method of Particular Solutions (MPS) finds eigenvalues as approximate zeros of
a *tension function* σ(λ). Given a basis of N particular solutions
{φ₁, ..., φ_N} — exact solutions to (Δ + λ)u = 0 in ℝ² — the tension at λ is:

```
σ(λ) = min_{‖c‖=1}  ‖∑ cᵢ φᵢ‖_{L²(∂Ω)} / ‖∑ cᵢ φᵢ‖_{L²(Ω)}
```

σ(λ) is small when there exists a linear combination of the basis functions that
nearly vanishes on ∂Ω while remaining non-trivial in the interior — i.e., when λ
is near a true eigenvalue. In the GSVD formulation used here, the tension is the
smallest generalized singular value of the (boundary matrix, interior matrix) pair.

The eigenvalue error is bounded by the tension:

```
|λ̂ - λ| / λ  ≲  C · σ(λ̂)
```

where C depends on the local spectral gap. Achieving relative accuracy `ltol`
therefore requires `σ(λ) ≲ ltol`.

---

## 2. Why convergence is exponential in N

The key theoretical fact underlying MPS is that the tension decays **exponentially**
in N for a well-chosen basis:

```
σ(λ) ~ exp(−γ · N)
```

where γ > 0 is a geometry-dependent convergence rate. This follows from
approximation theory for analytic functions:

**Fourier-Bessel (FB) basis.** At a corner with interior angle α, the Dirichlet
eigenfunctions have the form:

```
u(r, θ) = Σ_{n≥1} aₙ r^{nπ/α} sin(nπθ/α)
```

in local polar coordinates (r, θ) centered at the corner. This is the *exact*
singular form dictated by the geometry. The FB basis at that corner consists
precisely of J_{nπ/α}(√λ · r) sin(nπθ/α), which matches this structure term by
term. The residual after including N terms is therefore the tail of an analytic
series:

```
‖u - u_N‖ ~ C · exp(−γ_corner · N)
```

where γ_corner depends on the analyticity radius of u away from that corner.
Since the eigenfunction is analytic everywhere except at corners, and the FB basis
handles the corner singularities exactly, the remaining approximation error falls
off exponentially fast.

**Fundamental Solution (FS) basis.** Sources placed at distance d outside the
boundary generate functions that are analytic on all of Ω̄. For a smooth domain
(or as a supplement to FB functions on a polygon), N such sources achieve:

```
‖u - u_N‖ ~ C · exp(−γ_fs · N)
```

where γ_fs ~ π d / L for sources distributed at distance d from a boundary of
length L. The convergence rate decreases as d → 0, so source placement matters.

**Consequence for basis sizing.** The tension target `ltol` requires:

```
N  ≳  |log(ltol)| / γ
```

This is a *lower bound* on N from the accuracy requirement alone. As we will see,
the oscillation count often dominates in practice.

---

## 3. The oscillation count argument

The n-th eigenfunction oscillates at spatial frequency √λₙ. Along a boundary
segment of length ℓ, it executes roughly:

```
ℓ · √λₙ / (2π)  oscillations
```

A basis of N functions centered at a single point can accurately represent at most
O(N) oscillation modes. To capture the first k eigenvalues across all boundary
segments, we need the basis to span the oscillations at scale √λ_k. Summing over
all corners/sources:

```
N_oscillations ~ C · P · √λ_k / (2π)
```

where P is the perimeter. Substituting Weyl's law λ_k ~ 4πk/A:

```
√λ_k ~ 2√(πk/A)
```

gives:

```
N_oscillations ~ C · (P/√A) · √k
```

The factor P/√A is the **isoperimetric ratio**: it equals 2√π ≈ 3.54 for a
circle and grows for elongated or geometrically complex domains. A square has
P/√A = 4; an L-shape has P/√A ≈ 5–6 depending on proportions; GWW domains have
P/√A ~ 7–9.

---

## 4. The combined N_basis heuristic

The two constraints — accuracy and oscillation count — combine as:

```
N_basis  ~  C · (P/√A) · √k · F_geom · f(ltol)
```

where:

**f(ltol):** The accuracy factor. Empirically, once N exceeds the oscillation
count, the tension drops rapidly. The relationship is not purely linear in
log(1/ltol) because the exponential convergence means each additional basis
function buys a fixed multiplicative improvement. A reasonable approximation is:

```
f(ltol) = max(1,  α · |log(ltol)|^β)
```

with β ∈ [0.5, 1.0]. The benchmark data shows that going from ltol = 1e-10 to
1e-16 requires only ~30–50% more basis functions, which is consistent with β ~ 0.5.
We conservatively use β = 1 (linear) until we have more data.

**F_geom:** The geometric difficulty factor. It accounts for:

1. *Re-entrant corners (α > π):* The singularity exponent π/α < 1 means the
   eigenfunction is less smooth at the corner. More FB terms are needed to resolve
   it. A rough per-corner factor is (α/π)^(1/2) for α > π, giving:

   ```
   F_corners = ∏_{re-entrant corners i} (αᵢ/π)^{1/2}
   ```

   For example, the L-shape has one 270° corner (α = 3π/2), giving
   F_corners = √(3/2) ≈ 1.22. This is consistent with needing ~120 functions
   for the L-shape vs ~60–80 for the rectangle and triangles.

2. *Irrational corner angles:* If π/αᵢ is irrational, the FB orders needed are
   non-integer Bessel functions with slower convergence than integer-order ones.
   The disk sector at θ = π√2/2 is the clearest example in the benchmark data;
   it struggles even at n_basis = 160. For irrational angles, the convergence
   rate γ is degraded and a larger empirical multiplier is needed.

3. *Smooth curved boundaries:* Domains with curved sides but no corners (disk,
   ellipse, stadium) need no FB functions. The FS convergence rate depends on
   source placement distance d relative to the inradius r_in:
   γ_fs ~ π · d/r_in / P_normalized. A source distance of d ≈ 0.15 · r_in is
   the current default; the fs_placement sweep will quantify the optimal value.

**Putting it together.** A working preliminary heuristic is:

```
N_basis  ≈  round( C · (P/√A) · √k · F_geom · |log₁₀(ltol)| )
```

with C ≈ 2–4 (to be fit from the benchmark data). The log base-10 is used for
interpretability; it differs from the natural log by a factor of 2.303.

For the domains tested so far, rough estimates:

| Domain         | P/√A | F_geom | Predicted N (k=10, ltol=1e-14) | Observed N |
|----------------|------|--------|-------------------------------|------------|
| rect (2×1)     | 4.2  | 1.0    | ~55–70                        | ~60        |
| iso_right_tri  | 4.6  | 1.0    | ~60–75                        | ~70        |
| eq_tri         | 4.6  | 1.0    | ~60–75                        | ~60        |
| L_shape        | 5.5  | 1.22   | ~85–110                       | ~120       |
| GWW1           | ~8   | ~1.5   | ~150–200                      | >160 (tbc) |

The predictions are in reasonable agreement with observations, though the
constants need refinement.

---

## 5. Role of n_eigs

The number of requested eigenvalues k enters through the √k scaling of the
oscillation count. Key implications:

**Doubling n_eigs requires ~√2 ≈ 41% more basis functions,** not double. This
is a sublinear cost growth and is one of the attractive properties of MPS vs.
finite element methods (which typically scale as O(k) or worse).

**The basis must be sized for the largest eigenvalue requested.** If you ask for
k = 25 eigenvalues and use a basis sized for k = 10, the solver will reliably
find the first few eigenvalues but miss or inaccurately compute the upper range.
This is the open issue 2.7 (adaptive basis sizing): currently `evp.solve(k)` uses
whatever basis was constructed at initialization, regardless of k.

**Practical rule of thumb:**

```
N_basis  ≥  N_min(k) = round( C · (P/√A) · √k · F_geom )
```

where C ≈ 4–6 as a conservative starting point. When in doubt, err on the high
side: the cost of adding basis functions scales as O(N² · n_pts) for the GSVD,
so there is a penalty, but it is not catastrophic for moderate N.

**The ppl parameter** (points per level in the bracket search) also interacts with
k. The solver searches for k tension minima on [λ_min, λ_max]. With too few
points per level (ppl=5), eigenvalues can be missed entirely — this is what causes
the widespread failures in the benchmark data at small n_basis. The recommended
minimum is ppl=10; for difficult domains or large k, ppl=15–20 is safer.

---

## 6. The FB/FS split

Given a total budget of N_basis functions, how should they be divided between
Fourier-Bessel (n_fb) and Fundamental Solutions (n_fs)?

**Pure polygon, no curved sides:** FB only (n_fs = 0). The FB basis exactly
matches the singularity structure; FS functions add global smooth corrections that
are redundant once the FB orders are high enough.

**Pure smooth domain (no corners):** FS only (n_fb = 0). There is no singularity
structure for FB to capture.

**Mixed (polygon with some curved sides):** Use both. The FB functions handle
corner singularities; the FS functions supplement smooth regions.

**FB order allocation across corners.** With a total of n_fb functions distributed
across n_c corners, the `fb_strategy` controls allocation:

- `singular_angle_weighted` (default): allocates more orders to corners with larger
  (more re-entrant) angles, weighted by angle. This is the theoretically motivated
  choice since more singular corners require more FB terms.
- `uniform`: equal orders per corner. Suboptimal when corner angles vary widely.
- `singular_only`: all orders at non-regular corners (those where π/α is not an
  integer), zero at regular corners. Aggressive but correct in theory for polygons
  with a mix of regular and singular corners.

The `fb_corner_sweep` will quantify the accuracy difference between strategies.
Preliminary expectation: for domains with one dominant singular corner (like the
L-shape), `singular_only` and `singular_angle_weighted` should outperform
`uniform`.

---

## 7. Collocation points

The boundary and interior collocation points affect both accuracy and stability,
but are less sensitive than basis size.

**Boundary points (bdry_pts_factor):** The system is overdetermined: we use
`bdry_pts_factor × N_basis` boundary points. A factor of 2.0 (default) is
standard for MPS and gives good conditioning. Going below 1.5 risks an
underdetermined system; going above 3.0 adds cost without benefit.

The points should be distributed proportionally to segment length (the current
default). Gauss-Legendre quadrature points (the `'legendre'` option) are
preferable to uniform spacing because they avoid endpoint clustering artifacts.

**Interior points (int_pts_factor):** Used for basis normalization (L² norm
over Ω). A factor of 1.0 is typically sufficient; the normalization is not
sensitive to this choice as long as the points are reasonably distributed.

---

## 8. Regularization: rtol and ltol

The GSVD-based MPS involves two tolerance parameters with different roles:

**rtol** (regularization tolerance): singular values of the basis matrix below
`rtol × σ_max` are discarded. This controls ill-conditioning from near-linearly-
dependent basis functions. Too small: numerical noise dominates and the tension
function becomes noisy, causing missed or false eigenvalues. Too large: legitimate
but small singular values (corresponding to weakly excited modes) are discarded,
causing missed eigenvalues.

From the `regularization_sweep` data (preliminary): the sweet spot appears to be
rtol ∈ [1e-12, 1e-14] for well-conditioned domains. The GWW domain degrades
significantly at rtol < 1e-12, suggesting the basis becomes ill-conditioned at
large N for complex geometries.

A geometry-motivated heuristic for rtol:

```
rtol_suggested  ~  C_r · exp(−γ · N_basis)
```

i.e., set rtol to the expected tension floor for the given basis size and geometry.
In practice, rtol = 1e-14 is a safe default; tighten to 1e-16 only if you have
verified the basis is well-conditioned.

**ltol** (eigenvalue acceptance threshold): a tension minimum is accepted as an
eigenvalue only if σ(λ̂) < ltol. This should be set relative to the accuracy you
expect the basis to achieve:

```
ltol  ~  10 × exp(−γ · N_basis)  ~  10 × σ_floor(N_basis)
```

Setting ltol too tight causes real eigenvalues to be rejected. Setting it too loose
accepts noise peaks as eigenvalues. The current default of ltol = 1e-15 is
aggressive; for small bases (N < 50) or complex domains, ltol = 1e-10 to 1e-12
is more reliable.

**The rtol/ltol relationship:** Both tolerances should be set consistently. A
rough guideline:

```
ltol  ≈  100 · rtol
```

This ensures that basis regularization doesn't create artificial tension minima
above the eigenvalue acceptance threshold.

---

## 9. Practical recommendations (preliminary)

Until the benchmark sweeps are complete, the following settings are conservative
starting points:

### Step 1: estimate N_basis

```python
import numpy as np

def estimate_n_basis(domain, k, ltol=1e-14):
    P = domain.perim
    A = domain.area
    iso_ratio = P / np.sqrt(A)
    # geometric difficulty from re-entrant corners
    angles = domain.int_angles
    F_geom = np.prod([np.sqrt(a / np.pi) for a in angles if a > np.pi]) or 1.0
    C = 5.0   # conservative constant; refine from benchmarks
    return round(C * iso_ratio * np.sqrt(k) * F_geom * abs(np.log10(ltol)))
```

### Step 2: set FB/FS split

- Polygon with all straight sides: `n_fs = 0`, `n_fb = N_basis`
- Smooth domain: `n_fb = 0`, `n_fs = N_basis`
- Mixed: `n_fb = round(0.5 * N_basis)`, `n_fs = N_basis - n_fb`

### Step 3: set tolerances

| Use case                    | rtol   | ltol   | ppl |
|-----------------------------|--------|--------|-----|
| Quick / exploratory         | 1e-10  | 1e-8   | 10  |
| Standard                    | 1e-12  | 1e-10  | 10  |
| High accuracy               | 1e-14  | 1e-12  | 15  |
| Near-degenerate eigenvalues | 1e-14  | 1e-12  | 20  |

### Step 4: check the result

A result should be treated as reliable if:
- `len(eigs) == k` (all requested eigenvalues found)
- `max(tensions) < ltol` (all eigenvalues accepted with margin)
- `median(rel_errors) < ltol` (when reference is available)

If `len(eigs) < k`, increase `n_basis` by 25–50% and/or increase `ppl`.

---

## 10. Open questions for the benchmark data

The following will be addressed as more sweep results come in:

1. **The constant C.** The current range C ≈ 2–6 is too wide. The `basis_size`
   sweep will pin this down per domain class.

2. **The ltol exponent β.** Is f(ltol) ~ |log(ltol)| or ~ |log(ltol)|^{1/2}?
   The `regularization_sweep` data will answer this.

3. **Optimal FB/FS ratio for mixed domains.** The `basis_size_sweep` covers
   fb_fraction ∈ {0, 0.25, 0.5, 0.75, 1.0}; we expect a clear optimum around
   0.5–0.75 for polygons.

4. **FS source distance scaling.** The `fs_placement_sweep` will determine how
   accuracy depends on `fs_d_scale`. The current theoretical prediction is
   γ_fs ~ π · d / r_in; the data will test this.

5. **The irrational angle case.** The disk_sector at θ = π√2/2 is a red flag.
   We need to understand whether this is a basis sizing issue, an FB order
   allocation issue, or a fundamental limitation.

6. **n_eigs scaling exponent.** The √k prediction from Weyl's law should be
   directly testable from the `n_eigs_sweep`.
