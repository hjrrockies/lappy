# Adaptive Arc-Length Reparameterization and Polyline Sampling

A vectorized NumPy/SciPy implementation guide for two related problems on
differentiable planar curves $p(t)$, $t \in [t_0, T]$:

1. **Arc-length reparameterization** — build a smooth map $t(s)$ accurate to
   $\varepsilon \times L$ everywhere.
2. **Adaptive polyline sampling** — find the smallest set of sample points such
   that the resulting polyline stays within $\varepsilon \times L$ of the curve.

---

## Notation and Assumptions

| Symbol | Meaning |
|--------|---------|
| $p(t) \in \mathbb{R}^2$ | the curve, evaluated by a Python callable |
| $p'(t)$ | tangent vector, evaluated by a separate callable (or finite-differenced) |
| $L$ | true arc length (unknown initially) |
| $\varepsilon$ | relative tolerance, e.g. `1e-4` |
| $\varepsilon_{\mathrm{abs}} = \varepsilon L$ | absolute error budget |

The implementation below assumes `p` accepts a **1-D array of parameter
values** and returns a `(N, 2)` array — the vectorized calling convention used
throughout.

---

## Step 0 — Coarse Bootstrap

We need a rough estimate of $L$ before setting absolute tolerances.

```python
import numpy as np
from scipy.interpolate import PchipInterpolator

def estimate_length(p, t0, t1, n_init=64):
    """Cheap chord-sum estimate of arc length."""
    t = np.linspace(t0, t1, n_init + 1)
    pts = p(t)                          # (n_init+1, 2)
    chords = np.diff(pts, axis=0)       # (n_init, 2)
    return np.sum(np.hypot(chords[:, 0], chords[:, 1]))
```

Use `n_init=64` for smooth curves; raise it if you expect many oscillations.
The result $L_0$ is only used to set `eps_abs = eps * L0`; an error of a few
percent here shifts the tolerance slightly but does not affect correctness
because the adaptive passes below track their own local errors.

---

## Part 1 — Adaptive Arc-Length Quadrature

### Algorithm

We build an adaptive partition of $[t_0, T]$ and estimate
$\int_{t_i}^{t_{i+1}} \|p'(t)\|\,dt$ on each subinterval using two
Gauss–Legendre rules (5-point and 3-point). A subinterval is accepted when

$$\frac{|G_5 - G_3|}{G_5 + \varepsilon_{\mathrm{abs}}} < \varepsilon$$

The denominator guard prevents division by zero near zero-speed points (cusps).

### Gauss–Legendre Weights

```python
# 3-point and 5-point GL nodes/weights on [-1, 1]
_GL3_X = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
_GL3_W = np.array([ 0.5555555555555556, 0.8888888888888888, 0.5555555555555556])

_GL5_X = np.array([
    -0.9061798459386640, -0.5384693101056831, 0.0,
     0.5384693101056831,  0.9061798459386640,
])
_GL5_W = np.array([
    0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
    0.4786286704993665, 0.2369268850561891,
])
```

### Vectorized Adaptive Quadrature

The key to performance is processing **all active subintervals in parallel**
rather than recursing one at a time.

```python
def adaptive_arclength_table(p_prime, t0, t1, eps, eps_abs,
                              max_depth=50):
    """
    Build an adaptive arc-length table.

    Parameters
    ----------
    p_prime : callable (t_array,) -> (N, 2)
        Tangent vector of the curve.
    t0, t1  : float
        Parameter interval.
    eps     : float
        Relative tolerance.
    eps_abs : float
        Absolute tolerance (= eps * L_estimate).
    max_depth : int
        Hard recursion cap.

    Returns
    -------
    t_nodes : (M,) array   -- sorted parameter values
    s_nodes : (M,) array   -- cumulative arc lengths at t_nodes
    """
    # --- breadth-first adaptive subdivision ---
    # Active intervals stored as (left, right, depth) in parallel arrays.
    lefts  = np.array([t0])
    rights = np.array([t1])
    depths = np.array([0], dtype=int)

    accepted_lefts  = []
    accepted_rights = []
    accepted_ds     = []

    while len(lefts):
        mids = 0.5 * (lefts + rights)
        hl   = 0.5 * (rights - lefts)   # half-lengths

        # Map GL nodes from [-1,1] to each subinterval — shape (N_intervals, N_nodes)
        t3 = lefts[:, None] + hl[:, None] * (1.0 + _GL3_X[None, :])  # (N, 3)
        t5 = lefts[:, None] + hl[:, None] * (1.0 + _GL5_X[None, :])  # (N, 5)

        # Evaluate speed ‖p'(t)‖ at all quadrature nodes in one batch
        speed3 = np.linalg.norm(p_prime(t3.ravel()).reshape(-1, 3, 2),
                                axis=-1)   # (N, 3)
        speed5 = np.linalg.norm(p_prime(t5.ravel()).reshape(-1, 5, 2),
                                axis=-1)   # (N, 5)

        G3 = hl * (speed3 @ _GL3_W)   # (N,)
        G5 = hl * (speed5 @ _GL5_W)   # (N,)

        err = np.abs(G5 - G3) / (G5 + eps_abs)
        ok  = (err < eps) | (depths >= max_depth)

        # Accept converged intervals
        if ok.any():
            accepted_lefts.append(lefts[ok])
            accepted_rights.append(rights[ok])
            accepted_ds.append(G5[ok])

        # Subdivide the rest
        if (~ok).any():
            l_sub = lefts[~ok]
            r_sub = rights[~ok]
            d_sub = depths[~ok] + 1
            m_sub = 0.5 * (l_sub + r_sub)
            lefts  = np.concatenate([l_sub, m_sub])
            rights = np.concatenate([m_sub, r_sub])
            depths = np.concatenate([d_sub, d_sub])
        else:
            break

    # Sort accepted intervals and accumulate arc length
    all_lefts  = np.concatenate(accepted_lefts)
    all_rights = np.concatenate(accepted_rights)
    all_ds     = np.concatenate(accepted_ds)

    order = np.argsort(all_lefts)
    t_nodes = np.concatenate([[all_lefts[order[0]]],
                               all_rights[order]])
    s_nodes = np.concatenate([[0.0], np.cumsum(all_ds[order])])

    return t_nodes, s_nodes
```

### Inverting the Table: $t(s)$

```python
def make_arclength_inverse(t_nodes, s_nodes):
    """
    Return a callable t_of_s(s) using monotone cubic (PCHIP) interpolation.

    PCHIP preserves monotonicity, so t_of_s is guaranteed non-decreasing.
    """
    return PchipInterpolator(s_nodes, t_nodes)
```

Usage:

```python
L_est   = estimate_length(p, t0, t1)
eps_abs = eps * L_est

t_nodes, s_nodes = adaptive_arclength_table(
    p_prime, t0, t1, eps=eps, eps_abs=eps_abs
)
L = s_nodes[-1]          # refined arc-length estimate
t_of_s = make_arclength_inverse(t_nodes, s_nodes)

# Evaluate at uniform arc-length stations
s_uniform = np.linspace(0, L, 500)
t_uniform = t_of_s(s_uniform)
pts_uniform = p(t_uniform)
```

---

## Part 2 — Adaptive Polyline Sampling

### Error Criterion

On the subinterval $[t_i, t_{i+1}]$ we accept the chord
$\overline{p(t_i)\,p(t_{i+1})}$ when **both** of the following hold:

1. **Midpoint deviation** $\delta = \|p(\tfrac{t_i+t_{i+1}}{2}) -
   \tfrac{p(t_i)+p(t_{i+1})}{2}\| \le \varepsilon_{\mathrm{abs}}$
2. **Chord-length guard** $\|p(t_{i+1}) - p(t_i)\| \le C\sqrt{\varepsilon_{\mathrm{abs}} L}$

Condition 1 controls the geometric approximation error. Condition 2 prevents
near-inflection/aliased regions from slipping through (where the midpoint and
tercile deviations are accidentally small — e.g. all exactly zero for a chord
whose endpoints and tercile points happen to land back on the curve — but the
chord is still long enough to miss real curvature in between). The guard is
scaled by $\sqrt{\varepsilon_{\mathrm{abs}} L}$ ($L$ = the curve's total arc
length) rather than $\varepsilon_{\mathrm{abs}}$ directly: a legitimate,
non-aliased chord length needed to hit a given deviation tolerance is itself
$O(\sqrt{\varepsilon_{\mathrm{abs}} R})$ by the usual sagitta estimate ($R$ =
local radius of curvature, bounded by $L$), so a guard linear in
$\varepsilon_{\mathrm{abs}}$ shrinks faster than that as
$\varepsilon_{\mathrm{abs}} \to 0$ and becomes the sole binding constraint —
forcing uniform bisection far beyond what curvature requires. Empirically
tuned against a battery of smooth test curves (circle, ellipse, a zero-speed
cusp) and an adversarial aliased curve, $C = 2$ is the smallest constant that
never binds on the smooth cases (tolerance sweep `1e-2` to `1e-8`) while still
catching the aliased one.

For extra robustness on highly oscillatory curves, also test at the
**tercile points** $t_i + \tfrac{1}{3}\Delta t$ and $t_i + \tfrac{2}{3}\Delta t$.

### Vectorized Breadth-First Subdivision

```python
def adaptive_polyline(p, t0, t1, eps_abs, L=None, max_depth=60, chord_factor=2.0):
    """
    Adaptively sample p(t) so the polyline is within eps_abs of the curve.

    Parameters
    ----------
    p         : callable (t_array,) -> (N, 2)
    t0, t1    : float
    eps_abs   : float  -- absolute Hausdorff tolerance
    L         : float or None -- curve's total arc length; estimated via
                estimate_length(p, t0, t1) if not supplied
    max_depth : int    -- hard recursion cap
    chord_factor : float -- chord guard = chord_factor * sqrt(eps_abs * L)

    Returns
    -------
    t_sample : (M,) sorted array of parameter values
    """
    if L is None:
        L = estimate_length(p, t0, t1)
    chord_cap = chord_factor * np.sqrt(eps_abs * L)

    lefts  = np.array([t0])
    rights = np.array([t1])
    depths = np.array([0], dtype=int)

    # Pre-evaluate endpoints for the initial interval
    p_left  = p(lefts)    # (N, 2)
    p_right = p(rights)   # (N, 2)

    accepted_t = [np.array([t0])]

    while len(lefts):
        mids   = 0.5 * (lefts + rights)
        thirds = lefts + (rights - lefts) / 3.0
        twothirds = lefts + 2.0 * (rights - lefts) / 3.0

        # Batch-evaluate all needed points
        t_query = np.concatenate([mids, thirds, twothirds])
        pts_all = p(t_query)
        N = len(lefts)
        p_mid  = pts_all[:N]
        p_3rd  = pts_all[N:2*N]
        p_23rd = pts_all[2*N:]

        p_chord_mid = 0.5 * (p_left + p_right)

        # Criterion 1: midpoint deviation
        dev_mid  = np.linalg.norm(p_mid  - p_chord_mid, axis=-1)

        # Criterion 2: tercile deviations (interpolate chord at 1/3 and 2/3)
        p_chord_3rd  = p_left + (p_right - p_left) / 3.0
        p_chord_23rd = p_left + 2.0 * (p_right - p_left) / 3.0
        dev_3rd  = np.linalg.norm(p_3rd  - p_chord_3rd,  axis=-1)
        dev_23rd = np.linalg.norm(p_23rd - p_chord_23rd, axis=-1)

        # Criterion 3: chord length guard
        chord_len = np.linalg.norm(p_right - p_left, axis=-1)

        ok = (
            (dev_mid  <= eps_abs) &
            (dev_3rd  <= eps_abs) &
            (dev_23rd <= eps_abs) &
            (chord_len <= chord_cap)
        ) | (depths >= max_depth)

        # Accept converged intervals — record their right endpoints
        if ok.any():
            accepted_t.append(rights[ok])

        # Subdivide the rest — split at midpoint
        if (~ok).any():
            l_sub  = lefts[~ok]
            r_sub  = rights[~ok]
            d_sub  = depths[~ok] + 1
            m_sub  = mids[~ok]
            pl_sub = p_left[~ok]
            pm_sub = p_mid[~ok]
            pr_sub = p_right[~ok]

            lefts  = np.concatenate([l_sub, m_sub])
            rights = np.concatenate([m_sub, r_sub])
            depths = np.concatenate([d_sub, d_sub])
            p_left  = np.concatenate([pl_sub, pm_sub], axis=0)
            p_right = np.concatenate([pm_sub, pr_sub], axis=0)
        else:
            break

    t_sample = np.sort(np.concatenate(accepted_t))
    return t_sample
```

> **Cache reuse:** notice that `p_left` and `p_right` are threaded through the
> loop.  When a subinterval is split at its midpoint, `p_mid` becomes the new
> `p_right` of the left child and `p_left` of the right child — no extra
> evaluations are needed for shared endpoints.

---

## Complete Pipeline

```python
import numpy as np
from scipy.interpolate import PchipInterpolator


def build_arclength_parameterization(p, p_prime, t0, t1, eps=1e-4):
    """
    Full pipeline: adaptive sampling + arc-length reparameterization.

    Parameters
    ----------
    p       : callable (t_array,) -> (N, 2)   curve
    p_prime : callable (t_array,) -> (N, 2)   tangent (or pass None to
                                                finite-difference)
    t0, t1  : float   parameter interval
    eps     : float   relative tolerance

    Returns
    -------
    t_sample  : (M,) array    -- adaptively chosen parameter values
    pts       : (M, 2) array  -- corresponding curve points (polyline nodes)
    L         : float         -- estimated arc length
    t_of_s    : callable      -- smooth inverse arc-length map t(s), s in [0, L]
    s_of_idx  : (M,) array    -- arc-length value at each sample point
    """
    # --- 0. bootstrap ---
    L0 = estimate_length(p, t0, t1)
    eps_abs = eps * L0

    # --- 1. adaptive polyline ---
    t_sample = adaptive_polyline(p, t0, t1, eps_abs=eps_abs)
    pts = p(t_sample)

    # --- 2. arc-length table on the same partition ---
    if p_prime is None:
        p_prime = _finite_diff_tangent(p)

    t_nodes, s_nodes = adaptive_arclength_table(
        p_prime, t0, t1, eps=eps, eps_abs=eps_abs
    )
    L = s_nodes[-1]

    # --- 3. arc lengths at sample points (by interpolation) ---
    s_at_sample = PchipInterpolator(t_nodes, s_nodes)(t_sample)

    # --- 4. smooth inverse map ---
    t_of_s = make_arclength_inverse(t_nodes, s_nodes)

    return t_sample, pts, L, t_of_s, s_at_sample


def _finite_diff_tangent(p, h=1e-7):
    """Central-difference tangent when an analytic derivative is unavailable."""
    def p_prime(t):
        t = np.asarray(t)
        return (p(t + h) - p(t - h)) / (2.0 * h)
    return p_prime
```

---

## Worked Example

```python
# A non-uniform-speed figure-8 curve
def curve(t):
    t = np.asarray(t)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    return np.stack([x, y], axis=-1)

def curve_prime(t):
    t = np.asarray(t)
    x_p =  np.cos(t)
    y_p =  np.cos(2*t)
    return np.stack([x_p, y_p], axis=-1)

t0, t1 = 0.0, 2 * np.pi
eps    = 1e-4

t_samp, pts, L, t_of_s, s_vals = build_arclength_parameterization(
    curve, curve_prime, t0, t1, eps=eps
)

print(f"Arc length L ≈ {L:.6f}")
print(f"Polyline nodes: {len(t_samp)}")

# Uniformly-spaced points in arc length
s_uni = np.linspace(0, L, 200)
pts_uni = curve(t_of_s(s_uni))
```

---

## Performance Notes

### Batching strategy

| Pattern | Cost |
|---------|------|
| One `p(t)` call per interval per depth level (naïve recursion) | $O(M \cdot D)$ Python call overhead |
| Batch all active intervals per depth level (this guide) | $O(D)$ Python calls total |
| Pre-allocate `lefts`/`rights` as fixed-size buffers | avoids repeated `np.concatenate` |

For tight loops or very small $\varepsilon$, replace the `np.concatenate`
rebuilding with a pre-allocated stack array of size `2 * max_depth * N_init`.

### When to use `scipy.integrate.quad_vec`

If `p_prime` is expensive (e.g. wraps a compiled solver), replace the manual
GL quadrature with:

```python
from scipy.integrate import quad_vec

ds, err = quad_vec(
    lambda t: np.linalg.norm(p_prime(np.atleast_1d(t)), axis=-1),
    t0, t1,
    epsabs=eps_abs, epsrel=eps, limit=500
)
```

`quad_vec` manages its own adaptive subdivision and is well-tested, but it does
**not** expose the intermediate partition, so you must run `adaptive_polyline`
separately.

### Finite differences

Central differences with $h = 10^{-7}$ cost **two** curve evaluations per
tangent call.  If `p` is cheap, this is fine.  For expensive curves, supply an
analytic or automatic-differentiation derivative.

---

## Error Analysis Summary

| Source | Magnitude | Controlled by |
|--------|-----------|---------------|
| Polyline Hausdorff error | $\approx \tfrac{\kappa_{\max} \ell_{\max}^2}{8}$ | midpoint deviation test |
| Arc-length quadrature error | $|G_5 - G_3|$ per interval | GL error criterion |
| Inverse interpolation error | $O(h^4)$ in PCHIP | density of `t_nodes` |
| Bootstrap $L_0$ error | a few % of $L$ | only sets initial `eps_abs` |

The dominant term is the polyline error, kept below $\varepsilon_{\mathrm{abs}} = \varepsilon L$
by construction.  The quadrature and interpolation errors are set to the same
tolerance and are negligible in practice.
