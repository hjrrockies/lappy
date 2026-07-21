"""lappy.cubature — precision-driven cubature-rule generation for planar domains.

``polygon_cubature(poly, lam_max, precision)`` builds a set of positive cubature nodes
and weights on a polygon that integrates Laplacian eigenfunction L² norms and inner
products, up to spectral parameter ``lam_max``, to the requested relative ``precision``,
using a near-minimal number of nodes. It resolves reentrant-corner singularities with a
geometrically graded mesh.

Design (a-priori sizing; see docs/cubature.md and the verification suite in
tests/test_cubature.py):

- Interior eigenfunctions with λ ≤ lam_max are band-limited to wavenumber √λ_max, so the
  integrands u² and uᵢuⱼ carry Fourier content up to K = 2·√lam_max. The smooth-region
  element size and cubature degree are chosen (via an offline-calibrated capacity table)
  to integrate oscillations up to wavenumber K at the target precision with the fewest
  nodes.
- A reentrant corner of interior angle ω contributes a singular factor r^{π/ω}; a
  geometrically graded mesh toward the corner resolves it in a handful of layers.

The generation is a-priori: it meshes once, with no runtime iteration, so it is suitable
for use in the inner loop of a shape-optimization search. An optional ``verify=True`` runs
a one-shot a-posteriori surrogate check.
"""

import numpy as np
from functools import lru_cache

from .quad import (get_cubature_rule, _get_rules, tri_quad,
                   polygon_triangular_mesh, graded_polygon_mesh)

# Reference equilateral triangle (edge length 1) used for calibration.
_REF_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
_REF_AREA = np.sqrt(3) / 4

_GRADING_SIGMA = 0.17   # geometric mesh-grading ratio toward reentrant corners


# ── The positive-weight cubature ladder ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _rule_ladder():
    """All available cubature rules with strictly positive weights.

    Returns a tuple of (kind, deg, npts), sorted by degree. Only positive-weight rules
    are used so the generated cubature always satisfies the positivity requirement.
    """
    rules = _get_rules()
    ladder = []
    for kind in rules:
        for deg, arr in rules[kind].items():
            w = arr[:, 3]
            if (w > 0).all():
                ladder.append((kind, int(deg), int(len(w))))
    # sort by degree, then by fewest points (prefer the leaner rule at equal degree)
    ladder.sort(key=lambda t: (t[1], t[2]))
    # keep, for each degree, only the rule with the fewest points
    best = {}
    for kind, deg, npts in ladder:
        if deg not in best or npts < best[deg][2]:
            best[deg] = (kind, deg, npts)
    return tuple(sorted(best.values(), key=lambda t: t[1]))


# ── Offline calibration: plane-wave resolution capacity of each rule ──────────────

_DD_SPREAD_TAU = 1.0      # phase-spread threshold: below -> series, above -> direct
_DD_SERIES_TERMS = 32     # series length (converges to machine eps for spread <= tau)


def _fdd_exp(x, y):
    """Stable first divided difference (e^x - e^y)/(x - y), with the x->y limit e^y."""
    z = x - y
    if abs(z) < 1e-14:
        return np.exp(0.5 * (x + y))
    return np.exp(y) * np.expm1(z) / z


def _dd_exp(a):
    """Second divided difference exp[a0, a1, a2], evaluated cancellation-free.

    The naive form (differences of exp / differences of nodes) loses accuracy when the
    three phases are close together (small s): catastrophic cancellation floors the value
    at ~1e-13. Two complementary stable branches remove it:

    * **Small phase spread** (max pairwise gap ≤ tau): the complete-homogeneous-symmetric
      series exp[a0,a1,a2] = Σ_m h_m(b)/(m+2)! about the mean (b_j = a_j - mean), which is
      cancellation-free precisely when the b_j are small. h_m via the Newton recurrence
      m·h_m = Σ_{i=1}^m p_i h_{m-i} (p_i = power sums).
    * **Large phase spread**: the direct form, but with stabilized first divided
      differences (`_fdd_exp`, expm1-based) and the outer split taken on the *widest* phase
      gap so the outer denominator is never small.
    """
    a = np.asarray(a, dtype=complex)
    gaps = np.array([abs(a[0] - a[1]), abs(a[0] - a[2]), abs(a[1] - a[2])])

    if gaps.max() > _DD_SPREAD_TAU:
        # order (i, k, j) so the outer denominator a_i - a_j is the widest gap
        pair = (0, 1), (0, 2), (1, 2)
        i, j = pair[int(np.argmax(gaps))]
        k = ({0, 1, 2} - {i, j}).pop()
        d0 = _fdd_exp(a[i], a[k])
        d1 = _fdd_exp(a[k], a[j])
        return (d0 - d1) / (a[i] - a[j])

    # small spread: series about the mean (cancellation-free)
    b = a - a.sum() / 3
    p = np.array([(b ** i).sum() for i in range(1, _DD_SERIES_TERMS + 1)])
    h = np.empty(_DD_SERIES_TERMS + 1, dtype=complex)
    h[0] = 1.0
    for m in range(1, _DD_SERIES_TERMS + 1):
        h[m] = np.dot(p[:m], h[m - 1::-1]) / m
    fact = 2.0                                    # (0+2)!
    s = 0j
    for m in range(_DD_SERIES_TERMS + 1):
        s += h[m] / fact
        fact *= (m + 3)                           # (m+2)! -> (m+3)!
    return np.exp(a.sum() / 3) * s


def _planewave_exact(kx, ky):
    """Exact ∫ over the reference triangle of exp(i k·x), cancellation-free at all s."""
    a = 1j * (kx * _REF_VERTS[:, 0] + ky * _REF_VERTS[:, 1])
    return 2 * _REF_AREA * _dd_exp(a)


@lru_cache(maxsize=None)
def _capacity_curve(kind, deg):
    """(s_grid, E_grid): worst-case relative error E(s) of a rule integrating a unit
    plane wave over a triangle, as a function of s = K·h (nondimensional resolution).

    E is measured relative to the triangle area (the natural O(1) scale of a
    unit-modulus integrand). Scale invariance means this depends only on s and direction,
    so it is computed once on the unit reference triangle.
    """
    bc, bw = get_cubature_rule(kind, deg)
    nodes = bc @ _REF_VERTS
    wts = bw * _REF_AREA
    # generic directions (avoid axis alignment / vertex-phase confluence)
    thetas = np.linspace(0.0, np.pi, 17)[:-1] + 0.123
    s_grid = np.geomspace(0.1, 90.0, 240)
    E = np.empty_like(s_grid)
    for i, s in enumerate(s_grid):
        worst = 0.0
        for th in thetas:
            kx, ky = s * np.cos(th), s * np.sin(th)
            est = (wts * np.exp(1j * (kx * nodes[:, 0] + ky * nodes[:, 1]))).sum()
            err = abs(est - _planewave_exact(kx, ky)) / _REF_AREA
            worst = max(worst, err)
        E[i] = worst
    return s_grid, E


def capacity(kind, deg, eps):
    """Largest s = K·h a rule handles at relative precision eps (0 if none)."""
    s_grid, E = _capacity_curve(kind, deg)
    # first up-crossing of eps: capacity is the s just before E first exceeds eps
    over = np.nonzero(E > eps)[0]
    if len(over) == 0:
        return float(s_grid[-1])
    j = over[0]
    if j == 0:
        return 0.0
    # log-linear interpolation of the crossing in (s, E)
    s0, s1 = s_grid[j - 1], s_grid[j]
    e0, e1 = E[j - 1], E[j]
    t = (np.log(eps) - np.log(e0)) / (np.log(e1) - np.log(e0))
    return float(s0 * (s1 / s0) ** t)


def choose_rule(K, eps, area):
    """Pick the positive-weight rule minimizing estimated total nodes.

    For each rule the smooth element size is ``h = capacity/K``; the estimated element
    count is ``max(1, area / area(h))`` and total nodes ``= npts · n_elem``. In the
    asymptotic (many-element) regime this reduces to minimizing node density (high degree
    wins, spectral efficiency); in the few-element regime (h ≳ domain size) it avoids
    paying for a very-high-degree rule where a leaner one covers the domain in one
    element. Returns (kind, deg, npts, rho, h) with rho = capacity(kind, deg, eps).
    """
    a_ref = np.sqrt(3) / 4
    best = None
    for kind, deg, npts in _rule_ladder():
        rho = capacity(kind, deg, eps)
        if rho <= 0:
            continue
        h = rho / K
        n_elem = max(1.0, area / (a_ref * h**2))
        total = npts * n_elem
        if best is None or total < best[0]:
            best = (total, kind, deg, npts, rho, h)
    if best is None:
        raise RuntimeError("no positive-weight rule can reach the requested precision")
    _, kind, deg, npts, rho, h = best
    return kind, deg, npts, rho, h


# ── Corner grading parameters ────────────────────────────────────────────────────

def _corner_grading(poly, reentrant_idx, h_smooth, eps):
    """Per-reentrant-corner (h_corner, R0) for geometric grading.

    Layer count L = ceil( ln(eps) / ((2β+2)·ln σ) ), β = π/ω, from the truncation bound
    (h_corner/R0)^{2β+2} ≤ eps; h_corner = h_smooth·σ^L.
    """
    verts = poly.vertices
    angles = poly.int_angles
    n = len(verts)
    h_corners, R0s, corner_pts = [], [], []
    for i in reentrant_idx:
        omega = angles[i]
        beta = np.pi / omega
        L = int(np.ceil(np.log(eps) / ((2 * beta + 2) * np.log(_GRADING_SIGMA))))
        L = max(L, 1)
        h_corner = h_smooth * _GRADING_SIGMA**L
        # transition radius: a few smooth elements, but not past the adjacent edges
        e_prev = abs(verts[i] - verts[(i - 1) % n])
        e_next = abs(verts[(i + 1) % n] - verts[i])
        R0 = min(3 * h_smooth, 0.4 * min(e_prev, e_next))
        R0 = max(R0, 1.5 * h_corner)       # ensure room to grade
        h_corners.append(h_corner)
        R0s.append(R0)
        corner_pts.append(verts[i])
    return np.array(corner_pts), np.array(h_corners), np.array(R0s)


# ── Main entry point ─────────────────────────────────────────────────────────────

def polygon_cubature(poly, lam_max, precision, *, verify=False, safety=4.0):
    """Generate a positive cubature rule on a polygon for eigenfunction L² integration.

    Parameters
    ----------
    poly : Polygon
        The (CCW) polygonal domain.
    lam_max : float
        Approximate maximum spectral parameter λ the rule must handle.
    precision : float
        Target relative accuracy of computed L² norms / inner products.
    verify : bool, optional
        If True, run a one-shot a-posteriori surrogate check and raise on failure.
        Off by default (adds cost; unsuitable for inner-loop use).
    safety : float, optional
        Factor by which the sizing precision is tightened (eps = precision/safety) to
        cover the gap between the single-plane-wave surrogate and real eigenfunctions
        (direction spread, triangle-shape variation, sums of modes).

    Returns
    -------
    (nodes, weights) : complex ndarray, float ndarray
        Cubature nodes (complex coordinates) and positive weights.
    """
    eps = precision / safety
    K = 2.0 * np.sqrt(lam_max)
    K = max(K, 1e-12)

    kind, deg, npts, rho, h_smooth = choose_rule(K, eps, poly.area)

    reentrant = np.nonzero(poly.int_angles > np.pi + 1e-9)[0]
    if len(reentrant) == 0:
        mesh = polygon_triangular_mesh(
            poly.vertices, h_smooth,
            mesh_size_min=h_smooth * 0.5, mesh_size_max=h_smooth)
    else:
        corner_pts, h_corners, R0s = _corner_grading(poly, reentrant, h_smooth, eps)
        mesh = graded_polygon_mesh(
            poly.vertices, h_smooth, corner_pts, h_corners, R0s,
            mesh_size_max=h_smooth)

    z, w = tri_quad(mesh, kind, deg)
    if w.min() <= 0:
        raise RuntimeError("generated cubature has non-positive weights")

    if verify:
        _verify(poly, z, w, K, reentrant, precision)
    return z, w


# ── Optional a-posteriori verification (surrogate integrands) ─────────────────────

def _reference_rule(poly, lam_max, eps):
    """A much finer/higher rule used as a-posteriori ground truth for verify."""
    K = max(2.0 * np.sqrt(lam_max), 1e-12)
    kind, deg, npts, rho, h = choose_rule(K, eps, poly.area)
    reentrant = np.nonzero(poly.int_angles > np.pi + 1e-9)[0]
    if len(reentrant) == 0:
        mesh = polygon_triangular_mesh(poly.vertices, h,
                                       mesh_size_min=h * 0.5, mesh_size_max=h)
    else:
        corner_pts, h_corners, R0s = _corner_grading(poly, reentrant, h, eps)
        mesh = graded_polygon_mesh(poly.vertices, h, corner_pts, h_corners, R0s,
                                   mesh_size_max=h)
    return tri_quad(mesh, kind, deg)


def _verify(poly, z, w, K, reentrant, precision):
    """Check surrogate integrands against a finer reference; raise if error > precision."""
    ref_z, ref_w = _reference_rule(poly, (K / 2)**2, precision / 400.0)

    # (i) smooth: worst-case plane wave at wavenumber K over the polygon
    thetas = np.linspace(0.0, np.pi, 9)[:-1] + 0.123
    worst = 0.0
    for th in thetas:
        kx, ky = K * np.cos(th), K * np.sin(th)
        f = lambda zz: np.exp(1j * (kx * zz.real + ky * zz.imag))
        est = (w * f(z)).sum()
        ref = (ref_w * f(ref_z)).sum()
        worst = max(worst, abs(est - ref) / poly.area)
    if worst > precision:
        raise RuntimeError(
            f"verify: smooth plane-wave error {worst:.3e} exceeds precision {precision:.1e}")

    # (ii) corner-singular: r^{2β} per reentrant corner
    for i in reentrant:
        beta = np.pi / poly.int_angles[i]
        z0 = poly.vertices[i]
        f = lambda zz: np.abs(zz - z0)**(2 * beta)
        est = (w * f(z)).sum()
        ref = (ref_w * f(ref_z)).sum()
        err = abs(est - ref) / abs(ref)
        if err > precision:
            raise RuntimeError(
                f"verify: corner-singular error {err:.3e} exceeds precision {precision:.1e}")
