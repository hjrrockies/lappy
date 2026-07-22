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
- A reentrant corner of interior angle ω contributes a singular factor r^{π/ω}. It is
  chamfered out of the polygon (replaced by a straight chord at radius R0) and integrated
  separately by an explicit, hand-built geometrically-graded fan (``corner_fan_triangles``)
  rather than asking a general mesh generator to realize an extreme size field near a
  point -- gmsh does not reliably do that (verified empirically: even a size field
  demanding ~1e-4 element sizes at a corner still realizes ~0.2-0.4 there), which capped
  achievable accuracy on corner-singular integrands at ~1e-11 regardless of requested
  precision. The fan's accuracy is governed by its cubature rule's degree, chosen per
  corner from a calibration against the exact analytic wedge integral (``_choose_corner_rule``);
  a small fixed number of geometric layers suffices at any precision, since the singular
  integrand is scale-invariant under the grading (see docs/cubature.md discussion in the
  repo history for the derivation).
- Element size is only ever a *target* for the mesh generator: realized triangles vary
  around it. Applying one rule sized for the coarsest requirement to every triangle wastes
  nodes on every smaller one, so after meshing, each triangle is independently assigned
  the cheapest ladder rule sufficient for its own realized size (``_mixed_tri_quad``).

The generation is a-priori: it meshes once, with no runtime iteration, so it is suitable
for use in the inner loop of a shape-optimization search. An optional ``verify=True`` runs
a one-shot a-posteriori surrogate check.
"""

import numpy as np
from functools import lru_cache

from .quad import (get_cubature_rule, tri_quad, tri_quad_rule,
                   polygon_triangular_mesh, corner_fan_triangles)
from .cubature_registry import (iter_rules,
                                get_capacity_curve as _registry_capacity_curve,
                                get_singular_curve as _registry_singular_curve)

# Reference equilateral triangle (edge length 1) used for calibration.
_REF_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
_REF_AREA = np.sqrt(3) / 4


# ── The positive-weight cubature ladder ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _rule_ladder():
    """All available cubature rules with strictly positive weights.

    Returns a tuple of (kind, deg, npts), sorted by degree. Only positive-weight rules
    are used so the generated cubature always satisfies the positivity requirement.
    """
    ladder = [(r['kind'], r['deg'], r['npts']) for r in iter_rules(positive_only=True)]
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


# Plane-wave calibration grid (fixed; a rule's persisted capacity_E in the registry is
# only trusted if CAPACITY_GRID_VERSION matches -- bump this if these parameters change.
CAPACITY_GRID_VERSION = 1
_CAPACITY_THETAS = np.linspace(0.0, np.pi, 17)[:-1] + 0.123   # generic directions
_CAPACITY_S_GRID = np.geomspace(0.1, 90.0, 240)


def _compute_capacity_curve(kind, deg):
    """Computes E_grid fresh: worst-case relative error E(s) of a rule integrating a unit
    plane wave over a triangle, as a function of s = K·h (nondimensional resolution).

    E is measured relative to the triangle area (the natural O(1) scale of a
    unit-modulus integrand). Scale invariance means this depends only on s and direction,
    so it is computed once on the unit reference triangle.
    """
    bc, bw = get_cubature_rule(kind, deg)
    nodes = bc @ _REF_VERTS
    wts = bw * _REF_AREA
    E = np.empty_like(_CAPACITY_S_GRID)
    for i, s in enumerate(_CAPACITY_S_GRID):
        worst = 0.0
        for th in _CAPACITY_THETAS:
            kx, ky = s * np.cos(th), s * np.sin(th)
            est = (wts * np.exp(1j * (kx * nodes[:, 0] + ky * nodes[:, 1]))).sum()
            err = abs(est - _planewave_exact(kx, ky)) / _REF_AREA
            worst = max(worst, err)
        E[i] = worst
    return E


@lru_cache(maxsize=None)
def _capacity_curve(kind, deg):
    """(s_grid, E_grid), preferring a registry-persisted E_grid over recomputing it."""
    cached = _registry_capacity_curve(kind, deg)
    if cached is not None:
        version, E = cached
        if version == CAPACITY_GRID_VERSION and len(E) == len(_CAPACITY_S_GRID):
            return _CAPACITY_S_GRID, E
    return _CAPACITY_S_GRID, _compute_capacity_curve(kind, deg)


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


# ── Per-triangle rule assignment ─────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _capacity_frontier(eps):
    """Precomputes, for the whole positive-weight ladder at precision eps, the
    cheapest rule sufficient for any given required resolution s = K·h.

    Returns (rho_sorted, kind_of, deg_of, npts_of): rho_sorted is ascending; for a
    query s, searchsorted(rho_sorted, s) gives the index of the cheapest rule with
    capacity >= s (kind_of/deg_of/npts_of are precomputed suffix-minimum-npts
    selections, so this holds even though capacity doesn't grow monotonically with
    the ladder's degree ordering).
    """
    entries = []
    for kind, deg, npts in _rule_ladder():
        rho = capacity(kind, deg, eps)
        if rho > 0:
            entries.append((rho, npts, kind, deg))
    entries.sort(key=lambda e: e[0])

    n = len(entries)
    rho_sorted = np.array([e[0] for e in entries])
    kind_of = [None] * n
    deg_of = [None] * n
    npts_of = np.empty(n)
    best_npts, best_kind, best_deg = np.inf, None, None
    for i in range(n - 1, -1, -1):
        rho, npts, kind, deg = entries[i]
        if npts < best_npts:
            best_npts, best_kind, best_deg = npts, kind, deg
        kind_of[i], deg_of[i], npts_of[i] = best_kind, best_deg, best_npts
    return rho_sorted, kind_of, deg_of, npts_of


def _tri_max_edge(mesh_vertices, triangles):
    """Longest edge length of each triangle -- the calibration and mesh-size field both
    treat "h" as an edge length; area-derived equivalent sizes underestimate this for
    non-equilateral triangles (equilateral minimizes longest-edge at fixed area), so
    using area here would under-resolve skinny/irregular elements."""
    v = mesh_vertices[triangles]
    e0 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
    e1 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
    e2 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
    return np.maximum(np.maximum(e0, e1), e2)


def _mixed_tri_quad(mesh, fallback_rule, K, eps):
    """Applies the cheapest sufficient cubature rule to each triangle individually,
    based on its own realized size, rather than one rule sized for the whole mesh.

    Reentrant corners are chamfered out of this mesh entirely (see _generate_cubature)
    and integrated separately by an explicit fan, so every triangle here is genuinely in
    the smooth bulk region -- plane-wave capacity is the right criterion everywhere in it,
    unlike when a corner-graded mesh's singular region shared this same mesh.

    fallback_rule = (kind, deg) is used for any triangle whose required resolution
    exceeds every rule's capacity (shouldn't happen, since mesh generation already
    bounds element size to what fallback_rule was chosen to cover, but a triangle
    can occasionally come out larger than the target).
    """
    mesh_vertices = mesh.points[:, :2]
    triangles = mesh.cells[1].data
    h_equiv = _tri_max_edge(mesh_vertices, triangles)
    s_vals = K * h_equiv

    rho_sorted, kind_of, deg_of, _ = _capacity_frontier(eps)
    idx = np.searchsorted(rho_sorted, s_vals, side='left')
    n = len(rho_sorted)
    fb_kind, fb_deg = fallback_rule

    groups = {}
    for tri_i, i in enumerate(idx):
        rule = (fb_kind, fb_deg) if i >= n else (kind_of[i], deg_of[i])
        groups.setdefault(rule, []).append(tri_i)

    node_parts, weight_parts = [], []
    for (kind, deg), tri_idxs in groups.items():
        z_i, w_i = tri_quad_rule(mesh_vertices, triangles[tri_idxs], kind, deg)
        node_parts.append(z_i)
        weight_parts.append(w_i)
    return np.concatenate(node_parts), np.concatenate(weight_parts)


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


# ── Explicit corner fan: chamfering, geometry, and singular-integrand calibration ──

# Geometric layers and angular divisions for corner_fan_triangles, shared by production
# use (_generate_cubature) and calibration (_compute_singular_curve) so they match
# exactly. L=2 is validated to already reach machine precision -- accuracy is governed
# by the cubature rule's degree (chosen per corner by _choose_corner_rule), not by L.
_CORNER_L = 2
_CORNER_NTHETA = 8


def _corner_R0s(poly, reentrant_idx, h_smooth):
    """Transition radius per reentrant corner: a few smooth elements, but not past the
    adjacent edges (so the chamfer and fan never overlap the rest of the boundary)."""
    verts = poly.vertices
    n = len(verts)
    R0s = []
    for i in reentrant_idx:
        e_prev = abs(verts[i] - verts[(i - 1) % n])
        e_next = abs(verts[(i + 1) % n] - verts[i])
        R0s.append(min(3 * h_smooth, 0.4 * min(e_prev, e_next)))
    return np.array(R0s)


def _corner_fan_params(poly, idx, R0):
    """(z0, theta0, omega) for corner_fan_triangles at reentrant corner `idx`.

    theta0 is the direction from z0 toward the next vertex; the wedge sweeps CCW by
    omega (the corner's interior angle) to reach the direction toward the previous
    vertex -- matching _chamfer_polygon's replacement points exactly, so the fan and
    the chamfered bulk mesh share a boundary with no gap or overlap.
    """
    verts = poly.vertices
    n = len(verts)
    z0 = verts[idx]
    v_next = verts[(idx + 1) % n]
    theta0 = np.angle(v_next - z0)
    omega = poly.int_angles[idx]
    return z0, theta0, omega


def _chamfer_polygon(poly, reentrant_idx, R0s, n_theta=_CORNER_NTHETA):
    """Replaces each listed reentrant corner with the points of its fan's outer ring
    (radius R0, n_theta angular divisions -- must match corner_fan_triangles' n_theta
    exactly), removing the corner from the polygon boundary entirely.

    A single straight chord between the two adjacent-edge points is NOT correct here:
    for a reflex (>180 degree) corner, that chord cuts through the *complement* wedge
    (the actual notch), filling in a sliver of empty space as if it were material,
    rather than excluding the (large, >180 degree) material wedge. Matching the fan's
    own n_theta-gon outer boundary exactly is what makes the chamfered bulk polygon and
    the fan tile the original domain exactly, with no gap or overlap (verified: their
    areas sum to the polygon's area to machine precision).
    """
    verts = poly.vertices
    n = len(verts)
    replacements = {}
    for idx, R0 in zip(reentrant_idx, R0s):
        z0, theta0, omega = _corner_fan_params(poly, idx, R0)
        thetas = theta0 + np.linspace(0.0, omega, n_theta + 1)
        ring_pts = z0 + R0 * np.exp(1j * thetas)
        # reversed: ring_pts[-1] (angle theta0+omega, toward v_prev) comes first when
        # traversing from v[idx-1], ending at ring_pts[0] (angle theta0, toward v_next)
        replacements[idx] = ring_pts[::-1]

    new_verts = []
    for i in range(n):
        if i in replacements:
            new_verts.extend(replacements[i])
        else:
            new_verts.append(verts[i])
    return np.array(new_verts)


def _apply_rule_to_fan(tris, kind, deg):
    """tris: (n,3,2) raw (unshared) triangle vertex coords -> (nodes, weights)."""
    n = len(tris)
    mesh_vertices = tris.reshape(-1, 2)
    triangles = np.arange(n * 3).reshape(n, 3)
    return tri_quad_rule(mesh_vertices, triangles, kind, deg)


# Singular-integrand calibration grid (fixed; a rule's persisted singular_E in the
# registry is only trusted if SINGULAR_GRID_VERSION matches -- bump if these change.
# Uses the same _CORNER_L/_CORNER_NTHETA as production so calibration and actual use match.
SINGULAR_GRID_VERSION = 1
_SINGULAR_BETA_GRID = np.linspace(0.501, 0.999, 50)   # beta = pi/omega, omega in (pi,2pi)


def _analytic_wedge_value(beta, n_theta):
    """Exact int r^{2*beta} dA over the canonical unit-R0 straight-sided n_theta-gon
    wedge sector of angle omega=pi/beta (mpmath, closed form per angular slice) --
    matches exactly the polygonal shape corner_fan_triangles builds, so this is an
    apples-to-apples reference (no circular-arc-vs-chord mismatch)."""
    import mpmath as mp
    with mp.workdps(30):
        omega = mp.pi / mp.mpf(float(beta))
        twobeta = 2 * mp.mpf(float(beta))
        thetas = [omega * k / n_theta for k in range(n_theta + 1)]
        total = mp.mpf(0)
        for j in range(n_theta):
            t0, t1 = thetas[j], thetas[j + 1]

            def rmax(theta, t0=t0, t1=t1):
                # R0=1 for both endpoints (canonical unit wedge) -> ra=rb=1 cancels
                return mp.sin(t1 - t0) / (mp.sin(theta - t0) - mp.sin(theta - t1))

            def integrand(theta, t0=t0, t1=t1):
                return rmax(theta, t0, t1)**(twobeta + 2) / (twobeta + 2)

            total += mp.quad(integrand, [t0, t1])
        return float(total)


def _compute_singular_curve(kind, deg):
    """Computes E_grid fresh over _SINGULAR_BETA_GRID: relative error of rule (kind,deg)
    integrating r^{2*beta} over the canonical corner fan, for each grid beta."""
    E = np.empty_like(_SINGULAR_BETA_GRID)
    for i, beta in enumerate(_SINGULAR_BETA_GRID):
        omega = np.pi / beta
        tris = corner_fan_triangles(0.0 + 0.0j, 0.0, omega, 1.0,
                                     L=_CORNER_L, n_theta=_CORNER_NTHETA)
        z, w = _apply_rule_to_fan(tris, kind, deg)
        val = np.sum(w * np.abs(z)**(2 * beta))
        exact = _analytic_wedge_value(beta, _CORNER_NTHETA)
        E[i] = abs(val - exact) / exact
    return E


@lru_cache(maxsize=None)
def _singular_curve(kind, deg):
    """(beta_grid, E_grid), preferring a registry-persisted E_grid over recomputing it."""
    cached = _registry_singular_curve(kind, deg)
    if cached is not None:
        version, beta_grid, E = cached
        if (version == SINGULAR_GRID_VERSION and len(E) == len(_SINGULAR_BETA_GRID)
                and np.allclose(beta_grid, _SINGULAR_BETA_GRID)):
            return _SINGULAR_BETA_GRID, E
    return _SINGULAR_BETA_GRID, _compute_singular_curve(kind, deg)


def _singular_error(kind, deg, beta):
    """Interpolated relative error of rule (kind,deg) on a corner of exponent beta."""
    beta_grid, E = _singular_curve(kind, deg)
    beta_c = min(max(beta, beta_grid[0]), beta_grid[-1])
    log_E = np.interp(beta_c, beta_grid, np.log(np.maximum(E, 1e-300)))
    return float(np.exp(log_E))


def _choose_corner_rule(beta, eps, s_max):
    """Cheapest ladder rule satisfying BOTH: relative error <= eps on the singular
    r^{2*beta} corner integrand, AND plane-wave capacity >= s_max.

    Real eigenfunctions near a corner combine the leading power-law singularity with
    ordinary wavenumber-K oscillation; the fan's outer ring sits at radius R0 (not
    infinitesimal), so it must resolve genuine oscillation there too (s_max = K*R0) --
    a rule chosen for singular accuracy alone can be wildly insufficient for that (e.g.
    a cheap low-degree rule integrates r^{2*beta} perfectly well but has plane-wave
    capacity far below what K*R0 demands). Falls back to the highest-degree ladder rule
    if no rule satisfies both (machine-precision floor on one criterion).
    """
    best = None
    for kind, deg, npts in _rule_ladder():
        if capacity(kind, deg, eps) < s_max:
            continue
        if _singular_error(kind, deg, beta) > eps:
            continue
        if best is None or npts < best[2]:
            best = (kind, deg, npts)
    if best is None:
        kind, deg, npts = max(_rule_ladder(), key=lambda t: t[1])
        return kind, deg
    return best[0], best[1]


def _generate_cubature(poly, kind, deg, h, K, eps, use_mixed):
    """Builds (z, w) for a polygon: reentrant corners are chamfered out and integrated
    by an explicit fan, the remainder by an ordinary mesh. Shared by polygon_cubature
    (use_mixed=True, per-triangle rule assignment) and _reference_rule (use_mixed=False,
    uniform rule -- a simpler, independent construction for a-posteriori comparison).
    Returns ((z, w), reentrant_idx).
    """
    reentrant = np.nonzero(poly.int_angles > np.pi + 1e-9)[0]
    if len(reentrant) == 0:
        mesh = polygon_triangular_mesh(poly.vertices, h,
                                       mesh_size_min=h * 0.5, mesh_size_max=h)
        zw = _mixed_tri_quad(mesh, (kind, deg), K, eps) if use_mixed else tri_quad(mesh, kind, deg)
        return zw, reentrant

    R0s = _corner_R0s(poly, reentrant, h)
    chamfered_verts = _chamfer_polygon(poly, reentrant, R0s)
    mesh = polygon_triangular_mesh(chamfered_verts, h, mesh_size_min=h * 0.5, mesh_size_max=h)
    z_bulk, w_bulk = (_mixed_tri_quad(mesh, (kind, deg), K, eps) if use_mixed
                      else tri_quad(mesh, kind, deg))
    z_parts, w_parts = [z_bulk], [w_bulk]

    for idx, R0 in zip(reentrant, R0s):
        z0, theta0, omega = _corner_fan_params(poly, idx, R0)
        beta = np.pi / omega
        c_kind, c_deg = _choose_corner_rule(beta, eps, K * R0)
        tris = corner_fan_triangles(z0, theta0, omega, R0, L=_CORNER_L, n_theta=_CORNER_NTHETA)
        z_c, w_c = _apply_rule_to_fan(tris, c_kind, c_deg)
        z_parts.append(z_c)
        w_parts.append(w_c)

    return (np.concatenate(z_parts), np.concatenate(w_parts)), reentrant


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
    (z, w), reentrant = _generate_cubature(poly, kind, deg, h_smooth, K, eps, use_mixed=True)
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
    (z, w), _ = _generate_cubature(poly, kind, deg, h, K, eps, use_mixed=False)
    return z, w


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
