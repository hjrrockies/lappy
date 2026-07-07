"""Verification pipeline for cubature rules on planar domains.

A cubature rule is a set of nodes ``z`` (complex coordinates) and positive weights
``w`` such that ``sum(w * f(z)) ≈ ∫_Ω f dA``. In lappy these rules exist to compute
L² norms and inner products of Laplacian eigenfunctions (for orthonormalization in the
MPS solver). This module measures how accurately a rule integrates the functions that
matter, against several integrand families with independent ground truth.

The pipeline is deliberately defined against a plain ``(z, w)`` interface so it can
score *any* rule. Here it is exercised against the existing mesh + ``tri_quad`` path
(``baseline_rule``); when the new cubature generator lands, wire it into
``standin_make_rule`` / ``baseline_rule`` and the same metrics apply unchanged.

Integrand families
------------------
A. Constant  → weight sum vs analytic area (partition of unity; exposes curved-boundary
   geometric error).
B. Monomials → polynomial exactness up to the rule's degree (polygons; exact reference
   from a very-high-degree rule, which integrates polynomials exactly per triangle).
C. Eigenfunctions → L² norms (diagonal Gram) and orthogonality (off-diagonal Gram), the
   primary, most faithful metric (rect / disk / sector have closed forms).
D. Smooth manufactured bump → vs over-kill reference rule.
E. Corner-singular integrand → analytic over the reflex sector (r^{4/3} at a 270° apex).

Ground truth for non-analytic cases is a very-fine, very-high-degree "over-kill" rule.
"""

import numpy as np
import pytest

from lappy import geometry as geo
from lappy.geometry import Polygon
from lappy.quad import polygon_triangular_mesh, spline_mesh_with_curvature, tri_quad
from lappy import reference as ref


# ── Rule builders (candidate + over-kill reference) ──────────────────────────────

def _poly_mesh_rule(poly, kind, deg, mesh_size, mesh_size_min, mesh_size_max):
    mesh = polygon_triangular_mesh(poly.vertices, mesh_size, mesh_size_min, mesh_size_max)
    return tri_quad(mesh, kind, deg)


def _curved_mesh_rule(dom, kind, deg, pts_per_2pi, mesh_size_min, mesh_size_max):
    segs = [s.to_splineseg() for s in dom.bdry.segments]
    mesh = spline_mesh_with_curvature(segs, pts_per_2pi, mesh_size_min, mesh_size_max)
    return tri_quad(mesh, kind, deg)


def baseline_rule(domain, deg=10):
    """A candidate cubature rule from the existing mesh + tri_quad path.

    Uses a positive-weight Dunavant rule so the positivity property is meaningful.
    This is the stand-in for the (future) cubature generator under test.
    """
    if isinstance(domain, Polygon):
        return _poly_mesh_rule(domain, 'dunavant', deg, 0.08, 0.04, 0.2)
    return _curved_mesh_rule(domain, 'dunavant', deg, 60, 0.02, 0.06)


def overkill_rule(domain):
    """High-accuracy reference rule: fine mesh + high degree.

    A degree-19 rule integrates every polynomial of degree ≤ 19 exactly on each
    triangle, so on a polygon (exact triangulation) it gives exact moments (we only
    need ≤ 8). Combined with a fine mesh it is a trustworthy reference for smooth and
    mildly-singular integrands. On curved domains it still carries an O(h²) geometric
    error from the polygonal boundary approximation, so it is trusted only for
    interior-localized integrands there.
    """
    if isinstance(domain, Polygon):
        return _poly_mesh_rule(domain, 'dunavant', 19, 0.05, 0.015, 0.12)
    return _curved_mesh_rule(domain, 'dunavant', 19, 110, 0.01, 0.05)


# ── Analytic areas ───────────────────────────────────────────────────────────────

def analytic_area(name, **p):
    if name == 'polygon':
        return p['domain'].area          # exact shoelace
    if name == 'disk':
        return np.pi * p['R']**2
    if name == 'sector':
        return p['alpha'] * p['R']**2 / 2
    raise ValueError(name)


# ── Metric functions: each takes (z, w, ...) and returns a scalar/array error ─────

def area_error(z, w, area_exact):
    return abs(w.sum() - area_exact) / area_exact


def moment_error(z, w, ref_z, ref_w, p, q):
    """Relative error of the (p, q) monomial moment vs a reference rule."""
    val = (w * z.real**p * z.imag**q).sum()
    ref_val = (ref_w * ref_z.real**p * ref_z.imag**q).sum()
    scale = max(abs(ref_val), 1e-14)
    return abs(val - ref_val) / scale


def gram_errors(z, w, modes):
    """Max diagonal (norm) rel error and max normalized off-diagonal error.

    ``modes`` is a list of (label, u, norm2). Returns (diag_relerr, offdiag_err).
    """
    U = np.array([u(z) for _, u, _ in modes])          # (K, Npts), real
    norms2 = np.array([nrm2 for _, _, nrm2 in modes])
    G = (U * w) @ U.T                                    # Gram matrix, (K, K)
    diag_relerr = np.abs(np.diag(G) - norms2) / norms2
    D = np.sqrt(norms2)
    Gn = G / np.outer(D, D)                              # normalize rows/cols
    off = Gn - np.diag(np.diag(Gn))
    return diag_relerr.max(), np.abs(off).max()


def integrand_error(z, w, f, ref_value):
    val = (w * f(z)).sum()
    return abs(val - ref_value) / abs(ref_value)


# ── Eigenmode enumeration (λ ≤ lam_max) ──────────────────────────────────────────

def rect_modes(L, H, lam_max):
    modes = []
    mmax = int(np.ceil(np.sqrt(lam_max) * L / np.pi))
    nmax = int(np.ceil(np.sqrt(lam_max) * H / np.pi))
    for m in range(1, mmax + 1):
        for n in range(1, nmax + 1):
            if ref.rect_eig(m, n, L, H) <= lam_max:
                u, nrm2 = ref.rect_eigfun(m, n, L, H)
                modes.append((f'{m},{n}', u, nrm2))
    return modes


def disk_modes(R, lam_max):
    modes = []
    m = 0
    while ref.disk_eig(m, 1, R) <= lam_max:
        n = 1
        while ref.disk_eig(m, n, R) <= lam_max:
            parities = ['cos'] if m == 0 else ['cos', 'sin']
            for parity in parities:
                u, nrm2 = ref.disk_eigfun(m, n, R, parity)
                modes.append((f'{m},{n},{parity}', u, nrm2))
            n += 1
        m += 1
    return modes


def sector_modes(R, alpha, lam_max):
    modes = []
    m = 1
    while ref.sector_eig(m, 1, R, alpha) <= lam_max:
        n = 1
        while ref.sector_eig(m, n, R, alpha) <= lam_max:
            u, nrm2 = ref.sector_eigfun(m, n, R, alpha)
            modes.append((f'{m},{n}', u, nrm2))
            n += 1
        m += 1
    return modes


# ── Tolerances (calibrated empirically from the bootstrap run) ────────────────────

TOL_AREA_POLY = 1e-12       # exact triangulation → machine precision
TOL_AREA_CURVED = 1.5e-2    # polygonal-boundary geometric floor (O(h²))
TOL_MOMENT = 1e-9           # polynomial exactness up to the rule degree
TOL_EIGF_DIAG = 1e-5        # eigenfunction L² norm rel error (baseline hits ~1e-7)
TOL_EIGF_OFFDIAG = 1e-5     # orthogonality (normalized off-diagonal)
TOL_BUMP = 1e-3             # smooth manufactured integrand vs over-kill
TOL_SING_OVERKILL = 3e-3    # over-kill (non-graded) resolves r^{4/3} to ~1e-3


# =================================================================================
# Family A — constant / partition of unity
# =================================================================================

@pytest.mark.parametrize('fixture,kind,params', [
    ('unit_square_domain', 'polygon', {}),
    ('rect_domain', 'polygon', {}),
    ('eq_tri_domain', 'polygon', {}),
    ('iso_right_tri_domain', 'polygon', {}),
    ('Lshape_domain', 'polygon', {}),
    ('disk_domain', 'disk', dict(R=1.0)),
    ('sector_domain', 'sector', dict(R=1.0, alpha=np.pi/2)),
    ('sector_reflex_domain', 'sector', dict(R=1.0, alpha=3*np.pi/2)),
])
def test_area_and_positive_weights(request, fixture, kind, params):
    domain = request.getfixturevalue(fixture)
    z, w = baseline_rule(domain)

    # Property 2 of the spec: all cubature weights are positive.
    assert w.min() > 0, f"{fixture}: found non-positive weight {w.min():.3e}"

    area = analytic_area(kind, domain=domain, **params)
    err = area_error(z, w, area)
    tol = TOL_AREA_POLY if kind == 'polygon' else TOL_AREA_CURVED
    assert err < tol, f"{fixture}: area rel error {err:.3e} exceeds {tol:.1e}"


# =================================================================================
# Family B — polynomial exactness (polygons)
# =================================================================================

@pytest.mark.parametrize('fixture', [
    'unit_square_domain', 'rect_domain', 'eq_tri_domain',
    'iso_right_tri_domain', 'Lshape_domain',
])
def test_moment_exactness(request, fixture):
    domain = request.getfixturevalue(fixture)
    deg = 8
    z, w = _poly_mesh_rule(domain, 'dunavant', deg, 0.15, 0.05, 0.3)
    ref_z, ref_w = overkill_rule(domain)

    # A degree-`deg` rule must integrate every monomial of total degree ≤ deg exactly.
    max_err = 0.0
    for total in range(0, deg + 1):
        for p in range(0, total + 1):
            q = total - p
            max_err = max(max_err, moment_error(z, w, ref_z, ref_w, p, q))
    assert max_err < TOL_MOMENT, (
        f"{fixture}: degree-{deg} rule moment error {max_err:.3e} up to degree {deg}"
    )


# =================================================================================
# Family C — eigenfunction L² norms and orthogonality
# =================================================================================

@pytest.mark.parametrize('fixture,mode_fn,lam_max', [
    ('unit_square_domain', lambda: rect_modes(1.0, 1.0, 200.0), 200.0),
    ('rect_domain', lambda: rect_modes(2.0, 1.0, 200.0), 200.0),
    ('disk_domain', lambda: disk_modes(1.0, 60.0), 60.0),
    ('sector_domain', lambda: sector_modes(1.0, np.pi/2, 200.0), 200.0),
    ('sector_reflex_domain', lambda: sector_modes(1.0, 3*np.pi/2, 120.0), 120.0),
])
def test_eigenfunction_gram(request, fixture, mode_fn, lam_max):
    domain = request.getfixturevalue(fixture)
    modes = mode_fn()
    assert len(modes) >= 3, f"{fixture}: only {len(modes)} modes below lam_max"

    z, w = baseline_rule(domain, deg=12)
    diag_err, offdiag_err = gram_errors(z, w, modes)

    assert diag_err < TOL_EIGF_DIAG, (
        f"{fixture}: max eigenfunction norm rel error {diag_err:.3e} "
        f"over {len(modes)} modes (λ ≤ {lam_max})"
    )
    assert offdiag_err < TOL_EIGF_OFFDIAG, (
        f"{fixture}: max orthogonality error {offdiag_err:.3e}"
    )


# =================================================================================
# Family D — smooth manufactured integrand (all domains, incl. curved)
# =================================================================================

def _gaussian_bump(center, sigma):
    def f(z):
        return np.exp(-(np.abs(z - center)**2) / (2 * sigma**2))
    return f


@pytest.mark.parametrize('fixture', [
    'unit_square_domain', 'eq_tri_domain', 'Lshape_domain',
    'disk_domain', 'sector_domain',
])
def test_manufactured_bump(request, fixture):
    domain = request.getfixturevalue(fixture)
    ref_z, ref_w = overkill_rule(domain)

    # Center the bump at the (weighted) centroid — guaranteed interior — with a small
    # width so it decays well inside the domain.
    center = (ref_w * ref_z).sum() / ref_w.sum()
    sigma = 0.12
    f = _gaussian_bump(center, sigma)
    ref_value = (ref_w * f(ref_z)).sum()

    z, w = baseline_rule(domain, deg=12)
    err = integrand_error(z, w, f, ref_value)
    assert err < TOL_BUMP, f"{fixture}: bump integral rel error {err:.3e}"


# =================================================================================
# Family E — corner-singular integrand (reflex sector, analytic)
# =================================================================================

def test_corner_singular_reflex_sector(sector_reflex_domain):
    """f = r^{4/3} at a 270° reentrant apex; exact integral over the sector.

    The 270° wedge has eigenfunction singularity exponent π/α = 2/3, so u ~ r^{2/3}
    and u² ~ r^{4/3}. Over the full sector, ∫ r^{4/3} dA = α R^{10/3} / (10/3).
    """
    R, alpha, gamma = 1.0, 3*np.pi/2, 2/3
    exact = alpha * R**(2*gamma + 2) / (2*gamma + 2)
    f = lambda z: np.abs(z)**(2*gamma)

    # An over-kill graded rule must resolve the (mild) corner singularity.
    ok_z, ok_w = overkill_rule(sector_reflex_domain)
    ok_err = integrand_error(ok_z, ok_w, f, exact)
    assert ok_err < TOL_SING_OVERKILL, (
        f"over-kill rule fails to resolve corner singularity: err {ok_err:.3e}"
    )

    # The baseline (non-graded) rule is expected to do markedly worse — record the gap
    # so future graded generators can be compared against it.
    z, w = baseline_rule(sector_reflex_domain)
    base_err = integrand_error(z, w, f, exact)
    assert base_err >= ok_err, (
        "baseline unexpectedly beat the over-kill reference on the singular integrand"
    )


# =================================================================================
# Over-kill self-consistency (validates the reference itself)
# =================================================================================

def test_overkill_reproduces_analytic(disk_domain):
    """The over-kill reference must reproduce analytic disk eigenfunction norms.

    Cross-checks both the reference rule and the closed-form norm formulas.
    """
    modes = disk_modes(1.0, 40.0)
    z, w = overkill_rule(disk_domain)
    diag_err, offdiag_err = gram_errors(z, w, modes)
    assert diag_err < 1e-4, f"over-kill disk norm error {diag_err:.3e}"
    assert offdiag_err < 1e-4, f"over-kill disk orthogonality error {offdiag_err:.3e}"


# =================================================================================
# Family C calibration hook (ties to the deferred generator)
# =================================================================================

def standin_make_rule(domain, lam_max, resolution):
    """Stand-in for the future ``make_rule(domain, λ_max, precision)`` generator.

    Here ``resolution`` is an integer refinement knob (higher = finer/higher degree).
    When the real precision-driven generator exists, replace this body with a call to
    it and the calibration test below applies unchanged.
    """
    deg = min(6 + 2 * resolution, 19)
    if isinstance(domain, Polygon):
        h = 0.2 / (resolution + 1)
        return _poly_mesh_rule(domain, 'dunavant', deg, h, h/2, 0.3)
    pts = 30 * (resolution + 1)
    hmin = 0.05 / (resolution + 1)
    return _curved_mesh_rule(domain, 'dunavant', deg, pts, hmin, 3*hmin)


def test_calibration_convergence(disk_domain):
    """As the rule is refined, the eigenfunction norm error must decrease (converge).

    This is the generator-agnostic acceptance property: measured L² error is a
    monotone-decreasing function of resolution. The real generator plugs into
    ``standin_make_rule`` and gets checked for "measured error ≤ requested precision".
    """
    lam_max = 60.0
    modes = disk_modes(1.0, lam_max)
    errs = []
    for resolution in range(0, 4):
        z, w = standin_make_rule(disk_domain, lam_max, resolution)
        diag_err, _ = gram_errors(z, w, modes)
        errs.append(diag_err)

    # Overall convergence: finest is far better than coarsest, and the trend decreases.
    assert errs[-1] < errs[0], f"no convergence: errs = {errs}"
    assert errs[-1] < 1e-3, f"finest rule error {errs[-1]:.3e} not below 1e-3"
