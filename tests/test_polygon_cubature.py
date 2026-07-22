"""Certification of the polygon cubature generator (lappy.cubature.polygon_cubature).

Every rule the generator produces is scored with the Phase-1 verification metrics
(tests/test_cubature.py) and the closed-form eigenfunctions in lappy.reference. The
generator must, for a requested (lam_max, precision):
  * reach `precision` on eigenfunction L² norms and orthogonality (rectangle: closed form),
  * keep all weights positive,
  * resolve reentrant-corner singularities (L-shape),
  * use a near-minimal number of nodes (beats forced low-degree / uniform refinement),
  * and (opt-in) certify itself via verify=True.
"""

import numpy as np
import pytest

from lappy import geometry as geo
from lappy.geometry import Polygon
from lappy import cubature as cub
from lappy.quad import polygon_triangular_mesh, tri_quad

from tests.test_cubature import gram_errors, rect_modes, moment_error


def _irregular_polygon():
    """A rotated, non-axis-aligned convex polygon (no separability shortcut)."""
    ang = np.array([0.1, 1.3, 2.0, 3.1, 4.0, 5.2])
    rad = np.array([1.0, 1.4, 0.8, 1.2, 0.9, 1.3])
    return Polygon(rad * np.exp(1j * ang), val_simple=False)


# =================================================================================
# Precision: eigenfunction L² norms + orthogonality (rectangle has closed forms)
# =================================================================================

@pytest.mark.parametrize('L,H', [(1.0, 1.0), (2.0, 1.0)])
@pytest.mark.parametrize('lam_max', [100.0, 400.0])
@pytest.mark.parametrize('precision', [1e-3, 1e-5, 1e-8])
def test_precision_met_rectangle(L, H, lam_max, precision):
    dom = geo.rect(L, H)
    z, w = cub.polygon_cubature(dom, lam_max, precision)
    modes = rect_modes(L, H, lam_max)
    assert len(modes) >= 3
    diag_err, offdiag_err = gram_errors(z, w, modes)
    assert diag_err <= precision, (
        f"{L}x{H} lam={lam_max} prec={precision:.0e}: norm error {diag_err:.2e}")
    assert offdiag_err <= precision, (
        f"{L}x{H} lam={lam_max} prec={precision:.0e}: orthogonality error {offdiag_err:.2e}")


# =================================================================================
# Positivity + exact area (spec property 2; partition of unity)
# =================================================================================

@pytest.mark.parametrize('fixture', [
    'unit_square_domain', 'rect_domain', 'eq_tri_domain',
    'iso_right_tri_domain', 'Lshape_domain',
])
def test_positive_weights_and_area(request, fixture):
    dom = request.getfixturevalue(fixture)
    z, w = cub.polygon_cubature(dom, lam_max=150.0, precision=1e-6)
    assert w.min() > 0, f"{fixture}: non-positive weight {w.min():.3e}"
    assert abs(w.sum() - dom.area) / dom.area < 1e-12, f"{fixture}: area mismatch"


def test_positive_weights_irregular():
    dom = _irregular_polygon()
    z, w = cub.polygon_cubature(dom, lam_max=300.0, precision=1e-5)
    assert w.min() > 0
    assert abs(w.sum() - dom.area) / dom.area < 1e-12


# =================================================================================
# Polynomial exactness (a high-degree generated rule integrates monomials exactly)
# =================================================================================

def test_moment_exactness():
    dom = geo.rect(2.0, 1.0)
    # tight precision + high lam_max => high-degree rule
    z, w = cub.polygon_cubature(dom, lam_max=400.0, precision=1e-8)
    ref_mesh = polygon_triangular_mesh(dom.vertices, 0.2, 0.05, 0.4)
    ref_z, ref_w = tri_quad(ref_mesh, 'dunavant', 19)
    max_err = 0.0
    for total in range(0, 9):           # rule degree >> 8 at this precision
        for p in range(total + 1):
            max_err = max(max_err, moment_error(z, w, ref_z, ref_w, p, total - p))
    assert max_err < 1e-9, f"moment error {max_err:.3e}"


# =================================================================================
# Reentrant corner (L-shape): certified by self-convergence + verify
# =================================================================================

@pytest.mark.parametrize('precision', [1e-4, 1e-6])
def test_reentrant_corner_self_convergence(precision):
    """The corner-singular integrand r^{4/3} converges to within `precision`.

    No closed form on the L-shape, so certify by self-consistency: the rule at
    `precision` agrees with a 50x-tighter rule to within `precision`.
    """
    dom = geo.L_shape()
    beta = np.pi / dom.int_angles[0]     # 270° corner -> 2/3
    z0 = dom.vertices[0]
    f = lambda zz: np.abs(zz - z0) ** (2 * beta)

    z, w = cub.polygon_cubature(dom, lam_max=100.0, precision=precision)
    zf, wf = cub.polygon_cubature(dom, lam_max=100.0, precision=precision / 50)
    coarse = (w * f(z)).sum()
    fine = (wf * f(zf)).sum()
    rel = abs(coarse - fine) / abs(fine)
    assert rel <= precision, f"corner-singular self-convergence {rel:.2e} > {precision:.0e}"
    assert w.min() > 0


# =================================================================================
# verify=True: passes on valid rules, raises on a deliberately coarse one
# =================================================================================

@pytest.mark.parametrize('fixture', ['rect_domain', 'Lshape_domain'])
def test_verify_flag_passes(request, fixture):
    dom = request.getfixturevalue(fixture)
    # should not raise
    cub.polygon_cubature(dom, lam_max=120.0, precision=1e-5, verify=True)


@pytest.mark.parametrize('precision', [1e-3, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12])
def test_verify_flag_passes_reentrant_across_precisions(precision):
    """Regression test: per-triangle rule assignment (lappy.cubature._mixed_tri_quad)
    must not downgrade the rule on triangles inside a reentrant corner's grading
    radius -- their integrand is the singular factor itself, not a band-limited wave,
    so plane-wave capacity alone is the wrong criterion there and previously caused
    verify=True to fail at tight precision on corner-graded meshes. Also regression-
    guards the explicit corner fan's rule choice (_choose_corner_rule): a rule picked
    for singular-integrand accuracy alone can have far too little plane-wave capacity
    for the fan's own outer-ring oscillation requirement (s = K*R0).
    """
    dom = geo.L_shape()
    cub.polygon_cubature(dom, lam_max=400.0, precision=precision, verify=True)


def _analytic_corner_singular_value(poly, corner_idx, dps=40):
    """Exact int |z-z0|^{2*beta} dA over poly via mpmath, fan-triangulated from the
    corner vertex z0 (closed form per triangle via polar coordinates centered at z0) --
    independent ground truth, no mesh or cubature rule involved."""
    import mpmath as mp
    with mp.workdps(dps):
        verts = poly.vertices
        n = len(verts)
        z0 = mp.mpc(verts[corner_idx].real, verts[corner_idx].imag)
        beta = mp.pi / mp.mpf(poly.int_angles[corner_idx])
        twobeta = 2 * beta
        total = mp.mpf(0)
        for k in range(n):
            if k == corner_idx or (k + 1) % n == corner_idx:
                continue
            a = mp.mpc(verts[k].real, verts[k].imag) - z0
            b = mp.mpc(verts[(k + 1) % n].real, verts[(k + 1) % n].imag) - z0
            ra, ta = mp.fabs(a), mp.arg(a)
            rb, tb = mp.fabs(b), mp.arg(b)
            dt = tb - ta
            if dt > mp.pi:
                dt -= 2 * mp.pi
            if dt < -mp.pi:
                dt += 2 * mp.pi

            def rmax(theta, ra=ra, ta=ta, rb=rb, tb=tb):
                return ra * rb * mp.sin(tb - ta) / (rb * mp.sin(theta - ta) - ra * mp.sin(theta - tb))

            def integrand(theta, ra=ra, ta=ta, rb=rb, tb=tb):
                return rmax(theta, ra, ta, rb, tb)**(twobeta + 2) / (twobeta + 2)

            total += mp.fabs(mp.quad(integrand, [ta, ta + dt]))
        return float(total)


@pytest.mark.parametrize('precision', [1e-3, 1e-6, 1e-8, 1e-10, 1e-12])
def test_reentrant_corner_reaches_requested_precision(precision):
    """The explicit corner-fan construction must actually deliver the requested
    precision on a genuinely singular quantity, not just avoid raising under
    verify=True -- certified against an independent mpmath analytic ground truth
    (no mesh, no cubature rule), reaching machine precision at tight requests where
    the old gmsh-graded-mesh construction plateaued at a fixed ~1e-11 floor
    regardless of how tightly precision was requested.
    """
    dom = geo.L_shape()
    reentrant = np.nonzero(dom.int_angles > np.pi + 1e-9)[0]
    corner_idx = int(reentrant[0])
    beta = np.pi / dom.int_angles[corner_idx]
    z0 = dom.vertices[corner_idx]
    f = lambda zz: np.abs(zz - z0)**(2 * beta)

    true_val = _analytic_corner_singular_value(dom, corner_idx)
    z, w = cub.polygon_cubature(dom, lam_max=400.0, precision=precision)
    est = (w * f(z)).sum()
    rel_err = abs(est - true_val) / true_val
    assert rel_err <= precision, (
        f"precision={precision:.0e}: actual relative error {rel_err:.3e} exceeds request")


def test_verify_flag_catches_bad_rule():
    dom = geo.rect(2.0, 1.0)
    K = 2 * np.sqrt(120.0)
    # a deliberately too-coarse rule: one low-degree rule on a coarse mesh
    mesh = polygon_triangular_mesh(dom.vertices, 0.5, 0.25, 0.5)
    z, w = tri_quad(mesh, 'dunavant', 2)
    with pytest.raises(RuntimeError):
        cub._verify(dom, z, w, K, np.array([], dtype=int), precision=1e-8)


# =================================================================================
# Minimality
# =================================================================================

def test_minimality_high_degree_beats_low_degree():
    """The generator uses fewer nodes than a forced low-degree rule at equal precision."""
    dom = geo.rect(2.0, 1.0)
    lam_max, precision = 200.0, 1e-4
    z, w = cub.polygon_cubature(dom, lam_max, precision)
    diag, off = gram_errors(z, w, rect_modes(2.0, 1.0, lam_max))
    assert diag <= precision  # generator meets precision

    # forced degree-4 uniform rule, sized to also meet precision
    eps = precision / 4
    K = 2 * np.sqrt(lam_max)
    h4 = cub.capacity('cowper', 4, eps) / K
    mesh4 = polygon_triangular_mesh(dom.vertices, h4, h4 * 0.5, h4)
    z4, w4 = tri_quad(mesh4, 'cowper', 4)
    diag4, off4 = gram_errors(z4, w4, rect_modes(2.0, 1.0, lam_max))
    assert diag4 <= precision, "deg-4 alternative should also meet precision (fair comparison)"

    assert len(z) < len(z4), f"generator {len(z)} not fewer than deg-4 {len(z4)}"


def test_minimality_grading_beats_uniform_refinement():
    """Chamfered corner-fan treatment uses far fewer nodes than naive uniform mesh
    refinement to the fan's finest scale, applied over the whole domain."""
    dom = geo.L_shape()
    lam_max, precision = 100.0, 1e-6
    z, w = cub.polygon_cubature(dom, lam_max, precision)

    eps = precision / 4
    K = 2 * np.sqrt(lam_max)
    kind, deg, npts, rho, h = cub.choose_rule(K, eps, dom.area)
    reentrant = np.nonzero(dom.int_angles > np.pi + 1e-9)[0]
    R0s = cub._corner_R0s(dom, reentrant, h)
    h_finest = R0s[0] * (0.17 ** cub._CORNER_L)   # finest scale in the corner fan
    # nodes a uniform mesh at h_finest would need over the whole domain
    uniform_tris = dom.area / ((np.sqrt(3) / 4) * h_finest ** 2)
    uniform_nodes = uniform_tris * npts
    assert len(z) < 0.2 * uniform_nodes, (
        f"generator {len(z)} not << uniform estimate {uniform_nodes:.0f}")


# =================================================================================
# Calibration hook: the real generator plugged into the Phase-1 convergence check
# =================================================================================

def test_calibration_convergence_rectangle():
    """Tightening `precision` drives the measured eigenfunction error down and meets it."""
    dom = geo.rect(2.0, 1.0)
    lam_max = 200.0
    modes = rect_modes(2.0, 1.0, lam_max)
    errs = []
    for precision in (1e-2, 1e-4, 1e-6, 1e-8):
        z, w = cub.polygon_cubature(dom, lam_max, precision)
        diag, _ = gram_errors(z, w, modes)
        assert diag <= precision, f"prec={precision:.0e}: error {diag:.2e}"
        errs.append(diag)
    assert errs[-1] <= errs[0], f"no convergence: {errs}"


# =================================================================================
# Calibration reference accuracy (no spurious precision floor)
# =================================================================================

def test_planewave_reference_accurate_at_small_s():
    """The cancellation-free divided-difference reference is machine-accurate for all s.

    A naive divided difference floors at ~1e-13 for clustered phases (small s), which used
    to impose a spurious ~1e-11 precision limit on the generator. Cross-check against a
    high-order rule (accurate for these smooth, low-oscillation integrands).
    """
    from lappy.quad import get_cubature_rule
    verts, A = cub._REF_VERTS, cub._REF_AREA
    bc, bw = get_cubature_rule('xiao_gim', 50)
    nd = bc @ verts
    for s in (0.02, 0.1, 0.5, 1.0, 1.5):
        worst = 0.0
        for th in np.linspace(0, np.pi, 17)[:-1] + 0.123:
            kx, ky = s * np.cos(th), s * np.sin(th)
            hi = (bw * A * np.exp(1j * (kx * nd[:, 0] + ky * nd[:, 1]))).sum()
            worst = max(worst, abs(cub._planewave_exact(kx, ky) - hi) / A)
        assert worst < 1e-13, f"s={s}: reference error {worst:.2e}"


def test_precision_below_old_floor():
    """Requesting precision tighter than the old ~1e-11 wall now succeeds (no RuntimeError)
    and delivers accuracy far below it (down to float64 machine-epsilon accumulation)."""
    dom = geo.rect(2.0, 1.0)
    lam_max = 200.0
    modes = rect_modes(2.0, 1.0, lam_max)
    for precision in (1e-12, 1e-13):
        z, w = cub.polygon_cubature(dom, lam_max, precision)  # must not raise
        diag, offdiag = gram_errors(z, w, modes)
        assert w.min() > 0
        assert diag <= 1e-11, f"prec={precision:.0e}: error {diag:.2e} not below old floor"
