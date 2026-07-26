"""Tests for lappy.bases — FourierBesselBasis / ExPrecFBBasis."""

import numpy as np
import pytest

from lappy.bases import FourierBesselBasis, ExPrecFBBasis
from lappy.geometry import PointSet


def _make_basis(cls=FourierBesselBasis, phi0=0.0, phi1=np.pi, branch_cut=np.pi / 2,
                 order=3, kind='sin', **kwargs):
    sources = np.array([0 + 0j])
    phi0 = np.array([phi0])
    phi1 = np.array([phi1])
    branch_cuts = np.array([branch_cut])
    orders = np.array([order])
    return cls(sources, phi0, phi1, orders, branch_cuts, kind, **kwargs)


# ── _set_alphak / _phi_hat ──────────────────────────────────────────────────

def test_wedge_angle_excludes_zero_includes_2pi():
    """phi1 == phi0 (a degenerate full-circle wedge) must map to 2*pi, not 0."""
    basis = _make_basis(phi0=0.0, phi1=0.0, branch_cut=np.pi)
    assert basis.alpha[0] == pytest.approx(np.pi / (2 * np.pi))


def test_phi_hat_excludes_zero_includes_2pi():
    """A branch cut coincident with ray0 must map to 2*pi, not 0."""
    basis = _make_basis(phi0=0.0, phi1=np.pi, branch_cut=0.0)
    assert basis._phi_hat[0] == pytest.approx(2 * np.pi)


def test_phi_hat_precomputed_not_recomputed_per_call():
    basis = _make_basis()
    pts = PointSet(np.array([1 + 0.5j, -1 + 0.3j]))
    basis._theta(pts)
    assert hasattr(basis, '_phi_hat')


# ── on_boundary angle wrapping ──────────────────────────────────────────────

def test_on_boundary_tangent_at_pi_wraps_to_2pi():
    class _FakePts:
        def __init__(self, pts):
            self.pts = np.asarray(pts)

    class _FakeDomain:
        bc_type = 'dir'

        def bdry_pts(self, n_per_seg, kind):
            return _FakePts(np.array([0 + 0j]))

        def bdry_tangents(self, n_per_seg, kind):
            return _FakePts(np.array([-1 + 0j]))  # angle == pi

    basis = FourierBesselBasis.on_boundary(_FakeDomain(), n_per_seg=[1], order=1)
    assert basis._ray1.real == pytest.approx(np.cos(2 * np.pi))
    assert basis._ray1.imag == pytest.approx(np.sin(2 * np.pi))


# ── Vectorized _sin/_cos/_r_rep match ExPrecFBBasis reference ──────────────

@pytest.mark.parametrize('kind', ['sin', 'cos'])
def test_sin_cos_match_extended_precision_reference(kind):
    fb = _make_basis(FourierBesselBasis, order=4, kind=kind)
    eb = _make_basis(ExPrecFBBasis, order=4, kind=kind, dps=30)

    pts = PointSet(np.array([1 + 0.5j, -1 + 0.3j, 0.2 - 0.9j, -0.4 - 0.1j]))

    if kind == 'sin':
        got = fb._sin(pts)
        ref = np.array(eb._sin(pts).tolist(), dtype=float)
    else:
        got = fb._cos(pts)
        ref = np.array(eb._cos(pts).tolist(), dtype=float)

    assert got.shape == (len(pts), 4)
    assert np.allclose(got, ref)


def test_r_rep_matches_repeated_radii():
    basis = _make_basis(order=4)
    pts = PointSet(np.array([1 + 0.5j, -1 + 0.3j]))
    r_rep = basis._r_rep(pts)
    expected = np.repeat(basis._r(pts), basis.orders, axis=1)
    assert np.array_equal(r_rep, expected)


def test_index_maps_consistent_with_orders():
    basis = _make_basis(order=5)
    assert basis._src_idx.shape == (5,)
    assert basis._alphak_col.shape == (5,)
    assert np.array_equal(basis._alphak_col, basis.alphak_vec[0])


# ── corner_terms() — per-column corner-singularity structure (lappy.cauchy) ──

def test_corner_terms_fourier_bessel_matches_alphak():
    from lappy import Polygon
    domain = Polygon(np.array([0, 2, 2 + 1j, 1j]), bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[3, 3, 3, 3])
    corner_id, exponent = basis.corner_terms(domain)
    assert corner_id.shape == (len(basis),)
    assert np.array_equal(corner_id, basis._src_idx)  # sources built in domain.corners order
    assert np.array_equal(exponent, basis._alphak_col)
    assert np.array_equal(np.sort(np.unique(corner_id)), np.arange(4))


def test_corner_terms_fourier_bessel_sincos_duplicates_columns():
    domain_mixed_kind = 'sincos'
    from lappy import Polygon
    domain = Polygon(np.array([0, 2, 2 + 1j, 1j]), bc='dir')
    phi0, phi1 = domain.corner_angles
    basis = FourierBesselBasis(domain.corners, phi0, phi1, [3, 3, 3, 3],
                               domain.branch_cut_rays(), kind=domain_mixed_kind)
    corner_id, exponent = basis.corner_terms(domain)
    half = len(basis)//2
    assert np.array_equal(corner_id[:half], corner_id[half:])
    assert np.array_equal(exponent[:half], exponent[half:])


def test_corner_terms_unmatched_source_raises():
    basis = _make_basis(order=3)  # source at 0+0j, unrelated to any domain
    from lappy import Polygon
    domain = Polygon(np.array([5, 7, 7 + 1j, 5 + 1j]), bc='dir')  # doesn't contain 0+0j
    with pytest.raises(ValueError):
        basis.corner_terms(domain)


def test_corner_terms_fundamental_basis_all_regular():
    from lappy.bases import FundamentalBasis
    basis = FundamentalBasis(np.array([5+5j, -5-5j]), orders=2)
    corner_id, exponent = basis.corner_terms()
    assert np.all(corner_id == -1)
    assert corner_id.shape == (len(basis),)


def test_corner_terms_multibasis_concatenates():
    from lappy import Polygon
    from lappy.bases import FundamentalBasis
    domain = Polygon(np.array([0, 2, 2 + 1j, 1j]), bc='dir')
    fb = FourierBesselBasis.from_domain(domain, orders=[3, 3, 3, 3])
    fs = FundamentalBasis(np.array([5+5j, -5-5j]), orders=2)
    mb = fb + fs
    corner_id, exponent = mb.corner_terms(domain)
    cid_fb, exp_fb = fb.corner_terms(domain)
    cid_fs, exp_fs = fs.corner_terms(domain)
    assert np.array_equal(corner_id, np.concatenate([cid_fb, cid_fs]))
    assert np.array_equal(exponent, np.concatenate([exp_fb, exp_fs]))


def test_corner_terms_normalized_basis_delegates():
    from lappy import Polygon
    domain = Polygon(np.array([0, 2, 2 + 1j, 1j]), bc='dir')
    fb = FourierBesselBasis.from_domain(domain, orders=[3, 3, 3, 3])
    bdry = domain.bdry_pts(20)
    nb = fb.to_normalized(bdry)
    cid_nb, exp_nb = nb.corner_terms(domain)
    cid_fb, exp_fb = fb.corner_terms(domain)
    assert np.array_equal(cid_nb, cid_fb)
    assert np.array_equal(exp_nb, exp_fb)


# ── polyline+ray branch cuts ────────────────────────────────────────────────

from lappy import Polygon
from lappy.geometry import spiral, corner_branch_cut_rays, corner_branch_cut_polyline


def _spiral_single_source_basis(kind='sin', order=5):
    """A FourierBesselBasis with one source at a surrounded spiral corner, using a
    polyline+ray cut. Returns (basis, domain, corner_index, verts, beta)."""
    sp = spiral()
    phi0, phi1 = sp.corner_angles
    rays = corner_branch_cut_rays(sp)
    i = int(np.where(np.isnan(rays))[0][0])
    verts, beta = corner_branch_cut_polyline(sp, i)
    orders = np.zeros(len(sp.corners), int)
    orders[i] = order
    bpl = [None]*len(sp.corners)
    bpl[i] = (verts, beta)
    basis = FourierBesselBasis(sp.corners, phi0, phi1, orders, np.nan_to_num(rays), kind,
                               branch_polylines=bpl)
    return basis, sp, i, verts, beta


def test_plain_basis_has_no_polyline_sources():
    b = FourierBesselBasis.from_domain(Polygon([0, 1, 1+1j, 1j]), [4, 4, 4, 4])
    assert b._polyline_srcs == []


def test_branch_polylines_none_matches_uncorrected_theta():
    # with no polyline cuts, _theta is exactly the plain wrapped-angle formula
    b = _make_basis(order=3)
    assert b._polyline_srcs == []
    pts = PointSet(np.array([0.3+0.4j, -0.2+0.5j, 0.6-0.1j]))
    z = b._z(pts)
    theta = np.angle(z/b._ray0)
    theta[theta <= 0] += 2*np.pi
    theta[theta > b._phi_hat] -= 2*np.pi
    assert np.array_equal(b._theta(pts), theta)


def test_polyline_cut_continuous_across_initial_ray():
    # beyond q1 the initial ray threads the domain; the polyline cut must be continuous
    # there (jump ∝ eps), unlike a plain ray cut at the same angle (O(1) jump)
    basis, sp, i, verts, beta = _spiral_single_source_basis()
    phi0, phi1 = sp.corner_angles
    c = sp.corners[i]
    theta0 = np.angle(verts[0] - c)
    bc_plain = np.nan_to_num(corner_branch_cut_rays(sp))
    bc_plain[i] = theta0
    orders = np.zeros(len(sp.corners), int); orders[i] = 5
    plain = FourierBesselBasis(sp.corners, phi0, phi1, orders, bc_plain, 'sin')

    lam = 40.0
    n = 1j*np.exp(1j*theta0)
    p = c + 1.5*np.exp(1j*theta0)             # on the initial ray, beyond q1
    assert sp.contains(np.array([p + 1e-3*n]))[0] and sp.contains(np.array([p - 1e-3*n]))[0]
    jumps_poly, jumps_plain = [], []
    for eps in (1e-3, 1e-5):
        zp, zm = p + eps*n, p - eps*n
        jumps_poly.append(np.abs(basis(lam, PointSet(np.array([zp])))[0]
                                 - basis(lam, PointSet(np.array([zm])))[0]).max())
        jumps_plain.append(np.abs(plain(lam, PointSet(np.array([zp])))[0]
                                  - plain(lam, PointSet(np.array([zm])))[0]).max())
    # polyline jump shrinks ~100x with eps (continuous); plain stays O(1)
    assert jumps_poly[1] < 0.02*jumps_poly[0]
    assert jumps_plain[0] > 0.1 and jumps_plain[1] > 0.1


def test_polyline_cut_satisfies_helmholtz():
    basis, sp, i, verts, beta = _spiral_single_source_basis()
    lam = 40.0
    # an interior point comfortably away from the corner singularity
    ip = sp.int_pts(method='random', npts_rand=4000).pts
    z0 = ip[np.argmax(np.abs(ip - sp.corners[i]))]
    h = 1e-4
    val = lambda z: basis(lam, PointSet(np.array([z])))[0]
    lap = (val(z0+h) + val(z0-h) + val(z0+1j*h) + val(z0-1j*h) - 4*val(z0))/h**2
    assert np.max(np.abs(lap + lam*val(z0))) < 1e-5


def test_polyline_cut_gradient_matches_finite_differences():
    basis, sp, i, verts, beta = _spiral_single_source_basis()
    lam = 40.0
    c = sp.corners[i]
    z0 = c + 1.5*np.exp(1j*np.angle(verts[0] - c))
    h = 1e-5
    g = basis._grad_pointset(lam, PointSet(np.array([z0])))[0]
    v = lambda z: basis(lam, PointSet(np.array([z])))[0]
    gx = (v(z0+h) - v(z0-h))/(2*h)
    gy = (v(z0+1j*h) - v(z0-1j*h))/(2*h)
    assert np.max(np.abs(g.real - gx)) < 1e-6
    assert np.max(np.abs(g.imag - gy)) < 1e-6


def test_from_domain_auto_generates_polyline_cuts():
    sp = spiral()
    n_surrounded = int(np.sum(np.isnan(corner_branch_cut_rays(sp))))
    b = FourierBesselBasis.from_domain(sp, np.full(len(sp.corners), 3))
    assert len(b._polyline_srcs) == n_surrounded
    A = b(30.0, sp.int_pts(method='random', npts_rand=300))
    assert np.all(np.isfinite(A))


def test_from_domain_polyline_cuts_false_raises():
    sp = spiral()
    with pytest.raises(ValueError, match="no straight-ray branch cut"):
        FourierBesselBasis.from_domain(sp, np.full(len(sp.corners), 3), polyline_cuts=False)
