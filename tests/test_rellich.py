"""Tests for lappy.rellich — boundary-only L^2(Omega) Gram matrices (via the SS/SR/RS/RR
singularity-subtraction quadrature, lappy.cauchy.singular_gram), and their wiring into
MPSEigensolver.eigenfunction_coef."""

import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from lappy import Polygon
from lappy.bases import FourierBesselBasis
from lappy.reference import rect_eig
from lappy.geometry import L_shape, PointSet
from lappy.mps import MPSEigensolver, make_default_bdry_data, make_default_int_pts
from lappy.rellich import build_rellich_data, rellich_gram_basis, orthonormalize_coef, default_x0

L, H = 2.0, 1.0
RECT_VERTS = np.array([0, L, L + 1j*H, 1j*H])
SQ_VERTS = np.array([0, 1, 1 + 1j, 1j])


def _gl_norm_sq(f, L, H, nx=60, ny=60):
    """L^2 norm^2 of a scalar eigenfunction callable over [0,L]x[0,H] via
    tensor Gauss-Legendre quadrature (used only to independently validate the
    Rellich-based normalization -- not part of the implementation itself)."""
    xg, wx = leggauss(nx); xg = (xg + 1)/2*L; wx = wx/2*L
    yg, wy = leggauss(ny); yg = (yg + 1)/2*H; wy = wy/2*H
    X, Y = np.meshgrid(xg, yg, indexing='ij')
    W = np.outer(wx, wy)
    vals = f(X + 1j*Y)[:, :, 0]
    return np.sum(W*vals**2)


def _composite_gram_NN_reference(basis, domain, lam, x0, npanels=300, order=16):
    """Independent (no lappy.cauchy code involved) composite Gauss-Legendre reference for
    integral_{boundary} (r.N)(dN_u)(dN_v) ds, the whole-basis Gram matrix -- used to validate
    rellich_gram_basis's SS/SR/RS/RR quadrature against a completely different quadrature
    strategy, rather than checking self-consistency of the same machinery."""
    u, w = leggauss(order); u = (u + 1)/2; w = w/2
    N = len(basis)
    G = np.zeros((N, N))
    for seg in domain.bdry.segments:
        edges = np.linspace(0, 1, npanels + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            tau = a + u*(b - a)
            wts = (b - a)*w*seg.len
            pts, normals = seg.p(tau), seg.N(tau)
            ps, ns = PointSet(pts), PointSet(normals)
            Phi_N = basis.ddiff(lam, ps, ns)
            rN = (pts.real - x0.real)*normals.real + (pts.imag - x0.imag)*normals.imag
            G += (Phi_N*(wts*rN)[:, np.newaxis]).T@Phi_N
    return G


def test_rellich_gram_basis_matches_independent_reference_dirichlet():
    """rellich_gram_basis (SS/SR/RS/RR quadrature) must match a completely independent
    composite-quadrature computation of the same boundary integral, not just be self-consistent."""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[4, 4, 4, 4])
    lam = rect_eig(1, 2, L, H)
    x0 = default_x0(domain)

    rellich_data = build_rellich_data(domain, basis, x0, group_pts=32)
    G = rellich_gram_basis(basis, lam, rellich_data)
    Gref = _composite_gram_NN_reference(basis, domain, lam, x0)/(2*lam)
    assert np.allclose(G, Gref, atol=1e-8)


def test_rellich_identity_pure_dirichlet():
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)


def test_rellich_identity_pure_neumann():
    domain = Polygon(RECT_VERTS, bc='neu')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)


def test_rellich_identity_reentrant_corner():
    """L-shape has a genuinely singular (non-integer alpha=2/3) reentrant 3*pi/2 corner --
    exactly the case Phase 1's leading-mode-only grading handled worst. The per-mode-exact
    SS/SR/RS/RR quadrature should still normalize accurately here.

    Tolerance is looser than the rectangle checks: at this basis size the dominant error source
    is the MPS boundary-collocation residual (docs/rellich_hadamard_mps.pdf Sec. 9.3's "true
    precision floor"), not the Rellich quadrature itself -- confirmed by rerunning with several
    times the default quadrature density and seeing the result barely move."""
    from lappy.asymp import weyl_est
    domain = L_shape(bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8]*len(domain.corners))
    solver = MPSEigensolver.from_domain(domain, basis=basis, lam_max=weyl_est(6, domain))
    eigs, mults, _ = solver.solve_interval(1e-3, weyl_est(4, domain), 40)
    assert len(eigs) > 0
    eig = eigs[0]
    coef = solver.eigenfunction_coef(eig, mult=1)
    G = solver._cauchy_gram(eig)
    assert (coef.T@G@coef)[0, 0] == pytest.approx(1.0, abs=5e-3)


# ── end-to-end through MPSEigensolver ─────────────────────────────────────────

def test_eigenfunction_coef_normalizes_dirichlet():
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)


def test_eigenfunction_coef_normalizes_neumann():
    domain = Polygon(RECT_VERTS, bc='neu')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)


def test_eigenfunction_coef_orthonormal_degenerate():
    """Square has a genuine multiplicity-2 eigenvalue; the two coefficient
    vectors should come out mutually L^2(Omega)-orthonormal."""
    domain = Polygon(SQ_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[10, 10, 10, 10])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, 1, 1)
    assert rect_eig(2, 1, 1, 1) == pytest.approx(eig)

    coef = solver.eigenfunction_coef(eig, mult=2)
    G = solver._cauchy_gram(eig)
    Gram = coef.T@G@coef
    assert np.allclose(Gram, np.eye(2), atol=1e-8)


def test_eigenfunction_coef_normalize_false_recovers_raw():
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)
    coef_raw = solver.eigenfunction_coef(eig, mult=1, normalize=False)
    coef_norm = solver.eigenfunction_coef(eig, mult=1, normalize='rellich')
    # different scale in general -- raw GSVD scale isn't unit L^2 norm
    assert not np.allclose(coef_raw, coef_norm)


def test_robin_domain_falls_back_with_warning():
    """from_domain is the only place that knows about the domain's bc_type;
    the Robin warning fires there, at construction time."""
    domain = Polygon(RECT_VERTS, bc=0.5)
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])

    with pytest.warns(UserWarning, match='Robin'):
        # lam_max supplied explicitly: weyl_est doesn't support bc_type='rob'
        solver = MPSEigensolver.from_domain(domain, lam_max=100.0, basis=basis)
    assert solver._cauchy_data is None

    with pytest.warns(UserWarning, match='unavailable'):
        coef_raw = solver.eigenfunction_coef(30.0, mult=1, normalize=False)
        coef_norm = solver.eigenfunction_coef(30.0, mult=1, normalize='rellich')
    assert np.allclose(coef_raw, coef_norm)


def test_rellich_false_opts_out():
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis, rellich=False)
    assert solver._cauchy_data is None


def test_rellich_precision_knobs_accepted():
    """rellich_mult/rellich_min_per_seg/rellich_x0 are plumbed through to
    build_rellich_data without erroring, and still produce valid normalization."""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis, rellich_mult=4,
                                        rellich_min_per_seg=8, rellich_x0=1.0+0.5j)
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)


def test_manual_construction_without_domain_has_no_cauchy_data():
    """MPSEigensolver's raw constructor has no notion of a domain; without an
    explicit cauchy_data argument, normalization is simply unavailable."""
    verts = RECT_VERTS
    poly = Polygon(verts)
    basis = FourierBesselBasis.from_domain(poly, orders=[10, 0, 0, 0])
    bdry = poly.bdry_pts([0, 20, 20, 0], kind='even')
    ipts = poly.int_pts(method='random', npts_rand=30)
    solver = MPSEigensolver(basis, bdry, ipts)
    assert solver._cauchy_data is None
    assert solver.cauchy_data is None

    eig = rect_eig(1, 1, L, H)
    with pytest.warns(UserWarning, match='unavailable'):
        coef = solver.eigenfunction_coef(eig, mult=1, normalize='rellich')
    assert coef.shape[1] == 1


def test_manual_construction_with_explicit_cauchy_data():
    """A caller building the solver by hand can still opt into Rellich
    normalization by building cauchy_data themselves and passing it in --
    mirroring how bdry_pts/int_pts are built and passed in manually."""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    bdry_pts, bdry_normals, bc_param = make_default_bdry_data(domain, basis)
    int_pts = make_default_int_pts(domain, 'random', False, len(basis))
    basis_norm = basis.to_normalized((bdry_pts, int_pts))
    cauchy_data = build_rellich_data(domain, basis_norm)

    solver = MPSEigensolver(basis_norm, bdry_pts, int_pts, bdry_normals, bc_param,
                            cauchy_data=cauchy_data)
    assert solver.cauchy_data is cauchy_data
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)
