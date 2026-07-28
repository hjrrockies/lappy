"""Tests for lappy.rellich — boundary-only L^2(Omega) Gram matrices (via the Kress-style
graded-mesh boundary quadrature, lappy.cauchy.build_boundary_quadrature), and their wiring
into MPSEigensolver.eigenfunction_coef."""

import warnings

import numpy as np
import pytest
import scipy.linalg as la
from numpy.polynomial.legendre import leggauss

from lappy import Polygon, geometry, bases, cubature, bounds
from lappy.bases import FourierBesselBasis
from lappy.reference import rect_eig
from lappy.geometry import L_shape, PointSet
from lappy.mps import MPSEigensolver, NormalizedBasis, make_default_bdry_data, make_default_int_pts
import lappy.mps as mps_mod
from lappy.rellich import (build_rellich_data, rellich_gram_basis, orthonormalize_coef,
                           lowdin_transform, default_x0)

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

    rellich_data = build_rellich_data(domain, basis, lam_max=lam, x0=x0, mult=4)
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


def test_orthonorm_transforms_whiten_their_own_gram_degenerate():
    """Both independent Löwdin transforms (_orthonorm_transform_coef, _orthonorm_transform_eval)
    must whiten their OWN Gram matrix by construction, for a genuine mult=2 degenerate cluster --
    this does NOT assert they coincide (different GSVD pencils for mult>1 aren't guaranteed to
    pick the same rotation within the cluster, see MPSEigensolver._orthonorm_transform_eval's
    docstring)."""
    domain = Polygon(SQ_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[10, 10, 10, 10])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, 1, 1)
    assert rect_eig(2, 1, 1, 1) == pytest.approx(eig)

    D_coef, G_coef = solver._orthonorm_transform_coef(eig, mult=2)
    assert np.allclose(D_coef@G_coef@D_coef.T, np.eye(2), atol=1e-8)

    D_eval, G_eval = solver._orthonorm_transform_eval(eig, mult=2)
    assert np.allclose(D_eval@G_eval@D_eval.T, np.eye(2), atol=1e-8)


def test_orthonorm_coef_and_eval_agree_for_simple_eigenvalue():
    """For a simple (mult=1) eigenvalue, the coefficient-based (_orthonorm_transform_coef) and
    GSVD-eval-based (_orthonorm_transform_eval) orthonorm=True paths solve different GSVD
    pencils, but a 1-D nullspace has no rotation ambiguity -- they should agree at shared points
    up to a global sign. (docs/rellich_hadamard_mps.pdf Remark 2: large disagreement here would
    mean the raw coefficient vector is corrupted beyond safe pointwise evaluation, not merely
    mis-sandwiched.)"""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)

    pts = solver._cauchy_data.pts
    coef = solver.eigenfunction_coef(eig, mult=1, orthonorm=True)
    u_coef = (solver.basis(eig, pts)@coef)[:, 0]
    u_eval = solver.eigenfunction_eval_extras(eig, mult=1, extra_pts=pts, orthonorm=True)[2][:, 0]

    if np.dot(u_coef, u_eval) < 0:
        u_eval = -u_eval
    assert np.allclose(u_coef, u_eval, atol=1e-6, rtol=1e-6)


def test_lowdin_transform_deficient_returns_none_with_warning():
    """lowdin_transform must degrade gracefully (None + warning) rather than let w**-0.5 blow up
    on a near-singular Gram matrix (e.g. a requested multiplicity larger than the true one)."""
    G = np.array([[1.0, 0.0], [0.0, 1e-10]])
    with pytest.warns(UserWarning, match='deficient'):
        D = lowdin_transform(G)
    assert D is None


def test_orthonorm_caching_reuses_raw_coef_and_caches_independently():
    """orthonorm=False and orthonorm=True must cache independently (different kwarg -> different
    instance_lru_cache key), and orthonorm=True's internal raw-coefficient step must reuse
    orthonorm=False's cached result rather than re-solving the GSVD nullspace a second time."""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)

    calls = []
    original = mps_mod.nullspace_coef
    def counting_nullspace_coef(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    mps_mod.nullspace_coef = counting_nullspace_coef
    try:
        solver.eigenfunction_coef(eig, mult=1, orthonorm=False)
        assert len(calls) == 1
        solver.eigenfunction_coef(eig, mult=1, orthonorm=False)  # cache hit
        assert len(calls) == 1
        solver.eigenfunction_coef(eig, mult=1, orthonorm=True)  # reuses raw-coef cache
        assert len(calls) == 1
        solver.eigenfunction_coef(eig, mult=1, orthonorm=True)  # cache hit
        assert len(calls) == 1
    finally:
        mps_mod.nullspace_coef = original


def test_orthonorm_true_fixes_ill_conditioned_basis_norm():
    """Regression test for the CACT-sandwich bug (docs/rellich_hadamard_mps.pdf Sec. 3.1):
    an H-shape domain with a deliberately overcomplete Fourier-Bessel + fundamental-solution
    basis (collocation-matrix condition number ~1e16) previously gave a cubature-measured
    eigenfunction norm off from 1 by ~1e-4 under the old coef.T@G_NxN@coef sandwich; the new
    "evaluate first, sandwich never" orthonorm=True path should land within ~1e-6."""
    dom = geometry.H_shape()
    basis = bases.make_default_basis(dom, 200, fs_frac=0.5, fs_C=0.5)
    pps = mps_mod.pts_per_seg(dom, basis, mult=3)
    bdry_pts = dom.bdry_pts(pps)
    int_pts = dom.int_pts(npts_rand=len(basis))
    normed_basis = NormalizedBasis(basis, (bdry_pts, int_pts))
    rellich_data = build_rellich_data(dom, normed_basis, mult=2)
    solver = MPSEigensolver(normed_basis, bdry_pts, int_pts, cauchy_data=rellich_data, rtol=1e-14)
    eigs, mults, _ = solver.solve_interval(bounds.faber_krahn(dom), mps_mod.weyl_est(2, dom), 20)
    eig = eigs[0]

    coef = solver.eigenfunction_coef(eig, mult=1, orthonorm=True)
    nodes, weights = cubature.polygon_cubature(dom, eig, 1e-12)
    u = (normed_basis(eig, nodes)@coef)[:, 0]
    norm = la.norm(u*np.sqrt(weights))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_eigenfunction_coef_normalize_false_recovers_raw():
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis)
    eig = rect_eig(1, 2, L, H)
    coef_raw = solver.eigenfunction_coef(eig, mult=1, orthonorm=False)
    coef_norm = solver.eigenfunction_coef(eig, mult=1, orthonorm=True)
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
        coef_raw = solver.eigenfunction_coef(30.0, mult=1, orthonorm=False)
        coef_norm = solver.eigenfunction_coef(30.0, mult=1, orthonorm=True)
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
        coef = solver.eigenfunction_coef(eig, mult=1, orthonorm=True)
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
