"""Tests for lappy.mps — MPSEigensolver."""

import gc
import weakref

import numpy as np
import pytest

from lappy import Polygon
from lappy.bases import FourierBesselBasis
from lappy.reference import rect_eig
from lappy.mps import MPSEigensolver, make_default_bdry_data, bdry_jacobi_exponents, pts_per_seg
from lappy.geometry import L_shape, eq_tri, LineSegment, MultiSegment


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_rect_solver():
    """MPSEigensolver for the 2×1 rectangle."""
    verts = np.array([0, 2, 2 + 1j, 1j])
    poly = Polygon(verts)
    basis = FourierBesselBasis.from_domain(poly, orders=[10, 0, 0, 0])
    bdry = poly.bdry_pts([0, 20, 20, 0], kind='even')
    ipts = poly.int_pts(method='random', npts_rand=30)
    return MPSEigensolver(basis, bdry, ipts)


@pytest.fixture(scope='module')
def rect_solver():
    return _make_rect_solver()


# ── API surface ───────────────────────────────────────────────────────────────

def test_tensions_batch_removed(rect_solver):
    assert not hasattr(rect_solver, 'tensions_batch')


# ── tensions: scalar dispatch ─────────────────────────────────────────────────

def test_tensions_scalar_returns_array(rect_solver):
    result = rect_solver.tensions(10.0)
    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_tensions_scalar_nonnegative(rect_solver):
    result = rect_solver.tensions(10.0)
    assert all(t >= 0 for t in result)


# ── tensions: array dispatch ──────────────────────────────────────────────────

def test_tensions_array_returns_list(rect_solver):
    result = rect_solver.tensions(np.array([10.0, 11.0, 12.0]))
    assert isinstance(result, list)
    assert len(result) == 3


def test_tensions_array_each_entry_is_array(rect_solver):
    result = rect_solver.tensions(np.array([10.0, 11.0]))
    assert all(isinstance(r, np.ndarray) for r in result)


def test_tensions_array_matches_scalar(rect_solver):
    lams = np.array([10.0, 11.0, 12.0])
    arr_result = rect_solver.tensions(lams)
    for lam, arr_t in zip(lams, arr_result):
        scalar_t = rect_solver.tensions(float(lam))
        np.testing.assert_array_equal(arr_t, scalar_t)


def test_tensions_array_parallel_matches_serial(rect_solver):
    lams = np.linspace(10.0, 15.0, 8)
    serial = rect_solver.tensions(lams)
    parallel = rect_solver.tensions(lams, n_workers=2)
    for s, p in zip(serial, parallel):
        np.testing.assert_array_almost_equal(s, p)


def test_tensions_empty_array(rect_solver):
    assert rect_solver.tensions(np.array([])) == []


# ── _tensions_scalar: caching ─────────────────────────────────────────────────

def test_tensions_scalar_cache_hit(rect_solver):
    """Same lam must return the identical object on a cache hit."""
    r1 = rect_solver._tensions_scalar(10.0)
    r2 = rect_solver._tensions_scalar(10.0)
    assert r1 is r2


def test_tensions_scalar_cache_per_instance():
    """Two solver instances must not share their caches."""
    s1, s2 = _make_rect_solver(), _make_rect_solver()
    r1 = s1._tensions_scalar(10.0)
    r2 = s2._tensions_scalar(10.0)
    assert r1 is not r2


def test_tensions_scalar_cache_freed_on_gc():
    """Solver cache must not prevent garbage collection of the instance."""
    ref = None

    def _make():
        nonlocal ref
        solver = _make_rect_solver()
        solver._tensions_scalar(10.0)
        ref = weakref.ref(solver)

    _make()
    gc.collect()
    assert ref() is None


# ── solve_interval correctness ────────────────────────────────────────────────

def test_solve_interval_first_eigenvalue(rect_solver):
    """First Dirichlet eigenvalue of the 2×1 rectangle matches reference value."""
    exact = rect_eig(1, 1, 2, 1)
    eigs, mults, _ = rect_solver.solve_interval(exact * 0.9, exact * 1.1, 30)
    assert len(eigs) >= 1
    assert abs(eigs[0] - exact) / exact < 1e-3


# ── regularization strategies (svd / qrp / implicit) ────────────────────────────

@pytest.mark.parametrize("reg_type", ["svd", "qrp", "implicit"])
def test_tensions_scalar_returns_array_reg_type(rect_solver, reg_type):
    result = rect_solver.tensions(10.0, reg_type=reg_type)
    assert isinstance(result, np.ndarray)
    assert len(result) > 0


@pytest.mark.parametrize("reg_type", ["svd", "qrp", "implicit"])
def test_tensions_scalar_nonnegative_reg_type(rect_solver, reg_type):
    result = rect_solver.tensions(10.0, reg_type=reg_type)
    assert all(t >= 0 for t in result)


@pytest.mark.parametrize("reg_type", ["svd", "qrp", "implicit"])
def test_solve_interval_first_eigenvalue_reg_type(rect_solver, reg_type):
    """All regularization strategies converge to the same reference eigenvalue."""
    exact = rect_eig(1, 1, 2, 1)
    eigs, mults, _ = rect_solver.solve_interval(exact * 0.9, exact * 1.1, 30, reg_type=reg_type)
    assert len(eigs) >= 1
    assert abs(eigs[0] - exact) / exact < 1e-3


def test_invalid_reg_type_raises(rect_solver):
    with pytest.raises(ValueError):
        rect_solver.tensions(10.0, reg_type='bogus')


# ── bdry_jacobi_exponents / make_default_bdry_data ──────────────────────────────

class TestBdryJacobiExponents:
    def test_square_all_corners(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        a, b = bdry_jacobi_exponents(sq)
        assert np.allclose(a, 2.0)
        assert np.allclose(b, 2.0)

    def test_equilateral_triangle(self):
        tri = eq_tri(1.0)
        a, b = bdry_jacobi_exponents(tri)
        assert np.allclose(a, 3.0)
        assert np.allclose(b, 3.0)

    def test_l_shape_reflex_corner(self):
        lsh = L_shape()
        a, b = bdry_jacobi_exponents(lsh)
        # exactly one segment-pair straddling the 3*pi/2 reflex corner -> exponent 2/3
        assert np.isclose(a.min(), 2/3)
        assert np.isclose(b.min(), 2/3)
        # every other corner is a right angle -> exponent 2
        assert np.allclose(np.sort(a)[1:], 2.0)
        assert np.allclose(np.sort(b)[1:], 2.0)

    def test_order_zero_is_default(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        a0, b0 = bdry_jacobi_exponents(sq)
        a1, b1 = bdry_jacobi_exponents(sq, order=0)
        assert np.allclose(a0, a1)
        assert np.allclose(b0, b1)

    def test_order_scalar_shifts_every_segment(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        a, b = bdry_jacobi_exponents(sq, order=1)
        assert np.allclose(a, 1.0)
        assert np.allclose(b, 1.0)

    def test_order_not_affected_by_bc(self):
        # order is a direct, explicit shift -- it must NOT depend on seg.bc at
        # all (e.g. a Hadamard shape-derivative integral over a pure-Dirichlet
        # boundary still needs order=1, the outward-normal-derivative exponent,
        # regardless of the PDE's own boundary condition).
        verts = np.array([0, 1, 1+1j, 1j])
        bcs = ['dir', 'dir', 'neu', 'dir']
        segs = [LineSegment(verts[i], verts[(i+1) % 4], bc=bcs[i]) for i in range(4)]
        mixed = Polygon(bdry=MultiSegment(segs))

        a, b = bdry_jacobi_exponents(mixed, order=1)
        assert np.allclose(a, 1.0)
        assert np.allclose(b, 1.0)

    def test_order_per_segment_array(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        order = np.array([0, 1, 2, 0.5])
        a, b = bdry_jacobi_exponents(sq, order=order)
        assert np.allclose(a, 2.0 - order)
        assert np.allclose(b, 2.0 - order)

    def test_weights_do_not_match_perimeter_except_at_order_zero(self):
        # Gauss-Jacobi weights are constructed to exactly integrate functions with a
        # matching (1-x)^a(1+x)^b singular factor, not the constant function -- so
        # (unlike kind='legendre'/a=b=0) they do NOT sum to the exact perimeter once
        # any exponent is nonzero. This is expected behavior of jacgauss, not a bug --
        # it's exactly why bdry_jacobi_exponents is for accurate integration of known
        # singular boundary quantities, not for MPS collocation.
        lsh = L_shape()
        n_per_seg = np.full(len(lsh.bdry.segments), 20)

        leg_pts = lsh.bdry_pts(n_per_seg, kind='legendre', weights=True)
        assert np.isclose(leg_pts.wts.sum(), lsh.perimeter, rtol=1e-10)

        a, b = bdry_jacobi_exponents(lsh, order=1)
        jac_pts = lsh.bdry_pts(n_per_seg, kind='jacobi', weights=True, a=a, b=b)
        assert not np.isclose(jac_pts.wts.sum(), lsh.perimeter, rtol=1e-3)

    def test_sharp_corner_pushes_points_away(self):
        # a > 1 (sharp/regular convex corner): the eigenfunction vanishes rapidly
        # there, so Gauss-Jacobi grading pushes nodes AWAY from the corner relative
        # to plain Gauss-Legendre (verified directly against jacgauss/quad.py).
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))  # every corner exponent = 2.0
        seg = sq.bdry.segments[0]
        a, b = bdry_jacobi_exponents(sq)
        n = 15
        jac_pts = seg.pts(n, kind='jacobi', a=a[0], b=b[0])
        leg_pts = seg.pts(n, kind='legendre')
        jac_dist = np.abs(jac_pts.pts - seg.p0).min()
        leg_dist = np.abs(leg_pts.pts - seg.p0).min()
        assert jac_dist > leg_dist


class TestMakeDefaultBdryData:
    """make_default_bdry_data is for MPS boundary collocation specifically --
    a pointwise-residual task, not an integration task -- so it uses plain
    Gauss-Legendre points (counts from pts_per_seg), deliberately NOT the
    Gauss-Jacobi grading in bdry_jacobi_exponents (that's for accurately
    integrating a known boundary quantity, a different task)."""

    def _basis(self, domain, order=20):
        n = len(domain.bdry.segments)
        return FourierBesselBasis.from_domain(domain, orders=[order]*n)

    def test_return_shape_matches_bdry_data(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        basis = self._basis(sq)
        bdry_pts, bdry_normals, bc_param = make_default_bdry_data(sq, basis)
        assert len(bdry_pts) == len(bdry_normals) == len(bc_param)
        assert len(bdry_pts) > 0

    def test_weights_only_when_requested(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        basis = self._basis(sq)
        bdry_pts, bdry_normals, _ = make_default_bdry_data(sq, basis, weights=False)
        assert not hasattr(bdry_pts, 'wts')
        bdry_pts, bdry_normals, _ = make_default_bdry_data(sq, basis, weights=True)
        assert hasattr(bdry_pts, 'wts')
        assert hasattr(bdry_normals, 'wts')

    def test_weights_match_perimeter(self):
        # plain Legendre weights ARE genuine arclength quadrature weights,
        # unlike the Gauss-Jacobi case (see TestBdryJacobiExponents).
        lsh = L_shape()
        basis = self._basis(lsh)
        bdry_pts, _, _ = make_default_bdry_data(lsh, basis, weights=True)
        assert np.isclose(bdry_pts.wts.sum(), lsh.perimeter, rtol=1e-10)

    def test_matches_manual_legendre_bdry_data(self):
        # make_default_bdry_data(domain, basis) should be exactly
        # domain.bdry_data(pts_per_seg(domain, basis), kind='legendre')
        lsh = L_shape()
        basis = self._basis(lsh)
        n_per_seg = pts_per_seg(lsh, basis)

        bdry_pts, bdry_normals, bc_param = make_default_bdry_data(lsh, basis)
        exp_pts, exp_normals, exp_bc = lsh.bdry_data(n_per_seg, kind='legendre')

        assert np.allclose(bdry_pts.pts, exp_pts.pts)
        assert np.allclose(bdry_normals.pts, exp_normals.pts)
        assert np.allclose(bc_param, exp_bc)

    def test_counts_use_pts_per_seg(self):
        lsh = L_shape()
        basis = self._basis(lsh)
        n_per_seg = pts_per_seg(lsh, basis)
        bdry_pts, _, _ = make_default_bdry_data(lsh, basis)
        assert len(bdry_pts) == n_per_seg.sum()


class TestDomainBdryKwargsPassthrough:
    """Domain.bdry_pts/bdry_normals/bdry_data must forward **kwargs (e.g.
    Gauss-Jacobi a/b) down to MultiSegment/segment .pts -- needed for
    bdry_jacobi_exponents-based integration, independent of the collocation
    default above."""

    def test_domain_bdry_pts_kwargs_passthrough(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        n_per_seg = np.full(4, 10)
        a = np.full(4, 2.0)
        b = np.full(4, 2.0)

        via_domain = sq.bdry_pts(n_per_seg, kind='jacobi', a=a, b=b)
        via_bdry = sq.bdry.pts(n_per_seg, kind='jacobi', a=a, b=b)
        assert np.allclose(via_domain.pts, via_bdry.pts)

        default_pts = sq.bdry_pts(n_per_seg)
        assert len(default_pts) == 40

    def test_bdry_data_kwargs_passthrough(self):
        sq = Polygon(np.array([0, 1, 1+1j, 1j]))
        n_per_seg = np.full(4, 10)
        a = np.full(4, 2.0)
        b = np.full(4, 2.0)
        bdry_pts, bdry_normals, bc_param = sq.bdry_data(n_per_seg, kind='jacobi', a=a, b=b)
        assert len(bdry_pts) == len(bdry_normals) == len(bc_param) == 40


# ── SVD driver fallback ───────────────────────────────────────────────────────

def test_the_svd_fallback_fires_and_returns_the_same_factorization(monkeypatch):
    """`gesdd` failing must not be fatal, and `gesvd` must give the same answer.

    LAPACK's `gesdd` is iterative and raises `LinAlgError: SVD did not converge` on badly
    conditioned input. Measured in the wild: one shape out of ~1500 solves in a `douse`
    multi-start sweep raised inside `tensions`, and because nothing caught it the whole sweep
    died and lost six of ten starts.

    The failure is not reproducible on demand -- it depends on the exact pencil -- so the driver
    is made to fail here instead, which tests the branch deterministically. Numerical equivalence
    is checked separately below, on a real matrix, since the two drivers compute the same object.
    """
    from lappy import mps

    rng = np.random.default_rng(0)
    R = rng.normal(size=(12, 12))
    expected = mps.la.svd(R, lapack_driver='gesvd')

    calls = []
    real_svd = mps.la.svd

    def flaky_svd(a, *args, **kwargs):
        calls.append(kwargs.get('lapack_driver'))
        if kwargs.get('lapack_driver') is None:          # the default, gesdd
            raise np.linalg.LinAlgError('SVD did not converge for slice = 0.')
        return real_svd(a, *args, **kwargs)

    monkeypatch.setattr(mps.la, 'svd', flaky_svd)
    with pytest.warns(UserWarning, match='gesdd'):
        Z, s, Yt = mps._svd_gesdd_then_gesvd(R)

    assert calls == [None, 'gesvd'], 'must try the fast driver first, then fall back exactly once'
    assert s == pytest.approx(expected[1])
    assert Z @ np.diag(s) @ Yt == pytest.approx(R, abs=1e-12)


def test_the_two_svd_drivers_agree_so_the_fallback_changes_no_result():
    """The fallback is a robustness change, not a numerical one: both drivers factor the same
    matrix, so an eigenvalue computed through either path is the same eigenvalue."""
    from lappy import mps

    rng = np.random.default_rng(1)
    for shape in ((20, 20), (40, 15)):
        R = rng.normal(size=shape)
        R[:, 0] *= 1e-10                                  # ill-conditioned on purpose
        s_fast = mps.la.svd(R)[1]
        s_slow = mps.la.svd(R, lapack_driver='gesvd')[1]
        assert s_fast == pytest.approx(s_slow, rel=1e-10, abs=1e-14)


def test_regularize_pencil_survives_a_failing_default_driver(monkeypatch):
    """The fallback has to be reachable from `regularize_pencil` itself, not just in isolation --
    that is the call site that took down the sweep."""
    from lappy import mps

    rng = np.random.default_rng(2)
    A1, A2 = rng.normal(size=(8, 6)), rng.normal(size=(9, 6))
    good = mps.regularize_pencil(A1, A2)

    real_svd = mps.la.svd

    def flaky_svd(a, *args, **kwargs):
        if kwargs.get('lapack_driver') is None:
            raise np.linalg.LinAlgError('SVD did not converge for slice = 0.')
        return real_svd(a, *args, **kwargs)

    monkeypatch.setattr(mps.la, 'svd', flaky_svd)
    with pytest.warns(UserWarning, match='gesdd'):
        fell_back = mps.regularize_pencil(A1, A2)

    for a, b in zip(good, fell_back):
        assert np.abs(a) == pytest.approx(np.abs(b), abs=1e-10)   # sign/phase is not fixed by SVD
