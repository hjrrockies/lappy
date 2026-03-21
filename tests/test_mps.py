"""Tests for lappy.mps — MPSEigensolver."""

import gc
import weakref

import numpy as np
import pytest

from lappy import Polygon
from lappy.bases import FourierBesselBasis
from lappy.reference import rect_eig
from lappy.mps import MPSEigensolver


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
