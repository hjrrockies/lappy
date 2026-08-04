"""Regression tests for solver robustness fixes.

Each of these guards a failure that was silent or fatal, found during the
reference-value run. See benchmarks/suite/run/FINDINGS.md.
"""
import numpy as np
import pytest

from lappy import geometry as G, reference as R, bounds
from lappy import opt
from lappy.evp import Eigenproblem


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    """Leave numpy's global RNG exactly as we found it.

    Some tests here must exercise the legacy global-RNG path, and other test
    modules draw interior points from it without seeding, so reseeding here
    would change their results depending on test order. (That fragility is
    itself an argument for threading `rng` explicitly everywhere.)
    """
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


# --- 1. sharp lower bound as a search endpoint -----------------------------

def test_faber_krahn_is_sharp_for_the_disk():
    """The premise of the bug: the bound equals lambda_1 for a disk."""
    d = G.disk(1)
    assert np.isclose(bounds.faber_krahn(area=d.area), R.disk_eigs(1, 1)[0],
                      rtol=1e-12, atol=0)


def test_search_window_starts_strictly_below_lambda_1():
    """Eigenproblem must not put lambda_1 on the window edge.

    opt.discrete_locmin_idx ignores grid endpoints by construction, so a
    minimum sitting exactly on the lower edge can never be bracketed. Before
    the fix the unit disk silently returned modes 2..k+1: every value correct
    to ~14 digits, every certificate valid, and a Weyl gap of only ~1.
    """
    from lappy.evp import _WINDOW_PAD
    d = G.disk(1)
    edge = bounds.faber_krahn(area=d.area) * (1.0 - _WINDOW_PAD)
    assert edge < R.disk_eigs(1, 1)[0]


@pytest.mark.slow
def test_disk_ground_state_is_found():
    """End-to-end: the disk finds its own first eigenvalue."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'benchmarks', 'reference'))
    from common import build_solver
    np.random.seed(0)
    d = G.disk(1)
    eigs = Eigenproblem(d, build_solver(d, 120)).solve(4)
    exact = R.disk_eigs(4, 1)
    assert np.abs(eigs - exact).max() / exact[0] < 1e-6


# --- 2. bounded bracket refinement -----------------------------------------

def test_bracket_mins_is_depth_bounded_by_default():
    assert opt.bracket_mins.__defaults__ is not None
    import inspect
    assert inspect.signature(opt.bracket_mins).parameters['max_recurse'].default == 8


def test_bracket_mins_terminates_on_pure_noise():
    """A noisy objective must not refine forever.

    The "too many local minima" guard fires only at nrecurse == 0, so before
    the depth cap every deeper level could flag spurious minima and spawn
    another, finer, level -- compounding across levels and across runs.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(1.0, 2.0, 101)
    y = np.abs(rng.standard_normal((2, len(x)))) * 1e-14
    brackets, fevals = opt.bracket_mins(
        lambda t: np.abs(rng.standard_normal(2)) * 1e-14, x, y, xtol=1e-12)
    assert fevals < 200000, f'refinement ran away: {fevals} evaluations'


def test_degenerate_bracket_does_not_raise():
    """A bracket whose interior point coincides with an endpoint must fall
    back to golden search, not raise. Flat sigma at a high-multiplicity
    eigenvalue produces exactly this."""
    x = np.array([1.0, 1.0, 2.0])
    y = np.array([1.0, 1.0, 1.0])
    m, _ = opt.minimize_on_bracket(lambda t: (t - 1.5) ** 2, (x, y), xtol=1e-8)
    assert 1.0 <= m <= 2.0


# --- 4. reproducible interior points ---------------------------------------

@pytest.mark.parametrize('dom,label', [(G.L_shape(), 'Polygon'),
                                       (G.disk(1), 'Domain')])
def test_int_pts_rng_is_reproducible(dom, label):
    a = dom.int_pts(method='random', npts_rand=20, rng=42).pts
    b = dom.int_pts(method='random', npts_rand=20, rng=42).pts
    c = dom.int_pts(method='random', npts_rand=20, rng=7).pts
    assert np.array_equal(a, b), f'{label}: same seed gave different points'
    assert not np.array_equal(a, c), f'{label}: different seeds gave identical points'


def test_int_pts_without_rng_still_follows_global_seed():
    """Back-compat: rng=None must keep using numpy's global RNG."""
    np.random.seed(0)
    a = G.L_shape().int_pts(method='random', npts_rand=20).pts
    np.random.seed(0)
    b = G.L_shape().int_pts(method='random', npts_rand=20).pts
    assert np.array_equal(a, b)


def test_fundamental_int_pts_honours_rng():
    """`rng` was in this signature from the start but was ignored."""
    from lappy.symmetry import domain_symmetry, fundamental_int_pts
    d, g = G.L_shape(), domain_symmetry('L_shape')
    a = fundamental_int_pts(d, g, 20, rng=3).pts
    b = fundamental_int_pts(d, g, 20, rng=3).pts
    assert np.array_equal(a, b)
