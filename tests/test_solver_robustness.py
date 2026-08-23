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

def test_bracket_mins_has_a_generous_depth_backstop():
    """Depth is a backstop, not the control.

    A tight cap throttles well-resolved problems to protect against noisy
    ones, and has a bad regime in the middle: on rect(1, 1.00001), whose
    eigenvalues come in pairs 1.2e-5 apart, capping at 8 leaves them
    unresolved (4.8 true digits) while 12 resolves them partially and
    misaligns the list (0.2 digits). The noise test is the real control.
    """
    import inspect
    default = inspect.signature(opt.bracket_mins).parameters['max_recurse'].default
    assert default is None or default >= 20


def test_bracket_mins_stops_on_a_noise_floor():
    """A curve that is roundoff wiggle must stop refining.

    Models a real tension floor: values scattered around a small positive
    level and bounded away from zero. Every apparent minimum there is
    spurious, so subdividing only manufactures more of them.

    Two outcomes are acceptable and both correct: abort outright (the existing
    "too many local minima" guard, the right response to an unusable curve), or
    terminate cheaply via the noise test. What must not happen is unbounded
    refinement.
    """
    from lappy.core import EigensolverFailure
    rng = np.random.default_rng(0)
    x = np.linspace(1.0, 2.0, 101)
    floor = 1e-14

    def noisy(t):
        return floor * (1.0 + 0.5 * rng.random(2))

    y = floor * (1.0 + 0.5 * rng.random((2, len(x))))
    try:
        brackets, fevals = opt.bracket_mins(noisy, x, y, xtol=1e-12)
    except EigensolverFailure:
        return          # aborted rather than guessing -- the desired behaviour
    assert fevals < 200000, f'refinement ran away: {fevals} evaluations'


def test_bracket_mins_still_resolves_a_genuine_well():
    """The noise test must not stop refinement on real structure."""
    x = np.linspace(0.0, 2.0, 51)

    def f(t):
        v = abs(t - 1.0) + 1e-16
        return np.array([v, v])

    y = np.vstack([np.abs(x - 1.0) + 1e-16] * 2)
    brackets, _ = opt.bracket_mins(f, x, y, xtol=1e-12)
    assert brackets, 'lost a genuine minimum'
    lo, mid, hi = brackets[0][0]
    assert lo <= 1.0 <= hi


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


def test_make_default_int_pts_honours_rng():
    """The plumbing above `Domain.int_pts`: `make_default_int_pts` had no rng at all, so every
    default solver build drew from the global RNG no matter what the caller asked for."""
    from lappy.mps import make_default_int_pts
    dom = G.L_shape()
    a = make_default_int_pts(dom, 'random', npts_rand=20, rng=5).pts
    b = make_default_int_pts(dom, 'random', npts_rand=20, rng=5).pts
    c = make_default_int_pts(dom, 'random', npts_rand=20, rng=6).pts
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_from_domain_rng_makes_two_solver_builds_comparable():
    """The reason this matters. Without a seed, two solvers built for the same domain have
    different interior points and therefore different coefficients in every element -- which
    silently invalidates any A/B comparison between two builds."""
    from lappy import bases
    from lappy.mps import MPSEigensolver
    dom = G.L_shape()

    def build(**kw):
        return MPSEigensolver.from_domain(dom, basis=bases.make_default_basis(dom, 60),
                                          orthonorm=False, **kw)

    seeded = [build(rng=11).int_pts.pts for _ in range(2)]
    assert np.array_equal(*seeded), 'same seed must give the same interior points'
    assert not np.array_equal(seeded[0], build(rng=12).int_pts.pts)


def test_convergence_tests_default_draw_is_unchanged():
    """`make_solver` gained an `rng` knob; its DEFAULT must still be the legacy global-MT19937
    draw. np.random.seed(0) and default_rng(0) are different generators, so defaulting the new
    parameter to 0 would have moved every convergence curve the module has produced."""
    from lappy import convergence_tests as ct
    dom = G.L_shape()
    np.random.seed(0)
    expected = dom.int_pts(npts_rand=30).pts
    solver = ct.make_solver(dom, 30, 0, 0, {'d': 0.1}, {'C': 1.0, 'sigma': 1.0})
    assert np.array_equal(solver.int_pts.pts[:len(expected)], expected[:len(solver.int_pts.pts)])


# --- 5. curved domains reach the solver at all -----------------------------

def test_make_default_int_pts_handles_curved_domains():
    """`make_default_int_pts` raised a bare NotImplementedError for anything that was not a
    Polygon, which kept every curved domain out of MPSEigensolver.from_domain entirely --
    ellipse_a4 and stadium could not be built, so the smooth-panel plateau measured on them was
    unreachable through the solver. Domain.int_pts handled both kinds all along."""
    from lappy.mps import make_default_int_pts
    for dom in (G.ellipse(4, 1), G.stadium(1, 1), G.disk(1)):
        pts = make_default_int_pts(dom, 'random', npts_rand=30, rng=3)
        assert len(pts.pts) == 30
        assert np.all(dom.contains(pts.pts))


@pytest.mark.parametrize('weights', [False, True])
@pytest.mark.parametrize('dom,label', [(G.L_shape(), 'Polygon'), (G.ellipse(4, 1), 'curved')])
def test_mesh_int_pts_does_not_raise_on_an_array_truth_test(dom, label, weights):
    """`kind='mesh'` rebound its own `weights` parameter to the cubature weight ARRAY and then
    tested `if weights:`, so it raised "truth value of an array is ambiguous" for both values of
    the flag. It had therefore never worked, on any domain."""
    from lappy.mps import make_default_int_pts
    pts = make_default_int_pts(dom, 'mesh', weights=weights, lam_max=100.0)
    assert len(pts.pts) > 0
    # PointSet only HAS a .wts attribute when it was built with weights
    assert hasattr(pts, 'wts') == weights, f'{label}: weights flag ignored'


def test_make_default_int_pts_rejects_an_unknown_kind():
    from lappy.mps import make_default_int_pts
    with pytest.raises(ValueError, match="'random' or 'mesh'"):
        make_default_int_pts(G.L_shape(), 'nonsense')


@pytest.mark.slow
def test_disk_solves_end_to_end_against_exact_bessel_eigenvalues():
    """The acceptance test for the above: a curved domain through the full default pipeline,
    checked against closed-form truth (squared Bessel zeros). This could not run at all before
    -- from_domain raised NotImplementedError building the interior points.

    THE LOWER ENDPOINT MUST BE NUDGED, and the disk is the one domain where it matters.
    Faber--Krahn is sharp, with equality exactly for the disk, so `faber_krahn(disk)` and lam_1
    agree to ten digits -- 5.7831859629 either way. `opt.discrete_locmin_idx` cannot return a
    minimum sitting on the edge of its grid, which is why `evp._solve_dir_neu` multiplies the
    bound by `1 - _WINDOW_PAD` before searching and says so at length. This test used to scan
    from the raw bound and passed anyway, on roundoff luck: whether lam_1 was found came down to
    a last-ulp comparison against the ghost point below it, and a 1-ulp change in the Bessel
    evaluation (the order-0 fast path in `FundamentalBasis._bessel`) flipped it. Nothing about
    the solver changed -- `Eigenproblem.solve(2)`, which applies the pad itself, returns both
    modes before and after, and the assertion below now checks that path too.
    """
    sp = pytest.importorskip('scipy.special')
    from lappy import bases, Eigenproblem
    from lappy.evp import _WINDOW_PAD
    from lappy.mps import MPSEigensolver, weyl_est
    dom = G.disk(1)
    exact = np.sort(np.concatenate([sp.jn_zeros(m, 3)**2 for m in range(4)]))

    solver = MPSEigensolver.from_domain(dom, basis=bases.make_default_basis(dom, 120), rng=7)
    a = bounds.faber_krahn(dom)*(1.0 - _WINDOW_PAD)
    out = solver.solve_interval(a, weyl_est(2, dom), 20)
    eigs = np.atleast_1d(np.asarray(out[0] if isinstance(out, tuple) else out)).ravel()
    assert len(eigs) >= 2, eigs
    for computed in eigs[:2]:
        assert np.min(np.abs(exact - computed)/computed) < 1e-6, computed

    # ...and the same thing through the pipeline a caller actually uses, which owns the nudge.
    through_evp = np.asarray(Eigenproblem(dom, eval_solver=solver).solve(2))
    assert len(through_evp) == 2, through_evp
    for computed in through_evp:
        assert np.min(np.abs(exact - computed)/computed) < 1e-6, computed
