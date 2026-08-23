"""`Eigenproblem.track`: follow one eigenvalue across a shape family without rescanning.

WHY IT EXISTS. `solve(k)` re-scans the spectrum from the Faber--Krahn bound on every call --
2-8 s for four eigenvalues, against ~10 ms of solver construction -- and a shape-optimization
loop already knows where `lambda` was at the previous iterate. Tracking is also the answer to
`solve(k)`'s set-selection problem (see `tests/test_mode_completeness.py`): following a mode by
VALUE cannot silently hand back a different index, because there is no index involved.

THE GUARD IS THE INTERESTING PART. A local scan whose window the eigenvalue has outrun returns
the window EDGE -- a confident, wrong, perfectly finite number.
`benchmarks/basis_lab/PLAN_LAB.md` records that exact failure producing a reference wrong by 16%,
which then read as every basis under test being wrong by an identical 1.35e-01, and notes it was
the second time in that directory. So `track` raises rather than returning an edge, and the
window is sized from the local Weyl mean spacing rather than being a fixed relative fraction --
a fixed fraction is precisely what failed there.

The reference numbers are `PLAN_LAB.md`'s frozen-plan L-shape sweep, where one plan (n=158) is
realized on each member and sigma never leaves 6e-14.
"""
import warnings

import numpy as np
import pytest

from lappy import Eigenproblem, basis_plan as BP, geometry as geo
from lappy.asymp import weyl_est
from lappy.core import EigensolverFailure
from lappy.mps import MPSEigensolver


def _l_family(t):
    """L_shape() with the x=-1 edge translated outward by `t`. t=0 is exactly `geo.L_shape()`."""
    a = 1.0 + t
    return geo.Polygon([0, 1j, -a + 1j, -a - 1j, 1 - 1j, 1], bc='dir', val_simple=False)


# PLAN_LAB.md S3 `smooth`: one frozen plan realized on each member.
FAMILY = [(-0.030, 9.9147306531), (-0.015, 9.7767231280), (0.000, 9.6397238440),
          (+0.015, 9.5039405795), (+0.030, 9.3695663833)]


@pytest.fixture(scope='module')
def frozen():
    """A plan built once at t=0, plus a factory realizing it on any member.

    This is the architecture the whole design exists to support: term counts and arc endpoints
    are frozen, only positions move, so `n_basis` is identical across the family and a change in
    lambda came from the shape rather than from the basis.
    """
    dom0 = _l_family(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom0, weyl_est(6, dom0), target=1e-10)

    def make(t):
        dom = _l_family(t)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = MPSEigensolver.from_domain(dom, basis=BP.realize(plan, dom), prec=1e-12)
        return dom, Eigenproblem(dom, eval_solver=solver, precision=1e-12)

    return plan, make


@pytest.mark.slow
def test_track_walks_the_family_from_one_cold_start(frozen):
    """The inner loop, end to end: one `solve` at t=0, then `track` from the previous iterate.

    Each step is seeded from the PREVIOUS member's eigenvalue, not from lambda(0) -- continuation
    is what keeps the scan window on top of the mode as it moves.
    """
    plan, make = frozen
    _, evp0 = make(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lam = float(evp0.solve(1)[0])
    assert abs(lam - 9.6397238440) < 1e-9, lam

    sizes = []
    for t, expected in FAMILY:
        _, evp = make(t)
        sizes.append(len(evp.eval_solver.basis))
        lam = evp.track(lam)
        assert abs(lam - expected) < 1e-8, f't={t}: got {lam:.10f}, want {expected:.10f}'

    assert len(set(sizes)) == 1, f'the frozen plan must give one size across the family: {sizes}'


@pytest.mark.slow
def test_track_agrees_with_a_full_solve(frozen):
    """Tracking must not be a cheaper answer to a different question."""
    plan, make = frozen
    _, evp = make(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        full = float(evp.solve(1)[0])
    tracked = evp.track(full*1.0005)
    assert abs(tracked - full)/full < 1e-11, (tracked, full)


@pytest.mark.slow
def test_a_minimum_on_the_window_edge_raises(frozen):
    """The guard. A window the eigenvalue has left must not quietly return its edge.

    Forced with an absurdly narrow `window` offset from the mode, which is the shape of the real
    failure: the scan is centred where lambda USED to be and the minimum lies outside it.
    """
    plan, make = frozen
    _, evp = make(0.0)
    with pytest.raises(EigensolverFailure, match='edge of the scan window'):
        evp.track(9.6397238440 + 0.02, window=1e-3)


@pytest.mark.slow
def test_tracking_away_from_any_mode_raises(frozen):
    """A loop that has stepped off its mode must be told, not handed the nearest dip.

    Between lambda_1 = 9.640 and lambda_2 = 15.197 there is no eigenvalue, so a scan centred at
    12.5 has no interior minimum and the edge guard is what fires. Either guard is a correct
    refusal; what is forbidden is a return value.
    """
    plan, make = frozen
    _, evp = make(0.0)
    with pytest.raises(EigensolverFailure, match='edge of the scan window|is above ttol'):
        evp.track(12.5, window=0.35)


@pytest.mark.slow
def test_the_tension_check_is_applied_at_the_located_minimum(frozen):
    """The other half of the refusal, isolated from the tension landscape.

    `track` must not accept a minimum on the strength of its being a minimum: sigma has to be
    small there too. Driven by an unsatisfiable `ttol` so the assertion is about the guard's
    logic rather than about finding a domain with a conveniently spurious dip.
    """
    plan, make = frozen
    _, evp = make(0.0)
    evp.eval_solver.ttol = 1e-30
    with pytest.raises(EigensolverFailure, match='is above ttol'):
        evp.track(9.6397238440)


@pytest.mark.slow
def test_asking_for_the_wrong_multiplicity_raises(frozen):
    """lambda_1 on the L-shape is simple, so tracking it as a double must fail rather than
    return a number. `sigma[mult-1]` is the check, so a cluster that has split is caught too."""
    plan, make = frozen
    _, evp = make(0.0)
    with pytest.raises(EigensolverFailure, match='multiplicity 2'):
        evp.track(9.6397238440, mult=2)


def test_the_window_is_sized_from_the_local_spectrum():
    """Not a fixed relative fraction -- that is what failed in PLAN_LAB's reference. The spacing
    must shrink as the spectrum densifies, so one constant works at lambda_1 and at lambda_50."""
    dom = geo.L_shape()
    evp = Eigenproblem(dom)
    wide, narrow = evp._mean_spacing(9.64), evp._mean_spacing(500.0)
    assert wide > narrow > 0, (wide, narrow)
    # and it must not reach the neighbouring eigenvalue at lambda_1: gap is 15.197 - 9.640
    assert wide/3.0 < (15.19725193 - 9.63972384), wide


def test_track_rejects_nonsense_arguments():
    evp = Eigenproblem(geo.L_shape())
    with pytest.raises(ValueError, match="'lam_prev' must be positive"):
        evp.track(-1.0)
    with pytest.raises(ValueError, match="'n_pts' must be at least 3"):
        evp.track(9.64, n_pts=2)
