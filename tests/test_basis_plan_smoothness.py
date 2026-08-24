"""Does a frozen plan behave well as the shape moves?

This is the acceptance test for the whole plan/realize architecture. `test_basis_plan.py` checks
that a frozen plan realizes to a constant SIZE along a family; that is necessary and not
sufficient. What a shape optimizer actually needs is that the quantities it consumes -- lambda and
its shape derivative -- move smoothly and remain accurate as the domain is perturbed, using the
plan that was made for a *different* domain.

The rectangle is the instrument, because both truths are closed form there:

    lam_mn = pi^2 (m^2/L^2 + n^2/H^2)              exact for every member of the family
    dlam/dL = -2 pi^2 m^2 / L^3                    exact, translating the x = L edge

so no finite differencing and no reference table is involved, and a failure is the basis's. The
expensive finite-difference version on a singular-corner domain lives in
`benchmarks/basis_lab/plan_lab.py` (stage `smooth`), which is where slow things belong.

`docs/scope_and_downstream.md` section 4 requires lappy to keep a `dlambda` check permanently,
because it is the only instrument sensitive to a *systematic* error in `||u||` -- the certified
eigenvalue bound is scale-invariant and provably cannot see one. These tests are that check applied
to the planner.
"""
import warnings

import numpy as np
import pytest

from lappy import basis_plan as BP
from lappy import geometry as geo
from lappy import mps
from lappy import reference as ref
from lappy.asymp import weyl_est
from lappy.eigfun_integrals import boundary_quadrature, eigfun_cauchy_data, weighted_integral
from lappy.mps import MPSEigensolver

M_MODE, N_MODE = 2, 1
H = 1.0
LS = (1.7, 1.85, 2.0, 2.15, 2.3)      # the family; the plan is made at L = 2.0
L_PLAN = 2.0


def _solver(dom, basis, lam_max, seed=7):
    """Deliberately not `from_domain`: the shape-derivative contract needs `bdry_quad` attached and
    an interior draw that does not move between family members, and this mirrors
    `tests/test_shape_derivative.py::_solver`, which is the documented path for that contract."""
    bdry = dom.bdry_pts(mps.pts_per_seg(dom, basis, mult=2))
    interior = dom.int_pts(method='random', npts_rand=max(2*len(basis), 400),
                           rng=np.random.default_rng(seed))
    bq = boundary_quadrature(dom, lam_max, precision=1e-13, warn=False)
    return MPSEigensolver(basis.to_normalized((bdry, interior)), bdry, interior,
                          rtol=1e-14, ttol=1e-3, bdry_quad=bq)


def _moving_edge_mask(bq, L):
    seg = np.array([bq.panels[p].seg_idx for p in bq.panel_id])
    for i in np.unique(seg):
        if np.allclose(bq.pts[seg == i].real, L, atol=1e-9):
            return (seg == i).astype(float)
    raise RuntimeError('failed to identify the moving edge')


@pytest.fixture(scope='module')
def frozen():
    """One plan, made once on rect(2, 1), reused for every member of the family."""
    dom = geo.rect(L_PLAN, H)
    lam_max = 3*float(ref.rect_eig(M_MODE, N_MODE, L_PLAN, H))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom, lam_max, target=1e-10)
    return plan, lam_max


@pytest.fixture(scope='module')
def swept(frozen):
    """sigma, lambda-error and dlambda for the whole family, computed once."""
    plan, lam_max = frozen
    out = []
    for L in LS:
        dom = geo.rect(L, H)
        lam = float(ref.rect_eig(M_MODE, N_MODE, L, H))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            basis = BP.realize(plan, dom)
            solver = _solver(dom, basis, lam_max)
            sigma = float(np.atleast_1d(solver.sigma(lam))[0])
            Vn = _moving_edge_mask(solver.bdry_quad, L)
            coef = solver.eigenfunction_coef(lam, mult=1)
            ed = eigfun_cauchy_data(solver.basis, lam, coef, solver.bdry_quad)
            dlam = float(-weighted_integral(ed, 'NN', Vn)[0, 0])
        out.append(dict(L=L, n=len(basis), lam=lam, sigma=sigma, dlam=dlam,
                        dlam_exact=-2*np.pi**2*M_MODE**2/L**3))
    return out


def test_the_frozen_plan_gives_the_same_basis_size_everywhere(swept):
    assert len({r['n'] for r in swept}) == 1, [r['n'] for r in swept]


def test_the_tension_stays_small_across_the_family(swept):
    """A plan made at L=2 must still resolve L=1.7 and L=2.3 -- a 15% shape change, far larger than
    an optimizer step. If sigma at the known eigenvalue degraded, the frozen plan would have to be
    re-planned every few iterations and the architecture would buy nothing."""
    worst = max(r['sigma'] for r in swept)
    assert worst < 1e-8, [(r['L'], f"{r['sigma']:.2e}") for r in swept]


def test_the_shape_derivative_is_accurate_across_the_family(swept):
    """The headline. `dlam` from a plan built for a different domain, against closed form."""
    for r in swept:
        err = abs(r['dlam'] - r['dlam_exact'])/abs(r['dlam_exact'])
        assert err < 1e-9, (r['L'], r['dlam'], r['dlam_exact'], err)


def test_the_derivative_is_consistent_with_a_finite_difference_of_the_SOLVED_lambda(frozen):
    """`dlam` must be the derivative of the lambda this basis actually produces.

    The `lam` used elsewhere in this file is the closed form, so differencing it would test the
    closed forms against each other and never touch the basis. Here lambda is *solved for* at
    `L +- h` by minimizing the tension, with the frozen plan realized on each perturbed domain --
    which is exactly the quantity an optimizer differences when it estimates a gradient by hand.

    `h = 1e-3`, so the central difference's own truncation error is about `(h^2/6)|lam'''| ~ 1e-6`
    relative; the bar is set just above that. Do not widen h to make this "more stable": at
    h = 0.15 the truncation error alone is 1.1e-2 and the test stops being about the basis.
    """
    from lappy.opt import minimize_on_bracket
    plan, lam_max = frozen
    h = 1e-3

    def solved_lam(L):
        dom = geo.rect(L, H)
        lam0 = float(ref.rect_eig(M_MODE, N_MODE, L, H))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = _solver(dom, BP.realize(plan, dom), lam_max)
            f = lambda x: float(np.atleast_1d(solver.sigma(float(x)))[0])   # noqa: E731
            xs = lam0*(1.0 + np.linspace(-1e-4, 1e-4, 7))
            ys = np.array([f(x) for x in xs])
            i = int(np.clip(np.argmin(ys), 1, len(xs) - 2))
            lam, _ = minimize_on_bracket(f, ((xs[i-1], xs[i], xs[i+1]),
                                             (ys[i-1], ys[i], ys[i+1])), 1e-15)
        return float(lam)

    fd = (solved_lam(L_PLAN + h) - solved_lam(L_PLAN - h))/(2*h)
    exact = -2*np.pi**2*M_MODE**2/L_PLAN**3
    assert abs(fd - exact)/abs(exact) < 1e-5, (fd, exact)


def test_the_derivative_curve_has_no_basis_induced_kinks(swept):
    """Second differences of the COMPUTED `dlam` must match those of the exact `dlam`.

    Not "second differences are small": `dlam = -2 pi^2 m^2 / L^3` has real curvature, and over
    this grid its own second differences are 0.47 to 1.0 against values of order 16 -- so a test
    that asked for smallness would be measuring the function, not the basis. What a basis
    reorganizing itself mid-family would do is make the computed curve depart from the exact one,
    and that is what this compares.
    """
    got = np.diff(np.array([r['dlam'] for r in swept]), 2)
    exact = np.diff(np.array([r['dlam_exact'] for r in swept]), 2)
    assert np.max(np.abs(got - exact)) < 1e-9*np.max(np.abs(exact)), (got, exact)


def test_two_realizations_of_one_plan_agree_bitwise(frozen):
    """Determinism, at the level the inner loop needs: identical inputs, identical answer. Before
    the S1 fix to `make_default_int_pts` this failed for `from_domain` callers, and a
    finite-difference gradient of a nondeterministic objective is noise."""
    plan, lam_max = frozen
    dom = geo.rect(L_PLAN, H)
    lam = float(ref.rect_eig(M_MODE, N_MODE, L_PLAN, H))
    vals = []
    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = MPSEigensolver.from_domain(dom, basis=BP.realize(plan, dom), prec=1e-14)
            vals.append(float(np.atleast_1d(solver.sigma(lam))[0]))
    assert vals[0] == vals[1], vals


def test_cost_is_stable_along_the_family(frozen):
    """Per-iterate cost must not drift, or a time budget cannot be set. Column count is the proxy
    that matters (it drives both the evaluation and the factorization, PLAN_LAB.md S0a); wall time
    is too noisy to assert on."""
    plan, _ = frozen
    counts = {len(BP.realize(plan, geo.rect(L, H))) for L in LS}
    assert len(counts) == 1, counts


def test_replanning_every_iterate_would_change_the_basis_size(frozen):
    """The case FOR freezing, stated as a measurement rather than an argument.

    Re-planning at each member of the family gives 98, 100, 102, 104 columns across a 35% change in
    L -- correct behaviour for a planner (a longer edge genuinely wants more sources) and a moving
    target for an optimizer, which would be differencing two different bases. The frozen plan gives
    one size everywhere (`test_the_frozen_plan_gives_the_same_basis_size_everywhere`).

    The variation is also monotone in L, i.e. smooth rather than a threshold jump -- so on THIS
    family re-planning would merely add noise, not a discontinuity. The old recipe's weak/singular
    reclassification is the documented case where it would have added a discontinuity.
    """
    plan, lam_max = frozen
    sizes = []
    for L in LS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sizes.append(BP.plan_basis(geo.rect(L, H), lam_max, target=1e-10).n_total)
    assert len(set(sizes)) > 1, f'expected re-planning to vary the size, got {sizes}'
    assert sizes == sorted(sizes), sizes
    assert max(sizes) - min(sizes) < 0.15*min(sizes), sizes   # smooth drift, not a jump


# ── Moving a VERTEX, where the corner angles change too ─────────────────────────
#
# Everything above translates an edge, so every corner angle is fixed and only `clearance` and
# the arc lengths move. A polygon parametrization does not work that way: moving a vertex moves
# two edges and changes two interior angles at once, and `alpha = pi/omega` is what sets a corner
# block's exponents. `realize` freezes the term COUNT `M` and the arc endpoints as fractions, but
# recomputes alpha from the moved geometry -- so this asks whether that split is the right one.
#
# The rectangle cannot test this (a rectangle with a moved vertex is not a rectangle, and there is
# no closed form), so the instrument is a generic convex quadrilateral and the check is internal
# consistency: constant size, tension staying at the floor, and a smoothly varying nu.

QUAD = np.array([0.0, 1.3, 1.6 + 1.0j, 0.15 + 0.85j])
QUAD_DIR = 0.6 + 0.8j
QUAD_TS = (-0.04, -0.02, 0.0, 0.02, 0.04)


def _quad_moved(t, k=2):
    v = QUAD.copy()
    v[k] = v[k] + t*QUAD_DIR
    return geo.Polygon(v)


@pytest.fixture(scope='module')
def vertex_swept():
    """One plan frozen at t=0, realized on each member of a vertex-moving family."""
    from lappy import Eigenproblem
    from lappy.eigfun_integrals import corner_specs
    dom0 = _quad_moved(0.0)
    lam_max = weyl_est(6, dom0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom0, lam_max, target=1e-12)

    out, lam = [], None
    for t in QUAD_TS:
        dom = _quad_moved(t)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            basis = BP.realize(plan, dom)
            solver = MPSEigensolver.from_domain(dom, lam_max=weyl_est(6, dom), basis=basis,
                                                prec=1e-12)
            evp = Eigenproblem(dom, eval_solver=solver, precision=1e-12)
            lam = float(evp.solve(1)[0]) if lam is None else float(evp.track(lam))
            sigma = float(np.atleast_1d(solver.sigma(lam))[0])
        out.append(dict(t=t, n=len(basis), lam=lam, sigma=sigma,
                        nus=sorted(float(c.nu) for c in corner_specs(dom))))
    return out


def test_a_frozen_plan_survives_vertex_motion(vertex_swept):
    """Constant size and a tension still at the floor, though every angle has moved.

    This is the claim the edge-translating family above cannot make: `alpha` is recomputed from
    the moved geometry while `M` stays frozen, and that has to be enough.
    """
    assert len({r['n'] for r in vertex_swept}) == 1, [r['n'] for r in vertex_swept]
    worst = max(r['sigma'] for r in vertex_swept)
    assert worst < 1e-9, [(r['t'], r['sigma']) for r in vertex_swept]


def test_the_corner_exponents_move_smoothly_under_vertex_motion(vertex_swept):
    """`nu` is what the corner blocks are built on, so a kink in it would be a kink in the basis.

    Second differences are compared against the curve's own scale rather than against zero: nu(t)
    has real curvature here, and asking for it to be small is the badly-designed assertion
    PLAN_LAB records failing three of eight smoothness tests on the first run.
    """
    nus = np.array([r['nus'] for r in vertex_swept])          # (n_t, n_corner)
    for j in range(nus.shape[1]):
        d1 = np.diff(nus[:, j])
        if np.max(np.abs(d1)) < 1e-12:
            continue                                          # a corner that does not move
        d2 = np.abs(np.diff(d1))
        assert np.max(d2) < 0.25*np.max(np.abs(d1)), (j, nus[:, j])


def test_lambda_is_smooth_under_vertex_motion(vertex_swept):
    """Third differences of a smooth curve sampled at five points are dominated by its own
    curvature; a basis-induced kink would break the ordering."""
    lams = np.array([r['lam'] for r in vertex_swept])
    d1 = np.diff(lams)
    assert np.all(d1 < 0), lams                               # monotone: pushing out lowers lam_1
    d3 = np.abs(np.diff(np.diff(d1)))
    assert np.max(d3) < 0.02*np.max(np.abs(d1)), lams
