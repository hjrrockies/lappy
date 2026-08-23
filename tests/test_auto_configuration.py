"""The three-line API in CLAUDE.md principle 1, and the `precision` dial behind it.

    dom  = Domain(*args)
    evp  = Eigenproblem(dom)
    eigs = evp.solve(n)

Those three lines could not run before this: `MPSEigensolver.from_domain(basis=None)` raised
`NotImplementedError`, so every caller hand-built a basis, a collocation set and a quadrature --
and `docs/todo.md` records the reason it stayed that way, that `n_basis` is "the one quantity
[callers] have no principled way to choose". `lappy.basis_plan` chooses it from geometry,
`lam_max` and a target instead.

`precision` is one dial at both stages: it sizes the basis and becomes the search's `ltol`. These
tests assert it reaches both, and that the accuracy it buys is monotone -- not that it is a
guarantee, which it is not (see `benchmarks/basis_lab/PLAN_LAB.md`).
"""
import warnings

import numpy as np
import pytest

from lappy import Eigenproblem, geometry as geo, mps, reference as ref
from lappy.mps import MPSEigensolver


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def test_the_three_line_api_solves_the_L_shape():
    evp = Eigenproblem(geo.L_shape())
    eigs = evp.solve(4)
    truth = ref.L_shape_eigs(4)
    assert len(eigs) == 4
    digits = -np.log10(np.abs(eigs - truth)/truth)
    assert digits.min() > 8.0, dict(zip(np.round(eigs, 10), np.round(digits, 1)))


def test_from_domain_builds_a_basis_when_none_is_given():
    solver = MPSEigensolver.from_domain(geo.rect(1.0, 1.0))
    assert 20 < len(solver.basis) < 240        # geometry-derived, and under the rank ceiling
    assert solver.bdry_quad is not None


def test_precision_reaches_both_the_basis_and_the_search():
    """The dial's whole point: a basis good to `p` is wasted if the search stops at 1e-8, and a
    search asked for more than the basis supports is wasted too."""
    sizes, ltols = [], []
    for p in (1e-4, 1e-8, 1e-12):
        evp = Eigenproblem(geo.L_shape(), precision=p)
        evp.solve(1)
        sizes.append(len(evp.eval_solver.basis))
        ltols.append(evp.eval_solver.ltol)
    assert ltols == [1e-4, 1e-8, 1e-12]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1], sizes


def test_a_finer_precision_buys_accuracy():
    got = []
    for p in (1e-4, 1e-12):
        evp = Eigenproblem(geo.L_shape(), precision=p)
        eigs = evp.solve(3)
        truth = ref.L_shape_eigs(3)
        got.append(float(np.min(-np.log10(np.abs(eigs - truth)/truth))))
    assert got[1] > got[0] + 2.0, got


def test_no_solver_is_built_until_one_is_needed():
    """Constructing an `Eigenproblem` must stay cheap -- a basis, a collocation set and a boundary
    quadrature is not something to pay for on an object that may never be solved."""
    evp = Eigenproblem(geo.L_shape(), precision=1e-8)
    assert evp.eval_solver is None
    evp.solve(1)
    assert evp.eval_solver is not None


def test_the_solver_is_cached_across_solves():
    evp = Eigenproblem(geo.rect(1.0, 1.0))
    evp.solve(1)
    first = evp.eval_solver
    evp.solve(2)
    assert evp.eval_solver is first


def test_eigenfunctions_work_without_a_separate_evec_solver():
    evp = Eigenproblem(geo.rect(1.0, 1.0))
    eigs = evp.solve(1)
    u = evp.eigenfunction(float(eigs[0]))
    v = u(geo.PointSet(np.array([0.5 + 0.5j])))
    assert np.abs(v).max() > 0.1        # the mode does not vanish at the centre


def test_a_curved_domain_says_what_to_do_instead():
    """The planner is polygon-only. Inventing a size for a curved domain here would be exactly the
    unfounded `n_basis` guess the auto path exists to remove, so it raises and names the
    alternatives."""
    with pytest.raises(NotImplementedError, match='polygons only'):
        MPSEigensolver.from_domain(geo.disk(1.0))


def test_check_precision_reports_success_and_failure_honestly():
    """The gap this closes: before, nothing said whether a requested precision was met, and on the
    sharp-cornered domains it is not. `plan.capped` only reports what the planner knew in advance."""
    ok = Eigenproblem(geo.L_shape(), precision=1e-10)
    rep = ok.check_precision(ok.solve(3))
    assert rep['met'] and rep['achieved'] <= rep['target'], rep

    short = Eigenproblem(geo.iso_tri(4.0), precision=1e-10)
    rep = short.check_precision(short.solve(3))
    assert not rep['met'] and rep['achieved'] > rep['target'], rep
    assert rep['digits'] < 10.0


def test_check_precision_agrees_with_a_full_certification():
    """It is a cheap stand-in for Moler--Payne, so it has to track it. Measured agreement across
    150 cells is 0.16 digits; 1.0 here leaves room for the graded-sampling difference."""
    from benchmarks.basis_lab.plan_lab import certify
    evp = Eigenproblem(geo.L_shape(), precision=1e-10)
    eigs = evp.solve(1)
    rep = evp.check_precision(eigs)
    mp = certify(evp.eval_solver, evp.domain, float(eigs[0]))
    assert abs(rep['digits'] - mp['digits']) < 1.0, (rep['digits'], mp['digits'])


def test_check_precision_is_none_for_a_hand_built_basis():
    """No per-arc structure, nothing to attribute a residual to. Better than inventing one."""
    from lappy import bases
    dom = geo.L_shape()
    solver = MPSEigensolver.from_domain(dom, basis=bases.make_default_basis(dom, 120))
    evp = Eigenproblem(dom, eval_solver=solver)
    assert evp.check_precision(evp.solve(1)) is None


def test_refine_basis_closes_the_gap_it_reports():
    evp = Eigenproblem(geo.iso_tri(4.0), precision=1e-10)
    eigs = evp.solve(3)
    before = evp.check_precision(eigs)
    evp.refine_basis(eigs)
    after = evp.check_precision(evp.solve(3))
    assert after['n_basis'] > before['n_basis']
    assert after['digits'] > before['digits'] + 1.0, (before['digits'], after['digits'])


def test_refine_basis_is_a_no_op_when_the_target_is_already_met():
    evp = Eigenproblem(geo.L_shape(), precision=1e-10)
    eigs = evp.solve(2)
    n_before = len(evp.eval_solver.basis)
    plan = evp.refine_basis(eigs)
    assert plan.n_total == n_before
    assert len(evp.eval_solver.basis) == n_before


def test_refining_for_several_eigenvalues_serves_the_worst_of_them():
    """A plan refined only at lam_1 under-serves lam_k, because higher modes oscillate faster.
    `residual_by_arc` takes the worst over whatever eigenvalues it is given, so the choice of which
    modes matter stays with the caller -- an optimizer tracking one mode wants one."""
    from lappy import basis_plan as BP
    dom = geo.L_shape()
    evp = Eigenproblem(dom, precision=1e-10)
    eigs = evp.solve(4)
    solver, plan = evp.eval_solver, BP.plan_of(evp.eval_solver.basis)
    one = BP.residual_by_arc(plan, dom, solver, eigs[:1])
    many = BP.residual_by_arc(plan, dom, solver, eigs)
    worst_one = max(one[0].max(initial=0), one[1].max(initial=0))
    worst_many = max(many[0].max(initial=0), many[1].max(initial=0))
    assert worst_many >= worst_one*(1 - 1e-12)


def test_the_planner_reads_the_solver_s_rtol():
    """`basis_plan`'s conditioning ceilings are defined by the solver's rank truncation, not by
    machine epsilon. If the two constants drifted apart the ceilings would be mis-sized, and
    getting that wrong made achieved accuracy non-monotone in the target (PLAN_LAB.md, S2)."""
    from lappy.basis_plan import PlanConfig
    assert PlanConfig().rtol == mps.rtol_default

    loose = mps.default_basis_for(geo.L_shape(), 42.5, target=1e-10, rtol=1e-6)
    tight = mps.default_basis_for(geo.L_shape(), 42.5, target=1e-10, rtol=1e-14)
    assert len(loose) < len(tight), (len(loose), len(tight))
