"""Tests for L^2-orthonormalization of eigenfunctions: lappy.eigfun_integrals wired into
MPSEigensolver.eigenfunction_coef, plus the x0-invariance diagnostic (Leg 4).

Replaces the old test_rellich.py / test_cauchy.py. What those covered and this does not:
anything about the retired BASIS-LEVEL Gram (rellich_gram_basis, orthonormalize_coef,
_cauchy_gram). That path is gone by design, not untested -- a basis-level Gram mixes columns
centred at other corners, which are plain analytic there, so no corner-adapted rule can serve
it. The corner rule's own accuracy is validated in test_eigfun_integrals.py (Legs 1-3) and
test_quad.py.
"""

import warnings

import numpy as np
import pytest
import scipy.linalg as la
from numpy.polynomial.legendre import leggauss

from lappy import Polygon, geometry, bases, cubature, bounds
from lappy.bases import FourierBesselBasis
from lappy.reference import rect_eig
from lappy.geometry import L_shape, H_shape, plus_shape, PointSet
from lappy.mps import MPSEigensolver, NormalizedBasis, make_default_bdry_data, make_default_int_pts
import lappy.mps as mps_mod
from lappy.eigfun_integrals import (boundary_quadrature, EigfunData, gram, lowdin_transform,
                                    default_x0)

L, H = 2.0, 1.0
RECT_VERTS = np.array([0, L, L + 1j*H, 1j*H])
SQ_VERTS = np.array([0, 1, 1 + 1j, 1j])


def _gl_norm_sq(f, L, H, nx=60, ny=60):
    """L^2 norm^2 of a scalar eigenfunction callable over [0,L]x[0,H] via tensor
    Gauss-Legendre quadrature -- an INDEPENDENT check on the boundary-integral
    normalization, sharing no code with it."""
    xg, wx = leggauss(nx); xg = (xg + 1)/2*L; wx = wx/2*L
    yg, wy = leggauss(ny); yg = (yg + 1)/2*H; wy = wy/2*H
    X, Y = np.meshgrid(xg, yg, indexing='ij')
    W = np.outer(wx, wy)
    vals = f(X + 1j*Y)[:, :, 0]
    return np.sum(W*vals**2)


def _cluster_gram(solver, eig, coef, x0=None):
    """The (mult x mult) L^2 Gram of an already-computed coefficient cluster, built the safe
    way: evaluate the cluster's own Cauchy data at the shared node set, never sandwich a
    basis-level matrix."""
    bq = solver.bdry_quad
    U = solver.basis(eig, bq.pts)@coef
    U_N = solver.basis.ddiff(eig, bq.pts, bq.normals)@coef
    U_T = solver.basis.ddiff(eig, bq.pts, bq.tangents)@coef
    ed = EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, U, U_N, U_T)
    return gram(ed, eig, bq, x0)


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
    Gram = _cluster_gram(solver, eig, coef)
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

    pts = solver.bdry_quad.pts
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
    bq = boundary_quadrature(dom, mps_mod.weyl_est(2, dom), precision=1e-14)
    solver = MPSEigensolver(normed_basis, bdry_pts, int_pts, bdry_quad=bq, rtol=1e-14)
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
    assert solver.bdry_quad is None

    with pytest.warns(UserWarning, match='unavailable'):
        coef_raw = solver.eigenfunction_coef(30.0, mult=1, orthonorm=False)
        coef_norm = solver.eigenfunction_coef(30.0, mult=1, orthonorm=True)
    assert np.allclose(coef_raw, coef_norm)


def test_orthonorm_false_opts_out():
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis, orthonorm=False)
    assert solver.bdry_quad is None


@pytest.mark.parametrize("precision", [1e-6, 1e-10, 1e-14])
def test_orthonorm_precision_is_the_only_knob(precision):
    """The seven rellich_* tuning parameters collapsed to one accuracy target plus an
    optional x0. A looser target must still normalize correctly, just with fewer nodes."""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    solver = MPSEigensolver.from_domain(domain, basis=basis, orthonorm_precision=precision,
                                        orthonorm_x0=1.0+0.5j)
    assert solver.bdry_quad.x0 == 1.0+0.5j
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)


def test_manual_construction_without_domain_has_no_bdry_quad():
    """MPSEigensolver's raw constructor has no notion of a domain; without an
    explicit bdry_quad argument, normalization is simply unavailable."""
    verts = RECT_VERTS
    poly = Polygon(verts)
    basis = FourierBesselBasis.from_domain(poly, orders=[10, 0, 0, 0])
    bdry = poly.bdry_pts([0, 20, 20, 0], kind='even')
    ipts = poly.int_pts(method='random', npts_rand=30)
    solver = MPSEigensolver(basis, bdry, ipts)
    assert solver.bdry_quad is None
    assert solver.bdry_quad is None

    eig = rect_eig(1, 1, L, H)
    with pytest.warns(UserWarning, match='unavailable'):
        coef = solver.eigenfunction_coef(eig, mult=1, orthonorm=True)
    assert coef.shape[1] == 1


def test_manual_construction_with_explicit_bdry_quad():
    """A caller building the solver by hand can still opt into orthonormalization by
    building the boundary quadrature themselves and passing it in -- mirroring how
    bdry_pts/int_pts are built and passed in manually. Note it needs no basis."""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    bdry_pts, bdry_normals, bc_param = make_default_bdry_data(domain, basis)
    int_pts = make_default_int_pts(domain, 'random', False, len(basis))
    basis_norm = basis.to_normalized((bdry_pts, int_pts))
    bq = boundary_quadrature(domain, rect_eig(3, 3, L, H))

    solver = MPSEigensolver(basis_norm, bdry_pts, int_pts, bdry_normals, bc_param,
                            bdry_quad=bq)
    assert solver.bdry_quad is bq
    eig = rect_eig(1, 2, L, H)
    f = solver.eigenfunction(eig, mult=1)
    assert _gl_norm_sq(f, L, H) == pytest.approx(1.0, abs=1e-6)


# ── Leg 4: x0-invariance on the production path (a reference-free diagnostic) ──
#
# The Rellich identity holds for EVERY x0, so for an exact eigenfunction
# int c.n (du/dn)^2 ds = 0 for any constant vector c. The spread of the computed norm across
# x0 choices therefore needs no reference solution at all: it is pure quadrature error plus
# the MPS solution's own residual. That makes it the one check that works on domains with no
# analytic truth -- H_shape, GWW, the plus polyomino -- at the cost of being bounded below by
# the eigenfunction's own accuracy, so it is diagnostic rather than certifying.

X0S = [0.31 + 0.17j, -0.4 + 0.23j, 0.8 - 0.6j, 1.7 + 1.3j]


def _solver_for(dom, n_basis, precision=None):
    """Exercises from_domain's DEFAULT accuracy target unless told otherwise -- the whole
    point being that a caller supplies no quadrature configuration at all."""
    basis = bases.make_default_basis(dom, n_basis)
    kw = {} if precision is None else {'orthonorm_precision': precision}
    return MPSEigensolver.from_domain(dom, basis=basis, **kw)


@pytest.mark.parametrize("name,factory,n_basis,lo,hi", [
    ('L_shape', L_shape, 160, 8.0, 26.0),
    ('plus_shape', plus_shape, 160, 1.0, 8.0),
])
def test_leg4_x0_invariance_on_real_eigenfunctions(name, factory, n_basis, lo, hi):
    """A real MPS eigenfunction on a domain with reentrant corners, normalized with x0 at its
    default, must give the SAME norm when the Gram is recomputed at unrelated x0."""
    dom = factory()
    solver = _solver_for(dom, n_basis)
    out = solver.solve_interval(lo, hi, 400)
    eigs = np.atleast_1d(np.asarray(out[0] if isinstance(out, tuple) else out)).ravel()
    assert len(eigs) >= 1, f"{name}: no eigenvalue found in [{lo}, {hi}]"

    eig = float(eigs[0])
    coef = solver.eigenfunction_coef(eig, mult=1)
    base = _cluster_gram(solver, eig, coef)[0, 0]
    assert base == pytest.approx(1.0, abs=1e-11), f"{name}: not normalized ({base})"
    spread = max(abs(_cluster_gram(solver, eig, coef, x0)[0, 0] - base) for x0 in X0S)
    assert spread < 1e-9, f"{name}: x0-spread {spread:.2e} at lam={eig}"


def test_leg4_x0_invariance_is_a_real_test_not_a_tautology():
    """Guard: the alternative x0 must actually change the integrand, or the test above proves
    nothing. r.N differs pointwise between x0 choices by (x0_b - x0_a).n, which is nonzero
    somewhere on any closed boundary."""
    dom = L_shape()
    bq = boundary_quadrature(dom, 30.0, precision=1e-10, warn=False)
    from lappy.utils import complex_dot
    rN0 = complex_dot(bq.pts - bq.x0, bq.normals)
    for x0 in X0S:
        rN = complex_dot(bq.pts - x0, bq.normals)
        assert np.abs(rN - rN0).max() > 0.1, x0


def test_default_x0_sits_on_the_worst_singular_corner():
    """Free insurance: r.N vanishes identically on both edges at that corner, removing its
    singularity from the integrand outright. No longer necessary -- that is the whole point of
    the corner rule, and a multi-corner domain cannot zero every corner -- but still free."""
    for factory in (L_shape, H_shape, plus_shape):
        dom = factory()
        x0 = default_x0(dom)
        import lappy.eigfun_integrals as ei
        sing = [s for s in ei.corner_specs(dom) if s.singular]
        assert sing
        assert min(abs(x0 - s.point) for s in sing) < 1e-12


def test_orthonormal_eigenfunction_matches_interior_cubature():
    """Cross-check against a completely independent norm: interior cubature over a domain with
    a reentrant corner. Promoted from scripts/hshape_eigfunc_norm.py, which computed exactly
    this and only printed it."""
    dom = L_shape()
    solver = _solver_for(dom, 160)
    out = solver.solve_interval(8.0, 26.0, 400)
    eigs = np.atleast_1d(np.asarray(out[0] if isinstance(out, tuple) else out)).ravel()
    eig = float(eigs[0])
    coef = solver.eigenfunction_coef(eig, mult=1)
    nodes, weights = cubature.polygon_cubature(dom, eig, 1e-12)
    u = (solver.basis(eig, nodes)@coef)[:, 0]
    assert la.norm(u*np.sqrt(weights)) == pytest.approx(1.0, abs=1e-8)


# ── Lazy certification: the sizing model checked against the actual integrand ──
#
# `orthonorm_precision` sizes the rule from a model of the integrand's class. What the rule
# achieves on the integrand in hand is a different number, and on chevron(1,2) the two differ by
# two orders. These cover the lazy path that reports it, and the invariant that it only ever
# reports -- resizing under a caller who asked for a solve is what `certified_quadrature` is
# for, called deliberately.

def _chevron_solver(**kw):
    dom = geometry.chevron(1, 2)
    basis = bases.make_default_basis(dom, 200)
    solver = MPSEigensolver.from_domain(dom, basis=basis, **kw)
    out = solver.solve_interval(bounds.faber_krahn(dom), mps_mod.weyl_est(2, dom), 20)
    eigs = np.atleast_1d(np.asarray(out[0] if isinstance(out, tuple) else out)).ravel()
    return dom, solver, float(eigs[0])


def test_certification_is_off_by_default_and_costs_nothing():
    """The default path must be byte-for-byte the old one: no domain use, no extra basis
    evaluation, no certification recorded."""
    dom, solver, lam = _chevron_solver()
    solver.eigenfunction_coef(lam, mult=1)
    assert solver.certifications == {}


def test_certify_gram_measures_what_sizing_only_models():
    """The gap, on the production path: sized for 1e-13, measures ~3.7e-11."""
    dom, solver, lam = _chevron_solver()
    err = solver.certify_gram(lam)
    assert err > 10*solver.bdry_quad.sizing_precision, (err, solver.bdry_quad.sizing_precision)
    assert solver.certifications[(lam, 1)] == err


def test_certify_target_warns_lazily_without_changing_the_answer():
    """The lazy path fires during normalization and warns -- and changes nothing else.

    Tested by toggling the flag on ONE solver rather than comparing two: `make_default_int_pts`
    is random by default, so two independently built solvers have different interior points and
    legitimately different coefficients. (`mesh=True` would make them comparable but takes
    minutes on this domain.) The invariant that matters is exactly this one: the same solver,
    the same eigenvalue, the same transform, with and without the hook."""
    dom, solver, lam = _chevron_solver()
    bq_before = solver.bdry_quad
    D0, G0 = solver._orthonorm_transform_coef(lam, 1)
    assert solver.certifications == {}

    # the transform is instance-cached, so the second call would never reach the hook; drop
    # that one cache entry (name derived, not hardcoded) so both paths really execute
    solver.__dict__.pop(
        f'_icache_{MPSEigensolver._orthonorm_transform_coef.__qualname__.replace(".", "_")}',
        None)
    solver._certify_target = 1e-12
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        D1, G1 = solver._orthonorm_transform_coef(lam, 1)

    assert solver.certifications, 'lazy certification did not run'
    assert any('is short at lam' in str(w.message) for w in caught)
    np.testing.assert_array_equal(D0, D1)
    np.testing.assert_array_equal(G0, G1)
    assert solver.bdry_quad is bq_before, 'certification must report, never resize'


def test_certify_target_is_silent_when_the_rule_is_good_enough():
    dom, solver, lam = _chevron_solver(certify_target=1e-8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solver.eigenfunction_coef(lam, mult=1)
    assert solver.certifications
    assert not any('is short at lam' in str(w.message) for w in caught)


def test_certification_needs_the_domain_and_says_so():
    """The solver has no notion of a domain by design; `_domain` exists only because
    refine_quadrature needs a segment's p/N/T, which a materialized BoundaryQuad cannot give.
    A hand-built solver must get a clear error, not an AttributeError."""
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[8, 8, 8, 8])
    bdry_pts, bdry_normals, bc_param = make_default_bdry_data(domain, basis)
    int_pts = make_default_int_pts(domain, 'random', False, len(basis))
    basis_norm = basis.to_normalized((bdry_pts, int_pts))
    bq = boundary_quadrature(domain, rect_eig(3, 3, L, H))
    solver = MPSEigensolver(basis_norm, bdry_pts, int_pts, bdry_normals, bc_param,
                            bdry_quad=bq)
    with pytest.raises(ValueError, match='needs the domain'):
        solver.certify_gram(rect_eig(1, 2, L, H))
