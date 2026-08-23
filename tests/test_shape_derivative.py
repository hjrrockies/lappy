"""The Hadamard contract: lappy's promise to a downstream shape-optimization package.

For a simple eigenvalue with `u` normalized in L^2(Omega), and `V.n` the outward normal
velocity of the boundary,

    dlam = - integral_{dOmega} (du/dn)^2 (V.n) ds  =  -weighted_integral(ed, 'NN', Vn)

lappy implements no shape-derivative formulas (docs/scope_and_downstream.md) -- those live
downstream. What lives here is one test that the three things lappy *does* promise are good
enough to build that formula on: `lam` accurate, `u` orthonormal in L^2, and Cauchy-data
boundary integrals accurate.

WHY THIS TEST EXISTS. Nothing else in the suite is sensitive to a systematic error in `||u||`.
The certified eigenvalue bound is scale-invariant -- during one re-run a domain's quadrature
improved by ten orders and its certified digits did not move -- so certification cannot see the
accuracy the boundary machinery was built for. A shape derivative can: it is linear in the
normalization. This is the regression test for `weighted_integral`, and it is deliberately
built only from public API.

The sign and scale are not free parameters. Take `V = x - x0`, uniform dilation: then
`lam(eps) = lam/(1+eps)^2`, so `dlam/deps = -2 lam`, and the formula demands
`integral (r.N)(du/dn)^2 ds = 2 lam ||u||^2` -- which is the Rellich identity `gram()` already
computes. `test_dilation_is_the_rellich_identity` pins that correspondence.
"""
import numpy as np
import pytest

from lappy import bases, eigfun_integrals, geometry, mps, reference as ref
from lappy.geometry import rect, disk_sector
from lappy.mps import MPSEigensolver
from lappy.eigfun_integrals import (boundary_quadrature, eigfun_cauchy_data, gram,
                                    normal_velocity, weighted_integral)
from lappy.utils import complex_dot


def _solver(domain, lam_max, n_basis=120, seed=0):
    """An MPS solver with the boundary quadrature attached, so eigenfunctions come out
    L^2-orthonormal. Small and fast: every eigenvalue used here is known in closed form, so no
    search is run."""
    np.random.seed(seed)
    basis = bases.make_default_basis(domain, n_basis)
    bdry = domain.bdry_pts(mps.pts_per_seg(domain, basis, mult=2))
    interior = domain.int_pts(method='random', npts_rand=max(2*n_basis, 400))
    bq = boundary_quadrature(domain, lam_max, precision=1e-13, warn=False)
    return MPSEigensolver(basis.to_normalized((bdry, interior)), bdry, interior,
                          rtol=1e-14, ttol=1e-3, bdry_quad=bq)


def _segment_of_node(bq):
    """Which boundary segment each quadrature node belongs to.

    `BoundaryQuad` now carries this per node (`seg_idx`, with `tau` beside it), so this is a
    one-line alias kept for readability. It used to rebuild the array from `panel_id`, which
    docs/scope_and_downstream.md section 3 named as the sign of a seam one field short -- the
    field is there now, and `test_the_quad_carries_node_provenance` pins that the two agree.
    """
    return bq.seg_idx


def _hadamard(solver, lam, Vn, mult=1):
    """The shape-derivative matrix `-integral (du_i/dn)(du_j/dn) (V.n) ds`.

    For `mult == 1` this is `dlam`. For a degenerate cluster it is the matrix whose EIGENVALUES
    are the directional derivatives -- the eigenfunction basis of a multiple eigenvalue is
    arbitrary (Loewdin returns one of many), so no individual entry means anything.
    """
    coef = solver.eigenfunction_coef(lam, mult=mult)          # orthonormal by default
    ed = eigfun_cauchy_data(solver.basis, lam, coef, solver.bdry_quad)
    return -weighted_integral(ed, 'NN', Vn)


# ── Tier 1: rectangle, exact truth ───────────────────────────────────────────
#
# lam_mn = pi^2 (m^2/L^2 + n^2/H^2) on [0,L]x[0,H], so translating the right edge outward
# (V.n = 1 there, 0 elsewhere) has dlam/dL = -2 pi^2 m^2 / L^3. Integrating the formula by hand
# over that edge gives the same thing, so both sides are closed form and the measured error is
# the machinery's alone.

@pytest.mark.parametrize('m,n', [(1, 1), (2, 1), (1, 2), (3, 2)])
def test_rectangle_edge_translation(m, n):
    L, H = 2.0, 1.0
    dom = rect(L, H)
    lam = ref.rect_eig(m, n, L, H)
    solver = _solver(dom, 3*lam)
    Vn = (_segment_of_node(solver.bdry_quad) == 1).astype(float)   # segment 1 is x = L
    got = _hadamard(solver, lam, Vn)[0, 0]
    exact = -2*np.pi**2*m**2/L**3
    assert abs(got - exact)/abs(exact) < 1e-12, (got, exact)


def test_the_derivative_is_localized_to_the_moving_edge():
    """A guard against a weight that is silently ignored: moving the LEFT edge of a rectangle
    gives the same derivative by symmetry, and moving a horizontal edge gives the H-derivative
    instead, which is a different number."""
    L, H = 2.0, 1.0
    m, n = 2, 1
    dom = rect(L, H)
    lam = ref.rect_eig(m, n, L, H)
    solver = _solver(dom, 3*lam)
    seg = _segment_of_node(solver.bdry_quad)

    d_left = _hadamard(solver, lam, (seg == 3).astype(float))[0, 0]     # x = 0
    d_right = _hadamard(solver, lam, (seg == 1).astype(float))[0, 0]    # x = L
    d_top = _hadamard(solver, lam, (seg == 2).astype(float))[0, 0]      # y = H

    assert abs(d_left - d_right)/abs(d_right) < 1e-10          # symmetric edges agree
    assert abs(d_right - (-2*np.pi**2*m**2/L**3))/abs(d_right) < 1e-12
    assert abs(d_top - (-2*np.pi**2*n**2/H**3))/abs(d_top) < 1e-12
    assert abs(d_top - d_right) > 1.0                          # and they are different numbers


def test_degenerate_cluster_splits_correctly():
    """The sharpest test here. On the unit square (1,2) and (2,1) share lam = 5 pi^2; under a
    right-edge translation they split into -2 pi^2 and -8 pi^2.

    The shape derivative of a multiple eigenvalue is not a derivative but the eigenvalues of the
    m x m matrix, and the cluster basis is arbitrary -- so this compares EIGENVALUES. The matrix
    entries are nothing like the answer (measured: diagonal -50.4, -48.3 with off-diagonal
    -29.6, against eigenvalues -19.7 and -79.0).
    """
    dom = rect(1.0, 1.0)
    lam = 5*np.pi**2
    solver = _solver(dom, 3*lam, n_basis=160)
    Vn = (_segment_of_node(solver.bdry_quad) == 1).astype(float)

    M = _hadamard(solver, lam, Vn, mult=2)
    assert M.shape == (2, 2)
    assert abs(M[0, 1] - M[1, 0]) < 1e-10*abs(M).max(), 'the matrix must be symmetric'

    got = np.sort(np.linalg.eigvalsh(M))
    exact = np.sort(np.array([-2*np.pi**2, -8*np.pi**2]))
    assert np.allclose(got, exact, rtol=1e-11), (got, exact)


# ── The sign and scale are pinned by the Rellich identity ────────────────────

def test_dilation_is_the_rellich_identity():
    """Uniform dilation about x0 is V = x - x0, so the Hadamard integral must equal -2 lam
    ||u||^2 -- i.e. exactly `2 lam * gram()` with the sign flipped. This is not an independent
    check of the physics; it is the statement that lappy's normalization machinery and its
    shape-derivative contract are the same object, which is what fixes the sign convention.
    """
    L, H = 2.0, 1.0
    dom = rect(L, H)
    lam = ref.rect_eig(2, 1, L, H)
    solver = _solver(dom, 3*lam)
    bq = solver.bdry_quad

    Vn = complex_dot(bq.pts - bq.x0, bq.normals)          # V = x - x0
    had = _hadamard(solver, lam, Vn)[0, 0]

    coef = solver.eigenfunction_coef(lam, mult=1)
    ed = eigfun_cauchy_data(solver.basis, lam, coef, bq)
    norm2 = gram(ed, lam, bq)[0, 0]

    assert abs(norm2 - 1.0) < 1e-11, norm2                 # u is orthonormal
    assert abs(had - (-2*lam*norm2))/abs(2*lam) < 1e-11, (had, -2*lam*norm2)


def test_the_quad_carries_node_provenance():
    """`seg_idx` and `tau` per node, so a downstream package can evaluate `V` without calling
    the private `_panel_rule` and redoing the affine map itself.

    Checked against the panel-walking reconstruction that used to be the only route, and against
    the segment's own parametrization: `seg.p(tau)` must land back on the node.
    """
    dom = rect(2.0, 1.0)
    bq = boundary_quadrature(dom, 3*ref.rect_eig(2, 1, 2.0, 1.0), precision=1e-13, warn=False)

    from_panels = np.array([bq.panels[p].seg_idx for p in bq.panel_id])
    assert np.array_equal(bq.seg_idx, from_panels)

    assert bq.tau.shape == bq.pts.shape
    assert np.all((bq.tau >= -1e-12) & (bq.tau <= 1 + 1e-12)), (bq.tau.min(), bq.tau.max())
    segs = dom.bdry.segments
    back = np.array([segs[s].p(np.array([t]))[0] for s, t in zip(bq.seg_idx, bq.tau)])
    assert np.max(np.abs(back - bq.pts)) < 1e-12, np.max(np.abs(back - bq.pts))


def test_dilation_through_the_converter_is_the_rellich_identity():
    """`normal_velocity` must reproduce `test_dilation_is_the_rellich_identity` exactly.

    That test builds `V.n` by hand with `complex_dot(bq.pts - bq.x0, bq.normals)`; this one
    passes the same displacement field through the public converter. It is the sign test: the
    Rellich identity fixes both the magnitude AND the orientation, so a flipped normal or a
    conjugation error cannot pass. Nothing else downstream would catch it -- it would return a
    plausible wrong gradient.
    """
    L, H = 2.0, 1.0
    dom = rect(L, H)
    lam = ref.rect_eig(2, 1, L, H)
    solver = _solver(dom, 3*lam)
    bq = solver.bdry_quad

    dp = bq.pts - bq.x0                                    # uniform dilation about x0
    Vn = eigfun_integrals.normal_velocity(bq, dp)
    assert np.allclose(Vn, complex_dot(dp, bq.normals), rtol=0, atol=0)

    had = _hadamard(solver, lam, Vn)[0, 0]
    assert abs(had - (-2*lam))/abs(2*lam) < 1e-11, (had, -2*lam)
    assert had < 0, 'outward dilation must DECREASE a Dirichlet eigenvalue'


def test_normal_velocity_rejects_a_mismatched_field():
    """The shape check is worth having: silently broadcasting a wrong-length `V` would produce a
    finite, plausible number."""
    dom = rect(2.0, 1.0)
    bq = boundary_quadrature(dom, 3*ref.rect_eig(2, 1, 2.0, 1.0), precision=1e-13, warn=False)
    with pytest.raises(ValueError, match='one displacement per quadrature node'):
        eigfun_integrals.normal_velocity(bq, np.ones(len(bq.pts) - 1, dtype=complex))


def test_a_tangential_velocity_moves_nothing():
    """Only the normal component enters, which is the content of the Hadamard formula: sliding
    the boundary along itself does not change the domain."""
    L, H = 2.0, 1.0
    dom = rect(L, H)
    lam = ref.rect_eig(2, 1, L, H)
    solver = _solver(dom, 3*lam)
    bq = solver.bdry_quad
    Vn = eigfun_integrals.normal_velocity(bq, 0.37*bq.tangents)
    assert np.max(np.abs(Vn)) < 1e-14, np.max(np.abs(Vn))


def test_scaling_the_eigenfunction_would_break_the_derivative():
    """`eps` is scale-invariant and `dlam` is not: this is why the shape derivative is the
    regression test for normalization that certification cannot provide. Un-normalized
    coefficients must give a visibly wrong answer."""
    L, H = 2.0, 1.0
    dom = rect(L, H)
    lam = ref.rect_eig(2, 1, L, H)
    solver = _solver(dom, 3*lam)
    Vn = (_segment_of_node(solver.bdry_quad) == 1).astype(float)
    exact = -2*np.pi**2*4/L**3

    raw = solver.eigenfunction_coef(lam, mult=1, orthonorm=False)
    ed = eigfun_cauchy_data(solver.basis, lam, raw, solver.bdry_quad)
    wrong = -weighted_integral(ed, 'NN', Vn)[0, 0]
    assert abs(wrong - exact)/abs(exact) > 1e-6, 'un-normalized coefficients should NOT agree'


# ── Tier 2: circular sector, a curved boundary and a singular corner ─────────

def test_sector_radius_derivative():
    """lam = (j_{nu,n}/R)^2, so dlam/dR = -2 lam / R, with V.n = 1 on the arc and 0 on the two
    radii. Adds a curved boundary and (at alpha > pi) a reentrant corner to the contract."""
    for alpha_over_pi in (0.5, 1.5):
        alpha = alpha_over_pi*np.pi
        R = 1.0
        dom = disk_sector(R, alpha)
        lam = ref.sector_eig(1, 1, R, alpha)
        solver = _solver(dom, 3*lam, n_basis=160)
        bq = solver.bdry_quad

        # the arc is the segment whose nodes sit at |z| = R (the radii run from the apex)
        seg = _segment_of_node(bq)
        on_arc = np.zeros(len(seg), dtype=float)
        for i in np.unique(seg):
            if np.allclose(np.abs(bq.pts[seg == i]), R, atol=1e-9):
                on_arc[seg == i] = 1.0
        assert on_arc.sum() > 0, 'failed to identify the arc'

        got = _hadamard(solver, lam, on_arc)[0, 0]
        exact = -2*lam/R
        assert abs(got - exact)/abs(exact) < 1e-9, (alpha_over_pi, got, exact)


# ── The limitation this contract test discovered ────────────────────────────
#
# A perturbation that MOVES A CORNER has `V.n = r` on the rotating edge, and that weight is
# outside the corner rule's exactness class. The rule substitutes `t = r^nu` to rationalize the
# Rellich family `{j nu + 2q}`; a weight `r^p` contributes `t^(p/nu)`, which is a polynomial
# only when `p/nu` is an integer. At `nu = 2/3` that is `t^(3p/2)`: fine for even `p`, not for
# odd. Measured on a 1.5pi sector with the EXACT eigenfunction, so the error is the quadrature's:
#
#     weight r^p    p=0      p=1      p=2      p=3      p=4
#     rel err     3.4e-14  6.2e-06  3.5e-14  3.2e-11  3.5e-14
#
# This was invisible until now because every weight lappy had ever integrated is in the class:
# `gram` uses `r.N`, identically zero on the straight edges when `x0` sits at the singular corner
# and `O(1)` otherwise. A corner-moving shape velocity is the first weight that is `O(r^1)`.
#
# See benchmarks/hadamard/sector_angle.py. The bounds below are measured, not aspirational: they
# will catch a regression and will not obstruct a fix.

def _sector_angle_setup(alpha_over_pi, m=1, n=1, R=1.0, weight_family='even'):
    from lappy import reference as ref
    alpha = alpha_over_pi*np.pi
    dom = disk_sector(R, alpha)
    lam = ref.sector_eig(m, n, R, alpha)
    bq = boundary_quadrature(dom, 3*lam, precision=1e-13, warn=False,
                             weight_family=weight_family)
    u, norm2 = ref.sector_eigfun(m, n, R, alpha)
    s = 1.0/np.sqrt(norm2)
    G = s*ref.sector_eigfun_grad(m, n, R, alpha)(bq.pts)
    ed = EigfunDataFor(bq, (s*u(bq.pts))[:, None],
                       complex_dot(G, bq.normals)[:, None],
                       complex_dot(G, bq.tangents)[:, None])
    seg = np.array([bq.panels[p].seg_idx for p in bq.panel_id])
    mask = None
    for i in np.unique(seg):
        pts = bq.pts[seg == i]
        th = np.angle(pts) % (2*np.pi)
        if np.allclose(th, alpha, atol=1e-6) and np.ptp(np.abs(pts)) > 0.5*R:
            mask = seg == i
    assert mask is not None
    return dom, bq, ed, mask, alpha


def EigfunDataFor(bq, U, U_N, U_T):
    from lappy.eigfun_integrals import EigfunData
    return EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, U, U_N, U_T)


def test_corner_moving_derivative_is_exact_at_integer_nu():
    """alpha = pi/2 gives nu = 2: the exponent family is even and the rule is exact even for a
    weight that does not vanish at the corner."""
    import mpmath as mp
    mp.mp.dps = 30
    _, bq, ed, mask, alpha = _sector_angle_setup(0.5)
    got = -weighted_integral(ed, 'NN', np.where(mask, np.abs(bq.pts), 0.0))[0, 0]
    nu = mp.mpf(1)*mp.pi/mp.mpf(alpha)
    j = mp.besseljzero(nu, 1)
    dj = mp.diff(lambda v: mp.besseljzero(v, 1), nu)
    exact = float(2*j*dj*(-nu/mp.mpf(alpha)))
    assert abs(got - exact)/abs(exact) < 1e-13, (got, exact)


def test_weight_parity_at_a_reentrant_corner():
    """`weight_family` decides whether an ODD power of `r` is integrated exactly at a singular
    corner -- which is what a corner-moving shape velocity supplies, and the whole reason the
    option exists.

    The default 'even' rule uses `sub = nu`, exact on the eigenfunction's own family
    `{gamma + j nu + 2q}` but sending `r^m` to the slowly-resolved `t^(m/nu)`. 'integer' uses
    `sub = 1/2`, making every integer power the exact polynomial `t^(2m)` at the cost of
    leaving `t^(2 j nu)` inexact -- a good trade, because those exponents grow by `2 nu` per
    term. Both columns are asserted here so a regression in either is caught."""
    import mpmath as mp
    # 40 dps, not 30: the p=0 reference integrand carries r^(2nu-2) = r^(-2/3) at the endpoint
    # and `mp.quad` resolves it to only 4.9e-12 at 30 dps (3.4e-14 at 40, stable to 50). This is
    # the third time in this project that mpmath.quad on a corner integrand has been the weakest
    # link -- docs/eigfun_integrals.md warns about exactly this. Verify the reference before
    # believing an error.
    mp.mp.dps = 40
    _, bq, ed, mask, alpha = _sector_angle_setup(1.5)
    nu = mp.mpf(1)*mp.pi/mp.mpf(alpha)
    j = mp.besseljzero(nu, 1)
    k = j
    norm2 = (mp.mpf(1)/2)*mp.besselj(nu+1, j)**2*(mp.mpf(alpha)/2)

    def truth(p):
        return float(nu**2*mp.cos(nu*mp.mpf(alpha))**2
                     * mp.quad(lambda r: mp.besselj(nu, k*r)**2*r**(p-2), [0, 1])/norm2)

    err = {}
    for p in (0, 1, 2, 3, 4):
        got = weighted_integral(ed, 'NN', np.where(mask, np.abs(bq.pts)**p, 0.0))[0, 0]
        err[p] = abs(got - truth(p))/abs(truth(p))

    for p in (0, 2, 4):
        assert err[p] < 1e-12, (p, err[p])          # in the class: machine precision
    assert err[1] < 1e-5, err[1]                    # out of the class: ~6e-6 today
    assert err[1] > 100*err[0], (err[0], err[1])    # and decisively worse than p=0

    # weight_family='integer' removes the parity distinction entirely.
    _, bq2, ed2, mask2, _ = _sector_angle_setup(1.5, weight_family='integer')
    err2 = {}
    for p in (0, 1, 2, 3, 4):
        got = weighted_integral(ed2, 'NN', np.where(mask2, np.abs(bq2.pts)**p, 0.0))[0, 0]
        err2[p] = abs(got - truth(p))/abs(truth(p))
    for p in (0, 1, 2, 3, 4):
        assert err2[p] < 1e-11, (p, err2[p])        # every power, ~1e-14 today
    assert err2[1] < 1e-4*err[1], (err[1], err2[1]) # the odd power gains >= 4 orders


def test_integer_weight_family_fixes_the_corner_moving_derivative():
    """The contract this option exists for: `dlam/dalpha` on a reentrant sector, where the
    velocity `V.n = r` moves the singular corner. The default rule is capped near 1e-06 and
    does not improve with refinement; 'integer' reaches the eigenfunction's own accuracy."""
    import mpmath as mp
    mp.mp.dps = 40
    for aop in (1.5, 1.75):
        alpha = aop*np.pi
        nu = mp.mpf(1)*mp.pi/mp.mpf(alpha)
        j = mp.besseljzero(nu, 1)
        dj = mp.diff(lambda v: mp.besseljzero(v, 1), nu)
        exact = float(2*j*dj*(-nu/mp.mpf(alpha)))
        _, bq, ed, mask, _ = _sector_angle_setup(aop, weight_family='integer')
        got = -weighted_integral(ed, 'NN', np.where(mask, np.abs(bq.pts), 0.0))[0, 0]
        assert abs(got - exact)/abs(exact) < 1e-11, (aop, got, exact)


# ── hadamard_quadrature: the wrapper, and the measurement that justifies it ────
#
# `weight_family='integer'` was validated on the SPARSE sector eigenfunction, and separately
# shown to fail on a dense family with the plain r^0 weight -- which is why it is not the
# default. The case a shape derivative actually meets, a dense corner series AND an integer
# weight, was measured only when this wrapper was written. It is the combination that decides
# whether the wrapper is well founded, and it is: the split runs along the WEIGHT's parity, not
# along the density of the eigenfunction's family.

def _weighted_corner_error(dom, weight_family, p, lam_max, seeds=(0, 1, 2, 3, 4)):
    """Worst relative error over corner panels for `int rN * r^p * un^2`, against closed form."""
    from lappy.utils import complex_dot
    import tests.test_eigfun_integrals as T
    import lappy.eigfun_integrals as ei

    bq = ei.boundary_quadrature(dom, lam_max, precision=1e-14, warn=False,
                                weight_family=weight_family)
    x0 = 0.37 + 0.181j
    worst = 0.0
    for pid, panel in enumerate(bq.panels):
        if panel.rule == 'legendre' or panel.nu > 1.0:
            continue
        seg = dom.bdry.segments[panel.seg_idx]
        at_end = panel.tau0 > panel.tau1
        mid = seg.p(np.array([0.5]))[0]
        rN = complex_dot(mid - x0, seg.N(np.array([0.5]))[0])
        h = panel.tau1 - panel.tau0
        u_local, w = ei._panel_rule(panel)
        s = seg.len*(panel.tau0 + h*u_local)
        wts = seg.len*abs(h)*w
        lo, hi = sorted((panel.tau0*seg.len, panel.tau1*seg.len))
        r = (seg.len - s) if at_end else s          # distance from the ANCHORED end
        a, b = (seg.len - hi, seg.len - lo) if at_end else (lo, hi)
        for seed in seeds:
            terms, _ = T._bessel_corner_series(panel.nu, seg.len, 3, 3, lam_max, seed + pid,
                                               at_end)
            un = sum(c*r**q for c, q in terms)
            got = float(np.sum(wts*rN*r**p*un**2))
            exact = rN*sum(c1*c2*(b**(q1 + q2 + p + 1) - a**(q1 + q2 + p + 1))/(q1 + q2 + p + 1)
                           for c1, q1 in terms for c2, q2 in terms)
            worst = max(worst, abs(got/exact - 1.0))
    return worst


@pytest.mark.parametrize("factory,name", [
    (lambda: geometry.L_shape(), 'L_shape (cornerjac, nu=2/3)'),
    (lambda: geometry.chevron(0.5, 3), 'chevron(0.5,3) (cornerinterp, nu=0.772)'),
    (lambda: geometry.chevron(2, 3), 'chevron(2,3) (cornerinterp, nu=0.587)'),
])
def test_integer_family_wins_for_a_corner_moving_weight_on_a_dense_family(factory, name):
    """The gating measurement. Weight r^1 at lam_max=100, dense series, closed-form truth."""
    dom = factory()
    even = _weighted_corner_error(dom, 'even', 1, 100.0)
    integer = _weighted_corner_error(dom, 'integer', 1, 100.0)
    assert integer < even/100, f"{name}: even {even:.2e} vs integer {integer:.2e}"
    assert integer < 1e-9, f"{name}: {integer:.2e}"


def test_the_trade_runs_the_other_way_for_the_rellich_weight():
    """Why this is a separate entry point and not a new default: with the plain r^0 weight the
    same rules reverse, and by as much. A caller who swaps hadamard_quadrature in for the Gram
    loses five orders."""
    dom = geometry.L_shape()
    even = _weighted_corner_error(dom, 'even', 0, 1.0)
    integer = _weighted_corner_error(dom, 'integer', 0, 1.0)
    assert even < 1e-13, even
    assert integer > 100*even, (even, integer)


def test_hadamard_quadrature_selects_the_integer_family():
    from lappy.eigfun_integrals import hadamard_quadrature, boundary_quadrature
    dom = geometry.chevron(2, 3)
    hq = hadamard_quadrature(dom, 100.0, warn=False)
    ref = boundary_quadrature(dom, 100.0, warn=False, weight_family='integer')
    corner = [p for p in hq.panels if p.rule != 'legendre']
    assert corner, 'expected corner panels'
    assert all(p.rule == 'cornerjac' and p.sub == 0.5 for p in corner), corner
    assert len(hq.pts) == len(ref.pts)
    np.testing.assert_array_equal(hq.wts, ref.wts)


def test_hadamard_quadrature_refuses_a_conflicting_weight_family():
    from lappy.eigfun_integrals import hadamard_quadrature
    with pytest.raises(TypeError, match='weight_family'):
        hadamard_quadrature(geometry.L_shape(), 100.0, weight_family='even')
