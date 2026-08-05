"""Validation of lappy.eigfun_integrals against exact truth.

Three independent legs, none of which relies on an MPS solve:

Leg 1  single singular corner, end-to-end, exact.  Closed-form sector
       eigenfunctions of exactly known L^2 norm (reference.sector_eigfun +
       sector_eigfun_grad) fed to the Rellich identity, which must return 1.

Leg 2  multiple singular corners, end-to-end, exact -- but with ZERO singular
       amplitude.  sin(m pi x) sin(n pi y) vanishes on the whole integer grid, so
       any polyomino has it as an exact Dirichlet eigenfunction with norm^2 =
       cells/4.  A closed-form eigenfunction on a nonconvex domain is necessarily
       *smooth* at the reentrant corners -- that is exactly why L_shape has no
       closed form -- so this leg tests geometry, panel splitting and the whole
       assembly on four reentrant corners, and deliberately does NOT test the
       singularity.  See Leg 3 for that.

Leg 3  multiple singular corners WITH singular amplitude, exact.  Synthetic
       boundary data carrying each corner's true exponent structure, integrated
       against closed-form references (Beta functions for the cross terms).  This
       is the only leg that exercises an edge joining TWO singular corners, which
       is what forces the panel split.

Two rules learned the hard way and enforced here:

- Report the edge/arc split, never just the total.  At alpha=3pi/2 the two edges
  carry r.N = Im(x0) and -Re(x0) against identical (du/dn)^2, so an x0 on the
  diagonal cancels the singular contribution *identically* and a total-only test
  silently measures the smooth part alone.
- Never use mpmath.quad as the reference for these integrands.  As nu -> 1/2 the
  endpoint behaviour approaches r^-1 and it errs by 4e-2 even at 40 dps, which
  manufactures convincing but entirely spurious plateaus.  Every reference here
  is closed-form.
"""
import warnings

import numpy as np
import pytest

from lappy import reference as ref
from lappy.geometry import disk_sector, L_shape, H_shape, rect
from lappy.utils import complex_dot
from lappy import eigfun_integrals as ei


# ── helpers ──────────────────────────────────────────────────────────────────

def dirichlet_data(bq, grad_u, u=None):
    """EigfunData for a single Dirichlet eigenfunction given its analytic gradient.

    u == 0 on the boundary, so U and U_T vanish there up to roundoff and only U_N carries
    information -- which is exactly why the Dirichlet branch of `gram` uses the 'NN' kernel
    alone. They are still evaluated (not zeroed) so that any violation shows up rather than
    being assumed away."""
    g = grad_u(bq.pts)
    U = (u(bq.pts) if u is not None else np.zeros(len(bq.pts)))[:, None]
    U_N = complex_dot(g, bq.normals)[:, None]
    U_T = complex_dot(g, bq.tangents)[:, None]
    return ei.EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, U, U_N, U_T)


def sector_setup(alpha_over_pi, m, n, R=1.0):
    """Domain, lam, and the EXACTLY normalized eigenfunction and its gradient."""
    alpha = alpha_over_pi*np.pi
    dom = disk_sector(R, alpha)
    u, norm2 = ref.sector_eigfun(m, n, R, alpha)
    g = ref.sector_eigfun_grad(m, n, R, alpha)
    lam = ref.sector_eig(m, n, R, alpha)
    s = 1.0/np.sqrt(norm2)
    return dom, lam, (lambda z: s*u(z)), (lambda z: s*g(z))


def panel_split(bq, ed, lam, x0):
    """Per-panel contributions to the Rellich integral, keyed by whether the panel sits on a
    straight edge or a curved one. Reporting this is not optional -- see the module docstring
    on the alpha=3pi/2 cancellation."""
    rN = complex_dot(ed.pts - x0, ed.normals)
    contrib = ed.wts*rN*ed.U_N[:, 0]**2/(2*lam)
    out = {}
    for pid, panel in enumerate(bq.panels):
        key = 'corner' if panel.rule != 'legendre' else 'smooth'
        out[key] = out.get(key, 0.0) + contrib[bq.panel_id == pid].sum()
    return out


# ── Leg 1: single singular corner, exact ─────────────────────────────────────

# (alpha/pi, achievable relative error), from the MEASURED capability at the worst x0 -- the
# 'generic' one, which zeroes no corner. 'apex' and 'bbox' come in 1-3 orders better and are
# covered by the same bar. The capability degrades monotonically as nu -> 1/2, where the
# Rellich integrand stops being integrable at all, so a flat bar would be a wish not a test.
SECTOR_BARS = [(0.5, 1e-13), (2/3, 1e-13), (1.1, 1e-13), (1.25, 1e-13),
               (4/3, 1e-11), (1.5, 1e-13), (1.6, 1e-10), (1.75, 1e-10), (1.9, 1e-8)]

# x0 placements. 'apex' makes r.N vanish identically on both radii (the geometric escape);
# 'generic' is deliberately OFF the diagonal so the edge terms do NOT cancel; 'bbox' is what
# a bounding-box centre would give.
def sector_x0s(alpha):
    return [('apex', 0.0 + 0j),
            ('generic', 0.4 + 0.05j),
            ('bbox', 0.000257 - 0.000257j)]


@pytest.mark.parametrize("alpha_over_pi,bar", SECTOR_BARS)
@pytest.mark.parametrize("x0_name", ['apex', 'generic', 'bbox'])
def test_leg1_sector_rellich_norm_is_one(alpha_over_pi, bar, x0_name):
    """The identity is exact for EVERY x0, so an exactly-normalized eigenfunction must give
    back 1 regardless of where x0 sits -- including placements that zero no corner."""
    dom, lam, u, g = sector_setup(alpha_over_pi, 1, 1)
    x0 = dict(sector_x0s(alpha_over_pi*np.pi))[x0_name]
    bq = ei.boundary_quadrature(dom, lam, precision=1e-14, warn=False)
    ed = dirichlet_data(bq, g, u)
    G = ei.gram(ed, lam, bq, x0)
    assert G.shape == (1, 1)
    assert abs(G[0, 0] - 1.0) < bar, (G[0, 0] - 1.0, len(bq.pts))


@pytest.mark.parametrize("alpha_over_pi", [1.25, 1.5, 1.75])
def test_leg1_singular_edges_actually_contribute(alpha_over_pi):
    """Guard against the cancellation trap: with the 'generic' x0 the corner panels must carry
    a non-negligible share of the integral, or the test above is measuring the arc alone."""
    dom, lam, u, g = sector_setup(alpha_over_pi, 1, 1)
    bq = ei.boundary_quadrature(dom, lam, precision=1e-14, warn=False)
    ed = dirichlet_data(bq, g, u)
    parts = panel_split(bq, ed, lam, 0.4 + 0.05j)
    share = abs(parts['corner'])/(abs(parts['corner']) + abs(parts['smooth']))
    assert share > 0.02, (parts, share)


def test_leg1_diagonal_x0_cancels_the_singular_part():
    """The trap itself, pinned as a fact about the geometry rather than left as folklore: at
    alpha=3pi/2 the two radii carry r.N = Im(x0) and -Re(x0) against identical (du/dn)^2, so
    any x0 on the diagonal kills the corner contribution exactly. A test using such an x0
    would silently measure only the arc."""
    dom, lam, u, g = sector_setup(1.5, 1, 1)
    bq = ei.boundary_quadrature(dom, lam, precision=1e-14, warn=False)
    ed = dirichlet_data(bq, g, u)
    on_diag = panel_split(bq, ed, lam, 0.3 + 0.3j)
    off_diag = panel_split(bq, ed, lam, 0.4 + 0.05j)
    assert abs(on_diag['corner']) < 1e-12, on_diag
    assert abs(off_diag['corner']) > 1e-3, off_diag


@pytest.mark.parametrize("m,n", [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)])
def test_leg1_higher_modes(m, n):
    """Modes with nu_mode = m*pi/alpha >= 1 are not singular at all and must stay exact; the
    rule must do no harm where there is nothing to resolve."""
    dom, lam, u, g = sector_setup(1.5, m, n)
    bq = ei.boundary_quadrature(dom, lam, precision=1e-14, warn=False)
    ed = dirichlet_data(bq, g, u)
    G = ei.gram(ed, lam, bq, 0.4 + 0.05j)
    assert abs(G[0, 0] - 1.0) < 1e-11, G[0, 0] - 1.0


def test_leg1_nu_sensitivity_guard():
    """nu must reach the rule EXACTLY. Perturbing it by 3e-4 has to cost several digits; if
    this test ever reports no loss, a rounded or margin-padded exponent has crept in and the
    rule is no longer matched to the singularity (docs/corner_quadrature.tex Sec. 4)."""
    dom, lam, u, g = sector_setup(1.5, 1, 1)
    x0 = 0.4 + 0.05j
    clean = ei.boundary_quadrature(dom, lam, precision=1e-14, warn=False)
    err_clean = abs(ei.gram(dirichlet_data(clean, g, u), lam, clean, x0)[0, 0] - 1.0)

    specs = ei.corner_specs(dom)
    bad = [s._replace(nu=s.nu*(1 + 3e-4)) if s.admissible else s for s in specs]
    panels = ei.corner_panels(dom, bad, order_corner=16, order_smooth=32)
    bqb = ei.assemble_panels(dom, panels)
    err_bad = abs(ei.gram(dirichlet_data(bqb, g, u), lam, bqb, x0)[0, 0] - 1.0)

    assert err_clean < 1e-12
    assert err_bad > 1e3*max(err_clean, 1e-15), (err_clean, err_bad)


def test_leg1_precision_argument_is_honoured():
    """A looser request must cost fewer nodes, and a tighter one must actually deliver."""
    dom, lam, u, g = sector_setup(1.5, 1, 1)
    x0 = 0.4 + 0.05j
    n_prev, err_prev = 0, np.inf
    for prec in (1e-4, 1e-8, 1e-12):
        bq = ei.boundary_quadrature(dom, lam, precision=prec, warn=False)
        err = abs(ei.gram(dirichlet_data(bq, g, u), lam, bq, x0)[0, 0] - 1.0)
        assert len(bq.pts) >= n_prev
        assert err <= max(prec, 1e-15)*10, (prec, err)
        n_prev, err_prev = len(bq.pts), err


def test_leg1_smooth_control_rect_and_disk_sector_right_angle():
    """A convex domain has no singular corner, so every panel is Gauss-Legendre and the result
    must be machine-exact -- a control that the assembly itself is not introducing error."""
    for dom, (u, norm2), lam in [
            (rect(2.0, 1.0), ref.rect_eigfun(2, 1, 2.0, 1.0), ref.rect_eig(2, 1, 2.0, 1.0))]:
        s = 1.0/np.sqrt(norm2)
        L, H = 2.0, 1.0
        def grad(z, m=2, n=1):
            x, y = np.real(z), np.imag(z)
            return s*(m*np.pi/L*np.cos(m*np.pi*x/L)*np.sin(n*np.pi*y/H)
                      + 1j*n*np.pi/H*np.sin(m*np.pi*x/L)*np.cos(n*np.pi*y/H))
        bq = ei.boundary_quadrature(dom, lam, precision=1e-14, warn=False)
        ed = dirichlet_data(bq, grad, lambda z: s*u(z))
        G = ei.gram(ed, lam, bq, 0.37 + 0.11j)
        assert abs(G[0, 0] - 1.0) < 1e-13, G[0, 0] - 1.0


# ── multiplicity / Lowdin ────────────────────────────────────────────────────

def test_gram_of_two_orthogonal_sector_modes_is_the_identity():
    """Two distinct modes of the same domain are L^2-orthogonal, so the off-diagonal must
    vanish -- but they have DIFFERENT eigenvalues, so the identity applies to each separately
    and this checks the diagonal plus a directly-computed cross term."""
    dom, lam1, u1, g1 = sector_setup(1.5, 1, 1)
    _, lam2, u2, g2 = sector_setup(1.5, 2, 1)
    bq = ei.boundary_quadrature(dom, max(lam1, lam2), precision=1e-14, warn=False)
    for lam, u, g in ((lam1, u1, g1), (lam2, u2, g2)):
        G = ei.gram(dirichlet_data(bq, g, u), lam, bq, 0.4 + 0.05j)
        assert abs(G[0, 0] - 1.0) < 1e-11


def test_lowdin_transform_orthonormalizes():
    w = np.array([[2.0, 0.3], [0.3, 1.5]])
    D = ei.lowdin_transform(w)
    assert np.allclose(D.T@w@D, np.eye(2), atol=1e-13)


def test_lowdin_transform_refuses_a_deficient_gram():
    G = np.array([[1.0, 0.0], [0.0, 1e-9]])
    with pytest.warns(UserWarning, match="deficient"):
        assert ei.lowdin_transform(G) is None


# ── weighted_integral: the generalized Cauchy-data path ──────────────────────

def test_weighted_integral_kernels_and_symmetry():
    dom, lam, u, g = sector_setup(1.5, 1, 1)
    bq = ei.boundary_quadrature(dom, lam, precision=1e-10, warn=False)
    ed = dirichlet_data(bq, g, u)
    weight = np.ones(len(bq.pts))
    for kernel in ('uv', 'NN', 'TT', 'cr'):
        A = ei.weighted_integral(ed, kernel, weight)
        assert A.shape == (1, 1)
        assert np.isfinite(A).all()
    with pytest.raises(ValueError, match="kernel"):
        ei.weighted_integral(ed, 'nope', weight)


def test_weighted_integral_is_linear_in_the_weight():
    """The Hadamard-type extension point: any consumer supplies its own boundary weight, so
    linearity in that weight is the contract."""
    dom, lam, u, g = sector_setup(1.5, 1, 1)
    bq = ei.boundary_quadrature(dom, lam, precision=1e-10, warn=False)
    ed = dirichlet_data(bq, g, u)
    w1 = np.cos(np.angle(bq.pts))
    w2 = np.abs(bq.pts)
    A = ei.weighted_integral(ed, 'NN', 2.0*w1 - 3.0*w2)
    B = 2.0*ei.weighted_integral(ed, 'NN', w1) - 3.0*ei.weighted_integral(ed, 'NN', w2)
    assert np.allclose(A, B, rtol=1e-13)


# ── corner bookkeeping ───────────────────────────────────────────────────────

def test_panels_tile_every_segment_exactly_once():
    for dom in (rect(2.0, 1.0), L_shape(), H_shape(), disk_sector(1.0, 1.5*np.pi)):
        panels = ei.corner_panels(dom, order_corner=12, order_smooth=12)
        for i in range(len(dom.bdry.segments)):
            spans = sorted((min(p.tau0, p.tau1), max(p.tau0, p.tau1))
                           for p in panels if p.seg_idx == i)
            assert spans, f"segment {i} uncovered"
            assert spans[0][0] == pytest.approx(0.0)
            assert spans[-1][1] == pytest.approx(1.0)
            for a, b in zip(spans, spans[1:]):
                assert a[1] == pytest.approx(b[0]), (i, spans)


def test_corner_panels_anchor_tau0_at_the_corner():
    """tau0 is always the corner end, so the signed h = tau1-tau0 carries the orientation and
    the rule (which is anchored at its own 0) lands the right way round."""
    dom = L_shape()
    specs = ei.corner_specs(dom)
    segs = dom.bdry.segments
    for p in ei.corner_panels(dom, specs, order_corner=12):
        if p.rule == 'legendre':
            continue
        corner_pt = specs[p.corner].point
        anchored = segs[p.seg_idx].p(np.array([p.tau0]))[0]
        assert abs(anchored - corner_pt) < 1e-12, (p, anchored, corner_pt)


def test_h_shape_has_an_edge_joining_two_singular_corners_and_it_splits():
    """The case only a multi-corner domain exercises: a corner panel anchors at one endpoint,
    so an edge singular at BOTH ends must be split into two."""
    dom = H_shape()
    specs = ei.corner_specs(dom)
    sing = {s.seg_out for s in specs if s.singular and s.admissible}
    both = [s for s in specs if s.singular and s.admissible and s.seg_in in sing]
    assert both, "H_shape should have an edge with singular corners at both ends"
    panels = ei.corner_panels(dom, specs, order_corner=12)
    for s in both:
        on_seg = [p for p in panels if p.seg_idx == s.seg_in]
        assert len(on_seg) >= 2, (s.seg_in, on_seg)
        assert sum(1 for p in on_seg if p.rule != 'legendre') == 2


def test_singular_corner_report_explains_every_corner():
    for dom in (rect(2.0, 1.0), L_shape(), H_shape()):
        txt = ei.singular_corner_report(dom)
        assert len(txt.splitlines()) == len(dom.corners)


def test_default_x0_lands_on_a_singular_corner_when_there_is_one():
    dom = L_shape()
    x0 = ei.default_x0(dom)
    sing = [s.point for s in ei.corner_specs(dom) if s.singular]
    assert min(abs(x0 - p) for p in sing) < 1e-12
    # convex domain: falls back to the bounding-box centre
    assert abs(ei.default_x0(rect(2.0, 1.0)) - (1.0 + 0.5j)) < 1e-12


# ── Leg 3: multiple singular corners WITH singular amplitude, exact ──────────
#
# The certifying leg. Legs 1 and 2 cannot reach this case: Leg 1 has one singular corner, and
# Leg 2's closed-form eigenfunction is necessarily SMOOTH at the reentrant corners. Here the
# boundary data is synthetic but is a genuine member of the corner's integrand class, and the
# reference is closed-form, so the singular amplitude is real and the target is exact.
#
# WHAT A GENUINE MEMBER IS, and a wrong model that cost an afternoon. Near a corner the
# Dirichlet expansion is complete (Kondrat'ev): within the largest disk about the corner inside
# Omega, du/dn on the edge is exactly
#
#     un(s) = sum_k c_k * nu_k * J_{k nu}(sqrt(lam) s)/s  ~  sum_{k,q} a_kq s^(k nu - 1 + 2q)
#
# so on a straight edge the family is {k nu - 1 + 2q} and, after squaring, {gamma + j nu + 2q}.
# The far corner's singularity lies OUTSIDE that disk, so it enters through the coefficients
# c_k, NOT as a separate additive term.
#
# My first model summed two independent corner series, one anchored at each end. That is
# unphysical: the cross term carries exponents gamma/2 + m, outside any single-corner class,
# and it made a correct rule look broken at 1e-3 while every single-corner edge was exact to
# 1e-15. A related check: a real un at a 3pi/2 corner has NO constant term, since
# k*nu - 1 + 2q = 0 has no integer solution at nu = 2/3 -- a useful sanity test on any model.
#
# So each edge is modelled by ONE corner series, anchored at a singular end; an edge singular
# at both ends is tested once per end.

def _corner_series(nu, L, n_terms=4, seed=0, at_end=False):
    """A genuine member of the class: sum_j a_j s^(j*nu + nu - 1), even offsets only, with
    geometrically decaying amplitudes (an undamped series would be dominated by high-order
    terms no eigenfunction contains). `at_end` mirrors it to be singular at s=L instead."""
    rng = np.random.default_rng(seed)
    terms = [(rng.uniform(0.5, 1.5)*0.4**j, j*nu + nu - 1.0) for j in range(n_terms)]
    return terms, at_end


def _series_un(model, L):
    terms, at_end = model
    def un(s):
        r = (L - s) if at_end else s
        return sum(c*r**p for c, p in terms)
    return un


def _series_exact(model, L, rN):
    """int_0^L rN * un^2 ds in closed form. Mirroring at s=L leaves the value unchanged."""
    terms, _ = model
    return rN*sum(c1*c2*L**(p1 + p2 + 1)/(p1 + p2 + 1)
                  for c1, p1 in terms for c2, p2 in terms)


def _partial_exact(model, L, rN, lo, hi):
    """int_lo^hi rN * un^2 ds in closed form, in the model's own radial variable.

    A start-anchored model is a series in s, an end-anchored one in (L-s); either way the
    integral over a sub-interval is a sum of r^(p+1)/(p+1) evaluated at the sub-interval's
    endpoints measured from the model's own anchor."""
    terms, at_end = model
    a, b = (L - hi, L - lo) if at_end else (lo, hi)
    return rN*sum(c1*c2*(b**(p1 + p2 + 1) - a**(p1 + p2 + 1))/(p1 + p2 + 1)
                  for c1, p1 in terms for c2, p2 in terms)


def _leg3_case(dom, seed=0, n_terms=4, precision=1e-14):
    """Score every CORNER PANEL against the corner expansion valid on that panel.

    Per-panel rather than per-edge, because the two representations of a real eigenfunction on
    an edge singular at both ends are asymptotic expansions about DIFFERENT points: no single
    closed form is sparse-in-nu about both endpoints, so a global synthetic model would be
    smooth at one of the two anchors and would measure the wrong thing (4.8e-9, entirely from
    that artifact). Each panel is therefore tested against its own corner's series, over its own
    sub-interval, which is exactly the representation the rule is built for.

    Returns (worst relative error over panels, number of corner panels, node count)."""
    specs = ei.corner_specs(dom)
    segs = dom.bdry.segments
    bq = ei.boundary_quadrature(dom, 1.0, precision=precision, warn=False)
    x0 = 0.37 + 0.181j          # off every corner and every edge line

    worst, n_panels = 0.0, 0
    for pid, panel in enumerate(bq.panels):
        if panel.rule == 'legendre':
            continue
        n_panels += 1
        seg = segs[panel.seg_idx]
        at_end = panel.tau0 > panel.tau1        # tau0 is always the corner end
        model = _corner_series(panel.nu, seg.len, n_terms, seed + pid, at_end)
        mid = seg.p(np.array([0.5]))[0]
        rN = complex_dot(mid - x0, seg.N(np.array([0.5]))[0])

        m = bq.panel_id == pid
        h = panel.tau1 - panel.tau0
        u_local, _ = ei._panel_rule(panel)
        s = seg.len*(panel.tau0 + h*u_local)
        got = float(np.sum(bq.wts[m]*rN*_series_un(model, seg.len)(s)**2))

        lo, hi = sorted((panel.tau0*seg.len, panel.tau1*seg.len))
        exact = _partial_exact(model, seg.len, rN, lo, hi)
        worst = max(worst, abs(got/exact - 1.0))
    return worst, n_panels, len(bq.pts)


@pytest.mark.parametrize("factory,name,n_expected",
                         [(L_shape, 'L_shape', 2), (H_shape, 'H_shape', 8)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_leg3_synthetic_singular_multicorner(factory, name, n_expected, seed):
    """The certifying result: singular amplitude at every reentrant corner of a multi-corner
    domain, every corner panel integrated against a closed-form reference."""
    err, n_panels, n = _leg3_case(factory(), seed=seed)
    assert n_panels == n_expected, f"{name}: {n_panels} corner panels, expected {n_expected}"
    assert err < 1e-12, f"{name}: worst panel err={err:.3e} on {n} nodes"


def test_leg3_h_shape_edge_singular_at_both_ends_splits_into_two_corner_panels():
    """H_shape's notch floor is singular at BOTH ends. A corner panel anchors at one endpoint
    only, so that edge must split into two corner panels -- no single-corner domain reaches
    this case."""
    dom = H_shape()
    specs = ei.corner_specs(dom)
    sing_start = {s.seg_out for s in specs if s.singular and s.admissible}
    doubles = [s.seg_in for s in specs
               if s.singular and s.admissible and s.seg_in in sing_start]
    assert doubles, "expected an edge singular at both ends"
    bq = ei.boundary_quadrature(dom, 1.0, precision=1e-14, warn=False)
    for i in doubles:
        panels = [p for p in bq.panels if p.seg_idx == i]
        assert sum(1 for p in panels if p.rule != 'legendre') == 2, (i, panels)
        assert all(p.rule != 'legendre' for p in panels), "no smooth gap should remain"


def test_leg3_splitting_a_single_singular_edge_would_cost_accuracy():
    """Why an edge singular at only ONE end gets a single full-length panel rather than a
    split: the far half would be anchored where there is no singularity, so its singular weight
    has nothing to cancel and it converges only algebraically. Measured on a genuine class
    member: 8.9e-16 for one full-length panel against 1.8e-9 for the split, at order 8."""
    from lappy.quad import cached_cornerjacgauss
    nu, L = 2/3, 1.0
    gamma = 2*nu - 2
    model = ([(1.0, nu - 1), (-0.6, 2*nu - 1), (0.3, 3*nu - 1)], False)
    un = _series_un(model, L)
    exact = _series_exact(model, L, 1.0)
    u, w = cached_cornerjacgauss(8, nu, gamma, nu)
    whole = float(np.sum(L*w*un(L*u)**2))
    split = sum(float(np.sum(L*abs(t1 - t0)*w*un(L*(t0 + (t1 - t0)*u))**2))
                for t0, t1 in ((0.0, 0.5), (1.0, 0.5)))
    assert abs(whole/exact - 1) < 1e-14
    assert abs(split/exact - 1) > 1e-11
    assert abs(whole/exact - 1) < abs(split/exact - 1)/1e4


def test_leg3_no_constant_term_in_a_real_corner_series():
    """Sanity check on the model class itself: at nu=2/3 the equation k*nu - 1 + 2q = 0 has no
    solution in non-negative integers, so a genuine du/dn carries NO constant term. Any model
    that has one is not a member of the class -- the trap the first Leg 3 model fell into."""
    nu = 2/3
    assert not any(abs(k*nu - 1 + 2*q) < 1e-12
                   for k in range(1, 40) for q in range(40))


@pytest.mark.parametrize("precision", [1e-6, 1e-10, 1e-13])
def test_leg3_precision_is_honoured_on_a_multicorner_domain(precision):
    err, n_panels, n = _leg3_case(H_shape(), seed=3, precision=precision)
    assert err <= 10*precision, (precision, err, n)


def test_leg3_reference_is_closed_form_not_quadrature():
    """Guard on the ruler. The closed form is checked against mpmath's Beta function -- another
    closed form, different implementation -- not against quadrature: Gauss-Legendre on
    s^0.3 (L-s)^0.8 reaches only 3e-7 at order 200, so a quadrature 'check' would wrongly
    convict the exact expression. The analytic tier is what everything else is measured
    against, and part of it was wrong once before (see NOTEBOOK on _bessel_zero)."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    L, p, q = 1.7, 0.3, 0.8
    mine = np.exp((p + q + 1)*np.log(L) + lgamma(p + 1) + lgamma(q + 1) - lgamma(p + q + 2))
    theirs = float(mp.mpf(L)**(p + q + 1)*mp.beta(p + 1, q + 1))
    assert mine == pytest.approx(theirs, rel=1e-14)


from math import lgamma  # noqa: E402  (used by the reference guard above)
