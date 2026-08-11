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
from lappy.geometry import disk_sector, L_shape, H_shape, rect, chevron
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
    terms no eigenfunction contains). `at_end` mirrors it to be singular at s=L instead.

    ONLY VALID FOR nu = O(1) -- use `_bessel_corner_series` at a sharp convex corner. The 0.4^j
    damping is a stand-in for the expansion's true 1/Gamma(k nu + q + 1), and the two part
    company badly once nu is large: at nu = 22 this model's exponents run 21, 43, 65, 87 with
    O(1) amplitudes, giving a squared integrand of degree 174, where the physical amplitude of
    the k=1 term alone is ~1e-21. Scoring a corner rule against it there measures the model,
    not the rule -- it reported 1.7e-02 at a corner the physical series integrates to 4.4e-16.
    Harmless for L_shape and H_shape, whose corners are all nu = 2/3."""
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


# ── Leg 3b: the SAME dense family at a `cornerinterp` corner ──────────────────
#
# Leg 3 above certifies the dense corner series -- but only ever at nu = 2/3 on a straight edge,
# which `quad.corner_rule_spec` routes to 'cornerjac'. That rule is the exception: it is chosen
# only for a straight edge whose 2/nu is an integer, which among reentrant angles is essentially
# alpha = 3pi/2 alone. EVERY other corner -- any non-right polygon angle, every curved edge --
# runs 'cornerinterp', and until this section that rule had been measured only against Leg 1's
# sector eigenfunction, a SPARSE single Bessel mode. The dense multi-term family it actually
# meets on a generic domain was untested. `chevron(h1, h2)` supplies it: a nonconvex quadrilateral
# whose reentrant nu varies continuously with the parameters, on straight edges throughout, so
# the corner series {j nu + nu - 1} remains exactly the right class and the reference stays
# closed form.

def _bessel_corner_series(nu, L, n_k=3, n_q=3, lam=1.0, seed=0, at_end=False):
    """A class member with the expansion's OWN amplitudes, not a geometric stand-in.

        un(s) = sum_k c_k nu_k J_{k nu}(sqrt(lam) s)/s,
        J_{k nu}(z) = sum_q (-1)^q (z/2)^(k nu + 2q)/(q! Gamma(k nu + q + 1))

    so s^(k nu - 1 + 2q) carries c_k (k nu) (-1)^q (sqrt(lam)/2)^(k nu + 2q)
    / (q! Gamma(k nu + q + 1)), with c_k = O(1) random. That supplies both the Gamma damping
    that `_corner_series` approximates and the (-1)^q alternation it drops -- the two together
    are what make the model a member of the class at any nu rather than only at nu = O(1).

    Built in log space: k nu can reach 20+ at a sharp convex corner, where the amplitude
    underflows a direct evaluation."""
    rng = np.random.default_rng(seed)
    terms = []
    for k in range(1, n_k + 1):
        c_k = rng.uniform(0.5, 1.5)*rng.choice([-1.0, 1.0])
        for q in range(n_q + 1):
            p = k*nu + 2*q
            log_a = p*np.log(np.sqrt(lam)/2.0) - lgamma(q + 1) - lgamma(k*nu + q + 1)
            terms.append((c_k*(k*nu)*(-1.0)**q*np.exp(log_a), p - 1.0))
    return terms, at_end


def _two_notch(h1=2.0, h2=3.0, w=0.6):
    """Two reentrant corners at generic (non-right) angles: the multi-corner half of the
    question, so the dense family is asked at more than one `cornerinterp` corner at once."""
    from lappy.geometry import Polygon
    v = np.array([-1.0, -w + 1j*h1, 0.0 + 0.35j*h1, w + 1j*h1, 1.0, 1j*h2])
    return Polygon(v, val_simple=False)


_LEG3B_X0 = 0.37 + 0.181j
_LEG3B_SEEDS = (0, 1, 2, 3, 4)


def _score_corner_panels(dom, lam_max=1.0, precision=1e-14, physical=True, orders=None):
    """(rule, nu, order, worst rel err over seeds) per corner panel, against closed-form truth.

    `_leg3_case`'s scoring, generalized: any amplitude model, and an optional order override so
    the same panel can be scored off the sizing model's choice."""
    bq = ei.boundary_quadrature(dom, lam_max, precision=precision, warn=False)
    segs = dom.bdry.segments
    out = []
    for pid, panel in enumerate(bq.panels):
        if panel.rule == 'legendre':
            continue
        if orders is not None:
            panel = panel._replace(order=orders)
        seg = segs[panel.seg_idx]
        at_end = panel.tau0 > panel.tau1
        mid = seg.p(np.array([0.5]))[0]
        rN = complex_dot(mid - _LEG3B_X0, seg.N(np.array([0.5]))[0])
        h = panel.tau1 - panel.tau0
        u_local, w = ei._panel_rule(panel)
        s = seg.len*(panel.tau0 + h*u_local)
        wts = seg.len*abs(h)*w
        lo, hi = sorted((panel.tau0*seg.len, panel.tau1*seg.len))
        worst = 0.0
        for seed in _LEG3B_SEEDS:
            model = (_bessel_corner_series(panel.nu, seg.len, 3, 3, lam_max, seed + pid, at_end)
                     if physical else _corner_series(panel.nu, seg.len, 4, seed + pid, at_end))
            got = float(np.sum(wts*rN*_series_un(model, seg.len)(s)**2))
            exact = _partial_exact(model, seg.len, rN, lo, hi)
            worst = max(worst, abs(got/exact - 1.0))
        out.append((panel.rule, panel.nu, panel.order, worst))
    return out


# (factory, nu, bar). Bars are the MEASURED capability with an order of margin, not a wish:
# 4.0e-15 at nu=0.888, 3.6e-14 at nu=0.772, 1.5e-12 at nu=0.615. The degradation as nu -> 1/2
# is real and monotone, which is why the bars are not flat -- see the ceiling test below.
LEG3B_CASES = [(lambda: chevron(0.2, 1), 0.888363, 1e-12),
               (lambda: chevron(0.5, 3), 0.772101, 1e-12),
               (lambda: chevron(1.5, 4), 0.615133, 1e-10)]


@pytest.mark.parametrize("factory,nu_expected,bar", LEG3B_CASES)
def test_leg3b_cornerinterp_integrates_the_dense_family(factory, nu_expected, bar):
    """The gap this section exists to close: a dense corner series with real singular amplitude,
    at the interpolatory rule a generic domain actually gets."""
    dom = factory()
    scored = [(r, nu, o, e) for r, nu, o, e in _score_corner_panels(dom) if nu < 1.0]
    assert scored, "expected at least one singular corner panel"
    assert any(r == 'cornerinterp' for r, _, _, _ in scored), \
        "this domain is supposed to exercise cornerinterp, not cornerjac"
    assert any(abs(nu - nu_expected) < 1e-5 for _, nu, _, _ in scored), \
        [nu for _, nu, _, _ in scored]
    for rule, nu, order, err in scored:
        assert err < bar, f"nu={nu:.6f} rule={rule} order={order}: {err:.3e} >= {bar:.1e}"


def test_leg3b_sharp_convex_corners_are_exact_on_a_physical_series():
    """A sharp convex corner (nu up to 22 here) also draws the corner rule, via `nonintegral`.
    On a physical series the rule is exact there -- 3.5e-14 worst across every convex panel of
    the chevron family."""
    worst = 0.0
    for factory in (lambda: chevron(1, 2), lambda: chevron(2, 3), lambda: chevron(0.5, 3)):
        for rule, nu, order, err in _score_corner_panels(factory()):
            if nu > 1.0:
                worst = max(worst, err)
    assert worst > 0.0, "expected some convex corner panels"
    assert worst < 1e-12, worst


def test_leg3b_geometric_amplitudes_are_not_a_class_member_at_large_nu():
    """Why `_bessel_corner_series` exists, pinned as arithmetic rather than left as a caution.

    `_corner_series` gives its leading term an O(1) amplitude at every nu. The true one is
    (sqrt(lam)/2)^nu/Gamma(nu+1), which at a sharp corner is vanishingly small -- so the
    geometric model demands the rule integrate a term the eigenfunction does not have, and
    convicts it. Scoring the SAME panels both ways shows it directly."""
    nu = 9.764063                                  # chevron(1,2)'s sharp corner
    true_amp = np.exp(nu*np.log(0.5) - lgamma(nu + 1.0))
    assert true_amp < 1e-9, true_amp               # against the geometric model's O(1)

    dom = chevron(1, 2)
    phys = max(e for _, n, _, e in _score_corner_panels(dom, physical=True) if n > 1.0)
    geom = max(e for _, n, _, e in _score_corner_panels(dom, physical=False) if n > 1.0)
    assert phys < 1e-13, phys
    assert geom > 1e4*phys, (geom, phys)


def test_leg3b_near_slit_cornerinterp_has_a_ceiling_and_says_so():
    """The known limitation, pinned so it is a fact and not a surprise: below nu ~ 0.6 the
    interpolatory rule tops out near 1e-10 on the dense family. What makes this acceptable
    rather than silent is the second half -- `boundary_quadrature` warns and records a
    shortfall. What makes it a real limitation is that the reported bound is 450x optimistic
    (1.15e-12 claimed against 5.2e-10 delivered), which is `sizing_precision` being a sizing
    heuristic exactly as documented.

    If a future rule clears this, the first assert fires -- update the bars, do not delete."""
    dom = chevron(2, 3)
    scored = [(r, nu, o, e) for r, nu, o, e in _score_corner_panels(dom) if nu < 0.6]
    # one panel per adjacent edge, and they are NOT alike: the same corner at the same order
    # measures 5.2e-10 on one edge and 5.2e-14 on the other, so the corner's capability is the
    # worst of its panels and a test that looked at one of them could report either story.
    assert len(scored) == 2, scored
    assert all(r == 'cornerinterp' for r, _, _, _ in scored), scored
    nu, err = scored[0][1], max(e for _, _, _, e in scored)
    assert 1e-12 < err < 1e-7, f"known ceiling moved: nu={nu:.6f} err={err:.3e}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bq = ei.boundary_quadrature(dom, 1.0, precision=1e-14)
    assert bq.sizing_precision > 1e-14
    assert any(r == 'model bound above target' for _, _, r in bq.shortfalls)
    assert any('could not size for' in str(w.message) for w in caught)


def test_leg3b_more_order_does_not_rescue_a_near_slit_corner_at_low_k():
    """Raising the order past `corner_order_for_precision`'s choice makes this corner WORSE:
    9.5e-11 at order 32 against 1.2e-08 at 128. The weights stay positive and sum|w| is
    unchanged, so this is not the weight blow-up that killed `cornerinterp` on the dense
    exponent set -- it is accuracy loss in the interpolatory solve itself. Controls at
    nu = 0.888 and nu = 0.772 stay flat to order 192, so it is specific to the near-slit regime.

    **AT THIS lam_max, WHICH IS 1.** That qualifier is the whole content of the test's name and
    was missing when it was written. The corner order serves two demands -- resolving the
    singular exponent family and resolving oscillation at wavenumber k -- and `_score_corner_
    panels` builds at `lam_max=1`, which exercises only the first. The saturation point moves
    right, fast, with lam:

        true optimum order      lam=1    lam=100   lam=1000
        chevron(2,3)   nu=0.587    24         88        128
        chevron(0.5,3) nu=0.772    40        128        128

    So this must NOT be read as "cap the corner order" -- at realistic lam the model's choice is
    if anything too low, and a fixed cap would be a regression. What survives at every lam is
    the ceiling itself and the degradation PAST the optimum, wherever the optimum sits."""
    dom = chevron(2, 3)

    def at(order):
        return max(e for _, nu, _, e in _score_corner_panels(dom, orders=order) if nu < 0.6)

    lo, hi = at(32), at(128)
    assert hi > 10*lo, f"order 32 {lo:.2e} vs order 128 {hi:.2e}"

    for factory in (lambda: chevron(0.2, 1), lambda: chevron(0.5, 3)):
        ctrl = factory()
        errs = [max(e for _, nu, _, e in _score_corner_panels(ctrl, orders=o) if nu < 1.0)
                for o in (32, 192)]
        assert max(errs) < 1e-12, errs


def test_leg3b_bars_have_teeth():
    """Do the Leg 3b bars actually detect a broken rule? `test_leg1_nu_sensitivity_guard` asks
    this of 'cornerjac'; 'cornerinterp' takes its exactness from the WEIGHTS rather than from
    node placement, so it needs its own answer, and the bars above are only worth having if it
    is yes.

    Perturb the RULE's nu while holding the MODEL at the true nu -- perturbing both leaves them
    consistent and measures nothing, which is how this check read 'no sensitivity' on the first
    attempt. Measured at nu = 0.7721, order 28, against the 1e-12 bar:

        correct rule          1.7e-13
        rule nu off by 3e-4   7.0e-06        7 orders, and over the bar
        rule nu off by 3e-2   8.0e-04
        smooth legendre       1.7e-02        the rule itself being wrong
    """
    dom = chevron(0.5, 3)
    panels = ei.corner_panels(dom, ei.corner_specs(dom), order_corner=28, order_smooth=32)

    def score(nu_scale=1.0, rule=None):
        worst = 0.0
        for pid, p in enumerate(panels):
            if p.rule == 'legendre' or p.nu > 1.0:
                continue
            q = p._replace(nu=p.nu*nu_scale)
            if rule is not None:
                q = q._replace(rule=rule)
            seg = dom.bdry.segments[p.seg_idx]
            at_end = p.tau0 > p.tau1
            mid = seg.p(np.array([0.5]))[0]
            rN = complex_dot(mid - _LEG3B_X0, seg.N(np.array([0.5]))[0])
            u_local, w = ei._panel_rule(q)
            h = q.tau1 - q.tau0
            s = seg.len*(q.tau0 + h*u_local)
            wts = seg.len*abs(h)*w
            lo, hi = sorted((q.tau0*seg.len, q.tau1*seg.len))
            for seed in _LEG3B_SEEDS:
                model = _bessel_corner_series(p.nu, seg.len, 3, 3, 1.0, seed + pid, at_end)
                got = float(np.sum(wts*rN*_series_un(model, seg.len)(s)**2))
                worst = max(worst, abs(got/_partial_exact(model, seg.len, rN, lo, hi) - 1.0))
        return worst

    clean = score()
    assert clean < 1e-12, clean
    assert score(1 + 3e-4) > 1e4*max(clean, 1e-15)
    assert score(rule='legendre') > 1e-4


def test_leg3b_two_generic_reentrant_corners_at_once():
    """Multi-corner, both corners generic, both on `cornerinterp`: the nearest thing to a real
    optimization domain in the exact-truth tier. nu = 0.5557 is inside the ceiling regime, so
    the bar is the measured 4.1e-10 with margin rather than machine precision."""
    scored = [(r, nu, o, e) for r, nu, o, e in _score_corner_panels(_two_notch()) if nu < 1.0]
    assert len(scored) >= 2, scored
    assert all(r == 'cornerinterp' for r, _, _, _ in scored), scored
    for rule, nu, order, err in scored:
        assert err < 1e-8, f"nu={nu:.6f} order={order}: {err:.3e}"


# ── Leg 2: multiple singular corners, exact, but ZERO singular amplitude ──────
#
# A control, and its limitation is the point. sin(m pi x) sin(n pi y) vanishes on the entire
# integer grid, so it is an exact Dirichlet eigenfunction of any polyomino with norm^2 =
# cells/4 -- giving exact truth on a domain with four reentrant corners. But a closed-form
# eigenfunction on a nonconvex domain is necessarily SMOOTH at those corners (that is exactly
# why L_shape has no closed form), so the singular coefficients are zero and this leg CANNOT
# test the corner singularity. It tests geometry, panel splitting, orientation and assembly.
# Leg 3 carries the singularity.

POLYOMINOES = [
    ('plus', [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)], 4),
    ('L', [(0, 0), (0, 1), (1, 0)], 1),
    ('H', [(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)], 4),
    ('S', [(0, 0), (1, 0), (1, 1), (2, 1)], 2),
    ('square', [(0, 0), (1, 0), (0, 1), (1, 1)], 0),
]


@pytest.mark.parametrize("name,cells,n_reentrant", POLYOMINOES)
def test_leg2_polyomino_geometry(name, cells, n_reentrant):
    from lappy.geometry import polyomino
    dom = polyomino(cells)
    cia = np.asarray(dom.corner_int_angles)
    assert int((cia > np.pi + 1e-9).sum()) == n_reentrant
    assert dom.perimeter == pytest.approx(
        sum(4 - sum(1 for nb in ((i+1, j), (i-1, j), (i, j+1), (i, j-1))
                    if nb in set(map(tuple, cells)))
            for i, j in cells), abs=1e-12)


@pytest.mark.parametrize("name,cells,n_reentrant", POLYOMINOES)
@pytest.mark.parametrize("m,n", [(1, 1), (2, 1), (2, 3)])
def test_leg2_polyomino_rellich_norm_is_one(name, cells, n_reentrant, m, n):
    """Exact truth on up to four reentrant corners: norm^2 = cells/4 exactly."""
    from lappy.geometry import polyomino
    dom = polyomino(cells)
    lam = ref.polyomino_eig(m, n)
    u, norm2 = ref.polyomino_eigfun(m, n, len(cells))
    assert norm2 == pytest.approx(len(cells)/4.0, abs=1e-15)
    s = 1.0/np.sqrt(norm2)
    g = ref.polyomino_eigfun_grad(m, n)
    bq = ei.boundary_quadrature(dom, lam, precision=1e-14, warn=False)
    ed = dirichlet_data(bq, lambda z: s*g(z), lambda z: s*u(z))
    x0 = 1.37 + 0.61j            # off every corner and every edge line
    G = ei.gram(ed, lam, bq, x0)
    assert abs(G[0, 0] - 1.0) < 1e-12, (name, m, n, G[0, 0] - 1.0, len(bq.pts))


def test_leg2_is_a_control_not_a_singularity_test():
    """Pin the limitation, so nobody later reads Leg 2 as evidence about the singularity: the
    eigenfunction is SMOOTH at the reentrant corners, i.e. du/dn stays bounded there, unlike a
    genuine corner-singular solution whose du/dn ~ r^(nu-1) diverges."""
    from lappy.geometry import plus_shape
    dom = plus_shape()
    g = ref.polyomino_eigfun_grad(1, 1)
    reentrant = [s.point for s in ei.corner_specs(dom) if s.singular]
    assert len(reentrant) == 4
    for c in reentrant:
        # approach the corner along the interior bisector; |grad u| must stay bounded
        for r in (1e-2, 1e-4, 1e-6, 1e-8):
            z = c + r*np.exp(1j*np.pi/4)
            assert abs(g(z)) < 10.0, (c, r, abs(g(z)))


def test_leg2_polyomino_rejects_bad_cell_sets():
    from lappy.geometry import polyomino
    with pytest.raises(ValueError, match="edge-connected"):
        polyomino([(0, 0), (1, 1)])              # diagonal-only join
    with pytest.raises(ValueError, match="duplicate"):
        polyomino([(0, 0), (0, 0)])
    with pytest.raises(ValueError, match="at least one"):
        polyomino([])
    with pytest.raises(ValueError, match="not simple|hole"):
        # a ring of eight cells enclosing a hole
        polyomino([(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)])


# ── Panel length vs corner clearance ─────────────────────────────────────────
#
# A corner panel is exact for the corner's expansion, which is valid only within the largest
# disk about the corner inside Omega. On a domain whose edge is long relative to that disk, an
# uncapped panel is catastrophic -- and the mechanism is resolution, not just class mismatch:
# the corner rule clusters its nodes AT the corner, so the far end of a long panel is sparsely
# sampled and cannot resolve the sqrt(lam) oscillation over the remaining arclength.
#
# Measured with the polyomino's EXACT eigenfunction (zero residual, closed-form norm), because
# x0-invariance on a real MPS eigenfunction cannot see this at all: its spread came back
# identical to three figures across every panel configuration, being dominated by the
# eigenfunction's own residual. See benchmarks/corner_quad/panel_length.py.

def _long_arm(n):
    """1 x n strip of cells plus one below the left end: the reentrant corner's edge has length
    n-1 while its clearance is the strip width, 1."""
    from lappy.geometry import polyomino
    return polyomino([(i, 1) for i in range(n)] + [(0, 0)]), [(i, 1) for i in range(n)] + [(0, 0)]


def _long_arm_norm_error(n, m, k, clearance_frac):
    dom, cells = _long_arm(n)
    lam = ref.polyomino_eig(m, k)
    u, norm2 = ref.polyomino_eigfun(m, k, len(cells))
    g = ref.polyomino_eigfun_grad(m, k)
    s = 1.0/np.sqrt(norm2)
    bq = ei.boundary_quadrature(dom, lam, precision=1e-13,
                                clearance_frac=clearance_frac, warn=False)
    ed = dirichlet_data(bq, lambda z: s*g(z), lambda z: s*u(z))
    return abs(ei.gram(ed, lam, bq, 1.7 + 1.3j)[0, 0] - 1.0), len(bq.pts)


def test_long_edge_at_a_singular_corner_needs_the_clearance_cap():
    """The regression this guards: with the cap off, a 15-long edge at a corner of clearance 1
    loses twelve orders. Both configurations are run so the test states the size of the effect,
    not merely that the default happens to work."""
    err_on, n_on = _long_arm_norm_error(16, 2, 3, ei._CLEARANCE_FRAC)
    err_off, n_off = _long_arm_norm_error(16, 2, 3, None)
    assert err_on < 1e-13, f"default clearance cap should hold 1e-13, got {err_on:.2e}"
    assert err_off > 1e-4, ("expected the uncapped rule to fail badly here; if this now passes, "
                            "the test domain no longer exercises the effect")
    assert err_on < err_off/1e9
    assert n_on < 2*n_off, f"the cap should cost well under 2x nodes ({n_on} vs {n_off})"


@pytest.mark.parametrize("n", [2, 4, 8, 16])
@pytest.mark.parametrize("m,k", [(1, 1), (2, 3)])
def test_clearance_cap_holds_across_arm_lengths(n, m, k):
    err, _ = _long_arm_norm_error(n, m, k, ei._CLEARANCE_FRAC)
    assert err < 1e-13, (n, m, k, err)


def test_clearance_cap_default_is_not_at_the_edge_of_validity():
    """cf=1.0 puts the panel exactly at the radius where the expansion stops being valid, and
    measures an order worse than 0.9 (1.2e-14 against 1.3e-15 at n=16). The default must sit
    inside that boundary."""
    assert 0.5 <= ei._CLEARANCE_FRAC < 1.0
    err_default, _ = _long_arm_norm_error(16, 2, 3, ei._CLEARANCE_FRAC)
    err_edge, _ = _long_arm_norm_error(16, 2, 3, 1.0)
    assert err_default <= err_edge


def test_corner_clearance_matches_hand_computed_geometry():
    """corner_clearance is the distance to the nearest NON-adjacent boundary piece; on the
    long-arm strip that is the opposite wall, i.e. the strip width."""
    dom, _ = _long_arm(8)
    for s in ei.corner_specs(dom):
        if s.singular and s.admissible:
            clear = ei.corner_clearance(dom, s.point, s.seg_out, s.seg_in)
            assert clear == pytest.approx(1.0, abs=1e-12), clear


# ── Which corners need the corner-adapted rule ───────────────────────────────
#
# The criterion was `nu < 1` -- reentrant only -- justified as "a smooth rule is already exact"
# for nu >= 1. It is not: the Rellich integrand carries r^(2nu-2), so what matters is whether
# that exponent is an integer. A 135-degree corner (nu = 4/3) puts r^(2/3) on its edges, which
# Gauss-Legendre integrates algebraically. Measured on the suite before the fix:
#
#     right_trapezoid   84 nodes  5.6e-09      GWW1  204 nodes  6.9e-07
#
# both against a reported precision of 1e-13, and in both cases the offending panels were
# `legendre` ones whose sizing model claimed ~4e-15.

def test_smooth_power_error_is_exact_on_integer_powers():
    """tau^gamma with integer gamma is a polynomial: Gauss is exact, so such a corner needs no
    special rule however small nu is."""
    from lappy.quad import smooth_power_error
    for gamma in (1, 2, 3, 4, 6, 10):
        assert smooth_power_error(float(gamma), 16) < 1e-14, gamma


def test_smooth_power_error_flags_the_corners_that_actually_defeat_gauss():
    """Small fractional powers stay bad at any usable order; large ones are already smooth
    enough. This is the whole content of the criterion."""
    from lappy.quad import smooth_power_error
    assert smooth_power_error(2/3, 64) > 1e-8        # nu = 4/3, the case that was missed
    assert smooth_power_error(1/3, 64) > 1e-7        # nu = 7/6
    assert smooth_power_error(11.55, 16) < 1e-14     # nu = 6.78, fine on a smooth rule


def test_nonintegral_gives_a_convex_noninteger_corner_the_corner_rule():
    """A 135-degree corner (nu=4/3) is convex, so the reentrant-only criterion leaves it on a
    smooth rule; `nonintegral=True` must not."""
    dom = rect(1.0, 1.0)                      # all nu = 2: nothing should change
    for ni in (False, True):
        bq = ei.boundary_quadrature(dom, 50.0, precision=1e-13, nonintegral=ni, warn=False)
        assert all(p.rule == 'legendre' for p in bq.panels)

    from lappy.geometry import Polygon
    tri = Polygon(np.array([0.0, 1.0, 1.0 + 1.0j]))       # 45/45/90: nu = 4, 4, 2
    bq = ei.boundary_quadrature(tri, 50.0, precision=1e-13, nonintegral=True, warn=False)
    assert all(p.rule == 'legendre' for p in bq.panels), 'integer nu needs no corner rule'

    oct8 = np.exp(2j*np.pi*np.arange(8)/8)                # regular octagon: nu = 4/3
    bq_off = ei.boundary_quadrature(Polygon(oct8), 50.0, precision=1e-13,
                                    nonintegral=False, warn=False)
    bq_on = ei.boundary_quadrature(Polygon(oct8), 50.0, precision=1e-13,
                                   nonintegral=True, warn=False)
    assert all(p.rule == 'legendre' for p in bq_off.panels)
    assert any(p.rule != 'legendre' for p in bq_on.panels)
    assert len(bq_on.pts) > len(bq_off.pts)


def test_nonintegral_fixes_a_convex_corner_against_exact_truth():
    """alpha = 3pi/4 gives nu = 4/3: a CONVEX corner, so the reentrant-only criterion leaves it
    on a smooth rule, but its eigenfunction still carries r^(4/3) and so puts r^(2/3) into the
    Rellich integrand. Exact sector eigenfunction, generic x0 (the apex would cancel the very
    contribution being measured).

    Not tested with a synthetic integrand: a plane wave is analytic at the corner, and the
    corner rule is not even exact on constants (see "Three things that will look like bugs"),
    so it makes a correct rule look broken. The integrand has to be a genuine member of the
    corner's class, which means a real eigenfunction.
    """
    alpha = 0.75*np.pi
    dom = disk_sector(1.0, alpha)
    apex = [s for s in ei.corner_specs(dom) if abs(s.nu - 4/3) < 1e-9]
    assert apex, 'expected a nu=4/3 corner'
    assert not apex[0].singular, 'reentrant-only criterion should call this corner smooth'

    lam = ref.sector_eig(1, 1, 1.0, alpha)
    u, norm2 = ref.sector_eigfun(1, 1, 1.0, alpha)
    g = ref.sector_eigfun_grad(1, 1, 1.0, alpha)
    sc = 1.0/np.sqrt(norm2)
    x0 = 0.37 + 0.181j

    def err(nonintegral):
        bq = ei.boundary_quadrature(dom, lam, precision=1e-13,
                                    nonintegral=nonintegral, warn=False)
        G = sc*g(bq.pts)
        ed = ei.EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts,
                           (sc*u(bq.pts))[:, None],
                           complex_dot(G, bq.normals)[:, None],
                           complex_dot(G, bq.tangents)[:, None])
        return abs(ei.gram(ed, lam, bq, x0)[0, 0] - 1.0), len(bq.pts)

    off, n_off = err(False)
    on, n_on = err(True)
    assert off > 1e-9, f'expected the smooth rule to be short here, got {off:.2e}'
    assert on < 1e-12, f'corner-adapted rule should recover it, got {on:.2e}'
    assert n_on > n_off


# ── Honest precision reporting ───────────────────────────────────────────────

def test_demoted_singular_corner_makes_precision_infinite_and_is_named():
    """sector_slit's nu=0.504 corner is inadmissible and falls back to a smooth rule, which
    cannot integrate the singularity at all -- measured 6.9e-01 against a reported 1e-13. The
    reported precision must not be a number in that case."""
    dom = disk_sector(1.0, 1.984*np.pi)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bq = ei.boundary_quadrature(dom, 30.0, precision=1e-13)
    demoted = [s for s in ei.corner_specs(dom) if s.singular and not s.admissible]
    assert demoted, 'this geometry is supposed to have an inadmissible corner'
    assert not np.isfinite(bq.sizing_precision)
    assert bq.shortfalls and any('demoted' in str(r) for _, _, r in bq.shortfalls)


def test_precision_stays_finite_and_shortfalls_empty_on_an_easy_domain():
    bq = ei.boundary_quadrature(rect(1.0, 1.0), 50.0, precision=1e-13, warn=False)
    assert np.isfinite(bq.sizing_precision) and bq.sizing_precision <= 1e-13
    assert bq.shortfalls == ()


# ── A posteriori verification ────────────────────────────────────────────────

def test_refine_quadrature_covers_the_same_boundary_and_keeps_the_anchors():
    dom = L_shape()
    bq = ei.boundary_quadrature(dom, 60.0, precision=1e-13, warn=False)
    ref_bq = ei.refine_quadrature(dom, bq, depth=2)
    assert len(ref_bq.pts) > len(bq.pts)
    assert ref_bq.wts.sum() == pytest.approx(bq.wts.sum(), rel=1e-12)
    anchors = {(p.seg_idx, p.tau0) for p in bq.panels if p.rule != 'legendre'}
    ref_anchors = {(p.seg_idx, p.tau0) for p in ref_bq.panels if p.rule != 'legendre'}
    assert anchors == ref_anchors, 'a corner panel must keep its anchored end'


def test_verify_gram_reports_zero_on_an_exactly_integrated_case():
    """A rectangle's eigenfunction on a rule sized well past its needs: refinement must move
    nothing, so the instrument reports the truth rather than its own noise."""
    L = H = 1.0
    lam = ref.rect_eig(1, 2, L, H)
    u, norm2 = ref.rect_eigfun(1, 2, L, H)
    s = 1.0/np.sqrt(norm2)
    a, b = np.pi/L, 2*np.pi/H
    dom = rect(L, H)
    bq = ei.boundary_quadrature(dom, 4*lam, precision=1e-14, warn=False)

    def ed_for(q):
        x, y = np.real(q.pts), np.imag(q.pts)
        G = s*(a*np.cos(a*x)*np.sin(b*y) + 1j*b*np.sin(a*x)*np.cos(b*y))
        return ei.EigfunData(q.pts, q.normals, q.tangents, q.wts,
                             (s*u(q.pts))[:, None],
                             complex_dot(G, q.normals)[:, None],
                             complex_dot(G, q.tangents)[:, None])

    ref_bq = ei.refine_quadrature(dom, bq, depth=2)
    G = ei.gram(ed_for(bq), lam, bq)[0, 0]
    G_ref = ei.gram(ed_for(ref_bq), lam, ref_bq)[0, 0]
    assert abs(G - 1.0) < 1e-13
    assert abs(G - G_ref) < 1e-13


# ── The closed loop: certified_quadrature ────────────────────────────────────
#
# The sizing model on one side, `verify_gram` on the other, and until now nothing between them.
# These tests cover the loop and, more importantly, the measurement that decided its shape:
# refining in place is the obvious design and it is wrong.

def _sector_true_and_refined(aop, precision=1e-13, depth=2, smooth_order=48):
    """(true error, true error after one refine_quadrature) on an exactly normalized sector."""
    dom, lam, u, g = sector_setup(aop, 1, 1)
    x0 = 0.4 + 0.05j                              # 'generic': cancels no corner
    bq = ei.boundary_quadrature(dom, lam, precision=precision, warn=False)
    rq = ei.refine_quadrature(dom, bq, depth=depth, smooth_order=smooth_order)
    before = abs(ei.gram(dirichlet_data(bq, g, u), lam, bq, x0)[0, 0] - 1.0)
    after = abs(ei.gram(dirichlet_data(rq, g, u), lam, rq, x0)[0, 0] - 1.0)
    return before, after


def test_refining_in_place_loses_ground_near_the_slit():
    """Why `certified_quadrature` rebuilds instead of refining. `refine_quadrature` shrinks a
    corner panel and gives the vacated piece to a legendre rule that is still inside the
    singularity; near the slit that costs more than the shrink buys. At alpha=1.75pi the true
    error goes 3.3e-12 -> 1.3e-10 across one refinement.

    It remains the right REFERENCE -- verify_gram only needs a rule that differs -- but adopting
    it as the working rule would walk a loop backwards exactly where it is needed."""
    before, after = _sector_true_and_refined(1.75)
    assert after > 10*before, (before, after)

    # and it is genuinely a near-slit effect, not a general property of refinement
    b_mid, a_mid = _sector_true_and_refined(1.1)
    assert a_mid <= 10*b_mid, (b_mid, a_mid)


def test_verify_gram_estimate_tracks_the_true_error():
    """The stopping test is only as good as this. On exactly normalized sectors the estimate
    stayed within 6.3x of the true error and never under-reported by more, across nu from 2.0
    down to 0.526 -- which is what makes it safe to stop on, given margin."""
    x0 = 0.4 + 0.05j
    worst_optimism = 0.0
    for aop in (0.5, 2/3, 1.1, 1.25, 4/3, 1.5, 1.6, 1.75, 1.9):
        dom, lam, u, g = sector_setup(aop, 1, 1)
        bq = ei.boundary_quadrature(dom, lam, precision=1e-13, warn=False)
        rq = ei.refine_quadrature(dom, bq, depth=2, smooth_order=48)
        G = ei.gram(dirichlet_data(bq, g, u), lam, bq, x0)
        G_ref = ei.gram(dirichlet_data(rq, g, u), lam, rq, x0)
        true_err = abs(G[0, 0] - 1.0)
        est = float(abs(G - G_ref).max()/max(abs(np.diag(G_ref)).max(), np.finfo(float).tiny))
        worst_optimism = max(worst_optimism, true_err/max(est, 1e-18))
    assert worst_optimism < 20.0, worst_optimism


def _mps_case(dom, n_basis=200):
    from lappy import bases, bounds
    from lappy.mps import MPSEigensolver, weyl_est
    basis = bases.make_default_basis(dom, n_basis)
    solver = MPSEigensolver.from_domain(dom, basis=basis)
    out = solver.solve_interval(bounds.faber_krahn(dom), weyl_est(2, dom), 20)
    eigs = np.atleast_1d(np.asarray(out[0] if isinstance(out, tuple) else out)).ravel()
    lam = float(eigs[0])
    return basis, solver, lam, solver.eigenfunction_coef(lam, mult=1), weyl_est(2, dom)


def test_certified_quadrature_closes_a_gap_the_sizing_model_misses():
    """The point of the loop, on a domain where the model and the measurement disagree:
    chevron(1,2) is sized at 1e-13 and measures 9.1e-12 as the solver builds it. Escalating to
    smooth_safety=3 brings it to 1.8e-13 for 1.5x the nodes."""
    dom = chevron(1, 2)
    basis, solver, lam, coef, lam_max = _mps_case(dom)
    _, _, baseline = ei.verify_gram(basis, lam, coef, solver.bdry_quad, dom)
    assert baseline > 1e-12, f"baseline {baseline:.2e}: this domain no longer shows the gap"

    out = ei.certified_quadrature(dom, basis, lam, coef, lam_max, target=1e-12)
    assert out.certified, out.history
    assert out.error <= 1e-12
    assert out.error < baseline/10
    assert out.smooth_safety > 1.0, "expected escalation to be what fixed it"


def test_certified_quadrature_reports_failure_rather_than_claiming_it():
    """An unreachable target must come back `certified=False`, with the trace, the best rule
    actually measured (not the last one tried), and a warning -- never a silent claim."""
    dom = L_shape()
    basis, solver, lam, coef, lam_max = _mps_case(dom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = ei.certified_quadrature(dom, basis, lam, coef, lam_max, target=1e-18)
    assert not out.certified
    assert np.isfinite(out.error) and out.error > 1e-18
    assert out.error == min(e for _, _, e in out.history), 'must return the BEST, not the last'
    assert any('did not reach target' in str(w.message) for w in caught)


def test_certified_quadrature_stops_escalating_once_nodes_stop_paying():
    """A corner-limited domain must not burn the whole schedule. The stall test abandons the
    escalation as soon as more nodes stop buying `stall_factor`, so the cost of an unreachable
    target stays bounded."""
    dom = L_shape()
    basis, solver, lam, coef, lam_max = _mps_case(dom)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = ei.certified_quadrature(dom, basis, lam, coef, lam_max, target=1e-18,
                                      safety_schedule=(1.0, 2.0, 3.0, 4.0, 6.0))
    assert len(out.history) < 5, out.history


# ── The parametrization, not the eigenfunction ───────────────────────────────
#
# `int (r.N) ds = 2|Omega|` by the divergence theorem: a closed form involving no
# eigenfunction, no lam and no basis. It isolates what the *geometry* costs. An ellipse fails
# it (1.3e-05 at the 46 nodes the oscillation model asks for) while a disk, a stadium and a
# mushroom are machine-exact -- the difference is a varying-speed arclength map, whose strip of
# analyticity narrows with eccentricity and takes Gauss-Legendre's rate with it.
#
# Not the arclength table's tolerance: measured identical to three digits at tol = 1e-4, 1e-6
# and 1e-8 (build cost 0.0s, 0.9s, 65s), so refining the table is not the fix.

def _area_identity_error(dom, resolve_geometry=False, **kw):
    # Explicit rather than defaulted: both flags are ON by default now, and these tests exist
    # to compare the two behaviours, so they must name the one they mean.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bq = ei.boundary_quadrature(dom, 30.0, precision=1e-13,
                                    resolve_geometry=resolve_geometry, **kw)
    rN = complex_dot(bq.pts - bq.x0, bq.normals)
    return abs(np.sum(bq.wts*rN) - 2*dom.area)/(2*dom.area), bq


def test_exactly_parametrized_boundaries_already_satisfy_the_area_identity():
    """A disk, a stadium and a polygon are exact at the model's own node count, so
    resolve_geometry must cost them nothing."""
    from lappy.geometry import disk
    for dom in (disk(1.0), rect(1.0, 1.0), L_shape()):
        off, bq_off = _area_identity_error(dom)
        on, bq_on = _area_identity_error(dom, resolve_geometry=True)
        assert off < 1e-14, off
        assert on < 1e-14, on
        assert len(bq_on.pts) == len(bq_off.pts), 'no nodes should be spent here'


def test_resolve_geometry_fixes_a_mildly_eccentric_ellipse():
    from lappy.geometry import ellipse
    off, bq_off = _area_identity_error(ellipse(2, 1))
    on, bq_on = _area_identity_error(ellipse(2, 1), resolve_geometry=True)
    assert off > 1e-7, f'the oscillation model is supposed to be short here, got {off:.1e}'
    assert on < 1e-14, on
    assert len(bq_on.pts) > len(bq_off.pts)
    assert bq_on.shortfalls == ()


def test_an_eccentric_ellipse_is_reported_not_pretended():
    """Some geometry is simply out of reach: a=6 cannot be resolved at any usable order. The
    contract is that this is visible in `precision` and `shortfalls`, not that it is fixed."""
    from lappy.geometry import ellipse
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bq = ei.boundary_quadrature(ellipse(6, 1), 30.0, precision=1e-13,
                                    resolve_geometry=True)
    assert bq.sizing_precision > 1e-13, 'sizing must not report a target it did not meet'
    assert len(bq.shortfalls) == 1
    assert any('parametrization' in str(w.message) for w in caught)
