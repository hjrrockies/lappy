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
