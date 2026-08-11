"""Boundary integrals of *eigenfunctions*, with corner-adapted quadrature.

This module computes L^2(Omega) Gram matrices -- and, more generally, weighted
bilinear Cauchy-data integrals -- for a small cluster of eigenfunctions, from
boundary data alone, via the Rellich identity

    ||u||^2_{L2(Omega)} = (1/2 lam) * integral_{dOmega} (x - x0).n (du/dn)^2 ds

(Dirichlet; see docs/rellich.md for the Zaremba generalization). Its purpose is
to let MPSEigensolver return L^2-orthonormal eigenfunctions automatically, with
no quadrature tuning by the caller.

Scope, deliberately narrow
--------------------------
Everything here operates on *eigenfunction*-level data -- a handful of columns,
evaluated before any integral is formed ("evaluate first, sandwich never") --
never on a basis-level N x N Gram. That narrowing is what makes the corner
quadrature work, and it is not merely a simplification:

An exact Dirichlet eigenfunction's local expansion at a corner of interior angle
alpha is purely the corner family sum_k c_k J_{k nu}(sqrt(lam) r) sin(k nu theta)
with nu = pi/alpha, so on an edge leaving that corner
(du/dn)^2 = r^(2nu-2) G(r) with G a series in fractional powers of arclength.
That is exactly the class quad.cached_cornerjacgauss integrates to machine
precision. A basis-level Gram is *not* of that form: its columns centred at
other corners are plain analytic here, with O(1) amplitude at every corner
simultaneously. Eigenfunction-level data is therefore the matched consumer, and
the only one supported.

The node set is a pure function of geometry and lam_max: it needs no basis, and
it is built once and reused for every lam in a search, since only the
eigenfunction's Cauchy data depends on lam.

Curved boundaries
-----------------
Curved segments are supported, including two curved segments meeting at a
singular corner, and a corner where a straight edge meets an arc. Two things
differ from the straight case and both are handled by the substitution exponent
(quad.corner_substitution), not by special-casing the geometry:

- **r.N is not constant.** On a straight edge from x0 at the corner it is
  identically zero, which removes that corner's singularity outright. On a
  curved edge it is (kappa0/2) s^2 + O(s^3) -- vanishing to second order rather
  than identically, so the trick degrades gracefully rather than failing, and
  the integrand r.N (du/dn)^2 ~ s^(2nu) is still bounded. With x0 anywhere else
  r.N is an analytic series in s with nonzero constant and linear terms.
- **r is not arclength.** r = |x(s) - corner| = s - (kappa0^2/24) s^3 + ..., so
  r is s times a series in s^2.

Both add INTEGER powers of arclength to the exponent family, taking it from
{k nu + 2q} to {k nu + m}. The substitution t = r^nu does not rationalize that
(it leaves t^(1/nu) with 1/nu in (1,2), only C^1); t = r^(1/q) with nu = p/q
does, and covers the straight family too. Measured: on the curved family
sub = nu never reaches 1e-13 at any order up to 64, while sub = 1/q is exact by
order 6-14.

A varying-speed curve costs nodes, and that cost is a property of the
parametrization rather than of the eigenfunction. Normalized arclength is analytic
but its strip of analyticity narrows with eccentricity, and Gauss-Legendre's rate
narrows with it. Measured on `int (r.N) ds = 2|Omega|` -- a closed form with no
eigenfunction, no lam and no basis in it:

    order          16      32      64     128     192     256
    disk         3e-16   3e-16   3e-16   3e-16   3e-16   3e-16
    ellipse a=2  5e-03   2e-04   4e-07   3e-12   5e-15   2e-15
    ellipse a=4  4e-02   1e-02   2e-03   8e-05   3e-06   1e-07
    ellipse a=6  6e-02   3e-02   1e-02   2e-03   5e-04   1e-04

`boundary_quadrature(resolve_geometry=True)` sizes for this as well as for the
oscillation, which costs an a=2 ellipse 46 -> 168 nodes and takes its eigenfunction
integrals from 1.0e-06 to 3.9e-16. A disk, a stadium, a mushroom and every polygon
pay nothing: their arclength maps are exact, so the geometry order never exceeds
the oscillation order. Past about a=4 no usable order suffices, and the shortfall
is reported (`BoundaryQuad.shortfalls`, and `precision` stops claiming otherwise)
rather than chased.

An earlier version of this note blamed the adaptive arclength table's tolerance.
That is measured to be wrong: the error is identical to three digits at `tol` =
1e-4, 1e-6 and 1e-8, while the build cost goes 0.0s, 0.9s, 65s. Refining the table
is pure expense. What moves the error is quadrature order.
`_parametrization_quality` still reports the round-trip error, and spline
boundaries still need panels broken at knots.

Boundary conditions
-------------------
Dirichlet corners are implemented and validated. Pure-Neumann singular corners
are detected and fall back to a smooth rule: their 'uv' kernel has exponent
gamma=0 while 'TT' has gamma=2nu-2, and one shared node set cannot be matched to
both (the mismatch leaves a residual t^(2/nu-2), polynomial only when 2/nu is an
integer). Mixed Dirichlet/Neumann (Zaremba) *reentrant* corners are out of reach
on principle, not for want of a quadrature rule: the exponents there are
(k+1/2)nu, so (du/dn)^2 ~ r^(nu-2), which is NOT integrable for nu < 1 -- the
Rellich boundary integral diverges for any x0 off the adjacent edge lines.
"""
from collections import namedtuple
import warnings

import numpy as np
import scipy.linalg as la

from .quad import (cached_leggauss, cached_cornerjacgauss, cached_cornerinterpgauss,
                   cornerjac_order_cap, corner_rule_spec, corner_substitution,
                   corner_order_for_precision, smooth_order_for_precision,
                   smooth_power_error, _CORNER_NU_MIN, _CORNER_MAX_Q)
from .geometry import PointSet
from .utils import complex_dot

# A corner's geometry and its eligibility for the corner-adapted rule. `nu = pi/alpha` is taken
# from the geometry and must never be rounded or padded: a 3e-4 relative error in nu costs four
# digits (docs/corner_quadrature.tex Sec. 4). `seg_out` leaves the corner (distance from it is
# seg.len*tau), `seg_in` arrives (distance is seg.len*(1-tau)).
CornerSpec = namedtuple('CornerSpec',
                        ['idx', 'point', 'alpha', 'nu', 'kind', 'sub', 'rational', 'straight',
                         'seg_out', 'seg_in', 'singular', 'admissible', 'reason'])

# One quadrature panel on one segment. `tau0` is ALWAYS the corner-anchored end, so tau1 < tau0
# on a panel anchored at the segment's far endpoint; the signed h = tau1-tau0 carries the
# orientation and only |h| scales the weights. `corner` is the CornerSpec index, or -1.
CornerPanel = namedtuple('CornerPanel',
                         ['seg_idx', 'tau0', 'tau1', 'rule', 'order', 'nu', 'gamma', 'sub',
                          'curved', 'corner'])
# `rule` is 'legendre' (smooth stretch), 'cornerjac' (substitution; exact only for a straight
# edge with 2/nu integral) or 'cornerinterp' (interpolatory on the true exponent set; the
# general case, and the only one usable at large q or irrational nu -- see
# quad.corner_rule_spec).

# Geometry-only boundary quadrature: no domain reference, no lam dependence, no basis. Built
# once, reused across every lam. `panel_id` gives each node's index into `panels`, which is what
# makes per-corner diagnostics (and the edge/arc split that corner tests must report) possible.
BoundaryQuad = namedtuple('BoundaryQuad',
                          ['pts', 'normals', 'tangents', 'wts', 'dir_mask', 'neu_mask',
                           'panels', 'panel_id', 'sizing_precision', 'x0', 'shortfalls'],
                          defaults=((),))
# `sizing_precision` is the MODEL bound that chose the orders -- a sizing heuristic, not a
# certificate. It was called `precision` and read as an accuracy claim, which it is not:
# `chevron_1_2` reports 1e-13 here while `verify_gram` measures 4.9e-08 on the actual
# integrand, and that gap is identical under every corner model tried, because it comes from
# the smooth panels, whose requirement is set by the basis's own pole placement -- something no
# geometry-only model can see. Use `verify_gram` to certify a node set; use this to size one.
#
# `shortfalls` names every corner whose rule cannot honour the requested precision, so that
# `sizing_precision` is never a claim contradicted elsewhere in the same object. It is a tuple
# of (corner_idx, achieved_or_None, reason). A corner that was DEMOTED to a smooth rule (an
# inadmissible near-slit nu) has no bound at all -- `sector_slit` measured 6.9e-01 against a
# reported 1e-13 -- so it sets `sizing_precision` to inf rather than to a number that is not
# one. inf therefore means "the model declines to size this", the one case where the heuristic
# is also a genuine warning.

# A small cluster of eigenfunctions' Cauchy data at a BoundaryQuad's nodes. U/U_N/U_T are
# (n_nodes, mult) -- already-evaluated function values, never a basis matrix.
EigfunData = namedtuple('EigfunData', ['pts', 'normals', 'tangents', 'wts', 'U', 'U_N', 'U_T'])

# Margin above the nu=1/2 admissibility threshold. At nu=0.51 the Jacobi exponent is already
# -0.96 and the innermost node collides with the coordinate-collapse floor by order 32, so
# admitting corners arbitrarily close to a slit buys nothing.
_NU_MARGIN = 0.02
_ANGLE_TOL = 1e-10

# Which corners need the corner-adapted rule.
#
# The original criterion was `nu < 1` -- reentrant corners only -- on the reasoning that a
# convex corner's eigenfunction is bounded and "a smooth rule is already exact". It is not.
# The Rellich integrand is (du/dn)^2 ~ r^(2nu-2), so what matters is whether that exponent is
# an EVEN INTEGER, i.e. whether nu is an integer -- not whether it is bigger than one. At a
# 135-degree corner (nu = 4/3) the integrand carries r^(2/3): bounded, but with an infinite
# derivative at the corner, and Gauss-Legendre converges on it algebraically.
#
# Measured (per-panel attribution, benchmarks/eigfun_quad/sizing_audit.py):
#
#     right_trapezoid  nu=4/3   84 nodes   5.6e-09 -> 8.7e-19   at 212 nodes
#     GWW1             nu=4/3  204 nodes   6.9e-07 -> 1.1e-14   at 440 nodes
#
# and in both cases the offending panels were `legendre` ones whose sizing model claimed
# ~4e-15 while delivering 3e-09 -- the model assumes an analytic integrand, so it cannot see
# a fractional power at all. Every corner-adapted panel already met its model.
#
# True by tolerance rather than exactly: a nu that misses an integer by 1e-12 is an integer
# for this purpose, and the rules degrade gracefully anyway.
_NONINTEGRAL_TOL = 1e-9

# Default cap on a corner panel's arclength, as a multiple of the corner's clearance (the radius
# of the largest disk about it inside the domain). Calibrated in corner_panels' docstring: 1.0
# sits exactly at the edge of the expansion's validity and lands an order worse, while going
# below 0.9 buys nothing. Echoes cubature._corner_R0s's 0.4-of-an-edge cap for the interior rule.
_CLEARANCE_FRAC = 0.9


def _is_straight(seg):
    """A LineSegment, on which arclength from the corner equals r exactly and r.N is constant.

    Curved segments are NOT excluded -- see the module docstring on the curved case -- but the
    distinction is still worth recording, because on a straight edge the exponent family is
    the sparser {k nu + 2q} and r.N is exactly constant, while on a curved edge both pick up
    integer powers of arclength."""
    return type(seg).__name__ == 'LineSegment'


def _parametrization_quality(seg, n=257):
    """How exactly does this segment's `p(tau)` sit at arclength `tau*seg.len`? Returns the max
    relative round-trip error `|s(t(s)) - s| / seg.len`.

    This is the property the quadrature actually depends on. `assemble_panels` uses
    `seg.p/N/T(tau)` and `seg.len` and never calls `seg.dp`, so what matters is that the node
    really is at the arclength its weight assumes -- not that `|dp/dtau|` equals `seg.len`.

    An earlier version of this function measured `|dp/dtau| - seg.len` and reported a floor of
    ~1e-3 on varying-speed curves. That was the error of `ParametricSegment._dp_of_s`, which
    differentiates the PCHIP inverse interpolant -- a quantity nothing here uses, and three to
    five orders larger than the round-trip error at the same `tol`.

    LineSegments satisfy this exactly. A circular arc is machine-exact at any `tol` because its
    arclength map is linear. A curve of varying speed is limited by the PCHIP inverse: ~5e-6 at
    tol=1e-4, ~2e-8 at tol=1e-7. Note that the resulting boundary integrals are worse than that
    round-trip figure suggests (~1e-5 to 1e-6, and NOT improving with quadrature order),
    because a piecewise-cubic inverse makes `f(p(tau))` only C^1 and Gauss-Legendre needs
    analyticity. See docs/eigfun_integrals.md for the fix that removes both."""
    if type(seg).__name__ == 'LineSegment':
        return 0.0
    s_of_t, t_of_s = getattr(seg, '_s_of_t', None), getattr(seg, '_t_of_s', None)
    if s_of_t is None or t_of_s is None:
        seg.len                                     # force the lazy reparametrization
        s_of_t, t_of_s = getattr(seg, '_s_of_t', None), getattr(seg, '_t_of_s', None)
        if s_of_t is None or t_of_s is None:
            return 0.0
    s = np.linspace(0.0, seg.len, n)
    return float(np.abs(s_of_t(t_of_s(s)) - s).max()/seg.len)


def corner_clearance(domain, corner_pt, seg_out, seg_in):
    """Distance from a corner to the nearest NON-adjacent piece of the boundary.

    A proxy for the radius of the largest disk about the corner inside Omega, which bounds
    where the local corner expansion is valid -- and therefore how long a corner-anchored
    panel may usefully be. Reuses `bdry.polyline()`'s sub-segment decomposition and its
    `owner` array to drop the two adjacent segments, the same pattern
    `geometry.corner_branch_cut_rays` uses."""
    b0, b1, owner = domain.bdry.polyline()
    keep = (owner != seg_out) & (owner != seg_in)
    if not np.any(keep):
        return np.inf
    d = b1[keep] - b0[keep]
    t = np.clip(((corner_pt - b0[keep])*d.conjugate()).real/np.abs(d)**2, 0.0, 1.0)
    return np.abs(corner_pt - (b0[keep] + t*d)).min()


def corner_specs(domain, singular_test=None):
    """CornerSpec per genuine corner, corner-indexed like `domain.corners`.

    Uses `domain.corner_int_angles` (corner-indexed) rather than `int_angles` (segment-indexed)
    -- see geometry.Domain.corner_int_angles on why that distinction is load-bearing.

    `singular_test(nu, seg_out, seg_in) -> bool` decides which corners get the corner-adapted
    rule. `None` keeps the historical criterion, `nu < 1` (reentrant only);
    `boundary_quadrature(nonintegral=True)` supplies one that asks whether a smooth rule can
    actually integrate `tau^(2nu-2)` to the requested precision (`quad.smooth_power_error`).
    """
    segs = domain.bdry.segments
    n_segs = len(segs)
    corners = np.asarray(domain.corners)
    cidx = np.asarray(domain.corner_idx)
    alphas = np.asarray(domain.corner_int_angles)

    specs = []
    for c in range(len(corners)):
        seg_out = int(cidx[c])
        seg_in = (seg_out - 1) % n_segs
        alpha = float(alphas[c])
        nu = np.pi/alpha
        if singular_test is None:
            singular = nu < 1.0 - _ANGLE_TOL
        else:
            singular = bool(singular_test(nu, seg_out, seg_in))

        reason = ''
        if not singular:
            reason = ('not singular: a smooth rule integrates this corner\'s r^(2nu-2) to the '
                      'requested precision' if singular_test is not None else
                      'not singular (nu >= 1) under the reentrant-only criterion; pass '
                      'nonintegral=True to test whether a smooth rule really suffices')
        elif nu <= _CORNER_NU_MIN + _NU_MARGIN:
            reason = (f'nu={nu:.4f} too close to the slit limit 1/2: the Rellich integrand '
                      'is barely integrable and no rule recovers it; place x0 on this corner')
        elif segs[seg_out].bc_type != segs[seg_in].bc_type:
            reason = (f'mixed boundary conditions ({segs[seg_in].bc_type}/'
                      f'{segs[seg_out].bc_type}): the Zaremba corner integrand ~ r^(nu-2) '
                      'is not integrable for nu<1, so the identity itself fails here')
        elif segs[seg_out].bc_type != 'dir':
            reason = (f"bc_type={segs[seg_out].bc_type!r}: only Dirichlet corners are wired; "
                      "a Neumann corner's 'uv' and 'TT' kernels need different exponents")

        straight = _is_straight(segs[seg_out]) and _is_straight(segs[seg_in])
        kind, sub = corner_rule_spec(nu, curved=not straight)
        _, rational = corner_substitution(nu)
        specs.append(CornerSpec(c, complex(corners[c]), alpha, nu, kind, sub, rational,
                                straight, seg_out, seg_in, singular, reason == '', reason))
    return specs


def singular_corner_report(domain):
    """Human-readable summary of which corners get the corner-adapted rule and why not, for
    the rest. Cheap to call and the first thing to look at when accuracy disappoints."""
    lines = []
    for s in corner_specs(domain):
        tag = s.kind if s.admissible else ('SMOOTH' if not s.singular else 'FALLBACK')
        lines.append(f"  corner {s.idx:2d} at {s.point:+.4g} alpha={s.alpha/np.pi:.4f}pi "
                     f"nu={s.nu:.5f} -> {tag}" + (f"  [{s.reason}]" if s.reason else ""))
    return "\n".join(lines)


def corner_panels(domain, specs=None, order_corner=16, order_smooth=16, gamma=None,
                  panel_frac=1.0, clearance_frac=_CLEARANCE_FRAC, order_cap=True):
    """Panels tiling every segment's [0,1] exactly once, corner-anchored where admissible.

    `panel_frac` caps a corner panel's share of its segment. `clearance_frac` also caps the
    panel's *arclength* at that multiple of `corner_clearance`, and it is ON by default because
    leaving it off is catastrophic on a domain whose edges are long relative to the largest disk
    about the corner. Measured against the exact eigenfunction of a 1xN polyomino strip with one
    cell below an end, so the reentrant corner's edge has length N-1 against a clearance of 1
    (worst relative norm error over modes (1,1) and (2,3)):

        N     off      cf=1.0   cf=0.9   cf=0.7
         2  6.5e-14   6.5e-14  5.3e-15  2.2e-16
         4  4.7e-06   3.9e-14  3.6e-15  2.2e-16
         8  2.3e-02   2.2e-14  1.6e-15  0.0e+00
        16  7.7e-02   1.2e-14  1.3e-15  6.7e-16
        24  7.4e-03   9.3e-15  4.4e-16  2.2e-16

    Leaving it off costs up to twelve orders. A fixed `panel_frac` cannot substitute: it is a
    fraction of the EDGE, so it grows with the edge and stays too long -- panel_frac=0.25 with
    no clearance cap still gives 2.6e-04 at N=16. cf=1.0 sits exactly at the edge of the
    expansion's validity and lands an order worse than 0.9; below 0.9 buys nothing. Hence the
    0.9 default, which costs ~6% more nodes than 1.0 and 20-50% more than off.

    The mechanism measured here is resolution: the corner rule clusters its nodes AT the corner,
    so the far end of a long panel is sparsely sampled and cannot resolve the sqrt(lam)
    oscillation over the remaining arclength. The related concern that the integrand leaves the
    corner's exponent class beyond the clearance is not separately measured -- see
    benchmarks/corner_quad/panel_length.py on why it resists synthetic measurement.

    `order_cap` clamps each corner panel's order so its innermost node stays off the corner
    (quad.cornerjac_order_cap). Leave it on: past the cap the error *diverges* with order."""
    segs = domain.bdry.segments
    if specs is None:
        specs = corner_specs(domain)

    # per segment, which endpoint (if either) carries an admissible singular corner
    at_start = {}
    at_end = {}
    for s in specs:
        if s.admissible:
            at_start[s.seg_out] = s
            at_end[s.seg_in] = s

    panels = []
    for i, seg in enumerate(segs):
        s0, s1 = at_start.get(i), at_end.get(i)

        def corner_panel(spec, anchor_at_start, frac):
            """One corner-anchored panel; `frac` is its share of this segment."""
            if clearance_frac is not None:
                clear = corner_clearance(domain, spec.point, spec.seg_out, spec.seg_in)
                frac = min(frac, clearance_frac*clear/seg.len)
            frac = float(np.clip(frac, 1e-6, 1.0))
            order = order_corner
            if order_cap:
                order = min(order, cornerjac_order_cap(spec.nu, gamma, spec.sub, scale=frac))
            order = max(order, 2)
            t0, t1 = (0.0, frac) if anchor_at_start else (1.0, 1.0 - frac)
            return CornerPanel(i, t0, t1, spec.kind, order, spec.nu, gamma, spec.sub,
                               not spec.straight, spec.idx), frac

        if s0 is None and s1 is None:
            panels.append(CornerPanel(i, 0.0, 1.0, 'legendre', order_smooth,
                                      np.nan, None, np.nan, False, -1))
        elif s0 is not None and s1 is None:
            p, frac = corner_panel(s0, True, panel_frac)
            panels.append(p)
            if frac < 1.0:
                panels.append(CornerPanel(i, frac, 1.0, 'legendre', order_smooth,
                                          np.nan, None, np.nan, False, -1))
        elif s1 is not None and s0 is None:
            p, frac = corner_panel(s1, False, panel_frac)
            panels.append(p)
            if frac < 1.0:
                panels.append(CornerPanel(i, 0.0, 1.0 - frac, 'legendre', order_smooth,
                                          np.nan, None, np.nan, False, -1))
        else:
            # Both endpoints singular. The substitution anchors at one end only, so the edge
            # MUST split -- this is the case a single-corner domain never exercises.
            p0, f0 = corner_panel(s0, True, min(panel_frac, 0.5))
            p1, f1 = corner_panel(s1, False, min(panel_frac, 0.5))
            panels.append(p0)
            panels.append(p1)
            if f0 + f1 < 1.0 - 1e-12:
                panels.append(CornerPanel(i, f0, 1.0 - f1, 'legendre', order_smooth,
                                          np.nan, None, np.nan, False, -1))
    return _split_at_breaks(domain, panels, order_smooth)


def _split_at_breaks(domain, panels, order_smooth):
    """Subdivide panels at each segment's `break_taus`.

    A segment reports parameter values across which its integrand is not analytic -- spline
    knots being the motivating case, since a degree-k B-spline is only C^(k-1) there. A Gauss
    panel spanning one converges algebraically no matter how exact the parametrization is
    (measured: 8.3e-11 at 96 nodes with knot-aligned panels against 1.3e-06 at 128 without).

    A CORNER panel is subdivided only in its outer part: its anchored end carries the
    singularity the corner rule exists for and must not be cut off, so the corner sub-panel
    keeps the anchor and runs to the first break, and the remainder becomes smooth panels.
    Panels are emitted in the same (seg_idx, anchored-end-first) convention as the input.
    """
    out = []
    for p in panels:
        breaks = np.asarray(domain.bdry.segments[p.seg_idx].break_taus, dtype=float)
        lo, hi = min(p.tau0, p.tau1), max(p.tau0, p.tau1)
        inner = np.sort(breaks[(breaks > lo + 1e-12) & (breaks < hi - 1e-12)])
        if not len(inner):
            out.append(p)
            continue
        if p.rule == 'legendre':
            edges = np.concatenate([[lo], inner, [hi]])
            for a, b in zip(edges[:-1], edges[1:]):
                out.append(p._replace(tau0=a, tau1=b))
        else:
            # keep the anchor; cut only beyond the first break on the anchored side
            if p.tau0 < p.tau1:                       # anchored at lo
                edges = np.concatenate([[lo], inner, [hi]])
                out.append(p._replace(tau0=edges[0], tau1=edges[1]))
                rest = edges[1:]
            else:                                     # anchored at hi
                edges = np.concatenate([[lo], inner, [hi]])
                out.append(p._replace(tau0=edges[-1], tau1=edges[-2]))
                rest = edges[:-1]
            for a, b in zip(rest[:-1], rest[1:]):
                out.append(CornerPanel(p.seg_idx, a, b, 'legendre', order_smooth,
                                       np.nan, None, np.nan, False, -1))
    return out


def _panel_rule(panel):
    if panel.rule == 'legendre':
        return cached_leggauss(panel.order)
    if panel.rule == 'cornerjac':
        return cached_cornerjacgauss(panel.order, panel.nu, panel.gamma, panel.sub)
    if panel.rule == 'cornerinterp':
        return cached_cornerinterpgauss(panel.order, panel.nu, panel.gamma,
                                        None, panel.curved)
    raise ValueError(f"unknown panel rule {panel.rule!r}")


def assemble_panels(domain, panels, sizing_precision=None, x0=None, shortfalls=()):
    """BoundaryQuad from a panel plan, via each segment's own p/N/T at normalized arclength.

    Segments are parametrized by normalized arclength, so |dp/dtau| == seg.len identically and
    the weight is seg.len*|h|*w with no per-node Jacobian."""
    segs = domain.bdry.segments
    P, N, T, W, D, U, PID = [], [], [], [], [], [], []
    for pid, panel in enumerate(panels):
        seg = segs[panel.seg_idx]
        u, w = _panel_rule(panel)
        h = panel.tau1 - panel.tau0        # signed: carries the anchor orientation
        tau = panel.tau0 + h*u
        P.append(seg.p(tau))
        N.append(seg.N(tau))
        T.append(seg.T(tau))
        W.append(seg.len*abs(h)*w)
        D.append(np.full(len(u), float(seg.bc_type == 'dir')))
        U.append(np.full(len(u), float(seg.bc_type == 'neu')))
        PID.append(np.full(len(u), pid))
    if x0 is None:
        x0 = default_x0(domain)
    return BoundaryQuad(np.concatenate(P), np.concatenate(N), np.concatenate(T),
                        np.concatenate(W), np.concatenate(D), np.concatenate(U),
                        tuple(panels), np.concatenate(PID), sizing_precision, x0,
                        tuple(shortfalls))


def eigfun_cauchy_data(basis, lam, coef, bq):
    """Cauchy data of the eigenfunction cluster `basis(lam) @ coef` at `bq`'s nodes.

    `coef` is (n_basis, mult). The basis is evaluated once and contracted with `coef`
    immediately, so every downstream integral is formed from already-evaluated function
    values -- never by sandwiching a basis-level matrix between raw GSVD coefficient vectors,
    which multiplies the matrix's independently-rounded error through `coef` on both sides."""
    pts = PointSet(bq.pts) if not isinstance(bq.pts, PointSet) else bq.pts
    nrm = PointSet(bq.normals) if not isinstance(bq.normals, PointSet) else bq.normals
    tng = PointSet(bq.tangents) if not isinstance(bq.tangents, PointSet) else bq.tangents
    coef = np.atleast_2d(coef.T).T if coef.ndim == 1 else coef
    U = basis(lam, pts)@coef
    U_N = basis.ddiff(lam, pts, nrm)@coef
    U_T = basis.ddiff(lam, pts, tng)@coef
    return EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, U, U_N, U_T)


def weighted_integral(ed, kernel, weight):
    """A[i,j] = integral weight(s) * K(u_i, u_j) ds over the boundary node set.

    `kernel` is one of the four bilinear Cauchy-data kernels: 'uv' = u*v,
    'NN' = dNu*dNv, 'TT' = dTu*dTv, 'cr' = dTu*dNv + dNu*dTv. `weight` is a real array already
    evaluated at `ed.pts` (r.N for Rellich; a boundary velocity V(s) for a Hadamard-type shape
    derivative). Every formula in this module is a fixed linear combination of calls to this
    routine, and a consumer building its own boundary functional does the same."""
    w = (ed.wts*weight)[:, None]
    if kernel == 'uv':
        return (ed.U*w).T@ed.U
    if kernel == 'NN':
        return (ed.U_N*w).T@ed.U_N
    if kernel == 'TT':
        return (ed.U_T*w).T@ed.U_T
    if kernel == 'cr':
        return (ed.U_T*w).T@ed.U_N + (ed.U_N*w).T@ed.U_T
    raise ValueError(f"'kernel' must be one of 'uv', 'NN', 'TT', 'cr' (got {kernel!r})")


def default_x0(domain, singular_test=None):
    """Reference point x0 for the Rellich weight r.N.

    Placed at a singular corner when there is one, because r.N vanishes identically on both
    edges meeting it (the edges are parallel to x - x0 there), which removes that corner's
    singularity from the integrand outright rather than resolving it. With the corner-adapted
    rule this is no longer *necessary* -- that is the whole point of the rule, and multi-corner
    domains cannot zero every corner anyway -- but it is free, and it makes the one corner it
    covers exact independent of quadrature. Falls back to the bounding-box centre."""
    specs = [s for s in corner_specs(domain, singular_test) if s.singular]
    if specs:
        return min(specs, key=lambda s: s.nu).point
    tau = np.linspace(0, 1, 50)[:-1]
    pts = np.concatenate([seg.p(tau) for seg in domain.bdry.segments])
    return 0.5*(pts.real.min() + pts.real.max()) + 0.5j*(pts.imag.min() + pts.imag.max())


def gram(ed, lam, bq, x0=None):
    """L^2(Omega) Gram matrix G[i,j] = <u_i, u_j> for the eigenfunction cluster in `ed`.

    Zaremba-specialized Rellich identity (docs/rellich.md Sec. 2):
    G = (1/2lam)(I_1 - I_2) + (1/2)I_3, with the Dirichlet and Neumann parts restricted to
    their own segments by zeroing the per-node weight rather than by separate node sets.

    `x0` defaults to the node set's own `bq.x0`. The identity holds for EVERY x0, so passing a
    different one is legitimate and is exactly how the x0-invariance diagnostic works: any
    variation in the result across x0 is pure quadrature error."""
    if x0 is None:
        x0 = bq.x0
    rN = complex_dot(ed.pts - x0, ed.normals)
    n = ed.U.shape[1]
    G = np.zeros((n, n))
    if bq.dir_mask.any():
        G += weighted_integral(ed, 'NN', rN*bq.dir_mask)/(2*lam)
    if bq.neu_mask.any():
        G += weighted_integral(ed, 'uv', rN*bq.neu_mask)/2
        G -= weighted_integral(ed, 'TT', rN*bq.neu_mask)/(2*lam)
    return G


def lowdin_transform(G, ttol=1e-3):
    """Loewdin (symmetric) orthogonalization transform D for a small Gram matrix G:
    diagonalize G = Q diag(w) Q.T and return D = Q diag(w^-1/2) Q.T, so that for an
    (n, mult) array `vals` of already-evaluated function values, `vals @ D.T` is orthonormal
    in the inner product G represents.

    Returns None (after warning) if G is deficient -- w.min()/w.max() < ttol -- rather than
    letting w**-0.5 blow up on a near-zero or roundoff-negative eigenvalue and contaminate
    every column of D through the Q mixing. Callers should fall back to the raw values."""
    w, Q = la.eigh(G)
    if w.min()/w.max() < ttol:
        warnings.warn(f"Rellich Gram matrix is deficient (w.min()/w.max()="
                      f"{w.min()/w.max():.3e}<{ttol:.3e}); cluster may have wrong "
                      "multiplicity. Falling back to un-orthonormalized values.")
        return None
    return (Q*w**-0.5)@Q.T


def boundary_quadrature(domain, lam_max, precision=1e-13, x0=None, panel_frac=1.0,
                        clearance_frac=_CLEARANCE_FRAC, warn=True, nonintegral=True,
                        smooth_safety=1.0, resolve_geometry=True, weight_family='even'):
    """The entry point: a boundary node set for `domain`, SIZED for `precision` on
    eigenfunctions up to spectral parameter `lam_max`.

    No basis, and none of the tuning knobs the graded rule required -- the node set is a pure
    function of geometry, `lam_max` and the requested accuracy, so it can be built once and
    reused for every lam in a search. Every panel's rule is scored against a model integrand
    whose integral is closed-form (`quad.corner_rule_residual` on the corner's own exponent
    set, `quad.smooth_order_for_precision` on exp(i k tau)), and the orders follow from those
    scores.

    **`precision` sizes the rule; it does not certify it.** The models are of the integrand's
    CLASS, not the integrand in hand, and the gap can be large: `chevron_1_2` is sized at 1e-13
    and `verify_gram` measures 4.9e-08 on the real integrand. That gap was identical under every
    corner model tried, which is what identifies its source -- it is not the corners but the
    smooth panels, whose true requirement is set by the basis's own pole placement, invisible to
    anything that sees only geometry. To find out what a node set ACHIEVED, call `verify_gram`,
    which refines the rule and measures the change in the Gram it actually produces.

    Where a corner cannot reach `precision` -- a near-slit nu, or the coordinate-collapse
    order cap binding first -- the best achievable order is used and a warning names the
    corner and what its model bound came to, rather than silently missing the target. That
    value is recorded in the returned `BoundaryQuad.sizing_precision`.

    The default is 1e-13, not 1e-14: these integrals sit near the float64 roundoff floor, and a
    270-degree corner typically lands at ~1.9e-14, so a 1e-14 default would warn on the
    commonest domain in the suite while delivering essentially the same answer. Asking for 1e-14
    explicitly is legitimate and will warn if it falls short, which is the point.

    `nonintegral` and `resolve_geometry` are ON by default; both were validated opt-in first
    (commits bae147d, f4c57d5) and pass `False` to recover the earlier behaviour exactly.

    `nonintegral=True` gives the corner-adapted rule to every corner a smooth rule cannot
    actually integrate -- `quad.smooth_power_error` asks the question directly, rather than
    assuming `nu >= 1` is safe. **The reentrant-only default does not honour `precision` on a
    domain with a convex non-integer corner**, measured 5.6e-09 against a claimed 1e-13 on
    `right_trapezoid`, because the smooth model integrand is analytic and cannot see the
    fractional power `r^(2nu-2)` that such a corner puts on its edges. Opt in until the
    default flips.

    `smooth_safety` multiplies the wavenumber the smooth panels are sized for. It exists
    because the model integrand `exp(i k tau)`, with `k = 2 sqrt(lam_max) * arclength`, is not
    a reliable proxy for the real one's bandwidth in either direction: measured against the
    Cauchy data's own Fourier spectrum, a circle needs 4 harmonics where the model assumes 64,
    while an ellipse needs 80 where the model assumes 8. Verified error at `lam_2`
    (`verify_gram`, so the actual integrand, not a model):

        safety             1x        2x        3x        4x        6x
        ellipse_a2    1.0e-06   1.6e-09   4.6e-12   2.4e-14   2.1e-15
        ellipse_a4    9.2e-06   7.4e-07   8.0e-08   9.7e-09   1.9e-10
        stadium       1.5e-04   1.2e-06   1.0e-08   3.4e-08   5.4e-09
        disk          3.2e-15   4.5e-15   7.3e-15   4.1e-15   3.5e-15
        square        4.1e-16   2.1e-15   4.1e-15   4.3e-15   4.1e-16

    Polygons are already converged and pay only nodes. Curved boundaries improve sharply but
    `ellipse_a4` and `stadium` plateau above 1e-13, so a safety factor is a mitigation there
    and not yet a fix -- another reason `precision` is a sizing knob and `verify_gram` is the
    certificate.
    """
    segs = domain.bdry.segments
    k = 2.0*np.sqrt(max(lam_max, 0.0))     # a PRODUCT of eigenfunctions oscillates at 2*sqrt(lam)

    # smooth stretches: Nyquist in the segment's own arclength. Computed FIRST because the
    # singular-corner test below asks what those orders can actually deliver on tau^(2nu-2).
    smooth_orders, smooth_ach = {}, {}
    geom_short, order_max_geom = {}, 512
    x0_probe = x0 if x0 is not None else default_x0(domain)
    scale = 2.0*abs(domain.area)
    for i, seg in enumerate(segs):
        o, ach = smooth_order_for_precision(k*seg.len*smooth_safety, precision)
        if resolve_geometry:
            # The oscillation model assumes the parametrization is free. On a varying-speed
            # curve it is not, and the geometry alone can cost more nodes than the
            # eigenfunction does (see geometry_order_for_precision).
            go, gach = geometry_order_for_precision(seg, x0_probe, scale, precision,
                                                    order_max=order_max_geom)
            if go > o:
                o = go
            if gach > precision:
                geom_short[i] = gach
            ach = max(ach, min(gach, 1.0) if np.isfinite(gach) else 1.0)
        smooth_orders[i], smooth_ach[i] = o, ach

    singular_test = None
    if nonintegral:
        def singular_test(nu, seg_out, seg_in):
            if nu < 1.0 - _ANGLE_TOL:
                return True
            order = max(smooth_orders[seg_out], smooth_orders[seg_in])
            return smooth_power_error(2.0*nu - 2.0, order) > precision
    specs = corner_specs(domain, singular_test)
    if x0 is None:
        x0 = default_x0(domain, singular_test)

    # per-corner order, from the corner's own exponent set
    orders, achieved, subs, kinds = {}, {}, {}, {}
    for sp in specs:
        if not (sp.singular and sp.admissible):
            continue
        curved = not sp.straight
        frac = min(panel_frac, 0.5) if _both_ends_singular(specs, sp) else panel_frac
        seg_len = segs[sp.seg_out].len
        k_corner = np.sqrt(max(lam_max, 0.0))*seg_len*frac
        sub, kind = sp.sub, sp.kind
        o, ach = corner_order_for_precision(sp.kind, sp.nu, None, sub, curved,
                                            precision, scale=frac, k=k_corner)
        if weight_family == 'integer':
            # The corner rules are built for the exponent family the EIGENFUNCTION has,
            # `{gamma + j nu + ...}`, which `sub = nu` rationalizes. An integer-power
            # boundary weight -- what a shape derivative supplies, `V.n ~ r` for a
            # perturbation that moves the corner -- is outside it, and `sub = nu` sends
            # `r^m` to `t^(m/nu)`: non-integer with a SMALL exponent, so Gauss decays on it
            # only as `n^(-(2m/nu + 2))`. That is the whole defect. Measured against 40-digit
            # truth on the 1.5pi sector, weight `r^p`:
            #
            #     sub = nu,  order 32     p=0 2.9e-14  p=1 4.6e-07  p=2 8.5e-14  p=3 4.0e-14
            #     sub = 1/2, order 16     p=0 4.7e-15  p=1 1.2e-14  p=2 1.1e-14  p=3 1.0e-14
            #
            # `sub = 1/2` reverses the trade: every integer `m` becomes the exact polynomial
            # `t^(2m)`, while the Bessel family goes to `t^(2 j nu)` -- still non-integer, but
            # with exponents growing by `2 nu` per term. It needs NO rationality of nu, so it
            # covers the generic arc-arc corner where no exact substitution exists at all;
            # verified at nu = 1/1.37 and nu = 1/phi to 1e-15, and across nu in [0.57, 1.34] at
            # order 16. `tau_min` stays above 1e-6 throughout, far clear of the
            # coordinate-collapse floor that rules out `sub = 1/q` at q >= 4.
            #
            # **This is why it stays OPT-IN, and must not become the default.** "Gauss resolves
            # the shifted Bessel family at once" is true only for a SPARSE member -- the single
            # (m,n) sector mode, whose squared normal derivative has exponents spaced by 4. A
            # real multi-term corner series has cross terms spaced by `2 nu`, and on those
            # `sub = 1/2` converges only ALGEBRAICALLY. L_shape corner panel, nu = 2/3, the
            # Leg 3 synthetic series against closed-form truth:
            #
            #     order            8         16         32         64        128
            #     sub = nu    7.8e-16    2.7e-15    1.5e-14    6.1e-14    9.2e-14
            #     sub = 1/2   4.1e-06    2.8e-07    1.8e-08    1.2e-09    7.2e-11
            #
            # about `n^-4.7`, never reaching machine precision at a usable order, while
            # `sub = nu` is exact at order 8. Making this the default fails 12 tests including
            # all of Leg 3 (3.6e-06 against a 1e-12 bar, at the SAME node count). It is the
            # original defect with the roles swapped, and the two families are genuinely a
            # trade: `sub = nu` for the eigenfunction's own dense family (Rellich, Gram),
            # `sub = 1/2` for an integer-power weight on top of a sparse one (Hadamard
            # corner-moving). Leg 1 alone cannot see this -- its eigenfunction IS the sparse
            # case, where 'integer' ties or wins on half the nodes, which is exactly how the
            # claim came to be overstated.
            # Forced to 'cornerjac' even on a curved edge, where `corner_rule_spec` would
            # pick 'cornerinterp': that rule takes no substitution, so it cannot make the
            # weight exact, and its fixed-node interpolation is the weaker rule here anyway.
            o, ach = corner_order_for_precision('cornerjac', sp.nu, None, 0.5, curved,
                                                precision, scale=frac, k=k_corner)
            sub, kind = 0.5, 'cornerjac'
        orders[sp.idx], achieved[sp.idx] = o, ach
        subs[sp.idx], kinds[sp.idx] = sub, kind

    panels = []
    for p in corner_panels(domain, specs, order_corner=1, order_smooth=1,
                           panel_frac=panel_frac, clearance_frac=clearance_frac,
                           order_cap=False):
        if p.rule == 'legendre':
            panels.append(p._replace(order=smooth_orders[p.seg_idx]))
        else:
            panels.append(p._replace(order=orders.get(p.corner, 16),
                                     sub=subs.get(p.corner, p.sub),
                                     rule=kinds.get(p.corner, p.rule)))
    # Honest sizing. A corner short of the target contributes its model bound; a SINGULAR
    # corner demoted to a smooth rule contributes no bound at all, because the smooth model
    # cannot see the singularity it is being asked to integrate -- `sector_slit`'s nu=0.504
    # corner measured 6.9e-01 while this function reported 1e-13. inf is the honest value, and
    # `shortfalls` says which corner and why.
    demoted = [s for s in specs if s.singular and not s.admissible]
    shortfalls = tuple(
        [(c, a, 'model bound above target') for c, a in achieved.items() if a > precision]
        + [(s.idx, None, f'demoted to a smooth rule: {s.reason}') for s in demoted]
        + [(None, a, f'segment {i}: parametrization unresolved at order_max')
           for i, a in geom_short.items()])
    sizing_precision = max([precision] + list(achieved.values()) + list(smooth_ach.values()))
    if demoted:
        sizing_precision = float('inf')
    bq = assemble_panels(domain, panels, sizing_precision=sizing_precision, x0=x0,
                         shortfalls=shortfalls)

    if warn:
        short = {c: a for c, a in achieved.items() if a > precision}
        if short:
            msg = "; ".join(f"corner {c} at {specs[c].point:+.4g} "
                            f"(nu={specs[c].nu:.4f}) reached only {a:.2e}"
                            for c, a in short.items())
            warnings.warn(f"boundary_quadrature could not size for precision={precision:.1e}: "
                          f"{msg}. Model bound {bq.sizing_precision:.2e} overall; "
                          f"call verify_gram to measure what the rule achieves.")
        fallback = demoted
        if fallback:
            warnings.warn("singular corners on a smooth rule: "
                          + "; ".join(f"corner {s.idx} ({s.reason})" for s in fallback))
        if geom_short:
            warnings.warn(
                "boundary parametrization is the limit, not the eigenfunction: "
                + "; ".join(f"segment {i} converges only to {a:.1e} at order {order_max_geom}"
                            for i, a in geom_short.items())
                + ". A varying-speed curve's arclength map has a narrow strip of analyticity "
                  "and an eccentric one cannot be resolved at any usable order.")
    return bq


def hadamard_quadrature(domain, lam_max, precision=1e-13, **kw):
    """`boundary_quadrature` for SHAPE-DERIVATIVE integrals, where the weight is `V.n`.

    A Hadamard integrand is `(du_i/dn)(du_j/dn)(V.n)`, and a shape velocity that moves a corner
    supplies `V.n ~ r^1` there -- an integer power of distance from the corner, on top of the
    eigenfunction's own family. That weight is what the corner rules are NOT built for, and it
    is the whole reason `weight_family='integer'` exists. This is the thin wrapper that selects
    it, so downstream callers do not have to remember which family a shape derivative needs.

    Measured on a dense corner series against closed-form truth, worst relative error over five
    draws per panel, weight `r^1`, lam_max=100:

        domain                      even (sub=nu)   integer (sub=1/2)
        L_shape        nu=2/3             2.5e-09             1.5e-13
        chevron(1,2)   nu=2/3             2.8e-06             4.9e-11
        chevron(0.5,3) nu=0.772           2.7e-05             5.1e-12
        chevron(2,3)   nu=0.587           2.1e-06             2.2e-10

    Four to seven orders, at fewer or comparable nodes, and it holds at both nu regimes and on
    'cornerinterp' corners as well as 'cornerjac' -- the dense-family case that had never been
    measured before this function was written.

    **DO NOT use this node set for the Rellich/Gram normalization.** The trade runs the other
    way there, and by as much: with the plain `r^0` weight the same rules give 4.0e-15 (even)
    against 1.4e-05 (integer) on L_shape. `boundary_quadrature` remains correct for that, and
    the two are cheap enough to build separately. The split is by the WEIGHT's parity, not by
    the eigenfunction, which is why one function cannot serve both.
    """
    if 'weight_family' in kw:
        raise TypeError("hadamard_quadrature selects weight_family='integer'; to choose one "
                        "explicitly, call boundary_quadrature directly")
    return boundary_quadrature(domain, lam_max, precision=precision,
                               weight_family='integer', **kw)


def geometry_order_for_precision(seg, x0, scale, precision=1e-13, order_min=8, order_max=512):
    """Smallest Gauss order at which this segment's own `(r.N)` integral has converged.

    The oscillation model (`quad.smooth_order_for_precision`) sizes for the eigenfunction's
    wavenumber and silently assumes the *parametrization* is free. On a curved segment of
    varying speed it is not: normalized arclength is analytic but its strip of analyticity
    narrows with eccentricity, and Gauss-Legendre's rate narrows with it. Measured on
    `int (r.N) ds`, whose exact total over a closed boundary is `2|Omega|` -- geometry only, no
    eigenfunction, no lam:

        order          16      32      64     128     192     256
        disk         3e-16   3e-16   3e-16   3e-16   3e-16   3e-16
        ellipse a=2  5e-03   2e-04   4e-07   3e-12   5e-15   2e-15
        ellipse a=3  2e-02   5e-03   2e-04   8e-07   3e-09   4e-11
        ellipse a=4  4e-02   1e-02   2e-03   8e-05   3e-06   1e-07
        ellipse a=6  6e-02   3e-02   1e-02   2e-03   5e-04   1e-04

    A circle costs nothing (its arclength map is linear, so it is exact at any order) and a
    straight edge likewise. An eccentric ellipse is expensive and, past about a=4, cannot be
    bought at any sane order -- `order_max` binds and the caller records a shortfall rather
    than pretending otherwise.

    Self-convergence rather than the closed form, because the exact value is known only for
    the whole boundary, not per segment. `scale` normalizes the increment; pass `2|Omega|`.

    Two passes: double until the increment settles, which brackets the requirement, then scan
    up from `order_min` against the bracketing value as a reference and return the SMALLEST
    order that meets it. Doubling alone overshoots by up to 3x (an a=2 ellipse needs ~192 and
    doubling lands on 512) and, worse, never returns less than `2*order_min`, which would
    raise straight segments that are already exact.
    """
    def integral(order):
        u, w = cached_leggauss(order)
        return seg.len*np.sum(w*complex_dot(seg.p(u) - x0, seg.N(u)))

    prev, order, ref, increment = None, order_min, None, float('inf')
    while order <= order_max:
        I = integral(order)
        if prev is not None:
            increment = abs(I - prev)/scale       # kept: `prev` is overwritten below
            if increment <= precision:
                ref = I
                break
        prev, order = I, 2*order
    if ref is None:                       # order_max bound; report what it achieved
        return order_max, float(increment)

    step = max(8, order//16)
    for o in range(order_min, order + 1, step):
        err = abs(integral(o) - ref)/scale
        if err <= precision:
            return o, float(err)
    return order, 0.0


def refine_quadrature(domain, bq, depth=2, smooth_order=48):
    """A finer node set covering the same boundary, for a posteriori verification.

    Refinement respects what each rule is for. A legendre panel is split into `2**depth`
    equal pieces at `smooth_order`. A CORNER panel keeps its anchored end -- that end carries
    the singularity the corner rule exists for and must never be cut off, the same rule
    `_split_at_breaks` follows -- on a piece `2**depth` times shorter, and the vacated part
    becomes legendre panels. The singular end is refined by SHRINKING rather than by raising
    an order, because accuracy is not monotone in a corner rule's order past a nu-dependent
    threshold and `cornerjac_order_cap` binds besides.

    This is the honest half of `precision`: the model integrands that size the rule are
    members of the right class but carry model amplitudes, and on a convex non-integer corner
    they are pessimistic by several orders (measured 1.2e-10 predicted against 8.7e-19
    delivered on `right_trapezoid`). Comparing an integral across a refinement measures what
    was actually achieved on the actual integrand.
    """
    refined = []
    for p in bq.panels:
        lo, hi = p.tau0, p.tau1                  # signed: hi < lo means anchored at hi
        if p.rule == 'legendre':
            edges = np.linspace(lo, hi, 2**depth + 1)
            refined += [p._replace(tau0=a, tau1=b, order=max(p.order, smooth_order))
                        for a, b in zip(edges[:-1], edges[1:])]
            continue
        fracs = 0.5**np.arange(depth, -1, -1)
        taus = lo + (hi - lo)*fracs
        refined.append(p._replace(tau1=taus[0]))
        for a, b in zip(taus[:-1], taus[1:]):
            refined.append(CornerPanel(p.seg_idx, a, b, 'legendre', smooth_order,
                                       np.nan, None, np.nan, False, -1))
    return assemble_panels(domain, refined, sizing_precision=bq.sizing_precision, x0=bq.x0,
                           shortfalls=bq.shortfalls)


def verify_gram(basis, lam, coef, bq, domain, x0=None, depth=2, smooth_order=48):
    """Measure what the Gram on `bq` actually achieved, by recomputing it on a refined rule.

    Returns `(G, G_ref, err)` with `err` the largest entrywise change, relative to the
    diagonal. This is the certificate. Unlike `bq.sizing_precision` it is a statement about the
    integrand in hand rather than about a model of its class, and unlike the x0-spread it does
    not depend on where the reference point sits.

    Costs one extra evaluation of the basis at the refined nodes -- paid once per lam, against
    a node set that is otherwise built once per solve.
    """
    bq_ref = refine_quadrature(domain, bq, depth=depth, smooth_order=smooth_order)
    G = gram(eigfun_cauchy_data(basis, lam, coef, bq), lam, bq, x0=x0)
    G_ref = gram(eigfun_cauchy_data(basis, lam, coef, bq_ref), lam, bq_ref, x0=x0)
    scale = max(np.abs(np.diag(G_ref)).max(), np.finfo(float).tiny)
    return G, G_ref, float(np.abs(G - G_ref).max()/scale)


CertifiedQuad = namedtuple('CertifiedQuad',
                           ['bq', 'error', 'smooth_safety', 'certified', 'history'])
# What `certified_quadrature` returns. `error` is the `verify_gram` measurement for `bq` -- the
# certificate -- and `certified` says whether it met the target. `history` is the full
# (safety, n_nodes, error) trace, so a caller that did not reach its target can see whether it
# was close, plateauing, or getting worse.


def certified_quadrature(domain, basis, lam, coef, lam_max, target=1e-12, precision=1e-13,
                         safety_schedule=(1.0, 2.0, 3.0, 4.0, 6.0), x0=None, depth=2,
                         smooth_order=48, stall_factor=2.0, patience=2, warn=True, **kw):
    """A node set whose accuracy on the ACTUAL integrand is measured, not modelled.

    `boundary_quadrature` sizes from a model of the integrand's class; `verify_gram` measures
    what a node set achieved on the integrand in hand. This closes the loop between them:
    build, measure, escalate, stop when the measurement clears `target`.

    **It escalates the sizing knob and REBUILDS; it does not refine in place.** That is not a
    stylistic choice. `refine_quadrature` is the right reference for a measurement and the wrong
    thing to adopt as the working rule: shrinking a corner panel hands the vacated piece to a
    legendre rule that is still inside the singularity, and near the slit that loses ground.
    Measured on the sector, exact eigenfunction, true error before and after one refinement:

        alpha/pi     1.10      1.50      1.60      1.75      1.90
        before    1.2e-14   1.3e-15   5.9e-12   3.3e-12   9.5e-10
        after     9.4e-15   8.7e-15   3.9e-13   1.3e-10   4.9e-09

    -- better at 1.6, forty times WORSE at 1.75. A loop that adopted the refinement would walk
    backwards in exactly the regime that needs it. Rebuilding at a higher `smooth_safety`
    instead means every candidate is independently sized and independently measured, and the
    best measured one is what comes back (not merely the last).

    `smooth_safety` is the knob because that is where the residual lives: the corner rules are
    at their own ceiling well before the smooth panels are, and the smooth panels' true
    requirement is set by the basis's pole placement, which the oscillation model cannot see.

    ON THE STOPPING TEST. `verify_gram` reports the change across a refinement, which estimates
    the error of the COARSER rule. Against exact truth on the sectors it tracked the true error
    to within 6.3x and never under-reported by more -- good enough to stop on, not good enough
    to quote as an error bar without margin. Ask for a `target` an order below what you need.

    Where the reference is itself poor -- the near-slit corners above -- the measurement is
    dominated by the reference's error and becomes PESSIMISTIC (38x at alpha=1.75pi), which is
    the safe direction: such a domain will exhaust the schedule and return `certified=False`
    rather than claim something it has not got. `stall_factor` cuts that short, abandoning the
    escalation once more nodes stop buying at least that factor, since on a corner-limited
    domain they never will.

    `patience` is why that cutoff needs two strikes rather than one. Convergence in
    `smooth_safety` is NOT monotone -- on a disk it goes 7.1e-10, 6.4e-11, 4.6e-10, and then
    5.2e-15 at safety=4. A one-strike rule quits at the 4.6e-10 and reports `certified=False`
    one step before the answer, which is exactly what the first version of this function did.
    The cost of patience on a genuinely corner-limited domain is one extra build.

    COST. One `boundary_quadrature` build plus two basis evaluations per step, so a certified
    build is a few times a plain one -- paid once per solve, not once per lam. It needs a
    representative `lam` and `coef`, so certify at the top of the lam window and reuse the node
    set across the window, which is what `lam_max` sizes for anyway.
    """
    best = CertifiedQuad(None, float('inf'), None, False, ())
    history = []
    stalls = 0
    for safety in safety_schedule:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')     # sizing shortfalls are not the certificate
            bq = boundary_quadrature(domain, lam_max, precision=precision, x0=x0,
                                     smooth_safety=safety, warn=False, **kw)
        _, _, err = verify_gram(basis, lam, coef, bq, domain, x0=x0, depth=depth,
                                smooth_order=smooth_order)
        history.append((float(safety), len(bq.pts), float(err)))
        stalls = 0 if err < best.error/stall_factor else stalls + 1
        if err < best.error:
            best = CertifiedQuad(bq, float(err), float(safety), err <= target, ())
        if err <= target:
            return best._replace(certified=True, history=tuple(history))
        if stalls >= patience:
            # more nodes have stopped paying: a corner ceiling, not an under-resolved panel
            break
    if best.bq is None:
        raise ValueError("safety_schedule is empty: nothing was built or measured")
    best = best._replace(history=tuple(history))
    if warn and not best.certified:
        warnings.warn(
            f"certified_quadrature did not reach target={target:.1e}: best measured "
            f"{best.error:.2e} at smooth_safety={best.smooth_safety} on {len(best.bq.pts)} "
            f"nodes. Trace (safety, nodes, error): {best.history}. A plateau here is a corner "
            "ceiling rather than an under-resolved smooth panel -- more nodes will not fix it.")
    return best


def _both_ends_singular(specs, spec):
    """Does the segment leaving `spec`'s corner also END at an admissible singular corner?
    Such an edge must split, since a corner panel anchors at one endpoint only."""
    at_end = {s.seg_in for s in specs if s.admissible and s.singular}
    return spec.seg_out in at_end
