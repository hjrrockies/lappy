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

One caveat is real and is measured rather than assumed. The scheme assumes a node
at parameter `tau` really sits at arclength `tau*seg.len`, which holds exactly for
LineSegments and, for ParametricSegments, only as well as the adaptive arclength
table's PCHIP inverse does. A circular arc is machine-exact at any `tol` (its
arclength map is linear); a curve of varying speed is not, and the resulting
boundary integrals stall around 1e-5 to 1e-6 *regardless of quadrature order*,
because a piecewise-cubic inverse makes the integrand only C^1 and Gauss-Legendre
needs analyticity. `_parametrization_quality` reports the round-trip error behind
this. docs/eigfun_integrals.md describes the fix (a Newton-solved inverse, which
restores spectral convergence and makes the constant-speed property exact) and,
for spline boundaries, the additional requirement that panels break at knots.

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
                   _CORNER_NU_MIN, _CORNER_MAX_Q)
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
                           'panels', 'panel_id', 'precision', 'x0'])

# A small cluster of eigenfunctions' Cauchy data at a BoundaryQuad's nodes. U/U_N/U_T are
# (n_nodes, mult) -- already-evaluated function values, never a basis matrix.
EigfunData = namedtuple('EigfunData', ['pts', 'normals', 'tangents', 'wts', 'U', 'U_N', 'U_T'])

# Margin above the nu=1/2 admissibility threshold. At nu=0.51 the Jacobi exponent is already
# -0.96 and the innermost node collides with the coordinate-collapse floor by order 32, so
# admitting corners arbitrarily close to a slit buys nothing.
_NU_MARGIN = 0.02
_ANGLE_TOL = 1e-10

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


def corner_specs(domain):
    """CornerSpec per genuine corner, corner-indexed like `domain.corners`.

    Uses `domain.corner_int_angles` (corner-indexed) rather than `int_angles` (segment-indexed)
    -- see geometry.Domain.corner_int_angles on why that distinction is load-bearing."""
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
        singular = nu < 1.0 - _ANGLE_TOL

        reason = ''
        if not singular:
            reason = 'not singular (nu >= 1): a smooth rule is already exact'
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
    return panels


def _panel_rule(panel):
    if panel.rule == 'legendre':
        return cached_leggauss(panel.order)
    if panel.rule == 'cornerjac':
        return cached_cornerjacgauss(panel.order, panel.nu, panel.gamma, panel.sub)
    if panel.rule == 'cornerinterp':
        return cached_cornerinterpgauss(panel.order, panel.nu, panel.gamma,
                                        None, panel.curved)
    raise ValueError(f"unknown panel rule {panel.rule!r}")


def assemble_panels(domain, panels, precision=None, x0=None):
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
                        tuple(panels), np.concatenate(PID), precision, x0)


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


def default_x0(domain):
    """Reference point x0 for the Rellich weight r.N.

    Placed at a singular corner when there is one, because r.N vanishes identically on both
    edges meeting it (the edges are parallel to x - x0 there), which removes that corner's
    singularity from the integrand outright rather than resolving it. With the corner-adapted
    rule this is no longer *necessary* -- that is the whole point of the rule, and multi-corner
    domains cannot zero every corner anyway -- but it is free, and it makes the one corner it
    covers exact independent of quadrature. Falls back to the bounding-box centre."""
    specs = [s for s in corner_specs(domain) if s.singular]
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
                        clearance_frac=_CLEARANCE_FRAC, warn=True):
    """The entry point: a boundary node set for `domain`, accurate to `precision` for
    eigenfunctions up to spectral parameter `lam_max`.

    No basis, and none of the tuning knobs the graded rule required -- the node set is a pure
    function of geometry, `lam_max` and the requested accuracy, so it can be built once and
    reused for every lam in a search. Sizing is self-certifying rather than calibrated
    offline: every panel's rule is scored against a model integrand whose integral is
    closed-form (`quad.corner_rule_residual` on the corner's own exponent set,
    `quad.smooth_order_for_precision` on exp(i k tau)), so `precision` is honoured directly.

    Where a corner cannot reach `precision` -- a near-slit nu, or the coordinate-collapse
    order cap binding first -- the best achievable order is used and a warning names the
    corner and what it actually achieved, rather than silently missing the target. The
    achieved value is recorded in the returned `BoundaryQuad.precision`.

    The default is 1e-13, not 1e-14: these integrals sit near the float64 roundoff floor, and a
    270-degree corner typically lands at ~1.9e-14, so a 1e-14 default would warn on the
    commonest domain in the suite while delivering essentially the same answer. Asking for 1e-14
    explicitly is legitimate and will warn if it falls short, which is the point.
    """
    specs = corner_specs(domain)
    segs = domain.bdry.segments
    k = 2.0*np.sqrt(max(lam_max, 0.0))     # a PRODUCT of eigenfunctions oscillates at 2*sqrt(lam)

    # per-corner order, from the corner's own exponent set
    orders, achieved = {}, {}
    for sp in specs:
        if not (sp.singular and sp.admissible):
            continue
        curved = not sp.straight
        frac = min(panel_frac, 0.5) if _both_ends_singular(specs, sp) else panel_frac
        seg_len = segs[sp.seg_out].len
        o, ach = corner_order_for_precision(sp.kind, sp.nu, None, sp.sub, curved,
                                            precision, scale=frac,
                                            k=np.sqrt(max(lam_max, 0.0))*seg_len*frac)
        orders[sp.idx], achieved[sp.idx] = o, ach

    # smooth stretches: Nyquist in the segment's own arclength
    smooth_orders, smooth_ach = {}, {}
    for i, seg in enumerate(segs):
        o, ach = smooth_order_for_precision(k*seg.len, precision)
        smooth_orders[i], smooth_ach[i] = o, ach

    panels = []
    for p in corner_panels(domain, specs, order_corner=1, order_smooth=1,
                           panel_frac=panel_frac, clearance_frac=clearance_frac,
                           order_cap=False):
        if p.rule == 'legendre':
            panels.append(p._replace(order=smooth_orders[p.seg_idx]))
        else:
            panels.append(p._replace(order=orders.get(p.corner, 16)))
    bq = assemble_panels(domain, panels,
                         precision=max([precision] + list(achieved.values())
                                       + list(smooth_ach.values())),
                         x0=x0)

    if warn:
        short = {c: a for c, a in achieved.items() if a > precision}
        if short:
            msg = "; ".join(f"corner {c} at {specs[c].point:+.4g} "
                            f"(nu={specs[c].nu:.4f}) reached only {a:.2e}"
                            for c, a in short.items())
            warnings.warn(f"boundary_quadrature could not reach precision={precision:.1e}: "
                          f"{msg}. Achieved {bq.precision:.2e} overall.")
        fallback = [s for s in specs if s.singular and not s.admissible]
        if fallback:
            warnings.warn("singular corners on a smooth rule: "
                          + "; ".join(f"corner {s.idx} ({s.reason})" for s in fallback))
    return bq


def _both_ends_singular(specs, spec):
    """Does the segment leaving `spec`'s corner also END at an admissible singular corner?
    Such an edge must split, since a corner panel anchors at one endpoint only."""
    at_end = {s.seg_in for s in specs if s.admissible and s.singular}
    return spec.seg_out in at_end
