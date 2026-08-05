"""Stage 0 prototype of the corner-adapted boundary quadrature.

Lives in benchmarks/ deliberately: nothing goes into lappy/ until the gate in
the plan (>=4 orders of improvement over Kress on the sector sweep) is met.

Everything here goes through lappy's real segment API -- seg.p/N/T(tau),
seg.len, domain.int_angles[domain.corner_idx[c]] -- so that what is measured is
what Stage 2/3 will assemble, not a hand-rolled parametrization.

The rule itself (docs/corner_quadrature.tex Prop. 2): on a panel anchored at a
corner of interior angle alpha, with nu = pi/alpha and integrand
f(tau) ~ tau^gamma * (series in tau^nu, tau^2), substitute tau = t^(1/nu) and
apply Gauss-Jacobi. Reusing quad.jacgauss's weight-divided-out convention, the
gamma-dependence cancels from the weight and survives only in the exponent:

    t, W = jacgauss(order, (gamma+1)/nu - 1, 0)
    tau  = t**(1/nu)
    w    = (W/nu) * t**(1/nu - 1)
"""
import numpy as np

from lappy.quad import jacgauss, cached_leggauss
from lappy.utils import complex_dot

# nu <= 1/2 makes the Jacobi exponent <= -1 (inadmissible, and the integral
# itself diverges); leave a margin, since at nu=0.51 the exponent is already
# -0.96 and the innermost node collides with the coordinate-collapse floor.
NU_MIN = 0.5
TAU_FLOOR = 1e-9   # same float64 defect cached_kressgauss guards against


def corner_nus(domain):
    """Exact geometric nu per corner, corner-indexed like domain.corners.

    NOTE the index conventions: corners/corner_idx are corner-indexed while
    int_angles is segment-indexed, so the angle at corner c is
    int_angles[corner_idx[c]] -- never int_angles[c].
    """
    ia = np.asarray(domain.int_angles)
    return np.pi/ia[np.asarray(domain.corner_idx)]


def cornerjac(order, nu, gamma=None):
    """(tau, w) on [0,1], corner anchored at tau=0. See module docstring."""
    if nu <= NU_MIN:
        raise ValueError(f"nu={nu} <= 1/2: Jacobi exponent <= -1 (inadmissible)")
    if gamma is None:
        gamma = 2.0*nu - 2.0
    t, W = jacgauss(order, (gamma + 1.0)/nu - 1.0, 0.0)
    return t**(1.0/nu), (W/nu)*t**(1.0/nu - 1.0)


def seg_corner_map(domain):
    """(nu_start, nu_end) per segment: the exact nu at each of its endpoints,
    or nan where that junction is not a listed corner."""
    segs = domain.bdry.segments
    nus = corner_nus(domain)
    nu_start = np.full(len(segs), np.nan)
    for c, j in enumerate(np.asarray(domain.corner_idx)):
        nu_start[j] = nus[c]
    return nu_start, np.roll(nu_start, -1)


def is_straight(seg):
    return type(seg).__name__ == 'LineSegment'


def panel_plan(domain, order_corner=16, order_smooth=None, frac=1.0, nu_max=1.0):
    """Panels covering every segment's [0,1] exactly once.

    Each panel is (seg_idx, tau0, tau1, kind, order, nu), with tau0 ALWAYS the
    corner-anchored end -- so tau1 < tau0 on a panel anchored at the segment's
    far endpoint. `frac` is the panel's share of the segment (the open question
    from the plan: the corner expansion only converges within the largest disk
    about the corner inside Omega, so a full-length panel may overshoot it).
    """
    segs = domain.bdry.segments
    nu0, nu1 = seg_corner_map(domain)
    if order_smooth is None:
        order_smooth = order_corner

    def eligible(nu, seg):
        return (not np.isnan(nu)) and NU_MIN < nu < nu_max and is_straight(seg)

    panels = []
    for i, seg in enumerate(segs):
        e0, e1 = eligible(nu0[i], seg), eligible(nu1[i], seg)
        if not (e0 or e1):
            panels.append((i, 0.0, 1.0, 'leg', order_smooth, np.nan))
        elif e0 and not e1:
            panels.append((i, 0.0, frac, 'jac', order_corner, nu0[i]))
            if frac < 1.0:
                panels.append((i, frac, 1.0, 'leg', order_smooth, np.nan))
        elif e1 and not e0:
            panels.append((i, 1.0, 1.0 - frac, 'jac', order_corner, nu1[i]))
            if frac < 1.0:
                panels.append((i, 0.0, 1.0 - frac, 'leg', order_smooth, np.nan))
        else:
            # both endpoints singular: the substitution anchors at one end only,
            # so the edge must split. This is the case only Leg 3 exercises.
            s = min(frac, 0.5)
            panels.append((i, 0.0, s, 'jac', order_corner, nu0[i]))
            panels.append((i, 1.0, 1.0 - s, 'jac', order_corner, nu1[i]))
            if s < 0.5:
                panels.append((i, s, 1.0 - s, 'leg', order_smooth, np.nan))
    return panels


def assemble(domain, panels, gamma=None):
    """(pts, normals, tangents, wts, panel_id) from a panel plan, via seg.p/N/T."""
    segs = domain.bdry.segments
    P, N, T, W, PID = [], [], [], [], []
    for pid, (i, tau0, tau1, kind, order, nu) in enumerate(panels):
        seg = segs[i]
        if kind == 'leg':
            u, w = cached_leggauss(order)
        else:
            u, w = cornerjac(order, nu, gamma)
        h = tau1 - tau0                 # signed: carries the anchor orientation
        tau = tau0 + h*u
        P.append(seg.p(tau)); N.append(seg.N(tau)); T.append(seg.T(tau))
        W.append(seg.len*abs(h)*w)
        PID.append(np.full(order, pid))
    return (np.concatenate(P), np.concatenate(N), np.concatenate(T),
            np.concatenate(W), np.concatenate(PID))


def kress_reference(domain, basis, lam_max, mult=2, min_per_seg=4, margin=2.0, q=8,
                    c_lam=1.0, **kw):
    """The RETIRED Kress-graded rule, reconstructed here so the Stage 0 comparison in
    stage0_sector.py stays runnable after lappy.cauchy was deleted.

    Not a re-implementation of anything live: one Kress-graded Gauss-Legendre rule per segment,
    graded at the same order toward BOTH endpoints (the limitation that motivated the
    replacement), with the point count sized from basis size and lam as
    cauchy.graded_pts_per_seg did."""
    from lappy.quad import cached_kressgauss, cached_leggauss
    segs = domain.bdry.segments
    seg_lens = np.array([sg.len for sg in segs])
    base_n = mult*len(basis)*seg_lens/seg_lens.sum()
    lam_n = c_lam*np.sqrt(lam_max)*seg_lens
    n_per_seg = np.maximum(np.round(np.maximum(base_n, lam_n)).astype(int), min_per_seg)
    nus = corner_nus(domain)
    graded = {int(j) for c, j in enumerate(np.asarray(domain.corner_idx)) if nus[c] < 1.0}
    P, N, T, W = [], [], [], []
    for i, sg in enumerate(segs):
        n = int(n_per_seg[i])
        if i in graded or ((i + 1) % len(segs)) in graded:
            tau, w = cached_kressgauss(n, q)
        else:
            tau, w = cached_leggauss(n)
        P.append(sg.p(tau)); N.append(sg.N(tau)); T.append(sg.T(tau)); W.append(sg.len*w)
    pts = np.concatenate(P)
    return (pts, np.concatenate(N), np.concatenate(T), np.concatenate(W),
            np.zeros(len(pts)))


def rellich_norm2(pts, normals, wts, un, lam, x0, panel_id=None, groups=None):
    """||u||^2 = (1/2 lam) * int rN (du/dn)^2 ds, Dirichlet.

    `groups` optionally maps panel_id -> label, in which case the per-group
    contributions are returned too. Reporting that split is not optional: at
    alpha=3pi/2 the two edges carry rN = Im(x0) and -Re(x0) against identical
    (du/dn)^2, so an x0 on the diagonal cancels the singular part *identically*
    and a total-only test silently measures the arc alone.
    """
    rN = complex_dot(pts - x0, normals)
    contrib = wts*rN*un**2/(2*lam)
    total = contrib.sum()
    if groups is None:
        return total, None
    parts = {}
    for pid, label in groups.items():
        parts[label] = parts.get(label, 0.0) + contrib[panel_id == pid].sum()
    return total, parts
