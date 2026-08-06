"""Source placement that scales with the domain's LOCAL thickness.

`FundamentalBasis.by_boundary` offsets every source by the same `d`. Measured, the right `d`
is not a constant even across corner-free domains: `stadium` (width 1) wants 0.1 and keeps
320/320 columns there against 78/324 at 1.0, while `ellipse_a2` (width ~2.6) wants 1.0 and is
ten orders worse at 0.05. So the offset tracks thickness.

A domain with a *varying* thickness -- a mushroom, a cap joined to a stem by a thin neck --
therefore has no good single `d` at all, which is consistent with the mushrooms being the one
bucket-2 group that no uniform offset moves. This places each source at
`frac * (local thickness)` instead.
"""
import numpy as np

from lappy.bases import FundamentalBasis


def _dense_boundary(domain, n_probe=1500):
    segs = domain.bdry.segments
    lens = np.array([s.len for s in segs], dtype=float)
    per = lens.sum()
    P = []
    for seg, L in zip(segs, lens):
        m = max(int(round(n_probe*L/per)), 8)
        t = (np.arange(m) + 0.5)/m
        P.append(seg.p(t))
    return np.concatenate(P)


def local_thickness(domain, pts, normals, cone=0.5, n_probe=1500):
    """How far it is ACROSS the domain at each boundary point: the distance to the nearest
    boundary point lying in an inward cone about the inward normal (`cone` = minimum cosine,
    0.5 is 60 degrees).

    The cone is what makes this measure geometry rather than a parameter. A first version
    excluded boundary within a fixed arclength of the query point instead; that fails, because
    the nearest admissible point is then just outside the exclusion window on the SAME wall, so
    every domain reports a thickness equal to the exclusion radius. Measured that way, stadium
    (true width 1) and ellipse_a2 (true width ~2.6) both came back at 0.10 and 0.19 -- the
    parameter, not the domain.
    """
    P = _dense_boundary(domain, n_probe)
    z = pts.pts if hasattr(pts, 'pts') else np.asarray(pts)
    nz = normals.pts if hasattr(normals, 'pts') else np.asarray(normals)
    out = np.empty(len(z))
    for i, (p, n) in enumerate(zip(z, nz)):
        v = P - p
        r = np.abs(v)
        ok = r > 1e-9
        # cos angle between (q - p) and the INWARD normal -n
        cosang = np.where(ok, (v.real*(-n.real) + v.imag*(-n.imag))/np.where(ok, r, 1.0), -1.0)
        sel = ok & (cosang > cone)
        out[i] = r[sel].min() if sel.any() else np.inf
    return out


def fs_by_local_thickness(domain, n_per_seg, frac=0.3, floor=None, cap=None):
    """`FundamentalBasis` with each source offset outward by `frac * local_thickness`."""
    bdry = domain.bdry_pts(n_per_seg)
    nrm = domain.bdry_normals(n_per_seg)
    th = local_thickness(domain, bdry, nrm)
    d = frac*th
    if cap is not None:
        d = np.minimum(d, cap)
    if floor is not None:
        d = np.maximum(d, floor)
    return FundamentalBasis(bdry.pts + d*nrm.pts, 1), d
