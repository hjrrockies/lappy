"""Cauchy-data evaluation and generic boundary-integral kernel assembly for
particular-solution bases.

See docs/rellich_hadamard_mps.pdf. Every downstream boundary-integral quantity
built from MPS eigenfunctions -- Rellich L^2(Omega) inner products
(lappy.rellich), and Hadamard-type shape-derivative formulas (owned by
consumers *outside* lappy, e.g. a shape-optimization package) -- reduces to a
boundary integral of one of four bilinear "Cauchy-data kernels" (uv,
dNu*dNv, dTu*dTv, dTu*dNv+dTv*dNu) contracted against an application-specific
scalar weight (r.N/r.T for Rellich; a boundary velocity V(s) for Hadamard).

This module provides the two-layer machinery the paper proposes:
1. Evaluate each basis function's Cauchy data (value, normal derivative,
   tangent derivative) once at a shared boundary node set (`CauchyData`,
   `basis_cauchy_data`).
2. A generic kernel/weight assembler (`assemble_kernel`) that builds the
   N x N matrix for any kernel/weight pair from that precomputed data.

Rellich-specific formulas (lappy.rellich) are a fixed linear combination of
calls to `assemble_kernel`; a consumer building Hadamard-type (or other)
quantities does the same with its own weight arrays -- lappy has no notion
of shape derivatives, boundary velocities, or domain parameterizations.
"""
from collections import namedtuple

import numpy as np

from .geometry import PointSet
from .quad import jacgauss

# A basis's Cauchy data (value, normal deriv, tangent deriv) at a boundary node set (pts/normals/
# tangents plus plain, unsigned arc-length quadrature weights), for one spectral parameter lambda.
CauchyData = namedtuple('CauchyData', ['pts', 'normals', 'tangents', 'wts', 'Phi', 'Phi_N', 'Phi_T'])


def basis_cauchy_data(basis, lam, pts, normals, tangents, wts=None, cols=None):
    """Evaluates a particular-solution basis' Cauchy data (value, normal
    derivative, tangent derivative) at a caller-supplied boundary node set,
    bundled with the node geometry for reuse across every downstream
    boundary-integral quantity (`assemble_kernel`).

    `pts`, `normals`, `tangents` are PointSets (or anything accepted by
    PointSet) of matching length; `wts` are plain (unsigned) arc-length
    quadrature weights, defaulting to zero (an empty/placeholder node set).
    This is the general-purpose entry point: any caller -- including code
    outside lappy -- that wants Cauchy data at a node set/panel structure of
    its own choosing (e.g. tuned to its own weight function) can call this
    directly, rather than reusing lappy's precomputed Rellich node sets.

    `cols`, if given (a sorted integer index array -- see ParticularBasis.__call__),
    evaluates only those basis columns: the returned CauchyData's Phi/Phi_N/Phi_T have
    `len(cols)` columns, in that order, not the basis's full width. This is what makes
    lappy.cauchy.singular_gram's SS/RR sub-blocks cheap for bases with a per-column-localized
    cost (e.g. FourierBesselBasis): a corner's own small mode set, or "everything but that
    corner", can be evaluated without paying for the columns that block doesn't need."""
    if not isinstance(pts, PointSet):
        pts = PointSet(pts)
    if not isinstance(normals, PointSet):
        normals = PointSet(normals)
    if not isinstance(tangents, PointSet):
        tangents = PointSet(tangents)
    if wts is None:
        wts = np.zeros(len(pts))

    Phi = basis(lam, pts, cols=cols)
    Phi_N = basis.ddiff(lam, pts, normals, cols=cols)
    Phi_T = basis.ddiff(lam, pts, tangents, cols=cols)
    return CauchyData(pts, normals, tangents, wts, Phi, Phi_N, Phi_T)


def assemble_kernel(cauchy_data, kernel, weight, cols1=None, cols2=None):
    """Generic elemental-matrix assembler:
    A[j,l] = sum_i wts[i]*weight[i]*K_jl(node_i)
    for kernel in {'uv','NN','TT','cr'} -- the four bilinear Cauchy-data
    kernels K^uv=uv, K^NN=dNu*dNv, K^TT=dTu*dTv, K^cr=dTu*dNv+dTv*dNu -- using
    the precomputed Cauchy data in `cauchy_data`. `weight` is a real array
    already evaluated at `cauchy_data.pts` (e.g. r.N/r.T for Rellich, or a
    boundary velocity V(s) for a Hadamard-type formula built outside lappy).

    `cols1`/`cols2` optionally restrict the assembled matrix to a subset of
    basis columns on each side (e.g. to build a cross-block between a corner's
    own singular columns and the rest of the basis without re-evaluating it);
    both default to every column (today's full, symmetric N x N matrix).

    Every Rellich (and, by a consumer, Hadamard) formula is a fixed linear
    combination of calls to this one routine with different kernels/weights,
    reusing the same precomputed Cauchy data."""
    n = cauchy_data.Phi.shape[1]
    if cols1 is None:
        cols1 = np.arange(n)
    if cols2 is None:
        cols2 = cols1

    if len(cauchy_data.pts) == 0:
        return np.zeros((len(cols1), len(cols2)))

    w = (cauchy_data.wts*weight)[:, np.newaxis]
    if kernel == 'uv':
        Phi = cauchy_data.Phi
        return (Phi[:, cols1]*w).T@Phi[:, cols2]
    elif kernel == 'NN':
        Phi_N = cauchy_data.Phi_N
        return (Phi_N[:, cols1]*w).T@Phi_N[:, cols2]
    elif kernel == 'TT':
        Phi_T = cauchy_data.Phi_T
        return (Phi_T[:, cols1]*w).T@Phi_T[:, cols2]
    elif kernel == 'cr':
        Phi_N, Phi_T = cauchy_data.Phi_N, cauchy_data.Phi_T
        return (Phi_T[:, cols1]*w).T@Phi_N[:, cols2] + (Phi_N[:, cols1]*w).T@Phi_T[:, cols2]
    else:
        raise ValueError("'kernel' must be one of 'uv', 'NN', 'TT', 'cr'")


def default_x0(domain):
    """Bounding-box center of the domain's boundary, usable as a default
    reference point x0 for identities like Rellich's that admit any x0 (this
    default keeps |r| moderate for typical domains)."""
    tau = np.linspace(0, 1, 50)[:-1]
    pts = np.concatenate([seg.p(tau) for seg in domain.bdry.segments])
    x0_re = 0.5*(pts.real.min() + pts.real.max())
    x0_im = 0.5*(pts.imag.min() + pts.imag.max())
    return x0_re + 1j*x0_im


def graded_pts_per_seg(domain, basis, mult=2, min_per_seg=4):
    """Per-segment point counts for a graded boundary quadrature, sized
    relative to basis length and segment length (mirrors mps.pts_per_seg's
    generic branch)."""
    seg_lens = domain.seg_lens
    pps = np.round(mult*len(basis)*seg_lens/seg_lens.sum()).astype(int)
    return np.maximum(pps, min_per_seg).astype(int)


### Singularity-subtraction (SS/SR/RS/RR) quadrature, docs/rellich_hadamard_mps.pdf Sec. 4-5.
#
# Rather than one Gauss-Jacobi exponent per segment (graded from a corner's *leading* mode only and
# applied uniformly to every basis-function pair, an earlier/simpler approach), each basis column's
# *own* exact local behavior near each corner is used: column j with mode exponent p_j at corner c
# behaves like rho**(p_j - d) * entire(rho**2) there, d in {0,1} the number of derivatives the
# kernel takes of that factor (0 for a value factor as in K^uv, 1 for a normal/tangential-
# derivative factor, since both reduce the order by exactly 1). Grading a Gauss-Jacobi rule to a
# basis-function *pair*'s exact combined exponent -- rather than a segment-wide leading-mode proxy
# -- is standard, correct usage and needs no new special-function machinery: ordinary jv/jvp
# (already used by FourierBesselBasis) are evaluated only at the graded rule's interior nodes,
# which never land at rho=0.

def panel_radius(domain, corner_idx, frac=0.4):
    """Radius of the singularity-subtraction panel around domain.corners[corner_idx]: `frac` times
    the shorter of its two adjacent segment lengths. For a polygon, every singular point is a
    corner, so "nearest other singular point along an edge" is simply that edge's far endpoint --
    a correct, simpler specialization of the paper's Sec. 9.2 heuristic."""
    seg_idx = domain.corner_idx[corner_idx]
    n_segs = len(domain.bdry.segments)
    seg_lens = domain.seg_lens
    return frac*min(seg_lens[seg_idx], seg_lens[(seg_idx - 1) % n_segs])


def _panel_nodes(seg, R, exponent, side, n):
    """Gauss-Jacobi nodes/weights (plain complex/real arrays, not a PointSet) on an arc-length-R
    panel of `seg`, singular with the given exponent at its 'p0' or 'pf' end (wherever the panel's
    owning corner is) and regular (exponent 0) at the panel's other (interior) boundary.

    jacgauss's own convention grades its *left* (u=0) endpoint with `a` and its *right* (u=1)
    endpoint with `b` (quad.py:jacgauss docstring). The corner is always mapped to u=0 here (tau=0
    for side='p0', tau=1 for side='pf' -- see the tau formulas below), so both sides always grade
    with `a=exponent, b=0.0`; only the tau-mapping direction differs by side."""
    u, w = jacgauss(n, a=exponent, b=0.0)
    if side == 'p0':
        tau = u*(R/seg.len)
    elif side == 'pf':
        tau = 1.0 - u*(R/seg.len)
    else:
        raise ValueError("side must be 'p0' or 'pf'")
    wts = R*w
    return seg.p(tau), seg.N(tau), seg.T(tau), wts


def _check_exponent(a, corner_idx):
    if a <= -1:
        raise ValueError(f"invalid Gauss-Jacobi exponent {a} at corner {corner_idx} "
                         "(basis/domain data implies a non-integrable corner singularity)")


def _corner_panel_gram(basis, domain, lam, kernel, weight_fn, corner_idx, corner_id, exponent,
                       seg_mask, frac, group_pts):
    """(N,N) contribution of one corner's SS+SR+RS+RR panel (restricted to arc-length panel_radius
    on each of its two adjacent segments), or (None, 0.0) if this corner has no columns singular
    there (nothing to carve out; the caller's bulk region covers it untouched)."""
    N = len(corner_id)
    Sc = np.nonzero(corner_id == corner_idx)[0]
    if len(Sc) == 0:
        return None, 0.0

    seg_idx = domain.corner_idx[corner_idx]
    n_segs = len(domain.bdry.segments)
    prev_idx = (seg_idx - 1) % n_segs
    seg_out, seg_in = domain.bdry.segments[seg_idx], domain.bdry.segments[prev_idx]
    active_out = seg_mask is None or seg_mask[seg_idx]
    active_in = seg_mask is None or seg_mask[prev_idx]
    if not (active_out or active_in):
        return None, 0.0

    R = panel_radius(domain, corner_idx, frac)
    d = 0 if kernel == 'uv' else 1
    p_Sc = exponent[Sc]
    not_Sc = np.setdiff1d(np.arange(N), Sc)

    def panel_cauchy_data(a, cols=None):
        parts = []
        if active_out:
            parts.append(_panel_nodes(seg_out, R, a, 'p0', group_pts))
        if active_in:
            parts.append(_panel_nodes(seg_in, R, a, 'pf', group_pts))
        pts, normals, tangents, wts = (np.concatenate(x) for x in zip(*parts))
        cd = basis_cauchy_data(basis, lam, pts, normals, tangents, wts, cols=cols)
        return cd, weight_fn(pts, normals, tangents)

    G = np.zeros((N, N))

    # SS: Sc x Sc, grouped by each pair's exact combined exponent. Evaluates only Sc's own
    # columns (cols=Sc) -- the dominant cost otherwise, since SS calls are the overwhelming
    # majority of calls made across a whole singular_gram invocation, and previously each one
    # paid for evaluating every one of the basis's N columns to use only |Sc| of them (see
    # docs/rellich_hadamard_mps.pdf and the profiling in benchmarks/reference/rellich_profile.py).
    a_pairs = p_Sc[:, np.newaxis] + p_Sc[np.newaxis, :] - 2*d
    for a in np.unique(a_pairs):
        _check_exponent(a, corner_idx)
        mask = np.isclose(a_pairs, a)
        cd, weight = panel_cauchy_data(a, cols=Sc)
        M = assemble_kernel(cd, kernel, weight)   # cd already IS the Sc x Sc columns, in order
        sub = G[np.ix_(Sc, Sc)]
        sub[mask] = M[mask]
        G[np.ix_(Sc, Sc)] = sub

    # SR/RS: Sc x (rest). Genuinely needs (most of) every other column to correlate Sc's group
    # against, so no cols restriction here beyond what's already implied by grouping.
    for a in np.unique(p_Sc - d):
        _check_exponent(a, corner_idx)
        group = Sc[np.isclose(p_Sc - d, a)]
        cd, weight = panel_cauchy_data(a)
        M = assemble_kernel(cd, kernel, weight, cols1=group, cols2=not_Sc)
        G[np.ix_(group, not_Sc)] = M
        G[np.ix_(not_Sc, group)] = M.T

    # RR: (rest) x (rest), plain (ungraded) rule on the same panel -- restricted to not_Sc's
    # columns (a smaller, cheaper evaluation than the full basis whenever Sc is nonempty).
    cd, weight = panel_cauchy_data(0.0, cols=not_Sc)
    G[np.ix_(not_Sc, not_Sc)] = assemble_kernel(cd, kernel, weight)

    return G, R


def singular_gram(basis, domain, lam, kernel, weight_fn, seg_mask=None, panel_frac=0.4,
                  group_pts=16, bulk_mult=2, bulk_min_per_seg=8):
    """Boundary-integral N x N matrix for one (kernel, weight) pair, via the SS/SR/RS/RR
    singularity-subtraction quadrature (docs/rellich_hadamard_mps.pdf Sec. 4-5): each corner's own
    basis columns are graded to their *exact* local exponent, and the rest of the boundary uses
    plain Gauss-Legendre.

    `weight_fn(pts, normals, tangents) -> real array` is evaluated at whatever nodes this function
    builds internally -- unlike `assemble_kernel`'s precomputed `cauchy_data`, panel node sets here
    aren't shared/precomputed upfront (each corner/exponent group needs its own).

    `seg_mask`: optional boolean array over `domain.bdry.segments` restricting which segments
    contribute at all (e.g. Gamma_D/Gamma_N for Rellich's boundary-condition split); `None`
    includes every segment."""
    corner_id, exponent = basis.corner_terms(domain)
    N = len(basis)
    G = np.zeros((N, N))

    n_corners = len(domain.corners)
    n_segs = len(domain.bdry.segments)
    excl_start, excl_end = np.zeros(n_segs), np.zeros(n_segs)

    for c in range(n_corners):
        seg_idx = domain.corner_idx[c]
        prev_idx = (seg_idx - 1) % n_segs
        Gc, R = _corner_panel_gram(basis, domain, lam, kernel, weight_fn, c, corner_id, exponent,
                                   seg_mask, panel_frac, group_pts)
        if Gc is None:
            continue
        G += Gc
        if seg_mask is None or seg_mask[seg_idx]:
            excl_start[seg_idx] = R
        if seg_mask is None or seg_mask[prev_idx]:
            excl_end[prev_idx] = R

    # bulk RR: the remainder of each active segment beyond its corner panels
    n_per_seg = graded_pts_per_seg(domain, basis, bulk_mult, bulk_min_per_seg)
    for i, seg in enumerate(domain.bdry.segments):
        if seg_mask is not None and not seg_mask[i]:
            continue
        tau_lo, tau_hi = excl_start[i]/seg.len, 1 - excl_end[i]/seg.len
        if tau_hi <= tau_lo:
            continue
        u, w = jacgauss(n_per_seg[i], 0.0, 0.0)
        tau = tau_lo + u*(tau_hi - tau_lo)
        wts = seg.len*(tau_hi - tau_lo)*w
        pts, normals, tangents = seg.p(tau), seg.N(tau), seg.T(tau)
        cd = basis_cauchy_data(basis, lam, pts, normals, tangents, wts)
        weight = weight_fn(pts, normals, tangents)
        G += assemble_kernel(cd, kernel, weight)

    return G
