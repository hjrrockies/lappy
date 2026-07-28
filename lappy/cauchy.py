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
from .quad import cached_leggauss, cached_kressgauss
from .asymp import weyl_est

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
    `len(cols)` columns, in that order, not the basis's full width. Useful for any caller
    that only needs a subset of the basis at a given node set (e.g. a Hadamard-type
    consumer outside lappy); rellich_gram_basis itself always evaluates every column
    once, since build_boundary_quadrature's single shared node set serves the whole
    basis at once."""
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


def graded_pts_per_seg(domain, basis, lam_max=None, q_seg=None, mult=2, min_per_seg=4,
                       c_lam=1.0, beta=0.2):
    """Per-segment point counts for a graded boundary quadrature, sized from three terms:

    - `mult*len(basis)*seg_len/total_len` (the original basis-size/segment-length proxy;
      mirrors mps.pts_per_seg's generic branch).
    - `c_lam*sqrt(lam_max)*seg_len`, a Nyquist-style oscillation term: boundary Cauchy data
      oscillates at wavenumber ~sqrt(lam) (lam is the eigenvalue itself here, e.g.
      FourierBesselBasis evaluates jv(order, sqrt(lam)*r)), so resolving it needs points per
      unit arclength scaling with sqrt(lam_max) regardless of basis size. `lam_max` defaults
      to `weyl_est(6, domain)` (matching MPSEigensolver.from_domain's own default) when not
      supplied, so callers that don't pass it still get lam-aware sizing. weyl_est doesn't
      support mixed/Robin boundary conditions; for those, the lam term is skipped (not an
      error) when `lam_max` isn't given explicitly, falling back to the basis-size term alone.

    The two terms above are independent proxies for the same underlying "how much does the
    smooth part of the integrand need" quantity, so they're combined by max(), not sum().

    - `q_seg`, if given (per-segment Kress grading order, see corner_grading_orders/
      build_boundary_quadrature), inflates that smooth-content count by `1 + beta*(q-1)`
      for q>0: a higher grading order clusters more of a fixed node budget near the corner,
      so the segment's smooth interior needs proportionally more nodes to stay resolved.

    Every additional term only ever increases the point count relative to the original
    length-only heuristic, so this is a strictly more (or equally) conservative superset of
    the prior behavior for any fixed mult/min_per_seg."""
    seg_lens = domain.seg_lens
    base_n = mult*len(basis)*seg_lens/seg_lens.sum()
    if lam_max is None:
        try:
            lam_max = weyl_est(6, domain)
        except NotImplementedError:
            lam_max = None
    lam_n = c_lam*np.sqrt(lam_max)*seg_lens if lam_max is not None else 0.0
    smooth_n = np.maximum(base_n, lam_n)
    if q_seg is not None:
        smooth_n = smooth_n*np.where(q_seg > 0, 1.0 + beta*(q_seg - 1), 1.0)
    pps = np.round(smooth_n).astype(int)
    return np.maximum(pps, min_per_seg).astype(int)


### Kress-style graded-mesh quadrature, docs/rellich_hadamard_mps.pdf Sec. 6.1.
#
# Rather than grading a separate Gauss-Jacobi rule to each basis-function pair's exact combined
# corner exponent (the earlier SS/SR/RS/RR approach -- exact, but requires one full-basis
# Cauchy-data evaluation per distinct exponent *group* at each corner, which blows up whenever a
# corner's modes have generically-incommensurate exponents), each segment gets a single shared
# quadrature rule -- ordinary Gauss-Legendre, or Kress-graded toward whichever of its two corner
# endpoints are genuinely singular -- and the *raw* (undecomposed) Cauchy data of the whole basis
# is evaluated once at that rule's nodes. This trades an exact per-mode error certificate for an
# empirical one (node-doubling / grading-order-doubling), in exchange for evaluation cost that no
# longer depends on how many singular modes a corner has.

def corner_grading_orders(basis, domain, margin=2.0, q_min=4, q_max=12):
    """Per-corner Kress grading order q (see quad.kress_w), sized from that corner's worst (most
    negative/most singular) admissible exponent via basis.corner_terms(domain). Corners with no
    singular columns there, or whose columns are all entire (nonnegative-integer exponent), get
    q=0 (no grading needed -- plain Gauss-Legendre suffices).

    The quantity actually needing resolution is the *product* of two singular factors at the same
    corner (the old SS block), whose combined exponent for a derivative-type kernel (KNN/KTT/Kcr)
    is 2*(p1 - 1) for leading exponent p1; `margin` pads the grading order comfortably past that,
    per the doc's own qualitative guidance ("q chosen comfortably larger than the worst exponent
    present")."""
    corner_id, exponent = basis.corner_terms(domain)
    n_corners = len(domain.corners)
    q = np.zeros(n_corners, dtype=int)
    for c in range(n_corners):
        Sc = exponent[corner_id == c]
        if len(Sc) == 0:
            continue
        p1 = Sc.min()
        if np.isclose(p1, np.round(p1)) and p1 > -0.5:
            continue  # entire at this corner -- no singularity to grade for
        severity = max(0.0, 2*(1 - p1))
        q[c] = int(np.clip(np.ceil(severity + margin), q_min, q_max))
    return q


def build_boundary_quadrature(domain, basis, lam_max=None, mult=2, min_per_seg=4, margin=2.0,
                              q_min=4, q_max=12, c_lam=1.0, beta=0.2):
    """One shared quadrature rule per segment -- Kress-graded (quad.cached_kressgauss) toward
    either endpoint that basis.corner_terms(domain) marks as genuinely singular, plain
    Gauss-Legendre (quad.cached_leggauss) otherwise -- covering the whole boundary. Point counts
    per segment come from `graded_pts_per_seg`, sized from basis size, segment length, `lam_max`
    (the worst-case spectral parameter this shared node set must stay accurate for -- see
    rellich.build_rellich_data's docstring on why this node set is built once and reused across
    every lam in a solve_interval search) and each segment's own Kress grading order (so a
    heavily-graded segment doesn't starve its own smooth interior of nodes). Returns concatenated
    (pts, normals, tangents, wts, dir_mask, neu_mask) as plain arrays (pts/normals/tangents
    complex, wts/dir_mask/neu_mask real), covering every segment regardless of boundary-condition
    type -- dir_mask/neu_mask are 0/1 float masks over the SAME combined point array, letting a
    single Cauchy-data evaluation serve every boundary-condition split via multiplicative weight
    masking (see rellich.rellich_gram_basis) rather than separate per-condition node sets."""
    segs = domain.bdry.segments
    n_segs = len(segs)
    q_corner = corner_grading_orders(basis, domain, margin, q_min, q_max)

    # q_start[i]: grading order at segment i's p0 corner, if that point is a listed corner.
    q_start = np.zeros(n_segs, dtype=int)
    for c in range(len(domain.corners)):
        q_start[domain.corner_idx[c]] = q_corner[c]

    # q_seg[i]: grading order actually used on segment i (max of its two corner endpoints),
    # computed once so graded_pts_per_seg can size n_per_seg with it before quadrature assembly.
    q_seg_arr = np.maximum(q_start, np.roll(q_start, -1))

    n_per_seg = graded_pts_per_seg(domain, basis, lam_max, q_seg_arr, mult, min_per_seg,
                                   c_lam, beta)

    pts_parts, normals_parts, tangents_parts, wts_parts = [], [], [], []
    dir_parts, neu_parts = [], []
    for i, seg in enumerate(segs):
        q_seg = q_seg_arr[i]
        n = n_per_seg[i]
        if q_seg == 0:
            tau, w = cached_leggauss(n)
        else:
            tau, w = cached_kressgauss(n, q_seg)
        pts_parts.append(seg.p(tau))
        normals_parts.append(seg.N(tau))
        tangents_parts.append(seg.T(tau))
        wts_parts.append(seg.len*w)
        is_dir = float(seg.bc_type == 'dir')
        is_neu = float(seg.bc_type == 'neu')
        dir_parts.append(np.full(n, is_dir))
        neu_parts.append(np.full(n, is_neu))

    pts = np.concatenate(pts_parts)
    normals = np.concatenate(normals_parts)
    tangents = np.concatenate(tangents_parts)
    wts = np.concatenate(wts_parts)
    dir_mask = np.concatenate(dir_parts)
    neu_mask = np.concatenate(neu_parts)
    return pts, normals, tangents, wts, dir_mask, neu_mask
