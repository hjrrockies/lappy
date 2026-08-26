from .core import BaseSegment, BaseDomain
from .utils import (polygon_area, polygon_diameter, complex_form, real_form, rand_interior_points,
                    interior_angles, edge_lengths, as_generator, rand_uniform)
from shapely.geometry import Polygon as ShapelyPolygon
from shapely import points as shapely_points
from .opt import find_all_roots
from .quad import (spline_mesh_with_curvature, polygon_triangular_mesh,
                   tri_quad, cached_leggauss, cached_chebgauss, jacgauss)

from typing import Callable
import copy
import numpy as np
import warnings
from scipy.interpolate import make_interp_spline, BSpline, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar, linprog
import matplotlib.pyplot as plt
from pygmsh.geo import Geometry

def segment_intersection(a0, a1, b0, b1):
    """Computes the intersection of two line segments, if it exists."""
    # segment vectors
    d1 = a1 - a0
    d2 = b1 - b0

    # Cross product in 2D: d1 × d2
    cross = d1.real * d2.imag - d1.imag * d2.real

    # Solve for parameters s and t
    delta = b0 - a0
    if cross == 0:
        s,t = np.inf, np.inf
    else:
        s = (delta.real * d2.imag - delta.imag * d2.real) / cross
        t = (delta.real * d1.imag - delta.imag * d1.real) / cross

    # Check if intersection is within both segments
    if 0 <= s <= 1 and 0 <= t <= 1:
        return a0 + s*d1

    else: return None

class PointSet:
    """An immutable, hashable set of points in the plane. Exists to be a cache key.

    Numpy arrays are unhashable, so basis evaluations keyed on "these points at this lambda" need
    a wrapper. That is this class's whole job.

    HASHING IS BY VALUE, and that is load-bearing rather than cosmetic. Keying by identity is what
    the class did before, and it silently failed wherever a caller re-wrapped the same nodes:
    `eigfun_cauchy_data` builds a fresh PointSet per call, so a `douse` gradient over K
    eigenvalues produced K distinct keys for one quadrature. Measured on a hexagon at n_basis=186,
    the geometry caches held `2 + 4K` entries against a design intent of 2 -- 26 entries and
    139 MB at K=6, on ONE solver, and none of it evictable.

    Value hashing costs nothing on the path that matters. CPython compares key pointers before
    calling `__eq__`, so handing back the same object never reaches the array comparison; and the
    hash itself is computed once here, not per lookup. Measured: `hash(pts.tobytes())` is 2.9 us
    at n=1000 and 57 us at n=20000, against 1.2 ms and 24 ms for a single elementwise pass over
    the `(n, 300)` matrix being cached.

    IMMUTABILITY IS WHAT MAKES THE HASH SAFE. `complex_form(...).flatten()` always copies, so
    `pts` never aliases the caller's array, and `writeable=False` blocks mutation through this
    reference. Do not add a setter, and do not store a view of someone else's buffer.

    Weights used to live here too, for a quadrature-weighted least-squares variant of the pencil.
    That is gone: orthonormalization goes through the boundary-only Rellich identity in
    `eigfun_integrals`, and weights belong to the quadrature rule that owns them (`BoundaryQuad`
    carries its own `wts`), not to a set of points.
    """
    def __init__(self, points):
        self.pts = complex_form(points).flatten()
        self.pts.flags.writeable = False
        self._hash = hash(self.pts.tobytes())

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, PointSet):
            return NotImplemented
        return self._hash == other._hash and np.array_equal(self.pts, other.pts)

    @property
    def x(self):
        return self.pts.real

    @property
    def y(self):
        return self.pts.imag

    def __len__(self):
        return len(self.pts)
    
    def __str__(self):
        return f"PointSet(size={len(self.pts)})"
    
    def __add__(self, other):
        if not isinstance(other, PointSet):
            raise TypeError("'other' must be an instance of PointSet")
        return PointSet(np.concatenate((self.pts, other.pts)))

def pts_per_seg(domain, fb_basis, mult=2, min_per_seg=0):
    """Computes how many boundary points to have along each segment of a domain boundary so that each
    corner's basis has enough points on non-adjacent edges."""

    # get the number of basis functions associated to each corner of the domain
    n_basis = np.zeros(len(domain.bdry.segments), dtype='int')
    p0 = np.array([seg.p0 for seg in domain.bdry.segments])
    has_basis = np.any(np.isclose(np.subtract.outer(p0, fb_basis.sources), 0), axis=1)
    n_basis[has_basis] = fb_basis.orders

    # get the adjacent edge lengths to ith vertex into the first and last positions of column i, then drop rows
    seg_lengths = np.array([seg.len for seg in domain.bdry.segments])
    rolled_lengths = np.array([np.roll(seg_lengths, -j) for j in range(len(seg_lengths))])[1:-1]
    # normalize each column by sum of edge lengths
    normalized_lengths = rolled_lengths/rolled_lengths.sum(axis=0)
    # multiply by orders, take ceiling
    n_per_seg = np.ceil(mult*n_basis*normalized_lengths)
    # unroll and sum
    n_per_seg = np.array([np.roll(n_per_seg[i], i+1) for i in range(len(n_per_seg))]).sum(axis=0)
    # threshold with min_per_seg
    n_per_seg = np.maximum(n_per_seg, min_per_seg).astype('int')
    return n_per_seg
    
# segment classes
# --- Adaptive curve sampling -------------------------------------------------
# Gauss-Legendre nodes/weights on [-1, 1] for the 5-point rule used by the
# adaptive arc-length quadrature below.
_GL5_X = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                   0.5384693101056831, 0.9061798459386640])
_GL5_W = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                   0.4786286704993665, 0.2369268850561891])


def _estimate_length(p, t0, t1, n_init=64):
    """Cheap chord-sum estimate of the arc length of the complex curve ``p``."""
    t = np.linspace(t0, t1, n_init + 1)
    return np.sum(np.abs(np.diff(p(t))))


def _gl5(speed, lefts, rights):
    """5-point Gauss-Legendre arc-length integral of ``speed`` on each
    ``[lefts[i], rights[i]]`` (vectorized over intervals)."""
    hl = 0.5 * (rights - lefts)
    centers = 0.5 * (lefts + rights)
    nodes = centers[:, None] + hl[:, None] * _GL5_X[None, :]
    return hl * (speed(nodes.ravel()).reshape(-1, 5) @ _GL5_W)


def _lagrange4(xs, ys, xq):
    """Evaluate the cubic through the four points (xs[k], ys[k]) at ``xq``.
    All arguments are (4, N) / (N,) arrays; vectorized over the N intervals."""
    out = np.zeros_like(xq)
    for i in range(4):
        term = ys[i].astype(float).copy()
        for j in range(4):
            if j != i:
                term = term * (xq - xs[j]) / (xs[i] - xs[j])
        out = out + term
    return out


# Arc-length reparametrization constants -- see ParametricSegment._reparameterize.
# Calibrated in benchmarks/arclength/vectorized.py. Newton reaches 1.8e-16 in two steps.
# _REMAINDER_ORDER spans a fraction of one table panel, so it is insensitive -- order 6 is
# indistinguishable from 24. _ANCHOR_ORDER integrates a WHOLE table panel and is not: at 8 the
# total length still moved by 8e-11 between tol=1e-3 and tol=1e-6, which would leave a residual
# tol-dependence the solve is meant to remove; 12 brings that to 5e-15 for no measurable cost.
_NEWTON_MAX_ITERS = 60   # safeguarded: exits at _NEWTON_STOL, typically after 2-3
_NEWTON_STOL = 1e-15     # relative to the segment length
_REMAINDER_ORDER = 6
_ANCHOR_ORDER = 12


def adaptive_arclength_table(speed, t0, t1, eps, eps_abs, max_depth=50):
    """Build an adaptive arc-length table for a curve with the given ``speed``.

    The table is used to build the inverse map ``t(s)`` by monotone cubic (PCHIP)
    interpolation, so the refinement criterion must resolve *that inverse map* --
    not merely the value of the arc-length integral. A quadrature-error test fails
    here: a linear/low-degree speed is integrated exactly by Gauss-Legendre with
    no subdivision, yet its cumulative ``s(t)`` is curved and needs many nodes to
    invert (a zero-speed cusp, where ``t(s) ~ sqrt(s)``, is the extreme case). A
    piecewise-*linear* inverse test, on the other hand, over-refines every curved
    region by orders of magnitude relative to what the cubic interpolant needs.

    So each interval ``[l, r]`` is split into quarters and the arc length of each
    quarter is computed with a 5-point Gauss-Legendre rule, giving five inverse
    samples ``(s_k, t_k)``. The interval is accepted when both

    * the cubic through the four points ``(s_k, t_k)`` for ``k != 2`` predicts the
      arc-length-midpoint parameter to ``< eps * (t1 - t0)`` -- i.e. a cubic
      already represents ``t(s)`` here, matching the PCHIP interpolant, and
    * ``|G - G_full| / (G_full + eps_abs) < eps`` -- the composite (4x5-point) and
      single (5-point) quadratures agree, bounding the arc-length error,

    where ``G`` is the total arc length of the interval. Accepted intervals
    contribute all three interior quarter points as nodes, so ``t(s)`` is sampled
    wherever it curves. The ``eps_abs`` guard keeps the quadrature test
    well-behaved as a subinterval's arc length shrinks toward zero near a cusp.

    Parameters
    ----------
    speed : callable (t_array,) -> (N,)
        Real-valued speed ``|p'(t)|`` of the curve.
    t0, t1 : float
        Parameter interval.
    eps : float
        Relative tolerance.
    eps_abs : float
        Absolute tolerance (``eps * L_estimate``).
    max_depth : int
        Hard subdivision cap.

    Returns
    -------
    t_nodes : (M,) array  -- sorted parameter values
    s_nodes : (M,) array  -- cumulative arc lengths at ``t_nodes``
    """
    span = float(t1 - t0)
    lefts = np.array([float(t0)])
    rights = np.array([float(t1)])
    depths = np.array([0], dtype=int)

    accepted_lefts, accepted_rights, accepted_ds = [], [], []

    while len(lefts):
        h = rights - lefts
        q1 = lefts + 0.25 * h
        q2 = lefts + 0.50 * h            # arc-length test point (parameter midpoint)
        q3 = lefts + 0.75 * h

        g0 = _gl5(speed, lefts, q1)
        g1 = _gl5(speed, q1, q2)
        g2 = _gl5(speed, q2, q3)
        g3 = _gl5(speed, q3, rights)
        # cumulative arc length from the left endpoint at [l, q1, q2, q3, r]
        s0 = np.zeros_like(lefts)
        s1 = g0
        s2 = g0 + g1
        s3 = g0 + g1 + g2
        s4 = s3 + g3
        G = s4
        G_full = _gl5(speed, lefts, rights)

        # cubic through the four points off the midpoint, predicting t at s2
        t_pred = _lagrange4([s0, s1, s3, s4], [lefts, q1, q3, rights], s2)
        err_inv = np.abs(t_pred - q2) / span
        err_quad = np.abs(G - G_full) / (G_full + eps_abs)
        ok = ((err_inv < eps) & (err_quad < eps)) | (depths >= max_depth)

        if ok.any():
            # record all four quarter pieces so the interior points become nodes
            for a, b, ds in ((lefts, q1, g0), (q1, q2, g1),
                             (q2, q3, g2), (q3, rights, g3)):
                accepted_lefts.append(a[ok])
                accepted_rights.append(b[ok])
                accepted_ds.append(ds[ok])

        if (~ok).any():
            l_sub, r_sub = lefts[~ok], rights[~ok]
            d_sub = depths[~ok] + 1
            m_sub = q2[~ok]
            lefts = np.concatenate([l_sub, m_sub])
            rights = np.concatenate([m_sub, r_sub])
            depths = np.concatenate([d_sub, d_sub])
        else:
            break

    all_lefts = np.concatenate(accepted_lefts)
    all_rights = np.concatenate(accepted_rights)
    all_ds = np.concatenate(accepted_ds)

    order = np.argsort(all_lefts)
    t_nodes = np.concatenate([[all_lefts[order[0]]], all_rights[order]])
    s_nodes = np.concatenate([[0.0], np.cumsum(all_ds[order])])
    return t_nodes, s_nodes


def adaptive_polyline(p, t0, t1, eps_abs, L=None, max_depth=60, chord_factor=2.0):
    """Adaptively sample the complex curve ``p`` so the resulting polyline stays
    within ``eps_abs`` of the curve.

    A chord is accepted when the midpoint and tercile deviations are all within
    ``eps_abs`` and the chord length is below ``chord_factor * sqrt(eps_abs * L)``
    (the latter guards near-inflection/aliased regions where the 3-point
    deviation test is accidentally fooled despite a long chord -- e.g. a chord
    whose endpoints and tercile points all happen to land on the curve while it
    bulges out in between). The guard is scaled by ``sqrt(eps_abs * L)`` rather
    than ``eps_abs`` directly because the legitimate (non-aliased) chord length
    needed to hit a given deviation tolerance is itself ``O(sqrt(eps_abs * R))``
    by the usual sagitta estimate (``R`` = local radius of curvature, bounded
    here by the curve's total length ``L``); a guard linear in ``eps_abs``
    shrinks faster than that as ``eps_abs -> 0`` and becomes the sole binding
    constraint, forcing uniform bisection far beyond what curvature requires.
    ``L`` defaults to a cheap chord-sum estimate when not supplied. Endpoint
    evaluations are threaded through the loop so a midpoint becomes a child
    endpoint with no re-eval.

    Returns
    -------
    t_sample : (M,) sorted array of parameter values.
    """
    if L is None:
        L = _estimate_length(p, t0, t1)
    chord_cap = chord_factor * np.sqrt(eps_abs * L)

    lefts = np.array([float(t0)])
    rights = np.array([float(t1)])
    depths = np.array([0], dtype=int)

    p_left = p(lefts)
    p_right = p(rights)

    accepted_t = [np.array([float(t0)])]

    while len(lefts):
        mids = 0.5 * (lefts + rights)
        thirds = lefts + (rights - lefts) / 3.0
        twothirds = lefts + 2.0 * (rights - lefts) / 3.0

        N = len(lefts)
        pts_all = p(np.concatenate([mids, thirds, twothirds]))
        p_mid = pts_all[:N]
        p_3rd = pts_all[N:2 * N]
        p_23rd = pts_all[2 * N:]

        dev_mid = np.abs(p_mid - 0.5 * (p_left + p_right))
        dev_3rd = np.abs(p_3rd - (p_left + (p_right - p_left) / 3.0))
        dev_23rd = np.abs(p_23rd - (p_left + 2.0 * (p_right - p_left) / 3.0))
        chord_len = np.abs(p_right - p_left)

        ok = (
            (dev_mid <= eps_abs)
            & (dev_3rd <= eps_abs)
            & (dev_23rd <= eps_abs)
            & (chord_len <= chord_cap)
        ) | (depths >= max_depth)

        if ok.any():
            accepted_t.append(rights[ok])

        if (~ok).any():
            l_sub, r_sub = lefts[~ok], rights[~ok]
            d_sub = depths[~ok] + 1
            m_sub = mids[~ok]
            pl_sub, pm_sub, pr_sub = p_left[~ok], p_mid[~ok], p_right[~ok]

            lefts = np.concatenate([l_sub, m_sub])
            rights = np.concatenate([m_sub, r_sub])
            depths = np.concatenate([d_sub, d_sub])
            p_left = np.concatenate([pl_sub, pm_sub])
            p_right = np.concatenate([pm_sub, pr_sub])
        else:
            break

    return np.sort(np.concatenate(accepted_t))


def get_quadfunc(kind, **kwargs):
    if kind == 'legendre': return cached_leggauss
    elif kind == 'chebyshev': return cached_chebgauss
    elif kind == 'even': return lambda n: (np.linspace(0, 1, n+2)[1:-1], np.ones(n)/n)
    elif kind == 'jacobi': return lambda n: jacgauss(n, **kwargs)
    else: raise NotImplementedError(f"quadrature rule {kind} is not implemented")

class SegmentQuadratureMixin:
    """Provides pts/tangents/normals for any segment exposing p(tau), T(tau), N(tau), and len."""
    def pts(self, n, kind='legendre', **kwargs):
        """Gets n points spaced along the segment."""
        quadfunc = get_quadfunc(kind, **kwargs)
        tau, _wts = quadfunc(n)
        return PointSet(self.p(tau))

    def tangents(self, n, kind='legendre', **kwargs):
        """Gets the unit tangent vectors for n points spaced along the segment."""
        quadfunc = get_quadfunc(kind, **kwargs)
        tau, _wts = quadfunc(n)
        return PointSet(self.T(tau))

    def normals(self, n, kind='legendre', **kwargs):
        """Gets the unit outward normal vectors for n points spaced along the segment."""
        quadfunc = get_quadfunc(kind, **kwargs)
        tau, _wts = quadfunc(n)
        return PointSet(self.N(tau))

class ParametricSegment(SegmentQuadratureMixin, BaseSegment):
    """Class for boundary segments (lines, curves) given in terms of a differentiable function p(t). 
    Handles boundary point placement, boundary tangents and normals.
    All segments are automatically re-parameterized by tau in [0,1]"""
    def __init__(self, p, dp, t0, tf, bc='dir', tol=1e-4, val_simple=False, val_closed=False):
        super().__init__(bc)
        if tf <= t0:
            raise ValueError(f"tf ({tf}) must be greater than t0 ({t0})")
        if tol <= 0:
            raise ValueError(f"tol ({tol}) must be positive")

        # convert to complex vectorized form and store
        self.t0 = t0
        self.tf = tf
        self._p = self._complex_vectorize(p, t0, tf)
        self._dp = self._complex_vectorize(dp, t0, tf)
        self._len = None
        self._speed = lambda t: np.abs(self._dp(t))
        self.tol = tol

        # validation for simple and closed curve properties
        if val_simple:
            if not self._validate_simple():
                raise ValueError("segment be a simple curve (set val_simple=False to skip validation)")
            else: self._is_simple = True
        else:
            self._is_simple = None

        if val_closed:
            if not self._validate_closed():
                raise ValueError("segment must be a closed curve (set val_closed=False to skip validation)")
            else: self._is_closed = True
        else:
            self._is_closed = None

    def __str__(self):
        return f"ParametricSegment({self.p0},{self.pf})"
    
    def to_splineseg(self, spline_bc_type='natural'):
        """Returns a SplineSegment with the same geometry as this segment"""
        pts = self.polyline_pts
        return SplineSegment.interp_from_pts(pts, self.bc, spline_bc_type, self.tol)

    def _ensure_reparam(self):
        """Lazily runs the adaptive arc-length reparameterization + polyline
        sampling on first access of a quantity that depends on it."""
        if self._len is None:
            self._reparameterize()

    @property
    def polyline_tau(self):
        """Adaptively chosen parameter values (in [0,1]) whose chords approximate
        the curve to the segment's tolerance."""
        self._ensure_reparam()
        return self._poly_tau

    @property
    def polyline_pts(self):
        """Curve points at the adaptive polyline nodes."""
        return self.p(self.polyline_tau)
    
    @property
    def is_simple(self):
        if self._is_simple is None:
            self._is_simple = self._validate_simple()
        return self._is_simple

    def _validate_simple(self):
        p = self.polyline_pts
        for i in range(len(p)-1):
            for j in range(i+2,len(p)-1):
                pt = segment_intersection(p[i], p[i+1], p[j], p[j+1])
                if pt is not None and not np.isclose(pt, self.p0) and not np.isclose(pt, self.pf):
                    return False
        return True
    
    @property
    def is_closed(self):
        if self._is_closed is None:
            self._is_closed = self._validate_closed()
        return self._is_closed

    def _validate_closed(self):
        return np.isclose(self._p(self.t0), self._p(self.tf))
    
    @property
    def len(self):
        # lazily set from the adaptive arc-length table in _reparameterize()
        self._ensure_reparam()
        return self._len

    @property
    def p0(self):
        return self._p(self.t0)
    
    @property
    def pf(self):
        return self._p(self.tf)
    
    @property
    def T0(self):
        """Unit tangent vector at initial point"""
        return self._T(self.t0)
    
    @property
    def Tf(self):
        """Unit tangent vector at final point"""
        return self._T(self.tf)

    @staticmethod
    def _complex_vectorize(f: Callable, t0, tf) -> Callable:
        """
        Convert a scalar-argument function f to a vectorized complex-valued function.
        
        Accepts f that returns any of:
        - a complex scalar
        - a real scalar (passed through as-is)
        - a tuple/list/array of two reals (interpreted as real + imag)
        
        Returns a function that accepts scalar or array input and always returns
        a complex numpy array (or complex scalar for scalar input).
        """
        def _to_complex(val):
            val = np.asarray(val)
            if val.shape == (2,):
                return val[0] + 1j * val[1]
            scalar = val.item()
            return complex(scalar)

        def wrapped(t):
            t = np.asarray(t)
            scalar_input = t.ndim == 0
            t = np.atleast_1d(t)
            result = np.array([_to_complex(f(ti)) for ti in t], dtype=complex)
            return result[0] if scalar_input else result

        def wrapped_warned(t):
            warnings.warn(
                "ParametricSegment is falling back to a per-point Python loop: the supplied "
                "p/dp did not return a complex array of matching shape when probed. This is "
                "~130x slower than a vectorized callable and usually means the segment was "
                "built wrong -- most commonly by fitting a spline to COMPLEX points with "
                "scipy's make_interp_spline(..., bc_type='periodic'), which silently discards "
                "the imaginary part (ComplexWarning) and leaves a real-valued, degenerate "
                "curve. Fit to a real (n, 2) array instead; SplineSegment converts it.",
                RuntimeWarning, stacklevel=2)
            return wrapped(t)

        # Probe f to check if it's already vectorized and returns complex
        try:
            probe = np.linspace(t0, tf, 3)
            raw = f(probe)
            raw = np.asarray(raw)
            if raw.shape == probe.shape and np.iscomplexobj(raw):
                # Already vectorized and complex — return as-is
                return f
        except Exception:
            pass  # Fall through to wrapping

        return wrapped_warned
    
    def _reparameterize(self):
        """Arc-length reparametrization: an adaptive table for bracketing, then an exact solve.

        The table (`adaptive_arclength_table`) supplies bracketing nodes and a monotone cubic
        initial guess ONLY. Its arclengths are recomputed here to machine precision, and
        `_t_of_s` then *solves* s(t) = s by Newton rather than interpolating it.

        Interpolating the inverse was the old approach and it cost eight orders on any curve
        whose speed varies. Three separate defects, all removed together:

        * `t(s)` as a piecewise cubic makes `p(tau)` only C^1 in tau, so `f(p(tau))` is C^1 and
          Gauss-Legendre converges algebraically. Boundary integrals on an ellipse stalled at
          ~7e-5 *regardless of quadrature order* -- 32 nodes and 256 nodes gave the same answer.
          With the solve they reach 1.4e-13 at order 128, spectrally.
        * `dp` differentiated that interpolant, losing another order: |dp/dtau| - len was
          8.6e-3. Here `t'(s) = 1/|p'(t)|` analytically, so the constant-speed property the
          whole quadrature rests on holds by construction (3.7e-16).
        * accuracy depended on `tol`, and tightening it was ruinously slow (>30 s at 1e-8, for
          an error still only 6e-4). Now the table is a bracket, so `tol` sets the quality of an
          initial guess and nothing else.

        Newton's quadratic convergence is visible: one step reaches 4e-9, two 1.8e-16.
        `_NEWTON_ITERS = 3` is margin. `_REMAINDER_ORDER` integrates a span that is a fraction
        of one table panel, so it is insensitive -- order 6 is indistinguishable from 24.
        Measured cost is ~0.11 ms to build plus ~300 ns per point, i.e. under 5% of a typical
        boundary-quadrature build, and paid once per solve.
        See benchmarks/arclength/ and docs/eigfun_integrals.md.
        """
        L0 = _estimate_length(self._p, self.t0, self.tf)
        eps_abs = self.tol * L0
        t_nodes, _ = adaptive_arclength_table(self._speed, self.t0, self.tf,
                                              self.tol, eps_abs)
        # Force table nodes at the segment's own derivative breaks (spline knots). Without
        # this, an anchor or remainder Gauss rule integrates |p'| ACROSS a break and the
        # arc-length map itself is only ~1e-7 accurate, capping everything downstream.
        brk = np.asarray(self.break_ts, dtype=float)
        if len(brk):
            t_nodes = np.unique(np.concatenate([np.asarray(t_nodes, dtype=float), brk]))
        self._t_nodes = np.asarray(t_nodes, dtype=float)

        # machine-precision arclengths at the table's own nodes
        x, w = cached_leggauss(_ANCHOR_ORDER)          # nodes on [0,1], weights summing to 1
        lo, hi = self._t_nodes[:-1], self._t_nodes[1:]
        span = hi - lo
        vals = self._speed((lo[:, None] + span[:, None]*x[None, :]).ravel())
        self._s_nodes = np.concatenate([[0.0], np.cumsum(span*(vals.reshape(-1, len(x)) @ w))])
        self._len = float(self._s_nodes[-1])

        # monotone cubic initial guess for the Newton solve (never used as the answer)
        self._t_of_s_guess = PchipInterpolator(self._s_nodes, self._t_nodes)

        # Adaptive polyline nodes (in tau-space, on the constant-speed curve). Deliberately
        # driven by the CHEAP guess map, not the Newton solve: adaptive_polyline makes very
        # many small scalar-ish calls while recursing, and the exact map's per-call overhead
        # turns that into minutes. It only selects WHERE the nodes go, to a tolerance of
        # `tol` -- `polyline_pts` then evaluates them through the exact `self.p`, so nothing
        # downstream inherits the guess's error.
        def _p_guess(tau):
            return self._p(self._t_of_s_guess(self._len*np.asarray(tau)))

        self._poly_tau = adaptive_polyline(_p_guess, 0.0, 1.0,
                                           eps_abs=self.tol * self._len, L=self._len)

    @property
    def break_taus(self):
        """`break_ts` mapped into arc-length parameter. See BaseSegment.break_taus."""
        brk = np.asarray(self.break_ts, dtype=float)
        if not len(brk):
            return np.empty(0)
        self._ensure_reparam()
        taus = np.asarray(self._s_of_t(brk))/self._len
        return np.unique(taus[(taus > 1e-12) & (taus < 1 - 1e-12)])

    def _s_of_t(self, t):
        """Exact cumulative arc length from `t0`: anchor at the bracketing table node and
        integrate the remainder with a fixed Gauss rule over that short span."""
        self._ensure_reparam()
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i = np.clip(np.searchsorted(self._t_nodes, t, side='right') - 1,
                    0, len(self._t_nodes) - 2)
        lo = self._t_nodes[i]
        span = t - lo
        x, w = cached_leggauss(_REMAINDER_ORDER)
        vals = self._speed((lo[:, None] + span[:, None]*x[None, :]).ravel())
        return self._s_nodes[i] + span*(vals.reshape(-1, len(x)) @ w)

    def _t_of_s(self, s):
        """Inverse arc length: safeguarded Newton on `_s_of_t(t) - s`, with ds/dt = |p'(t)|.

        Plain Newton is enough almost everywhere and reaches 1.8e-16 in two steps. It is NOT
        enough at a near-cusp, where |p'| -> 0: the step divides by a vanishing derivative and
        overshoots, and a spline interpolated through points that align with a symmetry of the
        curve produces exactly that (measured: round-trip 2.3e-9 there against 2.0e-16 once the
        cusp is perturbed away). `adaptive_arclength_table` already anticipates the case -- its
        docstring calls a zero-speed cusp "the extreme case" -- so the solve must too.

        The bracket comes free: `s` is monotone in `t`, so the table interval containing `s`
        brackets the root. Each iteration takes the Newton step when it lands inside the current
        bracket and the derivative is usable, and bisects otherwise, tightening the bracket
        either way. Well-conditioned points converge in 2-3 iterations and the loop exits; only
        the pathological handful cost more.
        """
        self._ensure_reparam()
        scalar = np.ndim(s) == 0
        s = np.atleast_1d(np.asarray(s, dtype=float)).astype(float)
        i = np.clip(np.searchsorted(self._s_nodes, s, side='right') - 1,
                    0, len(self._s_nodes) - 2)
        lo, hi = self._t_nodes[i].copy(), self._t_nodes[i + 1].copy()
        t = np.clip(np.asarray(self._t_of_s_guess(s), dtype=float), lo, hi)

        # Iterate on an ACTIVE SET. Without it a couple of stragglers near a cusp drag the
        # whole vector through every iteration -- measured at 62 ms per 1000 points against
        # 0.4 ms once converged points are dropped.
        act = np.arange(len(s))
        tol_abs = _NEWTON_STOL*max(self._len, 1.0)
        for _ in range(_NEWTON_MAX_ITERS):
            if not len(act):
                break
            ta, sa = t[act], s[act]
            f = self._s_of_t(ta) - sa
            lo_a = np.where(f <= 0, ta, lo[act])
            hi_a = np.where(f > 0, ta, hi[act])
            lo[act], hi[act] = lo_a, hi_a
            d = self._speed(ta)
            with np.errstate(divide='ignore', invalid='ignore'):
                cand = ta - f/d
            ok = np.isfinite(cand) & (cand > lo_a) & (cand < hi_a)
            step = np.where(ok, cand, 0.5*(lo_a + hi_a))
            # `f` was evaluated at the CURRENT ta, so a point already within tolerance is done
            # and must keep that ta -- stepping it once more (possibly to a bisection midpoint)
            # before dropping it moves it back off the root. That bug cost four orders.
            conv = np.abs(f) <= tol_abs
            t[act] = np.where(conv, ta, step)
            keep = (~conv) & ((hi_a - lo_a) > 4*np.spacing(np.abs(ta) + 1.0))
            act = act[keep]
        return t[0] if scalar else t

    def _p_of_s(self, s):
        return self._p(self._t_of_s(s))

    def p(self, tau):
        return self._p_of_s(self.len*np.asarray(tau))

    def _dp_of_s(self, s):
        """Unit tangent: t'(s) = 1/|p'(t)| analytically, never a differentiated interpolant."""
        t = self._t_of_s(s)
        return self._dp(t)/self._speed(t)

    def dp(self, tau):
        return self._dp_of_s(self.len*np.asarray(tau))*self.len
    
    def _T(self, t):
        """Unit tangent vector in terms of t"""
        t = np.asarray(t)
        num = self._dp(t)
        denom = self._speed(t)
        out = np.divide(num, denom, out=np.full(num.shape, np.nan, dtype='complex128'), where=(denom!=0))
        t_ = t[denom==0]
        h = 1e-8
        ddp = (self._dp(t_+h)-self._dp(t_-h))/(2*h)
        out[denom==0] = ddp/np.abs(ddp)
        return out
    
    def _T_of_s(self, s):
        """Unit tangent vector in terms of arclength s"""
        return self._T(self._t_of_s(s))
    
    def T(self, tau):
        """Unit tangent vector in terms of tau from [0,1]"""
        return self._T_of_s(self.len*tau)
    
    def _N(self, t):
        """Unit outward normal vector in terms of t"""
        t = np.asarray(t)
        T = self._T(t)
        return T.imag - 1j*T.real
    
    def _N_of_s(self, s):
        """Unit outward normal vector in terms of arclength s"""
        return self._N(self._t_of_s(s))
    
    def N(self, tau):
        """Unit outward normal vector (unit tangent rotated 90deg clockwise) in terms of tau from [0,1]"""
        return self._N_of_s(self.len*tau)

    def intersection(self, other):
        if self is other:
            return self
        elif isinstance(other, LineSegment):
            return other.intersection(self)
        elif isinstance(other, ParametricSegment):
            # get points
            tau1 = self.polyline_tau
            tau2 = other.polyline_tau
            p1 = self.p(tau1)
            p2 = other.p(tau2)

            # distance func
            def dist(tau):
                p1, p2 = self.p(tau[0]), other.p(tau[1])
                d = p1 - p2
                out = np.abs(d)
                
                grad = np.zeros(2)
                dp1, dp2 = self.dp(tau[0]), other.dp(tau[1])
                grad[0] = (d.real*dp1.real + d.imag*dp1.imag)/(out+1e-16)
                grad[1] = -(d.real*dp2.real + d.imag*dp2.imag)/(out+1e-16)
                
                return out, grad

            # loop over discretized segments
            intersections = []
            for i in range(len(p1)-1):
                for j in range(len(p2)-1):
                    p = segment_intersection(p1[i], p1[i+1], p2[j], p2[j+1])
                    if p is not None:
                        tau0 = 0.5*np.array([tau1[i+1]+tau1[i],tau2[j+1]+tau2[j]])
                        res = minimize(dist, tau0, jac=True, bounds=[(tau1[i], tau1[i+1]),(tau2[j], tau2[j+1])])
                        if res.success:
                            intersections.append(self.p(res.x[0]))
            return np.array(intersections)

    def intersects(self, other):
        """Cheap boolean test for whether this segment crosses another. Stops at the
        first polyline crossing, skipping the point-localization (and the per-crossing
        optimization) done by ``intersection``."""
        if self is other:
            return True
        elif isinstance(other, LineSegment):
            return other.intersects(self)
        elif isinstance(other, ParametricSegment):
            p1 = self.polyline_pts
            p2 = other.polyline_pts
            return any(segment_intersection(p1[i], p1[i+1], p2[j], p2[j+1]) is not None
                       for i in range(len(p1)-1) for j in range(len(p2)-1))

    def dist(self, pt):
        """Computes the (minimum) distance from the given point to the segment"""
        tau = self.polyline_tau
        pts = self.p(tau)
        dists = np.abs(pts-pt)
        idx = dists.argmin()

        if idx == 0:
            tau0 = tau[idx]
            tau1 = tau[idx+1]
        elif idx == len(tau)-1:
            tau0 = tau[idx-1]
            tau1 = tau[idx]
        else:
            tau0 = tau[idx-1]
            tau1 = tau[idx+1]

        def f(tau):
            out = np.abs(self.p(tau)-pt)
            return out

        res = minimize_scalar(f, bounds=(tau0, tau1), options={'xatol':1e-14})
        return res.fun
    
    def __mul__(self, other):
        if not np.isscalar(other):
            raise ValueError("non-segment operand must be a scalar")
        p_new = lambda t: other*self._p(t)
        dp_new = lambda t: other*self._dp(t)
        return ParametricSegment(p_new, dp_new, self.t0, self.tf, self.bc, self.tol)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, BaseSegment):
            return MultiSegment([self, other])
        elif np.isscalar(other):
            p_new = lambda t: self._p(t) + other
            dp_new = self._dp
            return ParametricSegment(p_new, dp_new, self.t0, self.tf, self.bc, self.tol)
        else:
            raise TypeError("__add__ with Segment must be another Segment or complex scalar")
            
class LineSegment(SegmentQuadratureMixin, BaseSegment):
    """Class for straight line segments.
    Parameters
    ----------
    a : complex
        initial point of segment
    b : complex
        end point of segment
    """
    def __init__(self, p0, pf, bc='dir', tol=1e-4):
        if np.isclose(p0, pf):
            raise ValueError("'p0' and 'pf' are too close together to form a line segment")
        super().__init__(bc)
        self._p0 = complex_form(p0)
        self._pf = complex_form(pf)
        self._len = np.abs(self._pf-self._p0)
        self._tangent = (self._pf-self._p0)/self._len
        self._normal = self._tangent.imag - 1j*self._tangent.real
        self.tol = tol

    @property
    def polyline_tau(self):
        """A straight segment is exactly its two endpoints."""
        return np.array([0.0, 1.0])

    @property
    def polyline_pts(self):
        return np.array([self._p0, self._pf])

    def __str__(self):
        return f"LineSegment({self.p0},{self.pf})"
    
    @property
    def is_simple(self): 
        return True

    @property
    def is_closed(self): 
        return False

    @property
    def len(self):
        return self._len

    def p(self, tau):
        """Gets points along line segment"""
        return (1-tau)*self._p0 + tau*self._pf
    
    def dp(self, tau):
        """Gets derivatives along line segment"""
        return (self._pf - self._p0)*np.ones_like(tau)
    
    def T(self, tau):
        """Gets unit tangents along line segment"""
        tau = np.asarray(tau)
        return np.full_like(tau, self._tangent, dtype='complex')
    
    def N(self, tau):
        """Gets unit outward normal along line segment"""
        tau = np.asarray(tau)
        return np.full_like(tau, self._normal, dtype='complex')
    
    def intersection(self, other):
        """Finds the point(s) of intersection between a LineSegment and another Segment"""
        if self is other:
            return self
        # two LineSegments
        elif isinstance(other, LineSegment):
            p = segment_intersection(self.p0, self.pf, other.p0, other.pf)
            if p is not None:
                return np.array([p])
            else: return np.array([])
                
        # other is a Segment
        elif isinstance(other, ParametricSegment):
            d = self.pf - self.p0
            def signed_distance(tau):
                p = other.p(tau)
                d = p - self.p0
                return d.real*self._normal.real + d.imag*self._normal.imag
            roots = find_all_roots(signed_distance, 0, 1, len(other.polyline_tau))
            intersections = []
            for tau1 in roots:
                p = other.p(tau1)
                tau2 = ((p - self.p0) / d).real
                if 0 <= tau2 <= 1:
                    intersections.append(p)
            return np.array(intersections)

    def intersects(self, other):
        """Cheap boolean test for whether this segment crosses another, without
        localizing the intersection point(s) done by ``intersection``."""
        if self is other:
            return True
        elif isinstance(other, LineSegment):
            return segment_intersection(self.p0, self.pf, other.p0, other.pf) is not None
        elif isinstance(other, ParametricSegment):
            q = other.polyline_pts
            return any(segment_intersection(self.p0, self.pf, q[k], q[k+1]) is not None
                       for k in range(len(q) - 1))

    def to_splineseg(self, spline_bc_type='natural'):
        """Returns a SplineSegment with the same geometry as this segment"""
        pts = np.array([self.p0, self.pf])
        return SplineSegment.interp_from_pts(pts, self.bc, spline_bc_type, self.tol)
    
    @property
    def p0(self):
        return self._p0
    
    @property
    def pf(self):
        return self._pf
    
    @property
    def T0(self):
        """Unit tangent vector at initial point"""
        return self._tangent
    
    @property
    def Tf(self):
        """Unit tangent vector at final point"""
        return self._tangent

    def dist(self, pt):
        """Computes the distance from pt to this line segment."""
        d = self._pf - self._p0
        t = float(np.clip(((pt - self._p0) * d.conjugate()).real / (self._len ** 2), 0.0, 1.0))
        return float(np.abs(pt - (self._p0 + t * d)))
    
    def __mul__(self, other):
        if not np.isscalar(other):
            raise ValueError("non-segment operand must be a scalar")
        return LineSegment(other*self.p0, other*self.pf, self.bc, self.tol)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        if isinstance(other, BaseSegment):
            return MultiSegment([self, other])
        elif np.isscalar(other):
            return LineSegment(self.p0 + other, self.pf + other, self.bc, self.tol)
        else:
            raise TypeError("__add__ with Segment must be another Segment or complex scalar")

class SplineSegment(ParametricSegment):
    """Segments with spline boundary"""
    def __init__(self, spline, t0=None, tf=None, bc='dir', tol=1e-4, val_simple=False, val_closed=False):
        if not isinstance(spline, BSpline):
            raise TypeError("'spline' must be an instance of BSpline")
        if spline.c.ndim == 2 and spline.c.shape[1] == 2:
            spline = BSpline(spline.t, spline.c[:, 0] + 1j * spline.c[:, 1], spline.k,
                             spline.extrapolate, spline.axis)
        self.spline = spline
        p = lambda t: self.spline(t)
        dp = lambda t: self.spline(t, nu=1)
        if t0 is None:
            t0_idx = self.spline.k
            t0 = self.spline.t[t0_idx]
        if tf is None:
            tf_idx = len(self.spline.t)-spline.k-1
            tf = self.spline.t[tf_idx]
        super().__init__(p, dp, t0, tf, bc, tol, val_simple, val_closed)

    @property
    def break_ts(self):
        """Interior knots. A degree-k B-spline is only C^(k-1) there, so `|p'|` and every
        integrand built on it has a derivative break. See BaseSegment.break_ts."""
        k = self.spline.k
        knots = np.unique(self.spline.t[k:len(self.spline.t) - k])
        return knots[(knots > self.t0 + 1e-14) & (knots < self.tf - 1e-14)]

    @classmethod
    def interp_from_pts(cls, pts, bc='dir', spline_bc_type='natural',
                        tol=1e-4, val_simple=False, val_closed=False):
        """Builds a BSpline segment interpolating the given points."""
        t = np.linspace(0, 1, len(pts))
        spline = make_interp_spline(t, pts, bc_type=spline_bc_type)
        return cls(spline, 0, 1, bc, tol, val_simple, val_closed)

    def to_splineseg(self, spline_bc_type='natural'):
        return self
    
    def __mul__(self, other):
        if not np.isscalar(other):
            raise ValueError("non-segment operand must be a scalar")
        else:
            spline = self.spline
            newspline = BSpline(spline.t, other*spline.c, spline.k, spline.extrapolate, spline.axis)
            return SplineSegment(newspline, self.t0, self.tf, self.bc, self.tol)

    def __add__(self, other):
        if np.isscalar(other):
            spline = self.spline
            newspline = BSpline(spline.t, spline.c + other, spline.k, spline.extrapolate, spline.axis)
            return SplineSegment(newspline, self.t0, self.tf, self.bc, self.tol)
        else:
            return super().__add__(other)

class MultiSegment:
    """An ordered collection of segments forming a planar curve.

    Segments are joined end-to-end and may optionally be validated as closed
    (last endpoint matches first) or simple (no self-intersections).  The
    class exposes aggregated geometry: arc length, corner detection, boundary
    point placement, and minimum distance queries.

    Each segment may have different boundary conditions.

    Parameters
    ----------
    segments : list of BaseSegment
        Ordered list of curve segments.
    val_simple : bool, optional
        If True, raise ValueError unless the segments form a simple curve.
    val_closed : bool, optional
        If True, raise ValueError unless the segments form a closed curve.
    """
    def __init__(self, segments, val_simple=True, val_closed=False, val_contiguous=True):
        if not all(isinstance(seg, BaseSegment) or isinstance(seg, MultiSegment) for seg in segments):
            raise TypeError("segments must be an iterable of Segment or MultiSegment objects")
        self._is_simple = None
        self._is_closed = None
        self._is_contiguous = None
        self.segments = []
        for seg in segments:
            if isinstance(seg, BaseSegment):
                self.segments.append(seg)
            elif isinstance(seg, MultiSegment):
                self.segments += seg.segments
        if val_simple:
            if not self.is_simple:
                raise ValueError("segments must form a simple curve")
        if val_closed:
            if not self.is_closed:
                raise ValueError("segments must form a closed curve")
        if val_contiguous:
            if not self.is_contiguous:
                raise ValueError("segments must be contiguous")

        self._len = None
        self._corners = None
        self._corner_idx = None
        self._corner_angle0 = None
        self._corner_angle1 = None

    def __str__(self):
        return f"MultiSegment([{','.join(str(seg) for seg in self.segments)}])"

    @classmethod
    def from_vertices(cls, vertices, bc='dir', make_closed=True, val_simple=False, tol=1e-4):
        """Builds a polygonal MultiSegment from the given vertices"""
        segments = [LineSegment(vertices[i], vertices[i+1], bc, tol) for i in range(len(vertices)-1)]
        if make_closed:
            segments += [LineSegment(vertices[-1], vertices[0], bc, tol)]
        multiseg = cls(segments, val_simple, val_contiguous=False)
        multiseg._is_contiguous = True
        if make_closed: multiseg._is_closed = True
        return multiseg
    
    @staticmethod
    def _validate_contiguous(segments):
        p0 = np.array([seg.p0 for seg in segments])
        pf = np.array([seg.pf for seg in segments])
        return np.allclose(pf[:-1], p0[1:])

    @staticmethod
    def _validate_closed(segments):
        if len(segments) == 1:
            return segments[0].is_closed
        else:
            if MultiSegment._validate_contiguous(segments):
                # check if initial and final points match (closed loop)
                return np.isclose(segments[0].p0, segments[-1].pf)
            else: return False

    @staticmethod
    def _validate_simple(segments):
        if len(segments) == 1:
            return segments[0].is_simple
        elif len(segments) == 2:
            intersections = segments[0].intersection(segments[1])
            if len(intersections) > 2:
                return False
            elif len(intersections) >= 1:
                endpts = np.array([segments[0].p0,segments[0].pf,segments[1].p0,segments[1].pf])
                for intersection in intersections:
                    if not np.any(np.isclose(intersection, endpts)):
                        return False
                return True
            else:
                return True
        for i in range(len(segments)):
            if i == len(segments)-1: 
                j = 0
            else: 
                j = i+1
                
            # check next segment: at most one intersection at endpoint
            intersections = segments[i].intersection(segments[j])
            if len(intersections) > 1:
                return False
            elif len(intersections) == 1:
                if not np.isclose(intersections[0], segments[i].pf):
                    return False
                
            # check other segments: should have no intersections
            if i == 0: end = len(segments)-1
            else: end = len(segments)
            for j in range(i+2, end):
                if segments[i].intersects(segments[j]):
                    return False
        return True
    
    @property
    def p0(self):
        return np.array([seg.p0 for seg in self.segments])
    
    @property
    def pf(self):
        return np.array([seg.pf for seg in self.segments])
    
    @property
    def T0(self):
        return np.array([seg.T0 for seg in self.segments])
    
    @property
    def Tf(self):
        return np.array([seg.Tf for seg in self.segments])
    
    @property
    def int_angles(self):
        return np.angle(-np.roll(self.Tf,1)/self.T0) % (2*np.pi)
    
    @property
    def is_contiguous(self):
        if self._is_contiguous is None:
            self._is_contiguous = self._validate_contiguous(self.segments)
        return self._is_contiguous

    @property
    def is_simple(self):
        if self._is_simple is None:
            self._is_simple = self._validate_simple(self.segments)
        return self._is_simple

    @property
    def is_closed(self):
        if self._is_closed is None:
            self._is_closed = self._validate_closed(self.segments)
        return self._is_closed
        
    def _compute_length(self):
        return self.seg_lens.sum()

    @property
    def len(self):
        if self._len is None:
            self._len = self._compute_length()
        return self._len
    
    @property
    def seg_lens(self):
        return np.array([seg.len for seg in self.segments])

    @property
    def is_polyline(self):
        return all(isinstance(seg, LineSegment) for seg in self.segments)
    
    @property
    def bcs(self):
        return [seg.bc for seg in self.segments]
    
    @property
    def bc_types(self):
        return [seg.bc_type for seg in self.segments]

    def _broadcast_quad_kwargs(self, kwargs):
        """Broadcasts scalar quadrature kwargs (e.g. Gauss-Jacobi exponents 'a', 'b') to
        per-segment arrays, mirroring how N is broadcast for point counts."""
        n_seg = len(self.segments)
        return {key: np.full(n_seg, val) if np.isscalar(val) else val for key, val in kwargs.items()}

    def pts(self, N, kind='legendre', **kwargs):
        """Places N[i] points on ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self.segments), N)
        kwargs = self._broadcast_quad_kwargs(kwargs)
        return np.sum([self.segments[i].pts(N[i], kind, **{key: val[i] for key, val in kwargs.items()})
                       for i in range(len(self.segments)) if N[i] > 0])

    def tangents(self, N, kind='legendre', **kwargs):
        """Computes tangent vectors at N[i] points along the ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self.segments), N)
        kwargs = self._broadcast_quad_kwargs(kwargs)
        return np.sum([self.segments[i].tangents(N[i], kind, **{key: val[i] for key, val in kwargs.items()})
                       for i in range(len(self.segments)) if N[i] > 0])

    def normals(self, N, kind='legendre', **kwargs):
        """Computes normal vectors at N[i] points along the ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self.segments), N)
        kwargs = self._broadcast_quad_kwargs(kwargs)
        return np.sum([self.segments[i].normals(N[i], kind, **{key: val[i] for key, val in kwargs.items()})
                       for i in range(len(self.segments)) if N[i] > 0])
    
    def polyline(self):
        """Boundary as consecutive sub-segments (b0[k] -> b1[k]) with the owning
        segment index. LineSegments contribute one sub-segment; curved segments
        contribute len(seg.polyline_tau)-1, the adaptively chosen node count."""
        b0, b1, owner = [], [], []
        for j, seg in enumerate(self.segments):
            pts = seg.polyline_pts
            b0.append(pts[:-1])
            b1.append(pts[1:])
            owner.append(np.full(len(pts) - 1, j))
        return np.concatenate(b0), np.concatenate(b1), np.concatenate(owner)

    def to_splinesegs(self, spline_bc_type='natural'):
        """Returns a new MultiSegment with each segment converted to a
        SplineSegment (not a single SplineSegment for the whole boundary)."""
        segs = [seg.to_splineseg(spline_bc_type) for seg in self.segments]
        return MultiSegment(segs, val_simple=False, val_contiguous=False)

    def _find_corners(self):
        """Finds corners, i.e. where two boundary segments do not smoothly connect. Computes
        the angle wedge there.

        Every array returned here is CORNER-indexed, i.e. aligned with `corners` and
        `corner_idx`, so `corner_angles[k][c]` is the wedge ray at `corners[c]`. This matters:
        `int_angles` (a separate property) is SEGMENT-indexed, giving the angle at each
        segment's p0 whether or not that junction is a genuine corner, so the angle at corner
        `c` is `int_angles[corner_idx[c]]`. The two indexings coincide only when every
        junction is a corner. They diverge exactly when a junction is smooth -- a straight
        (pi) vertex, e.g. `disk_sector(r, pi)`, or a collinear polygon vertex -- and mixing
        them up previously made a half-disk unusable: `corners` had 2 entries while the
        angle arrays had 3, so no `orders` length could be masked consistently."""
        T0, Tf = self.T0, self.Tf
        turning_angles = np.angle(np.roll(Tf,1)/T0)
        is_corner = (np.abs(turning_angles) > 1e-10)
        corners = self.p0[is_corner]
        corner_idx = np.arange(len(self.segments))[is_corner]
        corner_angle0 = np.angle(T0)[is_corner]
        corner_angle1 = np.angle(-np.roll(Tf,1))[is_corner]
        return corners, corner_idx, corner_angle0, corner_angle1
    
    def dist(self, pt):
        """Returns the (minimum) distance from a given point to the MultiSegment"""
        b0, b1, owner = self.polyline()
        d = b1 - b0
        t = np.clip(((pt - b0) * d.conjugate()).real / np.abs(d) ** 2, 0.0, 1.0)
        seg_idx = owner[np.abs(pt - (b0 + t * d)).argmin()]
        return self.segments[seg_idx].dist(pt)
    
    @property
    def corners(self):
        if self._corners is None:
            self._corners, self._corner_idx, self._corner_angle0, self._corner_angle1 = self._find_corners()
        return self._corners
    
    @property
    def corner_idx(self):
        if self._corner_idx is None:
            self._corners, self._corner_idx, self._corner_angle0, self._corner_angle1 = self._find_corners()
        return self._corner_idx

    @property
    def corner_angles(self):
        if self._corner_angle0 is None or self._corner_angle1 is None:
            self._corners, self._corner_idx, self._corner_angle0, self._corner_angle1 = self._find_corners()
        return self._corner_angle0, self._corner_angle1
    
    def __add__(self, other):
        if isinstance(other, BaseSegment):
            return MultiSegment(self.segments + [other], val_simple=False)
        elif isinstance(other, MultiSegment):
            return MultiSegment(self.segments + other.segments, val_simple=False)
        elif np.isscalar(other):
            new_segments = [seg + other for seg in self.segments]
            return MultiSegment(new_segments, val_simple=False, val_contiguous=False)
        else:
            raise TypeError("__add__ with MultiSegment must be another MultiSegment, a Segment, or complex scalar")

    def __radd__(self, other):
        if isinstance(other, BaseSegment):
            return MultiSegment([other] + self.segments, val_simple=False)
        elif isinstance(other, MultiSegment):
            return MultiSegment(other.segments + self.segments, val_simple=False)
        elif np.isscalar(other):
            new_segments = [seg + other for seg in self.segments]
            return MultiSegment(new_segments, val_simple=False, val_contiguous=False)
        else:
            raise TypeError("__add__ with MultiSegment must be another MultiSegment, a Segment, or complex scalar")
    
    def plot(self, ax=None, showbc=False, **pltkwargs):
        """Plots the MultiSegment"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        lines = []
        for seg in self.segments:
            lines.append(seg.plot(ax=ax, showbc=showbc, **pltkwargs))
        return lines

    def plot_tangents(self, ax=None, **pltkwargs):
        """Plots the tangent vectors"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        quivs = []
        for seg in self.segments:
            quivs.append(seg.plot_tangents(ax=ax, **pltkwargs))
        return quivs

    def plot_normals(self, ax=None, **pltkwargs):
        """Plots the normal vectors"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        quivs = []
        for seg in self.segments:
            quivs.append(seg.plot_normals(ax=ax, **pltkwargs))
        return quivs
    
    def __mul__(self, other):
        if not np.isscalar(other):
            raise ValueError("non-MultiSegment operand must be a scalar")
        new_segments = [other*seg for seg in self.segments]
        return MultiSegment(new_segments, val_simple=False, val_contiguous=False)

    def __rmul__(self, other):
        return self.__mul__(other)
    
def corner_branch_cut_rays(domain):
    """
    For each corner of domain, return a ray angle (radians) such that the ray
    c + t*exp(i*theta), t > 0, lies entirely in the exterior of the domain.

    Each boundary sub-segment subtends an angular arc of directions from the
    corner; any ray in that arc hits it. The set of blocked directions is the
    union of these arcs over all non-adjacent sub-segments. The exterior sector
    bisector is returned when free (canonical, exact for convex corners);
    otherwise the midpoint of the largest free gap (maximal clearance from the
    boundary). Returns NaN for any corner with no free direction.

    Parameters
    ----------
    domain : Domain

    Returns
    -------
    branch_angles : ndarray, shape (n_corners,), float
        Branch cut angle in radians for each corner. NaN if none found.
    """
    bdry          = domain.bdry
    corners       = bdry.corners
    corner_idx    = bdry.corner_idx
    phi0, phi1    = bdry.corner_angles   # corner-indexed, like `corners` (see _find_corners)
    n_segs        = len(bdry.segments)
    b0, b1, owner = bdry.polyline()      # built once

    result = np.full(len(corners), np.nan)

    for i, (c, ci) in enumerate(zip(corners, corner_idx)):
        phi_in   = phi1[i]
        ext_span = (phi0[i] - phi_in) % (2 * np.pi)
        if ext_span < 1e-10:
            continue

        # arcs subtended by non-adjacent sub-segments, in coords relative to phi_in
        keep = (owner != ci) & (owner != (ci - 1) % n_segs)
        a0 = np.angle(b0[keep] - c)
        a1 = np.angle(b1[keep] - c)
        d  = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi        # signed minor arc, |d| < pi
        rel0 = (np.where(d >= 0, a0, a1) - phi_in) % (2 * np.pi)
        rel1 = rel0 + np.abs(d)                              # may exceed 2*pi (wraps)

        def covered(u):   # is relative direction u blocked by any arc? (wrap-aware)
            return np.any(((rel0 <= u) & (u <= rel1)) |
                          ((rel0 <= u + 2 * np.pi) & (u + 2 * np.pi <= rel1)))

        # canonical: bisector when free
        bisect = ext_span / 2.0
        if not covered(bisect):
            result[i] = (phi_in + bisect) % (2 * np.pi)
            continue

        # otherwise: midpoint of the largest free gap in (0, ext_span)
        eps = 1e-9
        edges = np.concatenate(([eps, ext_span - eps], rel0, rel1, rel1 - 2 * np.pi))
        edges = np.unique(edges[(edges > 0) & (edges < ext_span)])
        edges = np.concatenate(([0.0], edges, [ext_span]))
        best_w, best_mid = 0.0, np.nan
        for lo, hi in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (lo + hi)
            if (hi - lo) > best_w and not covered(mid):
                best_w, best_mid = hi - lo, mid
        if best_w > 0:
            result[i] = (phi_in + best_mid) % (2 * np.pi)

    return result

def free_ray_from_point(domain, p):
    """Return a ray angle (radians) such that the ray p + t*exp(i*theta), t > 0, never
    intersects the domain boundary — i.e. a clear sightline to infinity from the
    exterior point p. Uses the same angular-subtension idea as corner_branch_cut_rays,
    but over all boundary sub-segments (no adjacency exclusion) and the full circle.
    Returns the midpoint of the widest free angular gap, or NaN if p is in a pocket
    with no sightline."""
    b0, b1, _ = domain.bdry.polyline()
    a0 = np.angle(b0 - p)
    a1 = np.angle(b1 - p)
    d  = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi   # signed minor arc, |d| < pi
    lo = np.where(d >= 0, a0, a1) % (2 * np.pi)    # arc start in [0, 2pi)
    hi = lo + np.abs(d)                            # arc end (may exceed 2pi)

    def covered(u):   # is direction u (in [0,2pi)) blocked by any subtended arc?
        return np.any(((lo <= u) & (u <= hi)) |
                      ((lo <= u + 2 * np.pi) & (u + 2 * np.pi <= hi)))

    # candidate gap boundaries: all arc edges mod 2pi, swept around the circle
    edges = np.unique(np.concatenate((lo % (2 * np.pi), hi % (2 * np.pi))))
    swept = np.concatenate((edges, edges[:1] + 2 * np.pi))
    best_w, best_dir = 0.0, np.nan
    for a, b in zip(swept[:-1], swept[1:]):
        mid = 0.5 * (a + b) % (2 * np.pi)
        if (b - a) > best_w and not covered(mid):
            best_w, best_dir = b - a, mid
    return best_dir

def _path_crosses_boundary(domain, c, verts, atol=1e-9):
    """True if any segment of the path [c, *verts] crosses the boundary (touches at the
    corner c are allowed) or has a midpoint inside the domain."""
    b0, b1, _ = domain.bdry.polyline()
    path = np.concatenate(([c], verts))
    for a, b in zip(path[:-1], path[1:]):
        for z0, z1 in zip(b0, b1):
            hit = segment_intersection(a, b, z0, z1)
            if hit is not None and not np.isclose(hit, c, atol=atol):
                return True
    mids = 0.5 * (path[:-1] + path[1:])
    return bool(np.any(domain.contains(mids)))

def corner_branch_cut_polyline(domain, i, eps=None, max_steps=None):
    """Heuristic polyline+ray branch cut for the corner at ``domain.corners[i]`` when no
    straight ray is free (a "surrounded" corner). Wall-follows the outward-offset
    boundary from the corner until reaching a vertex with a clear sightline to infinity,
    then attaches that ray. Tries both directions around the boundary and returns the
    shorter valid cut.

    Returns
    -------
    vertices : ndarray of complex
        Interior polyline vertices [q1, ..., qm] (excluding the corner c and the ray).
    beta : float
        Angle of the final ray, emanating from vertices[-1].

    Raises
    ------
    RuntimeError if no valid polyline cut is found.
    """
    bdry = domain.bdry
    b0, b1, _ = bdry.polyline()
    K = len(b0)
    pts = b0                                   # ordered boundary points (closed loop)
    c = domain.corners[i]
    k0 = int(np.abs(pts - c).argmin())

    # outward normals (CCW boundary => outward = -1j * unit tangent)
    edge = b1 - b0
    seg_n = -1j * edge / np.abs(edge)
    vert_n = seg_n + np.roll(seg_n, 1)         # average of the two adjacent sub-seg normals
    mag = np.abs(vert_n)
    vert_n = np.where(mag > 1e-12, vert_n / np.where(mag > 1e-12, mag, 1.0), seg_n)

    if eps is None:
        eps = 0.05 * np.min(np.abs(edge))
    if max_steps is None:
        max_steps = K
    offset = pts + eps * vert_n

    def build(direction):
        verts = []
        for j in range(1, max_steps + 1):
            q = offset[(k0 + direction * j) % K]
            verts.append(q)
            beta = free_ray_from_point(domain, q)
            if not np.isnan(beta):
                return np.array(verts), float(beta)
        return None, None

    candidates = []
    for direction in (+1, -1):
        verts, beta = build(direction)
        if verts is not None and not _path_crosses_boundary(domain, c, verts):
            length = np.abs(np.diff(np.concatenate(([c], verts)))).sum()
            candidates.append((len(verts), length, verts, beta))

    if not candidates:
        raise RuntimeError(
            f"corner_branch_cut_polyline: no valid polyline cut found for corner {i}")
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2], candidates[0][3]

# domain class
class Domain(BaseDomain):
    """A planar domain whose boundary is a closed, simple MultiSegment.

    Provides geometric properties (area, diameter, perimeter), interior/boundary
    point placement, winding-number containment tests, and plotting utilities.
    Subclasses (e.g. Polygon) may override ``_compute_area`` and
    ``_compute_diameter`` with more efficient closed-form implementations.

    Parameters
    ----------
    bdry : MultiSegment
        Closed, simple boundary curve of the domain.
    val_simple : bool, optional
        If True, raise ValueError unless the boundary is a simple curve.
    val_closed : bool, optional
        If True, raise ValueError unless the boundary is a closed curve.
    val_orientation : bool, optional
        If True, raise ValueError unless the boundary is in CCW (positive)
        orientation. Only checked when the boundary is simple and closed,
        since orientation is meaningless otherwise.
    """
    def __init__(self, bdry, val_simple=True, val_closed=True, val_orientation=True):
        if not isinstance(bdry, MultiSegment):
            raise TypeError("'bdry' must be an instance of MultiSegment")
        if val_simple:
            if not bdry.is_simple:
                raise ValueError("boundary must be simple")
        if val_closed:
            if not bdry.is_closed:
                raise ValueError("boundary must be closed")

        self.bdry = bdry
        self._area = None
        self._diameter = None
        self._inradius = None
        super().__init__()

        if val_orientation and bdry.is_simple and bdry.is_closed:
            if self._polyline_signed_area() < 0:
                raise ValueError("boundary must be in CCW (positive) orientation")

    @property
    def bc_type(self):
        bc_types = self.bdry.bc_types
        if all([bc_type=='dir' for bc_type in bc_types]):
            return 'dir'
        elif all([bc_type=='neu' for bc_type in bc_types]):
            return 'neu'
        elif all([(bc_type=='dir' or bc_type=='neu') for bc_type in bc_types]):
            return 'mixed'
        else:
            return 'rob'

    @property
    def perimeter(self):
        return self.bdry.len

    @property
    def area(self):
        if self._area is None:
            self._area = self._compute_area()
        return self._area
            
    def _signed_area(self):
        """Composite 5-point Gauss-Legendre integral of Green's formula, with
        each segment's adaptive polyline partition as the quadrature panels.

        Reusing the polyline partition (rather than a single fixed-order rule
        spanning the whole segment) means segments needing more resolution
        automatically get proportionally more, better-placed panels: a single
        large-order rule still bets on the whole integrand being well fit by
        one polynomial across the entire segment, which can fail badly on a
        highly oscillatory curve even at large orders (see the aliasing case
        in test_polyline_signed_area_robust_to_gl_aliasing)."""
        I = 0.0
        for seg in self.bdry.segments:
            tau = seg.polyline_tau
            def integrand(t, seg=seg):
                Pt, dPt = seg.p(t), seg.dp(t)
                return Pt.real*dPt.imag - Pt.imag*dPt.real
            I += _gl5(integrand, tau[:-1], tau[1:]).sum()
        return I/2

    def _polyline_signed_area(self):
        """Shoelace-formula signed area on the boundary's adaptive polyline.

        Used only for the (sign-only) orientation check: each segment's
        polyline is already resolved to that segment's own ``tol``, so this
        tracks curve wiggliness adaptively rather than relying on a fixed
        quadrature order, and it's free -- the polyline is already built for
        intersection/containment checks. (Cheaper but lower-order than
        ``_signed_area``, which additionally accounts for curvature within
        each polyline panel via composite Gauss-Legendre.)"""
        b0, b1, _ = self.bdry.polyline()
        return np.sum(b0.real*b1.imag - b0.imag*b1.real)/2

    def _compute_area(self):
        return np.abs(self._signed_area())
    
    @property
    def diameter(self):
        if self._diameter is None:
            self._diameter = self._compute_diameter()
        return self._diameter
    
    @property
    def inradius(self):
        if self._inradius is None:
            self._inradius = self._compute_inradius()
        return self._inradius

    def _compute_inradius(self, ngrid=25):
        """Numerically computes the inradius (largest inscribed circle radius)."""
        # Build bounding box from coarse boundary samples
        tau = np.linspace(0, 1, 50)[:-1]
        bdry_pts = np.array([seg.p(tau) for seg in self.bdry.segments]).flatten()
        xmin, xmax = bdry_pts.real.min(), bdry_pts.real.max()
        ymin, ymax = bdry_pts.imag.min(), bdry_pts.imag.max()

        # Deterministic interior grid
        xs = np.linspace(xmin, xmax, ngrid)
        ys = np.linspace(ymin, ymax, ngrid)
        XX, YY = np.meshgrid(xs, ys)
        candidates = (XX + 1j * YY).flatten()
        interior = candidates[self.contains(candidates)]

        # Coarse best: maximum boundary distance over grid
        dists = np.array([self.bdry.dist(pt) for pt in interior])
        best_pt = interior[dists.argmax()]

        # Nelder-Mead refinement
        def neg_dist(xy):
            pt = complex(xy[0], xy[1])
            if not self.contains(np.array([pt]))[0]:
                return 0.0
            return -self.bdry.dist(pt)

        res = minimize(neg_dist, [best_pt.real, best_pt.imag], method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000})
        return float(-res.fun)

    def max_dist(self, pt, n=100):
        """compute the maximum distance from a given point to another point in the domain"""
        tau = np.linspace(0, 1, n)[:-1]
        pts = np.array([seg.p(tau) for seg in self.bdry.segments])
        dist = np.abs(pts-pt)
        seg_idx = dist.max(axis=1).argmax()
        tau_idx = dist[seg_idx].argmax()
        seg = self.bdry.segments[seg_idx]
        if tau_idx == 0:
            tau0 = tau[tau_idx]
            tau1 = tau[tau_idx+1]
        elif tau_idx == n-2:
            tau0 = tau[tau_idx-1]
            tau1 = tau[tau_idx]
        else:
            tau0 = tau[tau_idx-1]
            tau1 = tau[tau_idx+1]

        def f(tau):
            return -np.abs(seg.p(tau)-pt)

        res = minimize_scalar(f, bounds=(tau0,tau1), options={'xatol':1e-14})
        return -res.fun

    def _compute_diameter(self, n=100):
        # approximation using n points on each segment
        tau = np.linspace(0, 1, n)[:-1]
        pts = np.array([seg.p(tau) for seg in self.bdry.segments])
        dist = np.abs(np.subtract.outer(pts.flatten(),pts.flatten()))
        idx1 = dist.max(axis=0).argmax()
        idx2 = dist[idx1].argmax()

        seg1_idx, tau1_idx = np.unravel_index(idx1, pts.shape)
        seg2_idx, tau2_idx = np.unravel_index(idx2, pts.shape)
        seg1 = self.bdry.segments[seg1_idx]
        tau1 = tau[tau1_idx]
        seg2 = self.bdry.segments[seg2_idx]
        tau2 = tau[tau2_idx]

        def f(tau):
            p1, p2 = seg1.p(tau[0]), seg2.p(tau[1])
            diff = p2-p1
            out = -np.abs(diff)
            dp1 = seg1.dp(tau[0])
            dp2 = seg2.dp(tau[1])
            grad = np.array([(diff.real/out)*dp1.real + (diff.imag/out)*dp1.imag,
                             -(diff.real/out)*dp2.real - (diff.imag/out)*dp2.imag])
            return out, grad

        res = minimize(f, np.array([tau1,tau2]), jac=True, bounds=[(0,1),(0,1)], tol=1e-14)
        return float(-res.fun)
    
    def contains(self, pts):
        """Checks if the domain contains the given points using ray casting."""
        pts = complex_form(pts)
        pt_y = pts.imag
        pt_x = pts.real
        inside = np.zeros(len(pts), dtype=bool)

        b0, b1, _ = self.bdry.polyline()
        for z0, z1 in zip(b0, b1):
            x0, y0 = z0.real, z0.imag
            x1, y1 = z1.real, z1.imag
            crosses = ((y0 <= pt_y) & (pt_y < y1)) | ((y1 <= pt_y) & (pt_y < y0))
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(crosses, (pt_y - y0) / (y1 - y0), 0.0)
            x_cross = x0 + t * (x1 - x0)
            inside ^= crosses & (x_cross > pt_x)

        return inside
    
    def bdry_pts(self, n_per_seg, kind='legendre', **kwargs):
        return self.bdry.pts(n_per_seg, kind=kind, **kwargs)

    def bdry_tangents(self, n_per_seg, kind='legendre', **kwargs):
        return self.bdry.tangents(n_per_seg, kind=kind, **kwargs)

    def bdry_normals(self, n_per_seg, kind='legendre', **kwargs):
        return self.bdry.normals(n_per_seg, kind=kind, **kwargs)

    def bdry_data(self, n_per_seg, kind='legendre', **kwargs):
        if isinstance(n_per_seg, (int, np.integer)):
            n_per_seg = np.full(len(self.bdry.segments), n_per_seg)
        bdry_pts = self.bdry_pts(n_per_seg, kind=kind, **kwargs)
        bdry_normals = self.bdry_normals(n_per_seg, kind=kind, **kwargs)
        bc_param = np.concatenate([np.full(n, seg.bc, 'float') for seg, n in zip(self.bdry.segments, n_per_seg)])
        return bdry_pts, bdry_normals, bc_param
    
    def int_quad(self, method='mesh', kind='dunavant', deg=4, mesh_kwargs={}, n_bdry=100,
                 npts_rand=50, oversamp=2, rng=None):
        """``(nodes, weights)`` for INTEGRATING over the interior. A quadrature rule, not points.

        Weights live here, on the rule, rather than on the `PointSet` -- a set of points says
        where to evaluate, a rule says how to integrate, and conflating them is what put a
        `sqrt_wts` behind seven `hasattr` guards in `bases`. `int_pts` below returns the
        collocation points for the MPS pencil and has nothing to do with integration.

        The genuine consumer is `benchmarks/reference/certify.interior_l2`, which needs
        ``||u||_{L2(Omega)}`` for the Moler--Payne bound.

        ``rng`` (int or numpy Generator) makes ``method='random'`` reproducible.
        """
        nodes, wts = self._interior_nodes(method, kind, deg, mesh_kwargs, npts_rand, oversamp, rng)
        return nodes, wts

    def int_pts(self, method='random', kind='dunavant', deg=4, mesh_kwargs={}, n_bdry=100,
                npts_rand=50, oversamp=2, rng=None):
        """Interior COLLOCATION points for the MPS pencil. See `int_quad` to integrate.

        ``rng`` (int or numpy Generator) makes ``method='random'`` reproducible.
        Interior collocation points feed straight into the MPS pencil, and the
        draw genuinely moves the answer -- iso_right_tri spanned 2.5 to 5.8
        certified digits across draws -- so anything producing reference values
        should pass one. ``None`` keeps the global RNG.
        """
        nodes, _wts = self._interior_nodes(method, kind, deg, mesh_kwargs, npts_rand, oversamp, rng)
        return PointSet(nodes)

    def _interior_nodes(self, method, kind, deg, mesh_kwargs, npts_rand, oversamp, rng):
        if method == 'random':
            rng = as_generator(rng)
            pt = self.bdry.segments[0].p0
            xmin, xmax = pt.real - self.diameter, pt.real + self.diameter
            ymin, ymax = pt.imag - self.diameter, pt.imag + self.diameter
            box_area = (xmax-xmin)*(ymax-ymin)
            pts = []
            max_iters = 20
            for _ in range(max_iters):
                if len(pts) >= npts_rand:
                    break
                npts = int(np.ceil(npts_rand*oversamp*box_area/self.area))
                x = (xmax-xmin)*rand_uniform(rng, npts)+xmin
                y = (ymax-ymin)*rand_uniform(rng, npts)+ymin
                z = x + 1j*y
                pts_new = z[self.contains(z)]
                pts = np.concatenate((pts, pts_new))
                oversamp = 2*oversamp
            else:
                raise RuntimeError("int_pts: rejection sampling failed to collect enough interior points")
            int_pts = pts[:npts_rand]
            wts = np.full(len(int_pts), self.area / max(len(int_pts), 1))

        elif method == 'mesh':
            splinesegs = self.bdry.to_splinesegs().segments
            mesh = spline_mesh_with_curvature(splinesegs, **mesh_kwargs)
            int_pts, wts = tri_quad(mesh, kind, deg)

        return int_pts, wts
        
    @property
    def corners(self):
        return self.bdry.corners
    
    @property
    def corner_angles(self):
        return self.bdry.corner_angles
    
    @property
    def corner_idx(self):
        return self.bdry.corner_idx
    
    @property
    def int_angles(self):
        return self.bdry.int_angles

    @property
    def corner_int_angles(self):
        """Interior angles at the genuine corners, CORNER-indexed (aligned with `corners`,
        `corner_idx` and `corner_angles`).

        `int_angles` is SEGMENT-indexed -- one entry per segment's p0, corner or not -- so the
        angle at corner `c` is `int_angles[corner_idx[c]]`, never `int_angles[c]`. Use this
        property instead of doing that indexing by hand: any consumer that pairs an angle with
        `corners[c]` wants these. The distinction is invisible on a domain where every junction
        is a corner and silently wrong on one where a junction is smooth (a straight pi vertex,
        a collinear polygon vertex)."""
        return np.asarray(self.int_angles)[np.asarray(self.corner_idx)]

    def branch_cut_rays(self):
        """For each corner, a ray angle (radians) whose extension to infinity stays
        outside the domain — the branch cut placement for corner-centered
        Fourier-Bessel functions. NaN where no valid direction exists."""
        return corner_branch_cut_rays(self)

    def plot(self, ax=None, showbc=False, **plt_kwargs):
        if 'c' not in plt_kwargs.keys() and 'color' not in plt_kwargs.keys():
            plt_kwargs['color'] = 'k'
        return self.bdry.plot(ax, showbc, **plt_kwargs)
    
    def __add__(self, other):
        if not np.isscalar(other):
            raise TypeError("'other' must be a complex scalar")
        new_bdry = self.bdry + other
        return Domain(new_bdry, False, False)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not np.isscalar(other):
            raise TypeError("'other' must be a complex scalar")
        new_bdry = other*self.bdry
        return Domain(new_bdry, False, False)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def to_bc(self, bc):
        """Makes a copy of the domain with a different boundary condition"""
        new_dom = copy.deepcopy(self)
        for seg in new_dom.bdry.segments:
            seg.bc = bc
        return new_dom
    
    @property
    def seg_lens(self):
        return self.bdry.seg_lens

# polygon class
class Polygon(Domain):
    """Class for polygonal domains"""
    def __init__(self, vertices=None, bdry=None, bc='dir', val_simple=True, val_orientation=True):
        if not ((vertices is None)^(bdry is None)):
            raise ValueError("exactly one of 'vertices' and 'bdry' must be provided")
        elif vertices is not None:
            vertices = complex_form(vertices)
            bdry = MultiSegment.from_vertices(vertices, bc)   # always closed by construction
            if val_simple and not bdry.is_simple:
                raise ValueError("'bdry' must be simple")
            self.vertices = vertices
        elif bdry is not None:
            if not isinstance(bdry, MultiSegment) or not bdry.is_polyline:
                raise TypeError("'bdry' must be a polyline MultiSegment")
            if not bdry.is_closed:
                raise ValueError("'bdry' must be closed")
            if val_simple and not bdry.is_simple:
                raise ValueError("'bdry' must be simple")
            self.vertices = np.array([seg.p0 for seg in bdry.segments])
        if val_orientation and polygon_area(self.vertices) < 0:
            raise ValueError("'vertices' must be in CCW (positive) orientation")
        super().__init__(bdry, val_simple=False, val_closed=False, val_orientation=False)

    def _compute_area(self):
        return np.abs(polygon_area(self.vertices))

    def _compute_diameter(self):
        return polygon_diameter(self.vertices)

    def _compute_inradius(self):
        """Computes inradius exactly for convex polygons via LP; falls back to parent for non-convex."""
        if not np.all(self.int_angles <= np.pi):
            return super()._compute_inradius()

        # LP: maximize r  s.t.  dist(center, edge_k) >= r  for all k
        # Outward unit normals for CCW polygon: n_k = -i * (v_{k+1} - v_k) / |...|
        v = self.vertices
        edges = np.roll(v, -1) - v
        normals = -1j * edges / np.abs(edges)

        # Per-edge constraint: n_k.real*x + n_k.imag*y + r <= Re[v_k * conj(n_k)]
        A = np.column_stack([normals.real, normals.imag, np.ones(len(v))])
        b = (v * normals.conjugate()).real

        res = linprog([0.0, 0.0, -1.0], A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)])
        return float(-res.fun)

    @property
    def n_vertices(self):
        return len(self.vertices)
    
    @property
    def n_sides(self):
        return len(self.vertices)
    
    @property
    def edge_lengths(self):
        return edge_lengths(self.vertices)
    
    @property
    def int_angles(self):
        return interior_angles(self.vertices)
    
    @property
    def corner_idx(self):
        return np.arange(self.n_vertices)
    
    def int_quad(self, method='mesh', kind='dunavant', deg=4, mesh_size=1, npts_rand=50,
                 oversamp=2, rng=None):
        """``(nodes, weights)`` for integrating over the polygon. See `Domain.int_quad`."""
        return self._interior_nodes(method, kind, deg, mesh_size, npts_rand, oversamp, rng)

    def int_pts(self, method='random', kind='dunavant', deg=4, mesh_size=1, npts_rand=50, oversamp=2,
                rng=None):
        """Interior COLLOCATION points for the MPS pencil. See `int_quad` to integrate."""
        nodes, _wts = self._interior_nodes(method, kind, deg, mesh_size, npts_rand, oversamp, rng)
        return PointSet(nodes)

    def _interior_nodes(self, method, kind, deg, mesh_size, npts_rand, oversamp, rng):
        if method == 'random':
            nodes = rand_interior_points(self.vertices, npts_rand, oversamp, rng=rng)
            wts = np.full(len(nodes), self.area / max(len(nodes), 1))

        elif method == 'mesh':
            mesh = polygon_triangular_mesh(self.vertices, mesh_size)
            nodes, wts = tri_quad(mesh, kind, deg)

        return nodes, wts

    def contains(self, pts):
        """Checks containment using Shapely (exact, no approximation)."""
        pts = complex_form(pts)
        xy = real_form(pts)
        poly = ShapelyPolygon(real_form(self.vertices))
        return np.array(poly.contains(shapely_points(xy)))
    
    def __add__(self, other):
        if not np.isscalar(other):
            raise TypeError("'other' must be a complex scalar")
        new_bdry = self.bdry + other
        return Polygon(bdry=new_bdry, val_simple=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not np.isscalar(other):
            raise TypeError("'other' must be a complex scalar")
        new_bdry = other*self.bdry
        return Polygon(bdry=new_bdry, val_simple=False)

    def __rmul__(self, other):
        return self.__mul__(other)
    
### Sample domains
def rect(L, H, bc='dir'):
    return Polygon([0, L, L + 1j*H, 1j*H], bc=bc, val_simple=False)

def L_shape(bc='dir'):
    return Polygon([0, 1j, -1+1j, -1-1j, 1-1j, 1], bc=bc, val_simple=False)

def GWW1(bc='dir'):
    """first GWW domain"""
    vx = np.array([1,3,3,-1,-1,-3,-1,1])
    vy = np.array([-3,-1,1,1,3,1,-1,-1])
    vertices = vx + 1j*vy
    return Polygon(vertices, bc=bc, val_simple=False)

def GWW2(bc='dir'):
    """second GWW domain"""
    vx = np.array([1,1,3,1,-1,-1,-3,-3])
    vy = np.array([-3,-1,-1,1,1,3,3,1])
    vertices = vx + 1j*vy
    return Polygon(vertices, bc=bc, val_simple=False)

def disk(r=1, bc='dir', tol=1e-4):
    """circle of radius r"""
    seg = ParametricSegment(
        lambda t: r*np.exp(1j*t),
        lambda t: 1j*r*np.exp(1j*t),
        0, 2*np.pi, bc, tol
    )
    bdry = MultiSegment([seg])
    return Domain(bdry, val_simple=False, val_closed=False)

def chevron(h1=1, h2=2, bc='dir'):
    """chevron domain"""
    if h1 >= h2:
        raise ValueError("h1 must be less than h2")
    elif h1 < 0 or h2 < 0:
        raise ValueError("h1 and h2 must be nonnegative")
    
    vertices = np.array([-1, 1j*h1, 1, 1j*h2])
    return Polygon(vertices, bc=bc, val_simple=False)

def cut_square(r=0.5, bc='dir', tol=1e-4):
    """cut square domain"""
    if not (0 < r < 1):
        raise ValueError("r must be between 0 and 1 (strictly)")
    seg1 = LineSegment(0, 1, bc=bc)
    seg2 = LineSegment(1, 1 + (1-r)*1j, bc=bc)
    seg3 = ParametricSegment(lambda t: 1+1j+r*np.exp(-1j*t),
                             lambda t: -1j*r*np.exp(-1j*t),
                             np.pi/2, np.pi, bc, tol)
    seg4 = LineSegment((1-r)+1j, 1j, bc=bc)
    seg5 = LineSegment(1j, 0, bc=bc)
    bdry = MultiSegment([seg1, seg2, seg3, seg4, seg5])
    return Domain(bdry, val_simple=False, val_closed=False)

def polyomino(cells, bc='dir'):
    """Polygon bounding the union of unit grid cells `cells`, given as (i, j) integer pairs
    for the square [i, i+1] x [j, j+1].

    The point of this family: `sin(m pi x) sin(n pi y)` vanishes on the ENTIRE integer grid, so
    it is an exact Dirichlet eigenfunction of any polyomino, with eigenvalue pi^2 (m^2 + n^2)
    and squared L^2 norm exactly `len(cells)/4` (each cell contributes 1/2 * 1/2). That makes a
    polyomino the only multi-reentrant-corner domain with closed-form eigenfunctions -- see
    `reference.polyomino_eigfun`.

    The cells must form a simply-connected, edge-connected region; a diagonal-only join (two
    cells meeting at a single point) or an enclosed hole is rejected, since neither yields a
    simple boundary polygon.

    Boundary construction: every cell contributes its four edges counter-clockwise, an edge
    shared by two cells appears twice with opposite orientation and cancels, and what survives
    is the boundary already correctly oriented. Consecutive collinear edges are then merged, so
    a straight run of k cells becomes one segment rather than k -- without that, a 3-cell edge
    would arrive as three segments joined by smooth (pi) junctions, which is correct but wasteful.
    """
    cells = [(int(i), int(j)) for i, j in cells]
    if len(set(cells)) != len(cells):
        raise ValueError("duplicate cells")
    cellset = set(cells)
    if not cellset:
        raise ValueError("at least one cell is required")

    # edge-connectivity (diagonal joins do not count)
    seen, stack = {cells[0]}, [cells[0]]
    while stack:
        i, j = stack.pop()
        for nb in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            if nb in cellset and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if seen != cellset:
        raise ValueError("cells must be edge-connected (a diagonal-only join is not enough)")

    # cancel shared edges
    edges = {}
    for i, j in cells:
        for a, b in (((i, j), (i+1, j)), ((i+1, j), (i+1, j+1)),
                     ((i+1, j+1), (i, j+1)), ((i, j+1), (i, j))):
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = True
    if not edges:
        raise ValueError("no boundary edges: degenerate cell set")

    nxt = {}
    for a, b in edges:
        if a in nxt:
            raise ValueError("boundary is not simple (a vertex is left by two edges); "
                             "cells may be pinched at a point or enclose a hole")
        nxt[a] = b

    start = next(iter(nxt))
    loop, v = [start], nxt[start]
    while v != start:
        loop.append(v)
        v = nxt[v]
    if len(loop) != len(nxt):
        raise ValueError("boundary is not a single loop (the cells may enclose a hole)")

    # merge collinear runs
    pts = [complex(x, y) for x, y in loop]
    verts = []
    n = len(pts)
    for k in range(n):
        prev, cur, nxt_pt = pts[k-1], pts[k], pts[(k+1) % n]
        if abs((cur - prev).real*(nxt_pt - cur).imag
               - (cur - prev).imag*(nxt_pt - cur).real) > 1e-12:
            verts.append(cur)
    return Polygon(np.array(verts), bc=bc, val_simple=False)


def plus_shape(bc='dir'):
    """The 5-cell plus/cross polyomino: FOUR reentrant corners, and an exact eigenfunction.
    The multi-singular-corner test case with closed-form truth (see `polyomino`)."""
    return polyomino([(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)], bc=bc)


def H_shape(bc='dir'):
    vx = np.array([-1,  0,  0,  1,  1, 2, 2, 1, 1, 0, 0, -1])
    vy = np.array([-2, -2, -1, -1, -2,-2, 1, 1, 0, 0, 1,  1])
    vertices = vx + 1j*vy
    return Polygon(vertices, bc=bc, val_simple=False)

def reg_ngon(n, bc='dir'):
    theta = np.linspace(0, 2*np.pi, n+1)[:-1]
    vertices = np.exp(1j*theta)
    return Polygon(vertices, bc=bc, val_simple=False)

def spiral(turns=1.5, pitch=1.0, width=0.35, n=12, r0=0.6, bc='dir'):
    """A polygonal spiral strip. The inner coils surround several corners, leaving them
    with no straight-ray branch-cut sightline to infinity (see corner_branch_cut_rays /
    corner_branch_cut_polyline). Vertices are ordered counter-clockwise."""
    phi = np.linspace(0, 2*np.pi*turns, n)
    center = (r0 + pitch*phi/(2*np.pi))*np.exp(1j*phi)
    t = np.gradient(center)
    t /= np.abs(t)
    nrm = 1j*t
    verts = np.concatenate([center + width/2*nrm, (center - width/2*nrm)[::-1]])
    if polygon_area(verts) < 0:
        verts = verts[::-1]
    return Polygon(verts, bc=bc, val_simple=False)

def disk_sector(r=1, theta=np.pi/2, bc='dir', tol=1e-4):
    if not (0 < theta < 2*np.pi):
        raise ValueError("theta must be between 0 and 2pi (strictly)")
    seg1 = LineSegment(0, r, bc=bc)
    seg2 = ParametricSegment(
        lambda t: r*np.exp(1j*t),
        lambda t: 1j*r*np.exp(1j*t),
        0, theta, bc, tol,
        val_simple=False
    )
    seg3 = LineSegment(r*np.exp(1j*theta), 0, bc=bc)
    bdry = MultiSegment([seg1,seg2,seg3])
    return Domain(bdry, val_simple=False, val_closed=False)

def eq_tri(l=1, bc='dir'):
    return Polygon([0,l,l/2 + 1j*l*np.sqrt(3)/2], bc=bc, val_simple=False)

def iso_right_tri(l=1, bc='dir'):
    return Polygon([0, l, 1j*l], bc=bc, val_simple=False)

def iso_tri(h=1, bc='dir'):
    return Polygon([1,1j*h,-1], bc=bc, val_simple=False)

def mushroom(a=1, b=1, r=1.5, bc='dir', tol=1e-4):
    if r <= b:
        raise ValueError('b must be less than r')
    vert = np.array([-r, -b/2, -b/2 - 1j*a, b/2 - 1j*a, b/2, r])
    seg1 = MultiSegment.from_vertices(vert, bc, False)
    seg2 = ParametricSegment(
        lambda t: r*np.exp(1j*t),
        lambda t: 1j*r*np.exp(1j*t),
        0, np.pi, bc, tol
    )
    bdry = MultiSegment([seg1, seg2])
    return Domain(bdry, val_simple=False, val_closed=False)

def right_trapezoid(h1, h2, bc='dir'):
    vertices = np.array([1j*h1, 0, 1, 1+1j*h2])
    return Polygon(vertices, bc=bc, val_simple=False)

def parallelogram(b=1, h=1, alpha=np.pi/3, bc='dir'):
    """Parallelogram with base b, height h, and shear angle alpha (angle between base and left side)."""
    vertices = np.array([0, b, b + h/np.tan(alpha) + 1j*h, h/np.tan(alpha) + 1j*h])
    return Polygon(vertices, bc=bc, val_simple=False)

def stadium(L=1, H=1, bc='dir', tol=1e-4):
    """Bunimovich stadium"""
    seg1 = LineSegment(0-1j*(H/2), L-1j*(H/2), bc=bc)
    seg2 = ParametricSegment(
        lambda t: L + (H/2)*np.exp(1j*t),
        lambda t: 1j*(H/2)*np.exp(1j*t),
        -np.pi/2, np.pi/2, bc, tol
    )
    seg3 = LineSegment(L+1j*(H/2), 1j*(H/2))
    seg4 = ParametricSegment(
        lambda t: (H/2)*np.exp(1j*t),
        lambda t: 1j*(H/2)*np.exp(1j*t),
        np.pi/2, 3*np.pi/2, bc, tol
    )
    bdry = MultiSegment([seg1, seg2, seg3, seg4], val_contiguous=False)
    return Domain(bdry, val_simple=False, val_closed=False)

def ellipse(a=2, b=1, bc='dir', tol=1e-4):
    def p(t): return a*np.cos(t) + 1j*b*np.sin(t)
    def dp(t): return -a*np.sin(t) + 1j*b*np.cos(t)
    seg = ParametricSegment(p, dp, 0, 2*np.pi, bc, tol)
    bdry = MultiSegment([seg], val_simple=False)
    return Domain(bdry, val_simple=False, val_closed=False)
