from .core import BaseSegment, BaseDomain
from .utils import (polygon_area, polygon_diameter, complex_form, real_form, rand_interior_points,
                    interior_angles, edge_lengths, segment_intersection)
from shapely.geometry import Polygon as ShapelyPolygon
from shapely import points as shapely_points
from .opt import find_all_roots
from .quad import (spline_mesh_with_curvature, polygon_triangular_mesh, 
                   tri_quad, cached_leggauss, cached_chebgauss)

from typing import Callable
import numpy as np
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import minimize, minimize_scalar, linprog
import matplotlib.pyplot as plt
from pygmsh.geo import Geometry

class PointSet:
    """Class for pointsets. Written to make caching of basis evaluations easier. Optionally includes weights for 
    numerical quadrature. Designed to be immutable."""
    def __init__(self, points, weights=None):
        self.pts = complex_form(points)
        self.pts.flags.writeable = False
        if weights is not None:
            weights = np.asarray(weights)
            if (weights.ndim != 1) or (weights.shape[0] != self.pts.shape[0]):
                raise ValueError("provided weights must be a 1d array which matches the length of 'points'")
            else:
                self.wts = weights
                self.wts.flags.writeable = False
                self.sqrt_wts = np.sqrt(weights)[:,np.newaxis]
                self.sqrt_wts.flags.writeable = False

        self._hash = hash((self.pts.shape,self.pts.tobytes()))
    
    @property
    def x(self):
        return self.pts.real
    
    @property
    def y(self):
        return self.pts.imag
    
    def __len__(self):
        return len(self.pts)

    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return self.pts is other.pts
    
    def __str__(self):
        return f"PointSet(size={len(self.pts)}, hash={self._hash})"
    
    def __add__(self, other):
        if not isinstance(other, PointSet):
            raise TypeError("'other' must be an instance of PointSet")
        else:
            new_pts = np.concatenate((self.pts, other.pts))
            # neither have weights
            if (not hasattr(self, "wts")) and (not hasattr(other, "wts")):
                return PointSet(new_pts)
            else:
                if hasattr(self, "wts") and hasattr(other, "wts"):
                    new_wts = np.concatenate((self.wts, other.wts))
                elif hasattr(self, "wts"):
                    new_wts = np.concatenate((self.wts, np.ones(len(other.pts))))
                else:
                    new_wts = np.concatenate((np.ones(len(self.pts)), other.wts))
                return PointSet(new_pts, new_wts)

def pts_per_seg(domain, basis, mult=2, min_per_seg=0):
    """Computes how many boundary points to have along each segment of a domain boundary so that each
    corner's basis has enough points on non-adjacent edges."""

    # get the number of basis functions associated to each corner of the domain
    n_basis = np.zeros(len(domain.bdry.segments), dtype='int')
    p0 = np.array([seg.p0 for seg in domain.bdry.segments])
    has_basis = np.any(np.isclose(np.subtract.outer(p0, basis.sources), 0), axis=1)
    n_basis[has_basis] = basis.orders

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
def get_quadfunc(kind):
    if kind == 'legendre': return cached_leggauss
    elif kind == 'chebyshev': return cached_chebgauss
    elif kind == 'even': return lambda n: (np.linspace(0, 1, n+2)[1:-1], np.ones(n)/n)
    else: raise NotImplementedError(f"quadrature rule {kind} is not implemented")

class ParametricSegment(BaseSegment):
    """Class for boundary segments (lines, curves) given in terms of a differentiable function p(t). 
    Handles boundary point placement, boundary tangents and normals.
    All segments are automatically re-parameterized by tau in [0,1]"""
    def __init__(self, p, dp, t0, tf, bc='dir', nsamp=100, val_simple=False, val_closed=False):
        super().__init__(bc)
        if tf <= t0:
            raise ValueError(f"tf ({tf}) must be greater than t0 ({t0})")
        if nsamp < 2: 
            raise ValueError(f"nsamp ({nsamp}) must be at least 2")
            
        # convert to complex vectorized form and store
        self.t0 = t0
        self.tf = tf
        self._p = self._complex_vectorize(p, t0, tf)
        self._dp = self._complex_vectorize(dp, t0, tf)
        self._len = None
        self._speed = lambda t: np.abs(self._dp(t))
        self.nsamp = nsamp

        # reparameterize
        self._reparameterize(nsamp)

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
    
    def to_splineseg(self, spline_bc_type='natural', nsamp=None):
        """Returns a SplineSegment with the same geometry as this segment"""
        if nsamp is None:
            nsamp = self.nsamp
        tau = np.linspace(0, 1, nsamp)
        pts = self.p(tau)
        return SplineSegment.interp_from_pts(pts, self.bc, spline_bc_type, nsamp)
    
    @property
    def is_simple(self):
        if self._is_simple is None:
            self._is_simple = self._validate_simple()
        return self._is_simple

    def _validate_simple(self):
        t = np.linspace(self.t0, self.tf, self.nsamp)
        p = self._p(t)
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
        if self._len is None:
            self._len = self._compute_length()
        return self._len
    
    def _compute_length(self):
        return quad(self._speed, self.t0, self.tf)[0]
    
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

        return wrapped
    
    def _reparameterize(self, nsamp):
        self.nsamp = nsamp
        t_samp = np.linspace(self.t0, self.tf, nsamp)
        speeds = self._speed(t_samp)
        s_samp = np.concatenate([[0], cumulative_simpson(speeds, x=t_samp)])
        self._s_of_t = make_interp_spline(t_samp, s_samp)
        self._t_of_s = make_interp_spline(s_samp, t_samp)

    def _p_of_s(self, s):
        return self._p(self._t_of_s(s))

    def p(self, tau):
        return self._p_of_s(self.len*tau)
    
    def _dp_of_s(self, s):
        return self._dp(self._t_of_s(s))*self._t_of_s(s, nu=1)
    
    def dp(self, tau):
        return self._dp_of_s(self.len*tau)*self.len
    
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

    def pts(self, n, kind='legendre', weights=False):
        """Gets n points spaced along the segment, optionally with quadrature weights"""
        quadfunc = get_quadfunc(kind)
        tau, wts = quadfunc(n)
        pts = self.p(tau)
        if weights:
            return PointSet(pts, wts)
        else:
            return PointSet(pts)
    
    def tangents(self, n, kind='legendre', weights=False):
        """Gets the unit tangent vectors for n points spaced along the segment, optionally with quadrature weights"""
        quadfunc = get_quadfunc(kind)
        tau, wts = quadfunc(n)
        tangents = self.T(tau)
        if weights:
            return PointSet(tangents, wts)
        else:
            return PointSet(tangents)
        
    def normals(self, n, kind='legendre', weights=False):
        """Gets the unit outward normal vectors for n points spaced along the segment, optionally with quadrature weights"""
        quadfunc = get_quadfunc(kind)
        tau, wts = quadfunc(n)
        normals = self.N(tau)
        if weights:
            return PointSet(normals, wts)
        else:
            return PointSet(normals)

    def intersection(self, other):
        if self is other:
            return self
        elif isinstance(other, LineSegment):
            return other.intersection(self)
        elif isinstance(other, ParametricSegment):
            # get points
            tau1 = np.linspace(0, 1, self.nsamp)
            tau2 = np.linspace(0, 1, other.nsamp)
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
        
    def dist(self, pt):
        """Computes the (minimum) distance from the given point to the segment"""
        tau = np.linspace(0, 1, self.nsamp)
        pts = self.p(tau)
        dists = np.abs(pts-pt)
        idx = dists.argmin()

        if idx == 0:
            tau0 = tau[idx]
            tau1 = tau[idx+1]
        elif idx == self.nsamp-1:
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
        return ParametricSegment(p_new, dp_new, self.t0, self.tf, self.bc, self.nsamp)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, BaseSegment):
            return MultiSegment([self, other])
        elif np.isscalar(other):
            p_new = lambda t: self._p(t) + other
            dp_new = self._dp
            return ParametricSegment(p_new, dp_new, self.t0, self.tf, self.bc, self.nsamp)
        else:
            raise TypeError("__add__ with Segment must be another Segment or complex scalar")
            
class LineSegment(BaseSegment):
    """Class for straight line segments.
    Parameters
    ----------
    a : complex
        initial point of segment
    b : complex
        end point of segment
    """
    def __init__(self, p0, pf, bc='dir', nsamp=100):
        if np.isclose(p0, pf):
            raise ValueError("'p0' and 'pf' are too close together to form a line segment")
        super().__init__(bc)
        self._p0 = complex_form(p0)
        self._pf = complex_form(pf)
        self._len = np.abs(self._pf-self._p0)
        self._tangent = (self._pf-self._p0)/self._len
        self._normal = self._tangent.imag - 1j*self._tangent.real
        self.nsamp = nsamp

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
            roots = find_all_roots(signed_distance, 0, 1, other.nsamp)
            intersections = []
            for tau1 in roots:
                p = other.p(tau1)
                tau2 = ((p - self.p0) / d).real
                if 0 <= tau2 <= 1:
                    intersections.append(p)
            return np.array(intersections)
        
    def to_splineseg(self, spline_bc_type='natural', nsamp=None):
        """Returns a SplineSegment with the same geometry as this segment"""
        if nsamp is None:
            nsamp = self.nsamp
        pts = np.array([self.p0, self.pf])
        return SplineSegment.interp_from_pts(pts, self.bc, spline_bc_type, nsamp)
    
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
    
    def pts(self, n, kind='legendre', weights=False):
        """Gets n points spaced along the segment, optionally with quadrature weights"""
        quadfunc = get_quadfunc(kind)
        tau, wts = quadfunc(n)
        pts = self.p(tau)
        if weights:
            return PointSet(pts, wts)
        else:
            return PointSet(pts)
    
    def tangents(self, n, kind='legendre', weights=False):
        """Gets the unit tangent vectors for n points spaced along the segment, optionally with quadrature weights"""
        quadfunc = get_quadfunc(kind)
        tau, wts = quadfunc(n)
        tangents = self.T(tau)
        if weights:
            return PointSet(tangents, wts)
        else:
            return PointSet(tangents)
        
    def normals(self, n, kind='legendre', weights=False):
        """Gets the unit outward normal vectors for n points spaced along the segment, optionally with quadrature weights"""
        quadfunc = get_quadfunc(kind)
        tau, wts = quadfunc(n)
        normals = self.N(tau)
        if weights:
            return PointSet(normals, wts)
        else:
            return PointSet(normals)

    def dist(self, pt):
        """Computes the distance from pt to this line segment."""
        d = self._pf - self._p0
        t = float(np.clip(((pt - self._p0) * d.conjugate()).real / (self._len ** 2), 0.0, 1.0))
        return float(np.abs(pt - (self._p0 + t * d)))
    
    def __mul__(self, other):
        if not np.isscalar(other):
            raise ValueError("non-segment operand must be a scalar")
        return LineSegment(other*self.p0, other*self.pf, self.bc, self.nsamp)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        if isinstance(other, BaseSegment):
            return MultiSegment([self, other])
        elif np.isscalar(other):
            return LineSegment(self.p0 + other, self.pf + other, self.bc, self.nsamp)
        else:
            raise TypeError("__add__ with Segment must be another Segment or complex scalar")

class SplineSegment(ParametricSegment):
    """Segments with spline boundary"""
    def __init__(self, spline, t0=None, tf=None, bc='dir', nsamp=100, val_simple=False, val_closed=False):
        if not isinstance(spline, BSpline):
            raise TypeError("'spline' must be an instance of BSpline")
        self.spline = spline
        p = lambda t: self.spline(t)
        dp = lambda t: self.spline(t, nu=1)
        if t0 is None:
            t0_idx = self.spline.k
            t0 = self.spline.t[t0_idx]
        if tf is None:
            tf_idx = len(self.spline.t)-spline.k-1
            tf = self.spline.t[tf_idx]
        super().__init__(p, dp, t0, tf, bc, nsamp, val_simple, val_closed)
    
    @classmethod
    def interp_from_pts(cls, pts, bc='dir', spline_bc_type='natural',
                        nsamp=100, val_simple=False, val_closed=False):
        """Builds a BSpline segment interpolating the given points."""
        t = np.linspace(0, 1, len(pts))
        spline = make_interp_spline(t, pts, bc_type=spline_bc_type)
        return cls(spline, 0, 1, bc, nsamp, val_simple, val_closed)
    
    def _compute_length(self):
        # Get unique internal knots (excluding boundary repeats)
        T = np.unique(np.concatenate([[self.t0, self.tf], self.spline.t]))
        knots = np.sort(T[(T >= self.t0) & (T <= self.tf)])
        total_length = 0
        for i in range(len(knots) - 1):
            length, _ = quad(self._speed, knots[i], knots[i+1])
            total_length += length
        
        return total_length
    
    def to_splineseg(self):
        return self
    
    def __mul__(self, other):
        if not np.isscalar(other):
            raise ValueError("non-segment operand must be a scalar")
        else:
            spline = self.spline
            newspline = BSpline(spline.t, other*spline.c, spline.k, spline.extrapolate, spline.axis)
            return SplineSegment(newspline, self.t0, self.tf, self.bc, self.nsamp)
        
    def __add__(self, other):
        if np.isscalar(other):
            spline = self.spline
            newspline = BSpline(spline.t, spline.c + other, spline.k, spline.extrapolate, spline.axis)
            return SplineSegment(newspline, self.t0, self.tf, self.bc, self.nsamp)
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
    def from_vertices(cls, vertices, bc='dir', make_closed=True, val_simple=False, nsamp=100):
        """Builds a polygonal MultiSegment from the given vertices"""
        segments = [LineSegment(vertices[i], vertices[i+1], bc, nsamp) for i in range(len(vertices)-1)]
        if make_closed:
            segments += [LineSegment(vertices[-1], vertices[0], bc, nsamp)]
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
                intersections = segments[i].intersection(segments[j])
                if len(intersections) > 0:
                    return False
        return True
    
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
        return np.sum([seg.len for seg in self.segments])

    @property
    def len(self):
        if self._len is None:
            self._len = self._compute_length()
        return self._len

    @property
    def is_polyline(self):
        return all(isinstance(seg, LineSegment) for seg in self.segments)
    
    @property
    def bcs(self):
        return [seg.bc for seg in self.segments]
    
    @property
    def bc_types(self):
        return [seg.bc_type for seg in self.segments]

    def pts(self, N, kind='legendre', weights=False):
        """Places N[i] points on ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self.segments), N)
        return np.sum([self.segments[i].pts(N[i], kind, weights) for i in range(len(self.segments)) if N[i] > 0])

    def tangents(self, N, kind='legendre', weights=False):
        """Computes tangent vectors at N[i] points along the ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self.segments), N)
        return np.sum([self.segments[i].tangents(N[i], kind, weights) for i in range(len(self.segments)) if N[i] > 0])

    def normals(self, N, kind='legendre', weights=False):
        """Computes normal vectors at N[i] points along the ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self.segments), N)
        return np.sum([self.segments[i].normals(N[i], kind, weights) for i in range(len(self.segments)) if N[i] > 0])
    
    def _find_corners(self):
        """Finds corners, i.e. where two boundary segments do not smoothly connect. Computes
        angle wedge there."""
        segments = self.segments
        T0 = [seg.T0 for seg in segments]
        Tf = [seg.Tf for seg in segments]
        corner_idx = []
        corner_angle0 = []
        corner_angle1 = []
        if len(segments) == 1:
            if not np.isclose(T0[0],Tf[0]):
                corner_idx.append(0)
                corner_angle0.append(np.angle(T0[0]))
                corner_angle1.append(np.angle(-Tf[0]))
        else:
            for i in range(len(segments)):
                if not np.isclose(T0[i],Tf[i-1]):
                    corner_idx.append(i)
                    corner_angle0.append(np.angle(T0[i]))
                    corner_angle1.append(np.angle(-Tf[i-1]))
        corners = np.array([seg.p0 for seg in segments])[corner_idx]
        corner_angle0 = np.array(corner_angle0)
        corner_angle1 = np.array(corner_angle1)
        corner_angle0[corner_angle0 < 0] += 2*np.pi
        corner_angle1[corner_angle1 < 0] += 2*np.pi
        return corners, corner_idx, corner_angle0, corner_angle1
    
    def dist(self, pt, nsamp=100):
        """Returns the (minimum) distance from a given point to the MultiSegment"""
        tau = np.linspace(0, 1, nsamp)[:-1]
        pts = np.array([seg.p(tau) for seg in self.segments])
        dist = np.abs(pts-pt)
        seg_idx = dist.min(axis=1).argmin()

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
    """
    def __init__(self, bdry, val_simple=True, val_closed=True):
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
            
    def _compute_area(self, n=20):
        tau, wts = cached_leggauss(n)
        P = np.array([seg.p(tau) for seg in self.bdry.segments])
        dP = np.array([seg.dp(tau) for seg in self.bdry.segments])
        # Green's formula
        I = ((P.real*dP.imag - P.imag*dP.real)@wts).sum()
        return np.abs(I)/2
    
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

        for seg in self.bdry.segments:
            if isinstance(seg, LineSegment):
                vertices = np.array([seg.p0, seg.pf])
            else:
                vertices = seg.p(np.linspace(0, 1, seg.nsamp + 1))

            for k in range(len(vertices) - 1):
                x0, y0 = vertices[k].real, vertices[k].imag
                x1, y1 = vertices[k+1].real, vertices[k+1].imag
                crosses = ((y0 <= pt_y) & (pt_y < y1)) | ((y1 <= pt_y) & (pt_y < y0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = np.where(crosses, (pt_y - y0) / (y1 - y0), 0.0)
                x_cross = x0 + t * (x1 - x0)
                inside ^= crosses & (x_cross > pt_x)

        return inside
    
    def bdry_pts(self, n_per_seg, kind='legendre', weights=False):
        return self.bdry.pts(n_per_seg, kind=kind, weights=weights)
    
    def bdry_tangents(self, n_per_seg, kind='legendre', weights=False):
        return self.bdry.tangents(n_per_seg, kind=kind, weights=weights)
    
    def bdry_normals(self, n_per_seg, kind='legendre', weights=False):
        return self.bdry.normals(n_per_seg, kind=kind, weights=weights)
    
    def bdry_data(self, n_per_seg, kind='legendre', weights=False):
        if isinstance(n_per_seg, (int, np.integer)):
            n_per_seg = np.full(len(self.bdry.segments), n_per_seg)
        bdry_pts = self.bdry_pts(n_per_seg, kind=kind, weights=weights)
        bdry_normals = self.bdry_normals(n_per_seg, kind=kind, weights=False)
        bc_param = np.concatenate([np.full(n, seg.bc, 'float') for seg, n in zip(self.bdry.segments, n_per_seg)])
        return bdry_pts, bdry_normals, bc_param
    
    def int_pts(self, method='random', weights=False, kind='dunavant', deg=4, mesh_kwargs={}, n_bdry=100, 
                npts_rand=50, oversamp=2):
        """Gets interior points for the domain."""
        if method == 'random':
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
                x = (xmax-xmin)*np.random.rand(npts)+xmin
                y = (ymax-ymin)*np.random.rand(npts)+ymin
                z = x + 1j*y
                pts_new = z[self.contains(z)]
                pts = np.concatenate((pts, pts_new))
                oversamp = 2*oversamp
            else:
                raise RuntimeError("int_pts: rejection sampling failed to collect enough interior points")
            int_pts = pts[:npts_rand]
            wts = np.full(npts_rand, self.area / npts_rand)

        elif method == 'mesh':
            splinesegs = [seg.to_splineseg() for seg in self.bdry.segments]
            mesh = spline_mesh_with_curvature(splinesegs, **mesh_kwargs)
            int_pts, wts = tri_quad(mesh, kind, deg)

        if weights: return PointSet(int_pts, wts)
        else: return PointSet(int_pts)
        
    @property
    def corners(self):
        return self.bdry.corners
    
    @property
    def corner_angles(self):
        return self.bdry.corner_angles
    
    @property
    def corner_idx(self):
        return self.bdry.corner_idx

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

# polygon class
class Polygon(Domain):
    """Class for polygonal domains"""
    def __init__(self, vertices=None, bdry=None, bc='dir', val_simple=True):
        if not ((vertices is None)^(bdry is None)):
            raise ValueError("exactly one of 'vertices' and 'bdry' must be provided")
        elif vertices is not None:
            vertices = complex_form(vertices)
            bdry = MultiSegment.from_vertices(vertices, bc)
            self.vertices = vertices
        elif bdry is not None:
            if not isinstance(bdry, MultiSegment) or not bdry.is_polyline:
                    raise TypeError("'bdry' must be a polyline MultiSegment")
            elif val_simple:
                if not bdry.is_simple:
                    raise ValueError("'bdry' must be simple")
                self.vertices = np.array([seg.p0 for seg in bdry.segments])
            elif not bdry.is_closed:
                raise ValueError("'bdry' must be closed")
            else:
                self.vertices = np.array([seg.p0 for seg in bdry.segments])
        super().__init__(bdry, val_simple=False, val_closed=False)

    def _compute_area(self):
        return polygon_area(self.vertices)

    def _compute_diameter(self):
        return polygon_diameter(self.vertices)

    def _compute_inradius(self):
        """Computes inradius exactly for convex polygons via LP; falls back to parent for non-convex."""
        if not np.all(self.int_angles <= np.pi + 1e-10):
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
    
    def int_pts(self, method='random', weights=False, kind='dunavant', deg=4, mesh_size=1, npts_rand=50, oversamp=2):
        if method == 'random':
            pts = rand_interior_points(self.vertices, npts_rand, oversamp)
            if weights: int_pts = PointSet(pts, np.full(npts_rand, self.area / npts_rand))
            else: int_pts = PointSet(pts)

        elif method == 'mesh':
            mesh = polygon_triangular_mesh(self.vertices, mesh_size)
            int_pts, int_wts = tri_quad(mesh, kind, deg)
            if weights: int_pts = PointSet(int_pts, int_wts)
            else: int_pts = PointSet(int_pts)

        return int_pts

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

def circle(r=1, bc='dir', nsamp=100):
    """circle of radius r"""
    seg = ParametricSegment(
        lambda t: r*np.exp(1j*t),
        lambda t: 1j*r*np.exp(1j*t),
        0, 2*np.pi, bc, nsamp
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

def cut_square(r=0.5, bc='dir', nsamp=100):
    """cut square domain"""
    if not (0 < r < 1):
        raise ValueError("r must be between 0 and 1 (strictly)")
    seg1 = LineSegment(0, 1, bc=bc)
    seg2 = LineSegment(1, 1 + (1-r)*1j, bc=bc)
    seg3 = ParametricSegment(lambda t: 1+1j+r*np.exp(-1j*t),
                             lambda t: -1j*r*np.exp(-1j*t),
                             np.pi/2, np.pi, bc, nsamp)
    seg4 = LineSegment((1-r)+1j, 1j, bc=bc)
    seg5 = LineSegment(1j, 0, bc=bc)
    bdry = MultiSegment([seg1, seg2, seg3, seg4, seg5])
    return Domain(bdry, val_simple=False, val_closed=False)

def H_shape(bc='dir'):
    vx = np.array([-1,  0,  0,  1,  1, 2, 2, 1, 1, 0, 0, -1])
    vy = np.array([-2, -2, -1, -1, -2,-2, 1, 1, 0, 0, 1,  1])
    vertices = vx + 1j*vy
    return Polygon(vertices, bc=bc, val_simple=False)

def reg_ngon(n, bc='dir'):
    theta = np.linspace(0, 2*np.pi, n+1)[:-1]
    vertices = np.exp(1j*theta)
    return Polygon(vertices, bc=bc, val_simple=False)

def circle_sector(r=1, theta=np.pi/2, bc='dir', nsamp=100):
    if not (0 < theta < 2*np.pi):
        raise ValueError("theta must be between 0 and 2pi (strictly)")
    seg1 = LineSegment(0, r, bc=bc)
    seg2 = ParametricSegment(
        lambda t: r*np.exp(1j*t),
        lambda t: 1j*r*np.exp(1j*t),
        0, theta, bc, nsamp,
        val_simple=False
    )
    seg3 = LineSegment(r*np.exp(1j*theta), 0, bc=bc)
    bdry = MultiSegment([seg1,seg2,seg3])
    return Domain(bdry, val_simple=False, val_closed=False)

def iso_right_tri(l=1, bc='dir'):
    return Polygon([0, l, 1j*l], bc=bc, val_simple=False)

def iso_tri(h=1, bc='dir'):
    return Polygon([1,1j*h,-1], bc=bc, val_simple=False)

def mushroom(a=1, b=1, r=1.5, bc='dir', nsamp=100):
    if r <= b:
        raise ValueError('b must be less than r')
    vert = np.array([-r, -b/2, -b/2 - 1j*a, b/2 - 1j*a, b/2, r])
    seg1 = MultiSegment.from_vertices(vert, bc, False)
    seg2 = ParametricSegment(
        lambda t: r*np.exp(1j*t),
        lambda t: 1j*r*np.exp(1j*t),
        0, np.pi, bc, nsamp
    )
    bdry = MultiSegment([seg1, seg2])
    return Domain(bdry, val_simple=False, val_closed=False)
