from .core import BaseSegment, BaseDomain
from .utils import (polygon_area, polygon_diameter, complex_form, real_form, rand_interior_points,
                    interior_angles, edge_lengths, segment_intersection)
from .opt import find_all_roots
from .quad import (spline_mesh_with_curvature, polygon_triangular_mesh, 
                   tri_quad, cached_leggauss, cached_chebgauss)

import numpy as np
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import minimize, minimize_scalar
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
    def __init__(self, p, dp, t0, tf, bc='dir', nsamp=100, val_simple=True, val_closed=False):
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

        # reparameterize
        self._speed = lambda t: np.abs(self._dp(t))
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
    def _complex_vectorize(f, t0, tf):
        """Puts the function f in a vectorized form with complex-valued outputs"""
        out = f(t0)
        is_complex = np.isscalar(out) and np.iscomplexobj(out)
        is_real_pair = (isinstance(out, (list, tuple, np.ndarray)) and 
                    len(out) == 2)

        if (not is_complex) and (not is_real_pair):
            raise ValueError("f must have complex-scalar outputs or length-2 real outputs")

        t_test = np.linspace(t0, tf, 3)
        try:
            out = f(t_test)
            if out.shape == (2,3): f_vec = lambda t: f(t).T
            else: f_vec = f
        except:
            if is_complex:
                f_vec = np.vectorize(f, signature="() -> ()")
            elif is_real_pair:
                f_vec = np.vectorize(f, signature="() -> (2)")

        if is_real_pair:
            def f_cvec(t):
                arr = f_vec(t)
                return arr[...,0] + 1j*arr[...,1]
        elif is_complex:
            f_cvec = f_vec
        return f_cvec
    
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
        """Unit tanget vector in terms of t"""
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
        t = np.asarray(t)
        """Unit outward normal vector in terms of t"""
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
        idx = dists.argmax()

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

        res = minimize_scalar(f, bounds=(tau0, tau1), options={'xatol':1e-14})
        return res.fun
            
class LineSegment(BaseSegment):
    """Class for straight line segments.
    Parameters
    ----------
    a : complex
        initial point of segment
    b : complex
        end point of segment
    """
    def __init__(self, a, b, bc='dir', nsamp=100):
        if np.isclose(a,b):
            raise ValueError("'a' and 'b' are too close together to form a line segment")
        super().__init__(bc)
        self._p0 = complex_form(a)
        self._pf = complex_form(b)
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
    def interp_from_pts(cls, pts, bc='dir', alpha=None, beta=None, spline_bc_type='natural', 
                        nsamp=100, val_simple=False, val_closed=False):
        """builds a Bspline segment to interpolate the given points"""
        t = np.linspace(0, 1, len(pts))
        pts = real_form(pts)
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

class MultiSegment:
    """Class for simple planar curves composed of segments. Simplicity may be enforced, as well as closure"""
    def __init__(self, segments, val_closed=False, val_simple=False):
        if not all(isinstance(seg, BaseSegment) for seg in segments):
            raise TypeError("segments must be an iterable of Segment objects")
        if val_closed:
            if not self._validate_closed(segments):
                raise ValueError("segments must form a closed curve")
            else: self._is_closed = True
        else:
            self._is_closed = None
        if val_simple:
            if not self._validate_simple(segments):
                raise ValueError("segments must form a simple curve")
            else: self._is_simple = True
        else:
            self._is_simple = None
        self.segments = segments
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
        multiseg = cls(segments, val_simple=val_simple)
        if make_closed: multiseg._is_closed = True
        return multiseg

    @staticmethod
    def _validate_closed(segments):
        if len(segments) == 1:
            return segments[0].is_closed
        else:
            # get initial and final points for each segment
            p0 = np.array([seg.p0 for seg in segments])
            pf = np.array([seg.pf for seg in segments])
    
            # check if initial and final points match
            return np.allclose(p0, np.roll(pf, 1))

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
        seg_idx = dist.max(axis=1).argmax()

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

# domain class
class Domain(BaseDomain):
    def __init__(self, bdry):
        if not isinstance(bdry, MultiSegment):
            raise TypeError("'bdry' must be an instance of MultiSegment")
        if not bdry.is_closed:
            raise ValueError("boundary must be closed")
        if not bdry.is_simple:
            raise ValueError("boundary must be simple")
        self.bdry = bdry
        self._area = None
        self._diameter = None
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
        self._area = np.abs(I)/2
        return self._area
    
    @property
    def diameter(self):
        if self._diameter is None:
            self._diameter = self._compute_diameter()
        return self._diameter
    
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
        elif tau_idx == n-1:
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
        self._diameter = float(-res.fun)
        return self._diameter
    
    def _winding_number(self, pts, n=100):
        tau, wts = cached_leggauss(n)
        P = np.array([seg.p(tau) for seg in self.bdry.segments])
        dP = np.array([seg.dp(tau) for seg in self.bdry.segments])
        dx = np.subtract.outer(pts.real, P.real)
        dy = np.subtract.outer(pts.imag, P.imag)
        num = -dx*dP.imag + dy*dP.real
        denom = dx**2 + dy**2
        I = (np.divide(num, denom, out=np.full(num.shape, np.inf), where=(denom!=0))*wts).sum(axis=(1,2))/(2*np.pi)
        return I
    
    def contains(self, pts, n=100):
        """Checks if the domain contains the given points."""
        pts = complex_form(pts)
        w = np.abs(self._winding_number(pts, n))
        return (0.9 < w)&(w < 1.1)
    
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
            while len(pts) < npts_rand:
                npts = int(np.ceil(npts_rand*oversamp*box_area/self.area))
                x = (xmax-xmin)*np.random.rand(npts)+xmin
                y = (ymax-ymin)*np.random.rand(npts)+ymin
                z = x + 1j*y
                pts_new = z[self.contains(z, n_bdry)]
                pts = np.concatenate((pts, pts_new))
                oversamp = 2*oversamp
            int_pts = pts[:npts_rand]
            if weights: wts = self.area/npts_rand

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

# polygon class
class Polygon(Domain):
    """Class for polygonal domains"""
    def __init__(self, vertices=None, bdry=None, val_simple=True):
        if not ((vertices is None)^(bdry is None)):
            raise ValueError("exactly one of 'vertices' and 'bdry' must be provided")
        elif vertices is not None:
            vertices = complex_form(vertices)
            bdry = MultiSegment.from_vertices(vertices, val_simple=val_simple)
            self.vertices = vertices
        elif bdry is not None:
            if not isinstance(bdry, MultiSegment) or not bdry.is_polyline:
                    raise TypeError("'bdry' must be a polyline MultiSegment")
            elif val_simple:
                if not bdry.is_simple:
                    raise ValueError("'bdry' must be simple")
            elif not bdry.is_closed:
                raise ValueError("'bdry' must be closed")
            else:
                self.vertices = np.array([seg.p0 for seg in bdry.segments])
        super().__init__(bdry)

    def _compute_area(self):
        self._area = polygon_area(self.vertices)
        return self._area

    def _compute_diameter(self):
        self._diameter = polygon_diameter(self.vertices)
        return self._diameter

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
            if weights: int_pts = PointSet(pts, self.area/npts_rand)
            else: int_pts = PointSet(pts)

        elif method == 'mesh':
            mesh = polygon_triangular_mesh(self.vertices, mesh_size)
            int_pts, int_wts = tri_quad(mesh, kind, deg)
            if weights: int_pts = PointSet(int_pts, int_wts)
            else: int_pts = PointSet(int_pts)

        return int_pts