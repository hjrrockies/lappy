from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import norm
from .utils import (polygon_area, polygon_perimeter, polygon_diameter, complex_form, real_form, rand_interior_points,
                    interior_angles, edge_lengths, side_normals, polygon_edges, plot_polygon, segment_intersection)
from .opt import find_all_roots
from .quad import triangular_mesh, tri_quad, boundary_nodes_polygon
import shapely.geometry as geo
from shapely import points
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import minimize
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
from functools import cache
import matplotlib.pyplot as plt

### Quadrature rules for segments
@cache
def cached_leggauss(order):
    nodes,weights = leggauss(order)
    nodes = (nodes+1)/2 #adjust nodes to interval [0,1]
    weights = weights/2 #adjust weights to interval of unit length
    return nodes, weights

@cache
def cached_chebgauss(order):
    nodes,weights = chebgauss(order)
    # adjust the weights to cancel-out the Gauss-Cheb weighting function
    weights = weights*np.sqrt(1-nodes**2)
    nodes = (nodes+1)/2 # adjust nodes to interval [0,1]
    weights = weights/2 # adjust weights to interval of unit length
    return nodes[::-1], weights[::-1]

class PointSet:
    """Class for pointsets. Written to make caching of basis evaluations easier. Optionally includes weights for 
    numerical quadrature. Designed to be immutable."""
    def __init__(self, points, weights=None):
        self._pts = complex_form(points)
        self._pts.flags.writeable = False
        if weights is not None:
            weights = np.asarray(weights)
            if (weights.ndim != 1) or (weights.shape[0] != self.pts.shape[0]):
                raise ValueError("provided weights must be a 1d array which matches the length of 'points'")
            else:
                self._wts = weights
                self._wts.flags.writeable = False
                self._sqrt_wts = np.sqrt(weights)[:,np.newaxis]
                self._sqrt_wts.flags.writeable = False

        self._hash = hash((self._pts.shape,self._pts.tobytes()))

    @property
    def pts(self):
        return self._pts
    
    @property
    def wts(self):
        return self._wts
    
    @property
    def sqrt_wts(self):
        return self._sqrt_wts
    
    @property
    def x(self):
        return self._pts.real
    
    @property
    def y(self):
        return self._pts.imag
    
    def __len__(self):
        return len(self._pts)

    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return self._pts is other._pts
    
    def __str__(self):
        return f"PointSet(size={len(self._pts)}, hash={self._hash})"
    
    def __add__(self, other):
        if not isinstance(other, PointSet):
            raise TypeError("'other' must be an instance of PointSet")
        else:
            new_pts = np.concatenate((self._pts, other._pts))
            # neither have weights
            if (not hasattr(self, "_wts")) and (not hasattr(other, "_wts")):
                return PointSet(new_pts)
            else:
                if hasattr(self, "_wts") and hasattr(other, "_wts"):
                    new_wts = np.concatenate((self._wts, other._wts))
                elif hasattr(self, "_wts"):
                    new_wts = np.concatenate((self._wts, np.ones(len(other._pts))))
                else:
                    new_wts = np.concatenate((np.ones(len(self._pts)), other._wts))
                return PointSet(new_pts, new_wts)
    
def make_boundary_pts(domain, n_pts=None, basis=None, weights=True, **kwargs):
    if not isinstance(domain, Domain):
        raise TypeError("'domain' must be an instance of Domain")
    
    elif isinstance(domain, Polygon):
        if n_pts is not None:
            return polygon_boundary_pts(domain, n_pts, weights, **kwargs)
        elif basis is not None:
            n_per_edge = n_bdry_pts(domain, basis, **{key: value for key, value in kwargs.items() 
                                    if key in n_bdry_pts.__code__.co_varnames})
            return polygon_boundary_pts(domain, n_per_edge, weights, **{key: value for key, value in kwargs.items() 
                                        if key in polygon_boundary_pts.__code__.co_varnames})
        else:
            raise ValueError("One of 'n_pts' or 'basis' must be defined.")
    
def n_bdry_pts(poly, basis, mult=1, min_per_edge=3):
    """Computes how many boundary points to have along each edge of a polygon so that each
    corner's basis has enough points on non-adjacent edges."""
    edge_lengths = poly.edge_lengths
    # get the adjacent edge lengths to ith vertex into the first and last positions of column i, then drop rows
    rolled_lengths = np.array([np.roll(edge_lengths, -j) for j in range(len(edge_lengths))])[1:-1]
    # normalize each column by sum of edge lengths
    normalized_lengths = rolled_lengths/rolled_lengths.sum(axis=0)
    # multiply by orders, take ceiling
    n_per_edge = np.ceil(mult*basis.orders*normalized_lengths)
    # unroll and sum
    n_per_edge = np.array([np.roll(n_per_edge[i], i+1) for i in range(len(n_per_edge))]).sum(axis=0)
    # threshold with min_per_edge
    n_per_edge = np.maximum(n_per_edge, min_per_edge).astype('int')
    return n_per_edge
    
def polygon_boundary_pts(poly, n_pts, weights=True, kind="legendre", tans=False, norms=False, reentrant_mult=1.25):
    """makes boundary points (and weights) for the polygon"""
    if not isinstance(poly, Polygon):
        raise TypeError("'poly' must be an instance of Polygon")
    
    # space ~n_pts in total around edges
    if isinstance(n_pts, (int, np.integer)):
        pfrac = poly.edge_lengths/poly.perimeter

        # put more points on edges with reentrant corners
        pfrac_new = pfrac.copy()
        reentrant = poly.int_angles > np.pi
        reentrant = np.logical_or(reentrant,np.roll(reentrant, -1))
        pfrac_new[reentrant] = reentrant_mult*pfrac[reentrant]

        # n_pts proportional to edge length and reentrant weighting
        pts_per_edge = np.ceil(n_pts*pfrac_new).astype('int')
    else:
        pts_per_edge = n_pts

    bdry_pts = poly._bdry.pts(pts_per_edge, kind, weights)
    if tans: bdry_tans = poly._bdry.tangents(pts_per_edge, kind, weights)
    else: bdry_tans = None
    if norms: bdry_norms = poly._bdry.normals(pts_per_edge, kind, weights)
    else: bdry_norms = None
    if tans or norms:
        out = [pts for pts in [bdry_pts, bdry_tans, bdry_norms] if pts is not None]
        return tuple(out)
    else:
        return bdry_pts

def make_interior_pts(domain, method, weights=True, **kwargs):
    if not isinstance(domain, Domain):
        raise TypeError("'domain' must be an instance of Domain")
    
    elif isinstance(domain, Polygon):
        return polygon_interior_pts(domain, method, weights, **kwargs)
    
def polygon_interior_pts(poly, method, weights=True, mesh_size=None, kind='dunavant', order=14, eps=1e-14, npts_rand=50):
    """makes interior points (and weights) for the polygon"""
    if not isinstance(poly, Polygon):
        raise TypeError("'poly' must be an instance of Polygon")
    
    if method == "mesh":
        if mesh_size is None: mesh_size = poly.diameter*(eps**(1/order))
        else: mesh_size = mesh_size
        mesh = triangular_mesh(poly._vertices, mesh_size)
        int_pts, int_wts = tri_quad(mesh, kind=kind, deg=order)

    elif method == "random":
        int_pts = rand_interior_points(poly._vertices, npts_rand)
        int_wts = (poly.area/npts_rand)*np.ones(len(int_pts))

    if weights:
        return PointSet(int_pts, int_wts)
    else:
        return PointSet(int_pts)
    
# segment classes
class Segment:
    """Base class for boundary segments (lines, curves). Handles boundary point placement, boundary tangents and normals.
    All segments are automatically re-parameterized by tau in [0,1]"""
    def __init__(self, p, dp, t0, tf, nsamp=100, val_closed=False, val_simple=False):
        if tf <= t0:
            raise ValueError(f"tf ({tf}) must be greater than t0 ({t0})")
        if nsamp < 2: 
            raise ValueError(f"nsamp ({nsamp}) must be at least 2")
            
        self._t0 = t0
        self._tf = tf
        self._p = self._complex_vectorize(p)
        self._dp = self._complex_vectorize(dp)

        if val_closed:
            if not self._validate_closed():
                raise ValueError("segments must form a closed curve")
            else: self._is_closed = True
        else:
            self._is_closed = None
        if val_simple:
            if not self._validate_simple():
                raise ValueError("segments must form a simple curve")
            else: self._is_simple = True

        # cached properties
        self._len = None
        self._is_simple = None
        self._is_closed = None

        # reparameterize
        self._speed = lambda t: np.abs(self._dp(t))
        self._reparameterize(nsamp)

    def __str__(self):
        return f"Segment({self.p0},{self.pf})"

    def _validate_closed(self):
        return np.isclose(self._p(self._t0), self._p(self._tf))

    def _validate_simple(self):
        t = np.linspace(self._t0, self._tf, self.nsamp)
        p = self._p(t)
        for i in range(len(p)-1):
            for j in range(i+2,len(p)-1):
                pt = segment_intersection(p[i], p[i+1], p[j], p[j+1])
                if pt is not None and not np.isclose(pt, self.p0) and not np.isclose(pt, self.pf):
                    return False
        return True

    @property
    def is_simple(self):
        if self._is_simple is None:
            self._is_simple = self._validate_simple()
        return self._is_simple

    @property
    def is_closed(self):
        if self._is_closed is None:
            self._is_closed = self._validate_closed()
        return self._is_closed
        
    @property
    def len(self):
        if self._len is None:
            self._len = self._compute_length()
        return self._len

    def _compute_length(self):
        return quad(self._speed, self._t0, self._tf)[0]

    @property
    def t0(self):
        return self._t0

    @property
    def tf(self):
        return self._tf
    
    @property
    def p0(self):
        return self._p(self._t0)
    
    @property
    def pf(self):
        return self._p(self._tf)

    def _complex_vectorize(self, f):
        """Puts the function f in a vectorized form with complex-valued outputs"""
        out = f(self._t0)
        is_complex = np.isscalar(out) and np.iscomplexobj(out)
        is_real_pair = (isinstance(out, (list, tuple, np.ndarray)) and 
                    len(out) == 2)

        if (not is_complex) and (not is_real_pair):
            raise ValueError("f must have complex-scalar outputs or length-2 real outputs")

        t_test = np.linspace(self._t0, self._tf, 3)
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
    
    def _reparameterize(self, nsamp=100):
        self.nsamp = nsamp
        t_samp = np.linspace(self._t0, self._tf, nsamp)
        speeds = self._speed(t_samp)
        s_samp = np.concatenate([[0], cumulative_simpson(speeds, x=t_samp)])
        self.s_of_t = make_interp_spline(t_samp, s_samp)
        self.t_of_s = make_interp_spline(s_samp, t_samp)

    def _p_of_s(self, s):
        return self._p(self.t_of_s(s))

    def p(self, tau):
        return self._p_of_s(self.len*tau)
    
    def _dp_of_s(self, s):
        return self._dp(self.t_of_s(s))*self.t_of_s(s, nu=1)
    
    def dp(self, tau):
        return self._dp_of_s(self.len*tau)*self.len
    
    def _T(self, t):
        """Unit tanget vector in terms of t"""
        return self._dp(t)/self._speed(t)
    
    def _T_of_s(self, s):
        """Unit tangent vector in terms of arclength s"""
        return self._T(self.t_of_s(s))
    
    def T(self, tau):
        """Unit tangent vector in terms of tau from [0,1]"""
        return self._T_of_s(self.len*tau)
    
    def _N(self, t):
        """Unit outward normal vector in terms of t"""
        T = self._T(t)
        return T.imag - 1j*T.real
    
    def _N_of_s(self, s):
        """Unit outward normal vector in terms of arclength s"""
        return self._N(self.t_of_s(s))
    
    def N(self, tau):
        """Unit outward normal vector (unit tangent rotated 90deg clockwise) in terms of tau from [0,1]"""
        return self._N_of_s(self.len*tau)
    
    @property
    def T0(self):
        """Unit tangent vector at initial point"""
        return self._T(self.t0)
    
    @property
    def Tf(self):
        """Unit tangent vector at final point"""
        return self._T(self.tf)

    @cache
    def pts(self, n, kind='legendre', weights=True):
        """Gets n points spaced along the segment, optionally with quadrature weights"""
        if kind == 'chebyshev': quadfunc = cached_chebgauss
        elif kind == 'legendre': quadfunc = cached_leggauss
        elif kind == 'even': quadfunc = lambda n: (np.linspace(0,1,n+2)[1:-1], np.ones(n)/n)
        tau, w = quadfunc(n)
        pts = self.p(tau)
        if weights:
            return PointSet(pts, self.len*w)
        else:
            return PointSet(pts)
    
    @cache
    def tangents(self, n, kind='legendre', weights=True):
        """Gets the unit tangent vectors for n points spaced along the segment, optionally with quadrature weights"""
        if kind == 'chebyshev': quadfunc = cached_chebgauss
        elif kind == 'legendre': quadfunc = cached_leggauss
        elif kind == 'even': quadfunc = lambda n: (np.linspace(0,1,n+2)[1:-1], np.ones(n)/n)
        tau, w = quadfunc(n)
        tangents = self.T(tau)
        if weights:
            return PointSet(tangents, self.len*w)
        else:
            return PointSet(tangents)
        
    @cache
    def normals(self, n, kind='legendre', weights=True):
        """Gets the unit outward normal vectors for n points spaced along the segment, optionally with quadrature weights"""
        if kind == 'chebyshev': quadfunc = cached_chebgauss
        elif kind == 'legendre': quadfunc = cached_leggauss
        elif kind == 'even': quadfunc = lambda n: (np.linspace(0,1,n+2)[1:-1], np.ones(n)/n)
        tau, w = quadfunc(n)
        normals = self.N(tau)
        if weights:
            return PointSet(normals, self.len*w)
        else:
            return PointSet(normals)

    def intersection(self, other):
        if self is other:
            return self
        elif isinstance(other, LineSegment):
            return other.intersection(self)
        elif isinstance(other, Segment):
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
            

    def plot(self, nsamp=None, ax=None, **plotkwargs):
        if nsamp is None:
            nsamp = self.nsamp
        tau = np.linspace(0, 1, nsamp)
        pts = self.p(tau)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        return ax.plot(pts.real, pts.imag, **plotkwargs)

    def plot_tangents(self, nsamp=None, ax=None, **plotkwargs):
        if nsamp is None:
            nsamp = int(np.ceil(self.nsamp/10))
        tau = np.linspace(0, 1, nsamp)
        pts = self.p(tau)
        tangents = self.T(tau)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        return ax.quiver(pts.real, pts.imag, tangents.real, tangents.imag,
                         angles='xy', scale_units='xy', **plotkwargs)

    def plot_normals(self, nsamp=None, ax=None, **plotkwargs):
        if nsamp is None:
            nsamp = int(np.ceil(self.nsamp/10))
        tau = np.linspace(0, 1, nsamp)
        pts = self.p(tau)
        normals = self.N(tau)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        return ax.quiver(pts.real, pts.imag, normals.real, normals.imag,
                         angles='xy', scale_units='xy', **plotkwargs)
            
class LineSegment(Segment):
    """Class for straight line segments.
    Parameters
    ----------
    a : complex
        initial point of segment
    b : complex
        end point of segment
    """
    def __init__(self, a, b, nsamp=100):
        self._p0 = complex_form(a)
        self._pf = complex_form(b)
        self._len = np.abs(self._pf-self._p0)
        self._tangent = (self._pf-self._p0)/self._len
        self._normal = self._tangent.imag - 1j*self._tangent.real
        self.nsamp = nsamp
        self._t0, self._tf = 0, 1

        # overwrite unneeded reparameterizations
        self._p = self.p
        self._p_of_s = self.p
        self._dp = self.dp
        self._dp_of_s = self.dp
        self._T = self.T
        self._T_of_s = self.T
        self._N = self.N
        self._N_of_s = self.N

        # simple and closed settings
        self._is_simple = True
        self._is_closed = False

    def __str__(self):
        return f"LineSegment({self.p0},{self.pf})"

    @property
    def p0(self):
        return self._p0

    @property
    def pf(self):
        return self._pf

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

    def _reparameterize(self):
        """Line segments do not need reparameterization"""
        pass

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
        elif isinstance(other, Segment):
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
        
class SplineSegment(Segment):
    """Segments with spline boundary"""
    def __init__(self, spline, t0=None, tf=None, nsamp=100, val_simple=False, val_closed=False):
        if not isinstance(spline, BSpline):
            raise TypeError("'spline' must be an instance of BSpline")
        self._spline = spline
        p = lambda t: self._spline(t)
        dp = lambda t: self._spline(t, nu=1)
        if t0 is None:
            t0_idx = self._spline.k
            t0 = self._spline.t[t0_idx]
        if tf is None:
            tf_idx = len(self._spline.t)-spline.k-1
            tf = self._spline.t[tf_idx]
        super().__init__(p, dp, t0, tf, nsamp, val_simple, val_closed)
    
    @classmethod
    def interp_from_pts(cls, pts, bc_type='natural', nsamp=100, val_simple=False, val_closed=False):
        """builds a Bspline segment to interpolate the given points"""
        t = np.linspace(0, 1, len(pts))
        pts = real_form(pts)
        spline = make_interp_spline(t, pts, bc_type=bc_type)
        return cls(spline, 0, 1, nsamp, val_simple, val_closed)

class MultiSegment:
    """Class for simple planar curves composed of segments. Simplicity may be enforced, as well as closure"""
    def __init__(self, segments, val_closed=False, val_simple=False):
        if not all(isinstance(seg, Segment) for seg in segments):
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
        self._segments = segments

        # cached properties
        self._len = None

    def __str__(self):
        return f"MultiSegment([{','.join(str(seg) for seg in self._segments)}])"

    @classmethod
    def from_vertices(cls, vertices, make_closed=True, val_simple=False, nsamp=100):
        """Builds a polygonal MultiSegment from the given vertices"""
        segments = [LineSegment(vertices[i], vertices[i+1], nsamp) for i in range(len(vertices)-1)]
        if make_closed:
            segments += [LineSegment(vertices[-1], vertices[0], nsamp)]
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
            self._is_simple = self._validate_simple(self._segments)
        return self._is_simple

    @property
    def is_closed(self):
        if self._is_closed is None:
            self._is_closed = self._validate_closed(self._segments)
        return self._is_closed
    
    @property
    def segments(self):
        return self._segments
        
    def _compute_length(self):
        return np.sum([seg.len for seg in self._segments])

    @property
    def len(self):
        if self._len is None:
            self._len = self._compute_length()
        return self._len

    @property
    def is_polyline(self):
        return all(isinstance(seg, LineSegment) for seg in self._segments)

    def pts(self, N, kind='chebyshev', weights=True):
        """Places N[i] points on ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self._segments), N)
        return np.sum([self._segments[i].pts(N[i], kind, weights) for i in range(len(self._segments))])

    def tangents(self, N, kind='chebyshev', weights=True):
        """Computes tangent vectors at N[i] points along the ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self._segments), N)
        return np.sum([self._segments[i].tangents(N[i], kind, weights) for i in range(len(self._segments))])

    def normals(self, N, kind='chebyshev', weights=True):
        """Computes normal vectors at N[i] points along the ith segment of the MultiSegment"""
        if isinstance(N, (int, np.integer)):
            N = np.full(len(self._segments), N)
        return np.sum([self._segments[i].normals(N[i], kind, weights) for i in range(len(self._segments))])

    def plot(self, ax=None, **pltkwargs):
        """Plots the MultiSegment"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        for seg in self._segments:
            seg.plot(ax=ax, **pltkwargs)
        return ax

    def plot_tangents(self, ax=None, **pltkwargs):
        """Plots the tangent vectors"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        quivs = []
        for seg in self._segments:
            quivs.append(seg.plot_tangents(ax=ax, **pltkwargs))
        return quivs

    def plot_normals(self, ax=None, **pltkwargs):
        """Plots the normal vectors"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        quivs = []
        for seg in self._segments:
            quivs.append(seg.plot_normals(ax=ax, **pltkwargs))
        return quivs

# domain class
class Domain:
    def __init__(self, bdry):
        if not isinstance(bdry, MultiSegment):
            raise TypeError("'bdry' must be an instance of MultiSegment")
        if not bdry.is_closed:
            raise ValueError("boundary must be closed")
        if not bdry.is_simple:
            raise ValueError("boundary must be simple")
        self._bdry = bdry
        self._area = None
        self._diameter = None
        self._corners = None
        self._int_angles = None

    @property
    def bdry(self):
        return self._bdry

    @property
    def perimeter(self):
        return self._bdry.len

    @property
    def area(self):
        if self._area is None:
            self._area = self._compute_area()
        return self._area
            
    def _compute_area(self):
        tau, wts = cached_leggauss(20)
        P = np.array([seg.p(tau) for seg in self._bdry._segments])
        dP = np.array([seg.dp(tau) for seg in self._bdry._segments])
        # Green's formula
        I = ((P.real*dP.imag - P.imag*dP.real)@wts).sum()
        return np.abs(I)/2

    def _find_corners(self):
        """Finds corners, i.e. where two boundary segments do not smoothly connect. Computes
        interior angle there."""
        segments = self._bdry._segments
        T0 = [seg.T0 for seg in segments]
        Tf = [seg.Tf for seg in segments]
        corner_idxs = []
        int_angles = []
        if len(segments) == 1:
            if not np.isclose(T0[0],Tf[0]):
                corner_idxs.append(0)
                int_angles
        for i in range(len(segments)):
            if not np.isclose(T0[i],Tf[i-1]):
                corner_idxs.append(i)
                int_angles.append(np.angle(-Tf[i-1]/T0[i]))
        corner_idxs = np.array(corner_idxs)
        int_angles = np.array(int_angles)
        int_angles[int_angles<0] += 2*np.pi
        return corner_idxs, int_angles
    
    @property
    def corners(self):
        if self._corners is None:
            self._corners, self._int_angles = self._find_corners()
        return self._corners
    
    @property
    def int_angles(self):
        if self._int_angles is None:
            self._corners, self._int_angles = self._find_corners()
        return self._int_angles

    @property
    def diameter(self):
        if self._diameter is None:
            self._diameter = self._compute_diameter()
        return self._diameter

    def _compute_diameter(self):
        # approximation using 100 points on each segment
        tau = np.linspace(0, 1, 100)[:-1]
        pts = np.array([seg.p(tau) for seg in self._bdry._segments])
        dist = np.abs(np.subtract.outer(pts.flatten(),pts.flatten()))
        idx1 = dist.max(axis=0).argmax()
        idx2 = dist[idx1].argmax()

        seg1_idx, tau1_idx = np.unravel_index(idx1, pts.shape)
        seg2_idx, tau2_idx = np.unravel_index(idx2, pts.shape)
        seg1 = self._bdry._segments[seg1_idx]
        tau1 = tau[tau1_idx]
        seg2 = self._bdry._segments[seg2_idx]
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

    def plot(self, ax=None, **plt_kwargs):
        if 'c' not in plt_kwargs.keys() or 'color' not in plt_kwargs.keys():
            plt_kwargs['color'] = 'k'
        self._bdry.plot(ax=ax, **plt_kwargs)

# polygon class
class Polygon(Domain):
    """Class for polygonal domains"""
    def __init__(self,vertices=None, bdry=None, val_simple=True):
        if not ((vertices is None)^(bdry is None)):
            raise ValueError("exactly one of 'vertices' and 'bdry' must be provided")
        elif vertices is not None:
            vertices = complex_form(vertices)
            self._bdry = MultiSegment.from_vertices(vertices, val_simple=val_simple)
            self._vertices = vertices
        elif bdry is not None:
            if not isinstance(bdry, MultiSegment) or not bdry.is_polyline:
                    raise TypeError("'bdry' must be a polyline MultiSegment")
            elif val_simple:
                if not bdry.is_simple:
                    raise ValueError("'bdry' must be simple")
            elif not bdry.is_closed:
                raise ValueError("'bdry' must be closed")
            else:
                self._bdry = bdry
                self.vertices = np.array([seg.p0 for seg in self._bdry._segments])
        self._area = None
        self._perimeter = None
        self._diameter = None

    def _compute_area(self):
        return polygon_area(self._vertices)

    def _compute_perimeter(self):
        return polygon_perimeter(self._vertices)

    def _compute_diameter(self):
        return polygon_diameter(self._vertices)

    @property 
    def n_vertices(self):
        return len(self._vertices)
    
    @property
    def n_sides(self):
        return len(self._vertices)

    @property
    def vertices(self):
        return self._vertices
    
    @property
    def edge_lengths(self):
        return edge_lengths(self._vertices)
    
    @property
    def int_angles(self):
        return interior_angles(self._vertices)