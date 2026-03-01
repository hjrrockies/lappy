import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from shapely.geometry import Polygon
from shapely import points

# convert between real and complex representations of points in the plane
def complex_form(pts, y=None):
    """Converts points in the plane to complex form"""
    pts = np.asarray(pts)
    if y is not None and np.isrealobj(pts):
        y = np.asarray(y)
        if y.dtype != 'float64':
            raise ValueError('y must be a real-valued array')
        return pts + 1j*y
    elif np.iscomplexobj(pts):
        return pts
    elif pts.ndim <= 1:
        return pts.astype('complex128')
    elif pts.shape[-1] == 2 and np.isrealobj(pts):
        return np.transpose(pts.T[0] + 1j*pts.T[1])
    else:
        raise ValueError('pts must be a real array of shape (...,2)')

def real_form(pts):
    pts = np.asarray(pts)
    if np.iscomplexobj(pts):
        return np.array([pts.real,pts.imag]).T
    elif pts.shape[-1] == 2:
        return pts
    else:
        raise ValueError('pts must be a complex array, or already in real form with shape (...,2)')

# polygon common functions
def polygon_edges(vertices):
    """Computes the edges of a polygon with vertices, ordered counter-clockwise"""
    vertices = complex_form(vertices)
    return np.roll(vertices,-1)-vertices

def interior_angles(vertices):
    """Computes the interior angles of a polygon with given vertices, ordered
    counter-clockwise"""
    e = polygon_edges(vertices)
    phis = np.angle(np.roll(-e,1)/e)
    phis[phis<0] += 2*np.pi
    return phis

def edge_lengths(vertices):
    """Computes the edge lengths of a polygon with vertices in complex form, ordered
    counter-clockwise"""
    vertices = complex_form(vertices)
    return np.abs(polygon_edges(vertices))

def side_normals(vertices):
    """Computes the outward-pointing unit normals to the sides of a
    polygon with vertices in complex form, ordered counter-clockwise"""
    vertices = complex_form(vertices)
    e = polygon_edges(vertices)
    return (e.imag-1j*e.real)/np.abs(e)

def edge_angles(vertices):
    """Gets the angle of each side of a polygon with the given vertices, with respect to the real axis."""
    vertices = complex_form(vertices)
    e = polygon_edges(vertices)
    psis = np.angle(e)
    return psis

def singular_corner_check(angles,tol=1e-15):
    """Checks to see if polygon corners are singular (not an integer fraction of pi)"""
    alpha = np.pi/angles
    maxalph = np.ceil(alpha.max())
    diffs = np.abs(np.subtract.outer(alpha,np.arange(1,maxalph+1)))
    return (diffs>tol).min(axis=1)

def edge_midpoints(vertices):
    vertices = complex_form(vertices)
    return 0.5*(np.roll(vertices,-1)+vertices)

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

def boundary_pts_polygon(vertices,n_pts=20,rule='legendre',skip=None):
    from .quad import boundary_nodes_polygon
    return boundary_nodes_polygon(vertices,n_pts,rule,skip)

def rand_interior_points(vertices, m, oversamp=2):
    """Computes random interior points for a polygon with vertices in complex form,
    ordered counter-clockwise."""
    vertices = complex_form(vertices)
    poly_area = polygon_area(vertices)
    x_min, x_max = np.min(vertices.real), np.max(vertices.real)
    y_min, y_max = np.min(vertices.imag), np.max(vertices.imag)
    box_area = (x_max-x_min)*(y_max-y_min)
    npts = int(np.ceil(m*oversamp*box_area/poly_area))
    x_i = (x_max-x_min)*np.random.rand(npts)+x_min
    y_i = (y_max-y_min)*np.random.rand(npts)+y_min

    poly = Polygon(np.array([vertices.real,vertices.imag]).T)
    pts = points(np.array([x_i,y_i]).T)
    mask = poly.contains(pts)
    if mask.sum() < m:
        return rand_interior_points(vertices, m, oversamp=2*oversamp)
    return x_i[mask][:m] + 1j*y_i[mask][:m]

def plot_polygon(vertices,ax=None,**plotkwargs):
    """Plot a polygon with vertices in complex form, which are assumed to be in
    counter-clockwise order."""
    if ax is None: makeax = True
    else: makeax = False
    vertices = complex_form(vertices)
    v_ = np.append(vertices,vertices[0])
    if makeax:
        fig = plt.figure()
        ax = plt.gca()
    ax.plot(v_.real,v_.imag,**plotkwargs)
    if makeax:
        ax.set_aspect('equal')
        plt.show()

def plot_angles(vertices,ax=None):
    """Plots the angle arcs to confirm accurate calculation of angles. Also looks
    nice."""
    vertices = complex_form(vertices)
    phis = np.rad2deg(interior_angles(vertices))
    l = edge_lengths(vertices)
    d = .5*np.minimum(l,np.roll(l,-1))
    start = np.rad2deg(edge_angles(vertices))
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    for i in range(len(phis)):
        arc = patches.Arc((vertices.real[i],vertices.imag[i]),d[i],d[i],angle=start[i],theta2=phis[i])
        ax.add_patch(arc)

def polygon_area(vertices):
    """Computes the area of a polygon with vertices in complex form,
    ordered counter-clockwise. Uses the Shoelace formula."""
    vertices = complex_form(vertices)
    x,y = vertices.real, vertices.imag
    return 0.5*np.sum((x-np.roll(x,-1))*(y+np.roll(y,-1)))

def polygon_perimeter(vertices):
    vertices = complex_form(vertices)
    return np.sum(edge_lengths(vertices))

def polygon_diameter(vertices):
    vertices = complex_form(vertices)
    return np.max(np.abs(np.subtract.outer(vertices,vertices)))

def reg_polygon(n,r):
    """Generates a regular polygon with n vertices and radius r."""
    theta = np.linspace(0,2*np.pi,n+1)
    return r*np.exp(1j*theta)[:-1]

def random_polygon(n,r_avg,eps,sigma):
    """Generate the vertices of a random polygon, with vertices ordered
    counter-clockwise."""
    u = np.random.rand(n)
    dtheta = (2*np.pi/n+eps)*u + (2*np.pi/n-eps)*(1-u)
    k = dtheta.sum()/(2*np.pi)
    theta = np.zeros(n)
    theta[0] = dtheta[0]/k
    for i in range(1,n):
        theta[i] = theta[i-1] + dtheta[i]/k
    r = sigma*np.random.randn(n)+r_avg
    r[r<0] = 0
    r[r>2*r_avg] = 2*r_avg
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x+1j*y

### Eigenvalue Asymptotics
def dir_weyl_N(lam, area, perim):
    """Two-term Weyl asymptotics for the Dirichlet eigenvalue counting function"""
    return (area*lam - perim*np.sqrt(lam))/(4*np.pi)

def dir_weyl_k(k, area, perim):
    """Weyl asymptotic estimate for the kth Dirichlet eigenvalue"""
    return ((perim+np.sqrt(perim**2+16*np.pi*area*k))/(2*area))**2

### Eigenvalue Bounds
def dir_lbound(area):
    """Lower bound on Dirichlet spectrum"""
    return 5.76*np.pi/area

### Miscellaneous utils
def invert_permutation(p):
    """Invert a pertmutation vector. For use with column-pivoted QR solves"""
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s

### Eigenvalues of rectangles (for validation purposes)
def rect_eig(m,n,L,H):
    """Computes the (m,n) Dirichlet eigenvalue of an L-by-H rectangle"""
    return m**2*np.pi**2/L**2 + n**2*np.pi**2/H**2

def rect_eigs_k(k,L,H,ret_mn=False):
    """Gets first k Dirichlet eigenvalues for an LxH rectangle.
    Vectorized to handle arrays for L and H of various shapes, returning
    an array such that rect_eigs_k(L,H,k).flatten()[idx] is the first
    k eigenvalues of an L.flatten()[idx]xH.flatten()[idx] rectangle.

    Optionally returns the indices of the corresponding eigenvalues."""
    L,H = np.asarray(L), np.asarray(H)
    mn = np.arange(1,k+1)
    M,N = np.meshgrid(mn,mn,indexing='ij')
    eigs = rect_eig(M.flatten()[np.newaxis],
                    N.flatten()[np.newaxis],
                    L.flatten()[:,np.newaxis],
                    H.flatten()[:,np.newaxis])

    idx = np.argsort(eigs,axis=-1)
    eigs = np.take_along_axis(eigs,idx,axis=-1)[:,:k]
    eigs = eigs.reshape((*L.shape,k))
    if ret_mn:
        m = np.take_along_axis(M.flatten()[np.newaxis],idx,axis=-1)[:,:k]
        n = np.take_along_axis(N.flatten()[np.newaxis],idx,axis=-1)[:,:k]
        return eigs,m.reshape((*L.shape,k)),n.reshape((*L.shape,k))
    else:
        return eigs

def rect_eig_grad(m,n,L,H):
    """Derivatives of rectangular eigenvalues wrt length and width"""
    m,n = np.asarray(m), np.asarray(n)
    L,H = np.asarray(L), np.asarray(H)
    return (-2*(np.pi*m.T)**2/L**3).T, (-2*(np.pi*n.T)**2/H**3).T

def rect_eig_bound_idx(bound,L,H):
    """Identifies the indices of Dirichlet eigenvalues foran L-by-H rectangles
    which less than a given upper bound"""
    m_max = 1
    while True:
        eig = rect_eig(m_max,1,L,H)
        if eig > bound: break
        m_max += 1

    n_max = 1
    while True:
        eig = rect_eig(1,n_max,L,H)
        if eig > bound: break
        n_max += 1

    M = np.arange(1,m_max+1)[:,np.newaxis]
    N = np.arange(1,n_max+1)[np.newaxis]
    Lambda = rect_eig(M,N,L,H)
    return np.argwhere(Lambda <= bound)+1

def rect_eig_mult(lambda_,L,H,maxind=1000):
    """Compute the indices of rectangle Dirichlet eigenvalues which are
    duplicates of lambda_. For use in testing multiplicity."""
    Lam = rect_eig(np.arange(1,maxind)[np.newaxis],np.arange(1,maxind)[:,np.newaxis],L,H)
    diff = np.abs(lambda_-Lam)
    tot = (diff<1e-12).sum()
    ind = np.unravel_index(np.argsort(diff, axis=None), diff.shape)
    return (ind[0]+1)[:tot],(ind[1]+1)[:tot]

def rect_eig_mult_mn(m,n,L,H):
    """Compute the indices of rectangle Dirichlet eigenvalues which are
    duplicates of the (m,n) eigenvalue. For use in testing multiplicity."""
    return rect_eig_mult(rect_eig(m,n,L,H),L,H,maxind=10*max(m,n))

def loss_plot(L,H,loss,log=True,ax=None):
    n_levels = 50
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    if log:
        low = np.floor(np.log10(loss.min()+1e-16))
        high = np.ceil(np.log10(loss.max()))
        levels = 10.0**np.linspace(low,high,n_levels)
        norm = colors.LogNorm(vmin=loss.min()+1e-16, vmax=loss.max())
    else:
        levels = n_levels
        norm = 'linear'
    cs = ax.contourf(L,H,loss+1e-16,levels,norm=norm)
    cbar = ax.get_figure().colorbar(cs,ax=ax)
    if log: cbar.set_ticks(10.0**np.arange(low,(high+1)))
    ax.set_xlabel('L')
    ax.set_ylabel('H')
    return ax,cs

def eig_ratios(eigs):
    """Computes the ratios lambda_j/lambda_1, assuming the eigenvalues are
    indexed along the last dimension of the array"""
    return (eigs.T[1:]/eigs.T[0]).T

def rect_loss_std(L,H,eigs_true,weights=1,jac=False):
    """Computes the standard loss function for a rectangle against
    target eigenvalues. Can use weights."""
    weights = np.asarray(weights).flatten()
    k = len(eigs_true)
    eigs,m,n = rect_eigs_k(L,H,k,ret_mn=True)
    pweights = weights**(0.5)
    out = la.norm(pweights*(eigs-eigs_true),axis=-1)**2
    if jac:
        grad = weights*2*(eigs-eigs_true)
        deig_dL, deig_dH = rect_eig_grad(m,n,L,H)
        grad = np.array([(grad*deig_dL).sum(axis=-1), (grad*deig_dH).sum(axis=-1)])
        return out,grad
    else:
        return out

def rect_loss_reciprocal(L,H,eigs_true,weights=1):
    """Computes the standard loss function for a rectangle against
    target eigenvalues in reciprocal form. Can use weights."""
    weights = np.asarray(weights).flatten()
    k = len(eigs_true)
    eigs,m,n = rect_eigs_k(L,H,k,ret_mn=True)
    pweights = weights**(0.5)
    out = la.norm(pweights*(1/eigs-1/eigs_true),axis=-1)**2
    jac = False
    if jac:
        grad = weights*2*(eigs-eigs_true)
        deig_dL, deig_dH = rect_eig_grad(m,n,L,H)
        grad = np.array([(grad*deig_dL).sum(axis=-1), (grad*deig_dH).sum(axis=-1)])
        return out,grad
    else:
        return out

def rect_loss_outerlog(L,H,eigs_true,weights=1,eps=1e-200,jac=False):
    """Computes the 'outer-log loss' for rectangular eigenvalues"""
    L,H = np.asarray(L), np.asarray(H)
    weights = np.asarray(weights).flatten()
    k = len(eigs_true)
    if len(weights) == 1:
        weights = weights*np.ones(k)
    eigs,m,n = rect_eigs_k(L,H,k,ret_mn=True)
    eigsT = eigs.T
    eigs_trueT = eigs_true.reshape((-1,*np.ones(eigsT.ndim-1,dtype='int')))
    weightsT = weights.reshape((-1,*np.ones(eigsT.ndim-1,dtype='int')))
    out = weightsT[0]*np.log((eigsT[0]-eigs_trueT[0])**2+eps)
    eigdiff = eigsT[1:][np.newaxis]-eigs_trueT[1:][:,np.newaxis]
    out += (weightsT[1:]*np.log(eigdiff**2 + eps).sum(axis=0)).sum(axis=0)
    if jac:
        deig_dL, deig_dH = rect_eig_grad(m,n,L,H)
        C = np.empty((k,*L.shape))
        C[0] = (eigsT[0]-eigs_trueT[0])/((eigsT[0]-eigs_trueT[0])**2+eps)
        eigdiff_sum = eigdiff.sum(axis=0)
        C[1:] = eigdiff_sum/(eigdiff_sum**2+eps)
        C *= 2*weightsT
        grad = np.array([(C.T*deig_dL).sum(axis=-1),(C.T*deig_dH).sum(axis=-1)])
        return out,grad
    else:
        return out

def logify(obj):
    def log_obj(*args,**kwargs):
        if 'jac' in kwargs.keys():
            if kwargs['jac'] == True:
                jac = True
            else: jac = False
        else: jac = False
        if jac:
            out,grad = obj(*args,**kwargs)
            return np.log(out+1e-200), grad/(out+1e-200)
        else:
            return np.log(obj(*args,**kwargs)+1e-200)
    return log_obj

### interval arithmetic
from abc import ABC, abstractmethod
class RealSubset(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def union(self, other):
        pass

    @abstractmethod
    def intersection(self, other):
        pass

    @abstractmethod
    def complement(self):
        pass

    @property
    @abstractmethod
    def lb(self):
        pass

    @property
    @abstractmethod
    def ub(self):
        pass

    def __lt__(self, other):
        return self.ub < other.lb

    def intersects(self, other):
        return (self.intersection(other) is not None)

    def contains(self, other):
        return self.union(other) == self

    def __add__(self, other):
        return self.union(other)

    def __mul__(self, other):
        return self.intersection(other)

    def difference(self, other):
        return other.complement().intersection(self)

    def symm_difference(self, other):
        return (self.union(other)).difference(self.intersection(other))

    def __sub__(self, other):
        return self.difference(other)
    
class Interval(RealSubset):
    """(open) Interval class. Implements basic set logic for real (open) intervals."""
    def __init__(self, a, b):
        if b <= a:
            raise ValueError("a must be less than b")
        self.a = np.float64(a)
        self.b = np.float64(b)

    def __repr__(self):
        return f"Interval({repr(self.a)},{repr(self.b)})"

    def __str__(self):
        return f"Interval({str(self.a)},{str(self.b)})"
        
    def union(self, other):
        if isinstance(other, Interval):
            # no overlap, union is MultiInterval
            if self < other:
                return MultiInterval([self,other], checkvalid=False)
            elif other < self:
                return MultiInterval([other,self], checkvalid=False)
            # overlap, get new endpoints
            else:
                new_a = min(self.a, other.a)
                new_b = max(self.b, other.b)
                return Interval(new_a, new_b)
        elif isinstance(other, MultiInterval):
            return other.union(self)
        else:
            raise TypeError("'other' must be an instance of Interval or MultiInterval")

    def intersection(self, other):
        if isinstance(other, Interval):
            # no overlap, intersection is empty
            if self < other or other < self:
                return None
            # overlap, get new endpoints
            else:
                new_a = max(self.a, other.a)
                new_b = min(self.b, other.b)
                if new_b <= new_a:
                    return None
                else:
                    return Interval(new_a, new_b)
        elif isinstance(other, MultiInterval):
            return other.intersection(self)
        else:
            raise TypeError("'other' must be an instance of Interval or MultiInterval")

    def complement(self):
        return MultiInterval([Interval(float('-inf'),self.a),Interval(self.b,float('inf'))], checkvalid=False)

    @property
    def lb(self):
        return self.a

    @property
    def ub(self):
        return self.b

    def __eq__(self, other):
        if isinstance(other, Interval):
            return (self.a == other.a) and (self.b == other.b)
        else:
            return super.__eq__(other)

    def contains(self, other):
        if isinstance(other, Interval):
            return (self.a <= other.a) and (other.b <= self.b)
        else:
            return super.__eq__(other)
        
class MultiInterval(RealSubset):
    """Class for tracking (disjoint) unions of intervals"""
    def __init__(self, intervals, checkvalid=True):
        if checkvalid:
            if not all(isinstance(interval, Interval) for interval in intervals):
                raise ValueError("Each element of 'intervals' must be an instance of Interval")

        # process list/tuple inputs
        if isinstance(intervals, (list, tuple)):
            if checkvalid:
                if MultiInterval.is_valid(intervals):
                    self.intervals = intervals
                else:
                    raise ValueError("Intervals must be pairwise disjoint and in increasing order")
            else:
                self.intervals = intervals
        # process set inputs
        elif isinstance(intervals, set):
            intervals = list(intervals)
            pass
        else:
            raise TypeError("'intervals' must be a list/tuple of pairwise disjoint increasing intervals or a set of intervals")

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step <= 0:
                return MultiInterval(set(self.intervals[key]))
            else:
                return MultiInterval(self.intervals[key], checkvalid=False)
        else:
            return self.intervals[key]

    def __len__(self):
        return len(self.intervals)
            
    @property
    def lb(self):
        return self.intervals[0].a
    
    @property
    def ub(self):
        return self.intervals[-1].b

    @staticmethod
    def is_valid(intervals):
        """checks if a list of intervals is pairwise disjoint and in increasing order"""
        if not all(isinstance(interval, Interval) for interval in intervals):
            raise TypeError("each element of 'intervals' must be an instance of Interval")
        n = len(intervals)
        for i in range(n-1):
            for j in range(i+1,n):
                if not (intervals[i] < intervals[j]):
                    return False
        return True

    @staticmethod
    def _extend(intervals, new_interval):
        if new_interval > intervals[-1]:
            return intervals + [new_interval]
        for i in range(len(intervals)):
            # assume other does not intersect any of intervals before intervals[i] (handled in loop)
            # new_interval is left of ith interval
            if new_interval < intervals[i]:
                return intervals[:i] + [new_interval] + intervals[i:]
            # other intersects ith interval
            elif new_interval.intersects(intervals[i]):
                return MultiInterval._extend(intervals[:i] + intervals[i+1:], new_interval.union(intervals[i]))
            # new_interval is right of ith interval, pass to next iteration of loop
            else:
                pass
        
    def union(self, other):
        if not isinstance(other, (Interval, MultiInterval)):
            raise TypeError("'other' must be an instance of Interval or MultiInterval")
            
        if isinstance(other, Interval):
            intervals = MultiInterval._extend(self.intervals, other)
        elif isinstance(other, MultiInterval):
            intervals = self.intervals
            for interval in other.intervals:
                intervals = MultiInterval._extend(intervals, interval)

        if len(intervals) == 1:
            return intervals[0]
        else:
            return MultiInterval(intervals)
        
    def intersection(self, other):
        if not isinstance(other, (Interval, MultiInterval)):
            raise TypeError("'other' must be an instance of Interval or MultiInterval")
        
        # build intervals for intersection
        if isinstance(other, Interval):
            intervals = []
            for interval in self.intervals:
                if other.intersects(interval):
                    intervals.append(other.intersection(interval))
        elif isinstance(other, MultiInterval):
            intervals = []
            for interval1 in self.intervals:
                for interval2 in other.intervals:
                    if interval1.intersects(interval2):
                        intervals.append(interval1.intersection(interval2))
        if len(intervals) == 0:
            return None
        elif len(intervals) == 1:
            return intervals[0]
        else:
            return MultiInterval(intervals)

    def contains(self, other):
        if isinstance(other, Interval):
            return any(interval.contains(other) for interval in self.intervals)
        elif isinstance(other, MultiInterval):
            return all(self.contains(interval) for interval in other.intervals)
        else:
            return super().contains(other)

    def __eq__(self, other):
        if isinstance(other, MultiInterval):
            if len(self) != len(other):
                return False
            else:
                return all([int1 == int2 for int1,int2 in zip(self.intervals, other.intervals)])
        else:
            return super().__eq__(other)

    def complement(self):
        out = self.intervals[0].complement()
        for interval in self.intervals[1:]:
            out = out.intersection(interval.complement())
        return out

    def __repr__(self):
        return str(self)
        
    def __str__(self):
        return f"MultiInterval([{','.join([str(interval)[8:] for interval in self.intervals])}])"