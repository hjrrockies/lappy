import numpy as np
from scipy.special import jv, jvp
from scipy.linalg import norm
from .utils import complex_form, polygon_edges
from functools import cache, lru_cache
from abc import ABC, abstractmethod
from .geometry import PointSet, Domain, Polygon
from .utils import singular_corner_check
import mpmath as mp

class ParticularBasis(ABC):
    """Base class for function bases on the plane which depend on the spectral parameter λ."""
    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, lam, points):
        """Evaluate the basis on a given set of points in the plane for a given spectral parameter value."""
        if isinstance(points, PointSet):
            return self._eval_pointset(lam, points)
        else:
            ps = PointSet(points)
            return self._eval_pointset(lam, ps)
        
    def grad(self, lam, points):
        """Evaluate the basis on a given set of points in the plane for a given spectral parameter value."""
        if isinstance(points, PointSet):
            return self._grad_pointset(lam, points)
        else:
            ps = PointSet(points)
            return self._grad_pointset(lam, ps)
        
    @abstractmethod
    def _eval_pointset(self, lam, ps):
        pass

    def __add__(self, other):
        if not isinstance(other, ParticularBasis):
            raise TypeError("'other' must be an instance of ParticularBasis")
        return MultiBasis([self, other])
    
    def normalized(self, quad_pts, quad_wts=None):
        return NormalizedBasis(self, quad_pts, quad_wts)

class MultiBasis(ParticularBasis):
    """Basis composed of the union of several bases"""
    def __init__(self, bases):
        if not np.all([isinstance(basis, ParticularBasis) for basis in bases]):
            raise TypeError("all elements of 'bases' must be instances of ParticularBasis")
        self._bases = list(bases)

    @property
    def bases(self):
        return self._bases
    
    @bases.setter
    def bases(self, new_bases):
        if not np.all([isinstance(basis, ParticularBasis) for basis in new_bases]):
            raise TypeError("all elements of 'bases' must be instances of ParticularBasis")
        self._bases = list(new_bases)

    @lru_cache
    def _eval_pointset(self, lam, ps):
        return np.hstack((basis._eval_pointset(lam, ps) for basis in self.bases))
    
    @lru_cache
    def _grad_pointset(self, lam, ps):
        return np.hstack((basis._grad_pointset(lam, ps) for basis in self.bases))
    
    def __iadd__(self, other):
        if not isinstance(other, ParticularBasis):
            raise TypeError("'other' must be an instance of ParticularBasis")
        if isinstance(other, MultiBasis):
            self._bases = self._bases + other._bases
        else:
            self._bases.append(other)
        return self
    
    def __add__(self, other):
        if isinstance(other, MultiBasis):
            return MultiBasis(self._bases + other._bases)
        else:
            return super().__add__(other)
        
    def __str__(self):
        return f"MultiBasis({','.join([str(basis) for basis in self._bases])})"

class NormalizedBasis(ParticularBasis):
    """Class for particular bases which are normalized to be (approximately) unit norm in L^2. Wraps an existing
    basis, and normalizes it. Note that this is done *pointwise* with respect to the spectral parameter λ. This
    means that each evaluation of the basis (potentially) requires an additional evaluation of the L^2 norms (which may
    involve a Pointset which is different from the desired evaluation set). Automatically prunes basis terms that are 
    norm zero.
    """
    def __init__(self, basis, quad_pts, quad_wts=None):
        if not isinstance(basis, ParticularBasis):
            raise TypeError("'basis' must be an instance of ParticularBasis")
        self._basis = basis
        
        if isinstance(quad_pts, PointSet):
            self._quad_pts = quad_pts
        else:
            self._quad_pts = PointSet(quad_pts, quad_wts)

    @property
    def basis(self):
        return self._basis
    
    @basis.setter
    def basis(self, new_basis):
        if not isinstance(new_basis, ParticularBasis):
            raise TypeError("'new_basis' must be an instance of ParticularBasis")
        self._basis = new_basis

    @property
    def quad_pts(self):
        return self._quad_pts
    
    @quad_pts.setter
    def quad_pts(self, new_pts, new_wts=None):
        if isinstance(new_pts, PointSet):
            self._quad_pts = new_pts
        else:
            self._quad_pts = PointSet(new_pts, new_wts)

    @lru_cache
    def norms(self, lam):
        A = self._basis._eval_pointset(lam, self.quad_pts)
        if hasattr(self._quad_pts,"_wts"):
            A *= self._quad_pts._sqrt_wts
        norms = norm(A, axis=0)
        nonzero_cols = (norms > 0)
        return norms[nonzero_cols], nonzero_cols
    
    @lru_cache
    def _eval_pointset(self, lam, ps):
        A = self._basis._eval_pointset(lam, ps)
        norms, nonzero_cols = self.norms(lam)
        return A[:,nonzero_cols]/norms
    
    @lru_cache
    def _grad_pointset(self, lam, ps):
        Agrad = self._basis._grad_pointset(lam, ps)
        norms, nonzero_cols = self.norms(lam)
        return Agrad[:,nonzero_cols]/norms
    
    def __str__(self):
        return f"NormalizedBasis({self._basis})"
    
class FourierBesselBasis(ParticularBasis):
    """
        Parameters
        ----------
        sources : list or ndarray
            The locations of the source points (usually domain vertices) in the plane.
        phi0 : list or ndarray
            The principal angle of the first rays along which the basis functions will be zero.
            For polygons, corresponds to the "next" edge relative to vertices as source points.
        phi1 : list or ndarray
            The principal angle of the second rays.
        orders : list or ndarray
            The number of basis functions to use at each source point.
        branch_cuts : list or ndarray
            The angles of the branch cut rays for the trigonometric parts of the basis functions.
        """
    def __init__(self, sources, phi0, phi1, orders, branch_cuts):
        self._sources = complex_form(sources)
        self._phi0 = phi0
        self._phi1 = phi1
        self._orders = orders

        if isinstance(orders,int):
            self._orders = orders*np.ones(self.n_sources, dtype='int')
        else:
            self._orders = np.array(orders, dtype='int')

        if self._orders.shape[0] != self.n_sources:
            raise ValueError('orders must match length of vertices')
        
        self._set_alphak()
        self._branch_cuts = branch_cuts
        self._branch_rays = np.exp(1j*self._branch_cuts)

    def __str__(self):
        return f"FourierBesselBasis(n_sources={self.n_sources}, n_func={self._orders.sum()})"

    @classmethod
    def from_polygon(cls, poly, orders):
        """Builds a Fourier-Bessel basis with source points at the vertices of the given polygon."""
        if not isinstance(poly, Polygon):
            raise TypeError("'poly' must be an instance of Polygon")

        # get source points and phi0/phi1
        sources = poly._vertices
        edges0 = polygon_edges(poly._vertices)
        edges1 = np.roll(-edges0,1)
        phi0 = np.angle(edges0)
        phi0[phi0 < 0] += 2*np.pi
        phi1 = np.angle(edges1)
        phi1[phi1 < 0] += 2*np.pi

        # set branch cuts to match the half-division of the exterior angle
        psi = np.angle(edges0/edges1)
        psi[psi < 0] += 2*np.pi
        branch_cuts = phi1 + psi/2
        branch_cuts[branch_cuts >= 2*np.pi] -= 2*np.pi
        return cls(sources, phi0, phi1, orders, branch_cuts)
    
    def from_domain(cls, dom, orders):
        """Builds a Fourier-Bessel basis with source points at the corners of the given domain."""
        if not isinstance(dom, Domain):
            raise TypeError("'dom' must be an instance of Domain")
        
        # get source points and phi0/phi1
        sources = dom.corners
        v0 = np.array([seg.T0 for seg in dom._bdry._segments])
        v1 = -np.roll(np.array([seg.Tf for seg in dom._bdry._segments]),1)
        phi0 = np.angle(v0)
        phi0[phi0 < 0] += 2*np.pi
        phi1 = np.angle(v1)
        phi1[phi1 < 0] += 2*np.pi

        # set branch cuts to match the half-division of the exterior angle
        psi = np.angle(v0/v1)
        psi[psi < 0] += 2*np.pi
        branch_cuts = phi1 + psi/2
        branch_cuts[branch_cuts >= 2*np.pi] -= 2*np.pi
        return cls(sources, phi0, phi1, orders, branch_cuts)

    @property
    def n_sources(self):
        return len(self._sources)
    
    @property
    def sources(self):
        return self._sources
    
    @property
    def phi0(self):
        return self._phi0
    
    @property
    def phi1(self):
        return self._phi1
    
    @property
    def orders(self):
        return self._orders
    
    @property
    def branch_cuts(self):
        return self._branch_cuts
    
    @property
    def alphak(self):
        return self._alphak
    
    @property
    def alpha(self):
        return np.pi/np.angle(self._ray1/self._ray0)
    
    def _set_alphak(self):
        # write angles as complex rays
        self._ray0 = np.exp(1j*self._phi0)
        self._ray1 = np.exp(1j*self._phi1)

        # rewrite in standard form (for later!)
        self._phi0 = np.angle(self._ray0)
        self._phi0[self._phi0 < 0] += 2*np.pi
        self._phi1 = np.angle(self._ray1)
        self._phi1[self._phi1 < 0] += 2*np.pi

        # compute alpha[i]*k for k in [1,...,orders[i]] for the ith source point
        phi = np.angle(self._ray1/self._ray0)
        phi[phi < 0] += 2*np.pi
        alpha = np.pi/phi
        self._alphak = [alphai*np.arange(1,ki+1) for alphai,ki in zip(alpha,self._orders)]
        self._alphak_vec = np.concatenate(self._alphak)[np.newaxis]

    @cache
    def _z(self, ps):
        """Positions of PointSet ps relative to sources"""
        points = ps._pts
        return np.subtract.outer(points,self._sources)

    @cache
    def _theta(self, ps):
        """Computes the angles of the given PointSet ps with respect to the bases vertices
        with branch cuts as needed."""
        # evaluation points relative to source points
        z = self._z(ps)

        # angles relative to phi0/ray0 for each source point
        theta = np.angle(z/self._ray0)
        theta[theta < 0] += 2*np.pi # make all angles nonnegative

        # wrap angles relative to branch cut
        phi_hat = np.angle(self._branch_rays/self._ray0) # angle of branch cuts relative to wedge starts
        phi_hat[phi_hat < 0] += 2*np.pi # make nonnegative
        theta[theta > phi_hat] -= 2*np.pi # angles past branch cut wrapped-down

        return theta
    
    @cache
    def _r(self, ps):
        """Computes the distances from the PointSet ps to the source points"""
        # evaluation points relative to source points
        z = self._z(ps)
        r = np.abs(z)
        return r
    
    @cache
    def _r_rep(self, ps):
        r = self._r(ps)
        r_rep = np.repeat(r,self._orders,axis=1)
        return r_rep
    
    @cache
    def _sin(self, ps):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet ps"""
        theta = self._theta(ps)
        sin = np.empty((ps._pts.shape[0],self._orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(self._orders)))
        for i in range(self.n_sources):
            sin[:,cumk[i]:cumk[i+1]] = np.sin(np.outer(theta[:,i],self._alphak[i]))
        return sin
    
    @cache
    def _cos(self, ps):
        """Computes the cosine terms of Fourier-Bessel functions on the given PointSet ps"""
        theta = self._theta(ps)
        cos = np.empty((ps._pts.shape[0],self._orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(self._orders)))
        for i in range(self.n_sources):
            cos[:,cumk[i]:cumk[i+1]] = np.cos(np.outer(theta[:,i],self._alphak[i]))
        return cos
    
    @lru_cache
    def _bessel(self, lam, ps):
        """Computes the Bessel part of the Fourier-Bessel functions on the given PointSet ps"""
        r_rep = self._r_rep(ps)
        bessel = jv(self._alphak_vec, np.sqrt(lam)*r_rep)
        return bessel
    
    @lru_cache
    def _besselp(self, lam, ps):
        """Computes the derivatives of the Bessel part of the Fourier-Bessel functions on the given PointSet ps"""
        r_rep = self._r_rep(ps)
        besselp = jvp(self._alphak_vec, np.sqrt(lam)*r_rep)
        return besselp

    @lru_cache
    def _eval_pointset(self, lam, ps):
        # get (potentially cached) evaluations of sine part and Bessel part
        sin = self._sin(ps)
        bessel = self._bessel(lam, ps)
        return bessel*sin
    
    @cache
    def _dr_dz(self, ps):
         # partial derivatives of distance to source points w.r.t. x and y
        z = self._z(ps)
        r = self._r(ps)
        dr_dz = np.repeat(z/r, self._orders,axis=1)
        return dr_dz
        
    @cache
    def _dtheta_dz(self,ps):
        z = self._z(ps)
        r = self._r(ps)
        dtheta_dz_temp = (-z.imag + 1j*z.real)/(r**2)
        dtheta_dz = np.repeat(dtheta_dz_temp,self._orders,axis=1)
        return dtheta_dz
    
    @lru_cache
    def _grad_pointset(self, lam, ps):
        """Evaluates the gradients of the basis functions on the given PointSet. Returns in complex form, 
        with the x-partial in the real part and the y-partial in the imaginary part"""
        if not isinstance(ps, PointSet):
            raise TypeError("'ps' must be an instance of PointSet")

        # get (potentially cached) evaluations of components
        sin = self._sin(ps)
        cos = self._cos(ps)
        bessel = self._bessel(lam, ps)
        besselp = self._besselp(lam, ps)
        dr_dz = self._dr_dz(ps)
        dtheta_dz = self._dtheta_dz(ps)

        dA_dr = np.sqrt(lam)*besselp*sin
        dA_dtheta = self._alphak_vec*bessel*cos

        # combine using chain rule
        return dA_dr*dr_dz.real + dA_dtheta*dtheta_dz.real + 1j*(dA_dr*dr_dz.imag + dA_dtheta*dtheta_dz.imag)
    
class ExPrecFBBasis(FourierBesselBasis):
    """Evaluates a Fourier-Bessel Basis in extended precision."""
    def __init__(self, sources, phi0, phi1, orders, branch_cuts, dps):
        super().__init__(sources, phi0, phi1, orders, branch_cuts)
        self.dps = dps
    
    @classmethod
    def from_polygon(cls, poly, orders, dps):
        """Builds an extended precision Fourier-Bessel basis with source points at the vertices of the given polygon."""
        if not isinstance(poly, Polygon):
            raise TypeError("'poly' must be an instance of Polygon")

        # get source points and phi0/phi1
        sources = poly._vertices
        edges0 = polygon_edges(poly._vertices)
        edges1 = np.roll(-edges0,1)
        phi0 = np.angle(edges0)
        phi0[phi0 < 0] += 2*np.pi
        phi1 = np.angle(edges1)
        phi1[phi1 < 0] += 2*np.pi

        # set branch cuts to match the half-division of the exterior angle
        psi = np.angle(edges0/edges1)
        psi[psi < 0] += 2*np.pi
        branch_cuts = phi1 + psi/2
        branch_cuts[branch_cuts >= 2*np.pi] -= 2*np.pi
        return cls(sources, phi0, phi1, orders, branch_cuts, dps)

    @cache
    def _sin(self, ps):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet ps"""
        mp.mp.dps = self.dps
        theta = self._theta(ps)
        sin = mp.matrix(ps._pts.shape[0],int(self._orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(self._orders)))
        for i in range(self.n_sources):
            alphak_theta = np.outer(theta[:,i],self._alphak[i])
            for j in range(ps._pts.shape[0]):
                for k in range(cumk[i],cumk[i+1]):
                    kk = k-cumk[i]
                    sin[j,k] = mp.sin(alphak_theta[j,kk])
        return sin
    
    @lru_cache
    def _bessel(self, lam, ps):
        """Computes the Bessel part of the Fourier-Bessel functions on the given PointSet ps"""
        mp.mp.dps = self.dps
        r_rep = self._r_rep(ps)
        sqrtlam_r_rep = np.sqrt(lam)*r_rep
        bessel = mp.matrix(r_rep.shape[0], r_rep.shape[1])
        for i in range(r_rep.shape[0]):
            for j in range(r_rep.shape[1]):
                bessel[i,j] = mp.besselj(self._alphak_vec[0,j], sqrtlam_r_rep[i,j])
        return bessel
    
    @lru_cache
    def _eval_pointset(self, lam, ps):
        mp.mp.dps = self.dps
        mat = self._eval_pointset_mp(lam, ps)
        arr = np.array(mat.tolist(), dtype='float')
        return arr
    
    @lru_cache
    def _eval_pointset_mp(self, lam, ps):
        # get (potentially cached) evaluations of sine part and Bessel part
        mp.mp.dps = self.dps
        bessel = self._bessel(lam, ps)
        sin = self._sin(ps)

        # take product
        mat = mp.matrix(bessel.rows, bessel.cols)
        for i in range(bessel.rows):
            for j in range(bessel.cols):
                mat[i,j] = bessel[i,j]*sin[i,j]
        
        return mat
    
### Build bases for various domains
def make_default_basis(domain, n_eigs, etol=1e-12):
    if not isinstance(domain, Domain):
        raise TypeError("'domain' must be an instance of Domain")
    
    elif isinstance(domain, Polygon):
        return default_polygon_basis(domain, n_eigs, etol)
    
def default_polygon_basis(poly, n_eigs, etol=1e-12):
    """constructs a basis of Fourier-Bessel functions for the given polygon designed for computing the first n_eigs
    up to accuracy etol."""
    if not isinstance(poly, Polygon):
        raise TypeError("'poly' must be an instance of Polygon")
    
    # heuristic for number of basis functions
    phis = poly.int_angles
    sing_idx = np.abs(np.sin(np.pi**2/phis))**0.1 # maps regular corners to 0, and nearly-regular corners to nearly 0
    frac = sing_idx*(phis/phis.sum()) # now weight by angle measure
    wavenumber = np.sqrt(n_eigs * (4*np.pi)/poly.area)

    orders = np.rint(3 + frac*wavenumber).astype('int')
    return FourierBesselBasis.from_polygon(poly, orders)
    
        
### OLD CODE ###
# class FourierBesselBasis(ParticularBasis):
#     """A class for Fourier-Bessel particular bases. 
#     Allows the user to evaluate the basis on an arbitrary set. Caches subcomputations
#     which are independent of the spectral parameter lambda."""
#     def __init__(self,vertices,orders,normalize=True,branch_cuts='middle_out'):
#         # unpack vertices, put in complex form
#         vertices = np.array(vertices)
#         if vertices.ndim > 1:
#             if vertices.shape[0] == 2:
#                 vertices = complex_form(vertices[0],vertices[1])
#             elif vertices.shape[1] == 2:
#                 vertices = complex_form(vertices[:,0],vertices[:,1])
#         self.vertices = vertices
#         self.n_vert = len(self.vertices)
#         self.normalize = normalize

#         if isinstance(orders,int):
#             self.orders = orders*np.ones(self.n_vert,dtype='int')
#         else:
#             self.orders = np.array(orders,dtype='int')

#         if self.orders.shape[0] != self.n_vert:
#             raise ValueError('orders must match length of vertices')

#         self._set_alphak()
#         self.branch_cuts = branch_cuts

#     def _set_alphak(self):
#         alpha = np.pi/interior_angles(self.vertices)
#         self.alphak = [alphai*np.arange(1,ki+1) for alphai,ki in zip(alpha,self.orders)]
#         self.alphak_vec = np.concatenate(self.alphak)[np.newaxis]

#     @staticmethod
#     def fourier_bessel_angles(points,vertices,branch_cuts='middle_out'):
#         """Computes the angles of the given points with respect to the bases vertices
#         with branch cuts as needed."""
#         psis = edge_angles(vertices)
#         int_angles = interior_angles(vertices)
#         ext_angles = 2*np.pi-int_angles
#         angles = np.angle(np.subtract.outer(points,vertices))-psis
#         angles[angles<0] += 2*np.pi # puts into form for 'positive_edge' mode
#         if branch_cuts == 'positive_edge':
#             pass
#         elif branch_cuts == 'negative_edge':
#             angles[angles>int_angles] -= 2*np.pi
#         elif branch_cuts == 'middle_out':
#             angles[angles>(2*np.pi - ext_angles/2)] -= 2*np.pi
#         elif type(branch_cuts) is float:
#             c = 1-branch_cuts
#             if c > 1 or c < 0:
#                 raise ValueError('branch cuts must be placed at a fraction of the exterior angle')
#             angles[angles>(2*np.pi - c*ext_angles)] -= 2*np.pi
#         elif type(branch_cuts) in [list,np.ndarray]:
#             c = 1-np.asarray(branch_cuts)
#             if np.any(c > 1) or np.any(c < 0):
#                 raise ValueError('branch cuts must be placed at a fraction of the exterior angle')
#             angles[angles>(2*np.pi - c*ext_angles)] -= 2*np.pi
#         else:
#             raise ValueError(f"'{branch_cuts}' not a valid branch cut type")
        
#         return angles

#     def _set_basis_eval(self,points,y=None):
#         """
#         Set up evaluation of a Fourier-Bessel basis. Useful for when the evaluation
#         points will be held constant for a variety of lambdas.
#         """
#         if y is not None:
#             points = complex_form(points,y)

#         # unpack attributes
#         n = self.n_vert
#         m = len(points)

#         # compute radii and angles relative to corners with expansions
#         orders = self.orders
#         mask = orders>0
#         r = np.abs(np.subtract.outer(points,self.vertices[mask]))
#         theta = self.fourier_bessel_angles(points,self.vertices,self.branch_cuts)

#         # set up evaluations of Fourier-Bessel
#         # first calculate the fourier part (independent of λ!)
#         alphak = self.alphak
#         sin = np.empty((m,orders.sum()))
#         cumk = np.concatenate(([0],np.cumsum(orders)))
#         for i in range(n):
#             if orders[i] > 0:
#                 sin[:,cumk[i]:cumk[i+1]] = np.sin(np.outer(theta[:,i],alphak[i]))

#         # set up evaluations of bessel part
#         r_rep = np.repeat(r,orders[mask],axis=1)

#         return r_rep,sin

#     def set_default_points(self,points,y=None):
#         if y is not None:
#             points = complex_form(points,y)
#         self.points = points
#         self.r_rep,self.sin = self._set_basis_eval(points)
#         self.cos, self.dr_dx, self.dr_dy, self.dtheta_dx, self.dtheta_dy = self._set_gradient_eval(points)

#     @cache
#     def basis_norms(self,lam):
#         if self.r_rep is None:
#             raise ValueError('Basis has no default points. Provide evaluation '\
#                             'points or use FourierBesselBasis.set_default_points')
#         r_rep,sin = self.r_rep,self.sin
#         out = jv(self.alphak_vec,np.sqrt(lam)*r_rep)*sin
#         norms = la.norm(out,axis=0)
#         norms[norms==0] = 1
#         return norms

#     def __call__(self,lam,points=None,y=None):
#         if (points is None) and (y is not None):
#             raise ValueError('x coordinates must be provided when y coordinates are provided')
#         elif points is None:
#             if self.r_rep is None:
#                 raise ValueError('Basis has no default points. Provide evaluation '\
#                                 'points or use FourierBesselBasis.set_default_points')
#             r_rep,sin = self.r_rep,self.sin
#         else:
#             if y is not None:
#                 points = complex_form(points,y)
#             r_rep,sin = self._set_basis_eval(points)

#         out = jv(self.alphak_vec,np.sqrt(lam)*r_rep)*sin
#         if self.normalize:
#             out = out/self.basis_norms(lam)
#         return out

#     def _set_gradient_eval(self,points,y=None):
#         if y is None:
#             x,y = points.real,points.imag
#         else:
#             x,y = np.asarray(x),np.asarray(y)
#         if (not x.shape or not y.shape) or (type(x) is float and type(y) is float):
#             x,y = np.array([x]),np.array([y])
#         if x.shape != y.shape:
#             raise ValueError('x and y must have the same shape')
#         x,y = x.flatten(),y.flatten()

#         # unpack attributes
#         x_v,y_v = self.vertices.real, self.vertices.imag
#         n = self.n_vert
#         m = len(x)

#         # compute radii and angles relative to corners with expansions
#         orders = self.orders
#         mask = orders>0
#         r = np.abs(np.subtract.outer(points,self.vertices[mask]))
#         theta = self.fourier_bessel_angles(points,self.vertices,self.branch_cuts)

#         # set up evaluations of Fourier-Bessel
#         # first calculate the fourier part (independent of λ!)
#         alphak = self.alphak
#         cos = np.empty((m,orders.sum()))
#         cumk = np.concatenate(([0],np.cumsum(orders)))
#         for i in range(n):
#             if orders[i] > 0:
#                 cos[:,cumk[i]:cumk[i+1]] = np.cos(np.outer(theta[:,i],alphak[i]))

#         # set up x & y partial derivatives
#         deltax = np.subtract.outer(x,x_v[mask])
#         deltay = np.subtract.outer(y,y_v[mask])

#         dr_dx = np.repeat(deltax/r,orders[mask],axis=1)
#         dr_dy = np.repeat(deltay/r,orders[mask],axis=1)
#         dtheta_dx = np.repeat(-deltay/(r**2),orders[mask],axis=1)
#         dtheta_dy = np.repeat(deltax/(r**2),orders[mask],axis=1)

#         return cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy

#     def grad(self,lam,points=None,y=None):
#         """Computes the gradients of the basis functions at the given points, returning
#         the values in complex form (real part for partials w.r.t. to x_i, imaginary part 
#         for partials w.r.t. y_i)"""
#         if y is not None:
#             points = complex_form(points,y)
#         if points is not None:
#             r_rep,sin = self._set_basis_eval(points)
#             cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy = self._set_gradient_eval(points)
#         else:
#             r_rep,sin = self.r_rep,self.sin
#             cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy = self.cos, self.dr_dx, self.dr_dy, self.dtheta_dx, self.dtheta_dy

#         sqrtlam_r = np.sqrt(lam)*r_rep
#         jv_ = jv(self.alphak_vec,sqrtlam_r)
#         jvp_ = jvp(self.alphak_vec,sqrtlam_r)
#         dr = np.sqrt(lam)*jvp_*sin
#         dtheta = self.alphak_vec*jv_*cos

#         if self.normalize:
#             norms = self.basis_norms(lam)
#             dr = dr/norms
#             dtheta = dtheta/norms

#         return dr*dr_dx + dtheta*dtheta_dx + 1j*(dr*dr_dy + dtheta*dtheta_dy)
    
#     def Aprime(self,lam,points=None,y=None):
#         """Computes the derivatives of the basis functions with respect to the spectral
#         parameter lambda. In other words, computes the matrix A'(lambda)."""
#         if (points is None) and (y is not None):
#             raise ValueError('x coordinates must be provided when y coordinates are provided')
#         elif points is None:
#             if self.r_rep is None:
#                 raise ValueError('Basis has no default points. Provide evaluation '\
#                                 'points or use FourierBesselBasis.set_default_points')
#             r_rep,sin = self.r_rep,self.sin
#         else:
#             if y is not None:
#                 points = complex_form(points,y)
#             r_rep,sin = self._set_basis_eval(points)

#         return (0.5/np.sqrt(lam))*r_rep*jvp(self.alphak_vec,np.sqrt(lam)*r_rep)*sin