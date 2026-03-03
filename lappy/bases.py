from .utils import complex_form
from .asymp import weyl_est as _weyl_est
from .geometry import PointSet
from .core import BaseDomain

import numpy as np
from scipy.special import jv, jvp, yv, yvp
from scipy.linalg import norm
from functools import cache, lru_cache
from abc import ABC, abstractmethod

import mpmath as mp

def make_default_basis(domain, fs_orders=1, n_eig=10, ltol=1e-12, fs_mult=1, fb_mult=1, d_mult=1):
    """Make the default basis for a domain"""
    raise NotImplementedError("functionality under development")
    # approximate max spectral parameter
    lam_max = _weyl_est(n_eig+1, area=domain.area, perim=domain.perimeter, bc_type='dir')

    # get fundamental solution basis
    lens = np.array([seg.len for seg in domain.bdry.segments])
    n_per_seg = np.ceil(fs_mult*np.sqrt(lam_max)*lens).astype(int)
    d = d_mult*np.minimum(3/np.sqrt(lam_max), lens.mean()/4)
    basis = FundamentalBasis.from_domain(domain, n_per_seg, d, fs_orders)

    if len(domain.corners) > 0:
        corners = domain.corners
        # maximum from each corner
        R = np.array([domain.max_dist(corner) for corner in corners])

        # corner angles
        phi0, phi1 = domain.corner_angles
        ray0 = np.exp(1j*phi0)
        ray1 = np.exp(1j*phi1)
        angles = np.angle(ray1/ray0)
        angles[angles < 0] += 2*np.pi

        # compute alphas
        alpha = np.pi/angles
        
        # compute orders
        orders = np.ceil(fb_mult*np.sqrt(lam_max)*R/alpha).astype(int) + 2
        basis = FourierBesselBasis.from_domain(domain, orders) + basis
    
    return basis

class ParticularBasis(ABC):
    """Base class for function bases on the plane which depend on the spectral parameter λ."""
    def __call__(self, lam, pts, wts=False):
        """Evaluate the basis on a given set of points in the plane for a given spectral parameter value."""
        if not isinstance(pts, PointSet):
            pts = PointSet(pts)
        A = self._eval_pointset(lam, pts)
        if wts is True and hasattr(pts, 'wts'):
            return pts.sqrt_wts*A
        elif isinstance(wts, np.ndarray):
            return wts[:,np.newaxis]*A
        else: return A
        
    def grad(self, lam, pts, wts=False):
        """Evaluate the basis on a given set of points in the plane for a given spectral parameter value."""
        if not isinstance(pts, PointSet):
            pts = PointSet(pts)
        Agrad = self._grad_pointset(lam, pts)
        if wts is True and hasattr(pts, 'wts'):
            return pts.sqrt_wts*Agrad
        elif isinstance(wts, np.ndarray):
            return wts[:,np.newaxis]*Agrad
        else: return Agrad
        
    def ddiff(self, lam, pts, vecs, wts=False):
        if not isinstance(pts, PointSet):
            pts = PointSet(pts)
        Agrad = self._grad_pointset(lam, pts)
        if isinstance(vecs, PointSet):
            vecs = vecs.pts
        vecs = vecs[:,np.newaxis]
        Addiff = Agrad.real*vecs.real + Agrad.imag*vecs.imag
        if wts is True and hasattr(pts, 'wts'):
            return pts.sqrt_wts*Addiff
        elif isinstance(wts, np.ndarray):
            return wts[:,np.newaxis]*Addiff
        else: return Addiff

    @abstractmethod
    def _eval_pointset(self, lam, pts):
        pass

    def __add__(self, other):
        if not isinstance(other, ParticularBasis):
            raise TypeError("'other' must be an instance of ParticularBasis")
        if isinstance(other, MultiBasis):
            return MultiBasis([self]+other.bases)
        else:
            return MultiBasis([self, other])
    
    def to_normalized(self, quad_pts, quad_wts=None):
        return NormalizedBasis(self, quad_pts, quad_wts)
    
    @abstractmethod
    def __len__(self):
        pass

class MultiBasis(ParticularBasis):
    """Basis composed of the union of several bases"""
    def __init__(self, bases):
        if not np.all([isinstance(basis, ParticularBasis) for basis in bases]):
            raise TypeError("all elements of 'bases' must be instances of ParticularBasis")
        self.bases = list(bases)

    def _eval_pointset(self, lam, pts):
        return np.hstack([basis._eval_pointset(lam, pts) for basis in self.bases])
    
    def _grad_pointset(self, lam, pts):
        return np.hstack([basis._grad_pointset(lam, pts) for basis in self.bases])
    
    def __iadd__(self, other):
        if not isinstance(other, ParticularBasis):
            raise TypeError("'other' must be an instance of ParticularBasis")
        if isinstance(other, MultiBasis):
            self.bases = self.bases + other.bases
        else:
            self.bases.append(other)
        return self
    
    def __add__(self, other):
        if isinstance(other, MultiBasis):
            return MultiBasis(self.bases + other.bases)
        else:
            return super().__add__(other)
        
    def __str__(self):
        return f"MultiBasis({','.join([str(basis) for basis in self.bases])})"
    
    def __len__(self):
        return sum([len(basis) for basis in self.bases])
    
    def __getitem__(self, key):
        return self.bases[key]

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
        self.basis = basis
        
        if isinstance(quad_pts, PointSet):
            self.quad_pts = quad_pts
        else:
            self.quad_pts = PointSet(quad_pts)

        if quad_wts is not None:
            self.sqrt_wts = np.sqrt(quad_wts)[:,np.newaxis]
        elif hasattr(quad_pts, 'wts'):
            self.sqrt_wts = self.quad_pts.sqrt_wts
        else:
            self.sqrt_wts = None

    def __len__(self):
        return len(self.basis)

    @lru_cache
    def norms(self, lam):
        A = self.basis._eval_pointset(lam, self.quad_pts)
        if self.sqrt_wts is not None:
            A *= self.sqrt_wts
        norms = norm(A, axis=0)
        nonzero_cols = (norms > 0)
        return norms[nonzero_cols], nonzero_cols
    
    @lru_cache
    def _eval_pointset(self, lam, pts):
        A = self.basis._eval_pointset(lam, pts)
        norms, nonzero_cols = self.norms(lam)
        return A[:,nonzero_cols]/norms
    
    @lru_cache
    def _grad_pointset(self, lam, pts):
        Agrad = self.basis._grad_pointset(lam, pts)
        norms, nonzero_cols = self.norms(lam)
        return Agrad[:,nonzero_cols]/norms
    
    def __str__(self):
        return f"NormalizedBasis({self.basis})"
    
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
    def __init__(self, sources, phi0, phi1, orders, branch_cuts, kind='sin'):
        if kind not in ['cos','sin','sincos']:
            raise ValueError("'kind' must be one of 'sin', 'cos', or 'sincos'")
        if isinstance(orders, (int, np.integer)):
            orders = orders*np.ones(len(sources), dtype='int')
        else:
            orders = np.array(orders, dtype='int')
        orders = np.asarray(orders, dtype='int')
        mask = (orders > 0)
        if not np.any(mask):
            raise ValueError("at least one basis function must be included (orders must have at least one positive entry)")
        self.orders = orders[mask]
        self.kind = kind
        self.sources = complex_form(sources)[mask]
        self.orders = orders[mask]
        self._phi0 = phi0[mask]
        self._phi1 = phi1[mask]
        self.branch_cuts = branch_cuts[mask]
        self.branch_rays = np.exp(1j*self.branch_cuts)

        if self.orders.shape[0] != self.n_sources:
            raise ValueError('orders must match length of vertices')
        
        self._set_alphak()

    def __str__(self):
        return f"FourierBesselBasis(n_sources={self.n_sources}, n_func={len(self)}, kind={self.kind})"
    
    def __len__(self):
        if self.kind in ['sin','cos']:
            return self.orders.sum()
        elif self.kind == 'sincos':
            return 2*self.orders.sum()
    
    @classmethod
    def from_domain(cls, dom, orders):
        """Builds a Fourier-Bessel basis with source points at the corners of the given domain."""
        if not isinstance(dom, BaseDomain):
            raise TypeError("'dom' must be a valid Domain object")
        
        orders = np.asarray(orders, dtype='int')
        sources = dom.corners
        phi0, phi1 = dom.corner_angles

        # set branch cuts to bisect the exterior angle at each corner
        psi = np.angle(np.exp(phi0*1j)/np.exp(phi1*1j))
        psi[psi < 0] += 2*np.pi
        branch_cuts = phi1 + psi/2
        branch_cuts[branch_cuts >= 2*np.pi] -= 2*np.pi

        # determine kind of basis
        if dom.bc_type == 'dir': kind = 'sin'
        elif dom.bc_type == 'neu': kind = 'cos'
        else: kind = 'sincos'
        return cls(sources, phi0, phi1, orders, branch_cuts, kind)

    @property
    def n_sources(self):
        return len(self.sources)
    
    @property
    def alpha(self):
        phi = np.angle(self._ray1/self._ray0)
        phi[phi < 0] += 2*np.pi
        return np.pi/phi
    
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
        self.alphak = [alphai*np.arange(1,ki+1) for alphai,ki in zip(alpha,self.orders)]
        self.alphak_vec = np.concatenate(self.alphak)[np.newaxis]

    @cache
    def _z(self, pts):
        """Positions of PointSet pts relative to sources"""
        return np.subtract.outer(pts.pts, self.sources)

    @cache
    def _theta(self, pts):
        """Computes the angles of the given PointSet pts with respect to the bases vertices
        with branch cuts as needed."""
        # evaluation points relative to source points
        z = self._z(pts)

        # angles relative to phi0/ray0 for each source point
        theta = np.angle(z/self._ray0)
        theta[theta < 0] += 2*np.pi # make all angles nonnegative

        # wrap angles relative to branch cut
        phi_hat = np.angle(self.branch_rays/self._ray0) # angle of branch cuts relative to wedge starts
        phi_hat[phi_hat < 0] += 2*np.pi # make nonnegative
        theta[theta > phi_hat] -= 2*np.pi # angles past branch cut wrapped-down

        return theta
    
    @cache
    def _r(self, pts):
        """Computes the distances from the PointSet pts to the source points"""
        # evaluation points relative to source points
        z = self._z(pts)
        r = np.abs(z)
        return r
    
    @cache
    def _r_rep(self, pts):
        r = self._r(pts)
        r_rep = np.repeat(r,self.orders,axis=1)
        return r_rep
    
    @cache
    def _sin(self, pts):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet pts"""
        theta = self._theta(pts)
        sin = np.empty((len(pts),self.orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(self.orders)))
        for i in range(self.n_sources):
            sin[:,cumk[i]:cumk[i+1]] = np.sin(np.outer(theta[:,i],self.alphak[i]))
        return sin
    
    @cache
    def _cos(self, pts):
        """Computes the cosine terms of Fourier-Bessel functions on the given PointSet pts"""
        theta = self._theta(pts)
        cos = np.empty((len(pts),self.orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(self.orders)))
        for i in range(self.n_sources):
            cos[:,cumk[i]:cumk[i+1]] = np.cos(np.outer(theta[:,i],self.alphak[i]))
        return cos
    
    @lru_cache
    def _bessel(self, lam, pts):
        """Computes the Bessel part of the Fourier-Bessel functions on the given PointSet pts"""
        r_rep = self._r_rep(pts)
        bessel = jv(self.alphak_vec, np.sqrt(lam)*r_rep)
        return bessel
    
    @lru_cache
    def _besselp(self, lam, pts):
        """Computes the derivatives of the Bessel part of the Fourier-Bessel functions on the given PointSet pts"""
        r_rep = self._r_rep(pts)
        besselp = jvp(self.alphak_vec, np.sqrt(lam)*r_rep)
        return besselp

    @lru_cache
    def _eval_pointset(self, lam, pts):
        # get (potentially cached) evaluations of sine/cosine part and Bessel part
        bessel = self._bessel(lam, pts)
        if self.kind == 'sin':
            sin = self._sin(pts)
            return bessel*sin
        elif self.kind == 'cos':
            cos = self._cos(pts)
            return bessel*cos
        elif self.kind == 'sincos':
            sin = self._sin(pts)
            cos = self._cos(pts)
            return np.hstack((bessel*sin,bessel*cos))
    
    @cache
    def _dr_dz(self, pts):
         # partial derivatives of distance to source points w.r.t. x and y
        z = self._z(pts)
        r = self._r(pts)
        dr_dz = np.repeat(z/r, self.orders,axis=1)
        return dr_dz
        
    @cache
    def _dtheta_dz(self,pts):
        z = self._z(pts)
        r = self._r(pts)
        dtheta_dz_temp = (-z.imag + 1j*z.real)/(r**2)
        dtheta_dz = np.repeat(dtheta_dz_temp,self.orders,axis=1)
        return dtheta_dz
    
    @lru_cache
    def _grad_pointset(self, lam, pts):
        """Evaluates the gradients of the basis functions on the given PointSet. Returns in complex form, 
        with the x-partial in the real part and the y-partial in the imaginary part"""
        if not isinstance(pts, PointSet):
            raise TypeError("'pts' must be an instance of PointSet")

        # get (potentially cached) evaluations of components
        sin = self._sin(pts)
        cos = self._cos(pts)
        bessel = self._bessel(lam, pts)
        besselp = self._besselp(lam, pts)
        dr_dz = self._dr_dz(pts)
        dtheta_dz = self._dtheta_dz(pts)

        if self.kind == 'sin':
            dA_dr = np.sqrt(lam)*besselp*sin
            dA_dtheta = self.alphak_vec*bessel*cos
        elif self.kind == 'cos':
            dA_dr = np.sqrt(lam)*besselp*cos
            dA_dtheta = -self.alphak_vec*bessel*sin
        elif self.kind == 'sincos':
            arr1 = np.sqrt(lam)*besselp
            arr2 = self.alphak_vec*bessel
            dA_dr = np.hstack((arr1*sin,arr1*cos))
            dA_dtheta = np.hstack((arr2*cos,-arr2*sin))
            dr_dz = np.hstack((dr_dz, dr_dz))
            dtheta_dz = np.hstack((dtheta_dz, dtheta_dz))

        # combine using chain rule
        return dA_dr*dr_dz.real + dA_dtheta*dtheta_dz.real + 1j*(dA_dr*dr_dz.imag + dA_dtheta*dtheta_dz.imag)
    
class FundamentalBasis(ParticularBasis):
    """
    Basis of real-valued fundamental solutions to the Helmholtz equation
    -Δu = λu, placed at source points outside the domain.

    Each source point contributes basis functions of the form:
        Y_m(√λ · r_j) · cos(m · θ_j)     (m = 0, 1, ..., order-1)
        Y_m(√λ · r_j) · sin(m · θ_j)     (m = 1, 2, ..., order-1)
    where r_j and θ_j are polar coordinates relative to source point j.

    For m = 0 only the Y_0(√λ · r_j) monopole term is included (since sin(0) = 0).

    Parameters
    ----------
    sources : array_like
        Locations of the source points in the plane, as complex numbers.
        Should be placed *outside* the domain of interest.
    orders : int or array_like of int
        Maximum multipole order at each source point. If a single int,
        the same order is used at every source. order=1 gives monopoles only.
    """

    def __init__(self, sources, orders=1):
        self.sources = np.atleast_1d(np.asarray(sources, dtype=complex))
        if isinstance(orders, (int, np.integer)):
            self.orders = np.full(self.n_sources, orders, dtype=int)
        else:
            self.orders = np.asarray(orders, dtype=int)
        if self.orders.shape[0] != self.n_sources:
            raise ValueError("'orders' must match the number of source points")
        if not np.all(self.orders >= 1):
            raise ValueError("all orders must be >= 1")

        # Precompute the (source_index, m) pairs and the angular function type
        # for fast vectorized evaluation.
        self._build_index_maps()

    @classmethod
    def from_domain(cls, domain, n_per_seg, d=1, orders=1):
        if not isinstance(domain, BaseDomain):
            raise TypeError("'domain' must be a valid domain object")
        bdry_pts = domain.bdry_pts(n_per_seg, kind='even').pts
        bdry_normals = domain.bdry_normals(n_per_seg, kind='even').pts
        sources = bdry_pts + d*bdry_normals
        return cls(sources, orders)

    def _build_index_maps(self):
        """Build arrays that map each basis column to a source index, order m,
        and whether it is a cos or sin term."""
        source_indices = []
        ms = []
        is_sin = []  # False = cos (or m=0), True = sin

        for j, order in enumerate(self.orders):
            # m = 0: monopole (Y_0), counted once
            source_indices.append(j)
            ms.append(0)
            is_sin.append(False)
            # m >= 1: cos and sin pairs
            for m in range(1, order):
                source_indices.append(j)
                ms.append(m)
                is_sin.append(False)  # cos term
                source_indices.append(j)
                ms.append(m)
                is_sin.append(True)   # sin term

        self._src_idx = np.array(source_indices, dtype=int)
        self._m = np.array(ms, dtype=int)
        self._is_sin = np.array(is_sin, dtype=bool)

    def __str__(self):
        return (f"FundamentalBasis(n_sources={self.n_sources}, "
                f"n_func={len(self)}, orders={self.orders})")

    def __len__(self):
        # Each source with order K contributes 1 (m=0) + 2*(K-1) (m=1..K-1) = 2K - 1 functions
        return int(np.sum(2 * self.orders - 1))

    @property
    def n_sources(self):
        return len(self.sources)

    # ------------------------------------------------------------------ #
    #  Cached geometric computations (independent of λ)                  #
    # ------------------------------------------------------------------ #

    @cache
    def _z(self, pts):
        """Displacement vectors from each source to each evaluation point."""
        return np.subtract.outer(pts.pts, self.sources)

    @cache
    def _r(self, pts):
        """Distances from each evaluation point to each source."""
        return np.abs(self._z(pts))

    @cache
    def _theta(self, pts):
        """Angles from each evaluation point to each source."""
        return np.angle(self._z(pts))

    @cache
    def _r_cols(self, pts):
        """r values broadcast to basis columns: shape (n_pts, n_basis)."""
        r = self._r(pts)
        return r[:, self._src_idx]

    @cache
    def _angular(self, pts):
        """Cosine and sine angular factors for every basis column."""
        theta = self._theta(pts)
        theta_cols = theta[:, self._src_idx]  # (n_pts, n_basis)
        m_theta = self._m[np.newaxis, :] * theta_cols
        ang = np.where(self._is_sin, np.sin(m_theta), np.cos(m_theta))
        return ang

    @cache
    def _angular_deriv(self, pts):
        """Derivative of angular factor w.r.t. θ for every basis column.
        d/dθ cos(mθ) = -m sin(mθ),  d/dθ sin(mθ) = m cos(mθ)."""
        theta = self._theta(pts)
        theta_cols = theta[:, self._src_idx]
        m_theta = self._m[np.newaxis, :] * theta_cols
        m = self._m[np.newaxis, :]
        dang = np.where(self._is_sin,
                        m * np.cos(m_theta),
                        -m * np.sin(m_theta))
        return dang

    # ------------------------------------------------------------------ #
    #  Cached radial (Bessel) computations (depend on λ)                 #
    # ------------------------------------------------------------------ #

    @lru_cache
    def _bessel(self, lam, pts):
        """Y_m(√λ · r) for every basis column."""
        k = np.sqrt(lam)
        r_cols = self._r_cols(pts)
        return yv(self._m[np.newaxis, :], k * r_cols)

    @lru_cache
    def _besselp(self, lam, pts):
        """Y_m'(√λ · r) for every basis column (derivative w.r.t. the argument)."""
        k = np.sqrt(lam)
        r_cols = self._r_cols(pts)
        return yvp(self._m[np.newaxis, :], k * r_cols)

    # ------------------------------------------------------------------ #
    #  Core evaluation methods                                            #
    # ------------------------------------------------------------------ #

    @lru_cache
    def _eval_pointset(self, lam, pts):
        """Evaluate all basis functions at the given points.

        Returns array of shape (n_pts, n_basis).
        """
        return self._bessel(lam, pts) * self._angular(pts)

    @lru_cache
    def _grad_pointset(self, lam, pts):
        """Evaluate gradients of all basis functions at the given points.

        Returns a complex array of shape (n_pts, n_basis) where the
        real part is ∂/∂x and the imaginary part is ∂/∂y.
        """
        k = np.sqrt(lam)
        bessel = self._bessel(lam, pts)
        besselp = self._besselp(lam, pts)
        ang = self._angular(pts)
        dang = self._angular_deriv(pts)

        # Radial contribution:  dA/dr = k · Y_m'(kr) · angular
        dA_dr = k * besselp * ang

        # Angular contribution: (1/r) · Y_m(kr) · d(angular)/dθ
        r_cols = self._r_cols(pts)
        dA_dtheta = bessel * dang  # will divide by r below

        # Chain rule: ∂r/∂x = (x-x0)/r,  ∂r/∂y = (y-y0)/r
        #             ∂θ/∂x = -(y-y0)/r², ∂θ/∂y = (x-x0)/r²
        z = self._z(pts)
        z_cols = z[:, self._src_idx]
        dx = z_cols.real  # x - x_j
        dy = z_cols.imag  # y - y_j

        dr_dx = dx / r_cols
        dr_dy = dy / r_cols
        dtheta_dx = -dy / (r_cols ** 2)
        dtheta_dy = dx / (r_cols ** 2)

        grad_x = dA_dr * dr_dx + dA_dtheta * dtheta_dx
        grad_y = dA_dr * dr_dy + dA_dtheta * dtheta_dy

        return grad_x + 1j * grad_y
    
class ExPrecFBBasis(FourierBesselBasis):
    """Evaluates a Fourier-Bessel Basis in extended precision."""
    def __init__(self, sources, phi0, phi1, orders, branch_cuts, kind, dps):
        super().__init__(sources, phi0, phi1, orders, branch_cuts, kind)
        self.dps = dps
    
    @classmethod
    def from_domain(cls, dom, orders, dps):
        """Builds a Fourier-Bessel basis with source points at the corners of the given domain."""
        if not isinstance(dom, BaseDomain):
            raise TypeError("'dom' must be a valid Domain object")
        
        orders = np.asarray(orders, dtype='int')
        sources = dom.corners
        phi0, phi1 = dom.corner_angles

        # set branch cuts to bisect the exterior angle at each corner
        psi = np.angle(np.exp(phi0*1j)/np.exp(phi1*1j))
        psi[psi < 0] += 2*np.pi
        branch_cuts = phi1 + psi/2
        branch_cuts[branch_cuts >= 2*np.pi] -= 2*np.pi

        # determine kind of basis
        if dom.bc_type == 'dir': kind = 'sin'
        elif dom.bc_type == 'neu': kind = 'cos'
        else: kind = 'sincos'
        return cls(sources, phi0, phi1, orders, branch_cuts, kind, dps)
    
    @cache
    def _sin(self, pts):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet ps"""
        mp.mp.dps = self.dps
        theta = self._theta(pts)
        sin = mp.matrix(pts.pts.shape[0], int(self.orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(self.orders)))
        for i in range(self.n_sources):
            alphak_theta = np.outer(theta[:,i],self.alphak[i])
            for j in range(pts.pts.shape[0]):
                for k in range(cumk[i],cumk[i+1]):
                    kk = k-cumk[i]
                    sin[j,k] = mp.sin(alphak_theta[j,kk])
        return sin
    
    @cache
    def _cos(self, pts):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet ps"""
        mp.mp.dps = self.dps
        theta = self._theta(pts)
        cos = mp.matrix(pts.pts.shape[0],int(self.orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(self.orders)))
        for i in range(self.n_sources):
            alphak_theta = np.outer(theta[:,i],self.alphak[i])
            for j in range(pts.pts.shape[0]):
                for k in range(cumk[i],cumk[i+1]):
                    kk = k-cumk[i]
                    cos[j,k] = mp.cos(alphak_theta[j,kk])
        return cos
    
    @lru_cache
    def _bessel(self, lam, pts):
        """Computes the Bessel part of the Fourier-Bessel functions on the given PointSet ps"""
        mp.mp.dps = self.dps
        r_rep = self._r_rep(pts)
        sqrtlam_r_rep = np.sqrt(lam)*r_rep
        bessel = mp.matrix(r_rep.shape[0], r_rep.shape[1])
        for i in range(r_rep.shape[0]):
            for j in range(r_rep.shape[1]):
                bessel[i,j] = mp.besselj(self.alphak_vec[0,j], sqrtlam_r_rep[i,j])
        return bessel
    
    @lru_cache
    def _eval_pointset(self, lam, pts):
        mp.mp.dps = self.dps
        mat = self._eval_pointset_mp(lam, pts)
        arr = np.array(mat.tolist(), dtype='float')
        return arr
    
    @lru_cache
    def _eval_pointset_mp(self, lam, pts):
        # get (potentially cached) evaluations of sine part and Bessel part
        mp.mp.dps = self.dps
        bessel = self._bessel(lam, pts)
        if self.kind == 'sin':
            sin = self._sin(pts)
            # take product
            mat = mp.matrix(bessel.rows, bessel.cols)
            for i in range(bessel.rows):
                for j in range(bessel.cols):
                    mat[i,j] = bessel[i,j]*sin[i,j]
            return mat
        elif self.kind == 'cos':
            cos = self._cos(pts)
            # take product
            mat = mp.matrix(bessel.rows, bessel.cols)
            for i in range(bessel.rows):
                for j in range(bessel.cols):
                    mat[i,j] = bessel[i,j]*cos[i,j]
            return mat
        elif self.kind == 'sincos':
            sin = self._sin(pts)
            cos = self._cos(pts)
            # take product
            m, n = bessel.rows, bessel.cols
            mat = mp.matrix(m, 2*n)
            for i in range(m):
                for j in range(n):
                    mat[i,j] = bessel[i,j]*sin[i,j]
                    mat[i,j+n] = bessel[i,j]*cos[i,j]
            return mat
        
class NormalizedExPrecFBBasis(ExPrecFBBasis):
    def __init__(self, basis, norm_pts):
        if not isinstance(basis, ExPrecFBBasis):
            raise TypeError("'basis' must be an extended-precision Fourier-Bessel basis")
        self.basis = basis
        self.dps = self.basis.dps
        
        if isinstance(norm_pts, PointSet):
            self.quad_pts = norm_pts
        else:
            self.quad_pts = PointSet(norm_pts)

    def __len__(self):
        return len(self.basis)

    @lru_cache
    def norms(self, lam):
        mp.mp.dps = self.dps
        A = self.basis._eval_pointset_mp(lam, self.quad_pts)
        m, n = A.rows, A.cols
        norms = mp.matrix(n, 1)
        for j in range(n):
            norms[j,0] = mp.norm(A[:,j])
        return norms
    
    @lru_cache
    def _eval_pointset_mp(self, lam, pts):
        mp.mp.dps = self.dps
        A = self.basis._eval_pointset_mp(lam, pts)
        norms = self.norms(lam)
        mat = mp.matrix(A.rows, A.cols)
        for j in range(A.cols):
            mat[:,j] = A[:,j]/norms[j,0]
        return mat
    
class InfNormalizedExPrecFBBasis(ExPrecFBBasis):
    def __init__(self, basis, norm_pts):
        if not isinstance(basis, ExPrecFBBasis):
            raise TypeError("'basis' must be an extended-precision Fourier-Bessel basis")
        self.basis = basis
        self.dps = self.basis.dps
        
        if isinstance(norm_pts, PointSet):
            self.quad_pts = norm_pts
        else:
            self.quad_pts = PointSet(norm_pts)

    def __len__(self):
        return len(self.basis)

    @lru_cache
    def norms(self, lam):
        mp.mp.dps = self.dps
        A = self.basis._eval_pointset_mp(lam, self.quad_pts)
        m, n = A.rows, A.cols
        norms = mp.matrix(n, 1)
        for j in range(n):
            norms[j,0] = mp.norm(A[:,j], p=mp.inf)
        return norms
    
    @lru_cache
    def _eval_pointset_mp(self, lam, pts):
        mp.mp.dps = self.dps
        A = self.basis._eval_pointset_mp(lam, pts)
        norms = self.norms(lam)
        mat = mp.matrix(A.rows, A.cols)
        for j in range(A.cols):
            mat[:,j] = A[:,j]/norms[j,0]
        return mat