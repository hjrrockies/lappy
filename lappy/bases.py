from .utils import complex_form
from .asymp import weyl_est
from .geometry import PointSet
from .core import BaseDomain

import numpy as np
from scipy.special import jv, jvp, yv, yvp
from scipy.linalg import norm
from .cache import instance_cache, instance_lru_cache
from abc import ABC, abstractmethod

import mpmath as mp

def _points_in_polygon(pts, poly):
    """Vectorized even-odd point-in-polygon test. pts, poly are complex arrays."""
    x, y = pts.real, pts.imag
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    for k in range(n):
        x0, y0 = poly[k].real, poly[k].imag
        x1, y1 = poly[(k+1) % n].real, poly[(k+1) % n].imag
        crosses = ((y0 <= y) & (y < y1)) | ((y1 <= y) & (y < y0))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(crosses, (y - y0)/(y1 - y0), 0.0)
        x_cross = x0 + t*(x1 - x0)
        inside ^= crosses & (x_cross > x)
    return inside

def make_default_basis(domain, n_basis, fs_frac=0.5, fs_bdry_order=1, fs_d=1.0, 
                       fs_corner_order=2, fs_sigma=1.0, fs_C=10.0):
    """Make the default basis for a domain of size n_basis"""
    # smooth domains: pure boundary fundamental solutions
    if len(domain.corners) == 0:
        n_sources = np.round(n_basis/(2*(fs_bdry_order-1)+1)).astype(int)
        # single segment case
        if len(domain.bdry.segments) == 1:
            t = np.linspace(0,1,n_sources+1)[:-1]
            bdry_pts = domain.bdry.segments[0].p(t)
            bdry_normals = domain.bdry.segments[0].N(t)
            sources = bdry_pts + fs_d*bdry_normals
        # multiple segments
        else:
            lens = domain.seg_lens
            sources_per_seg = np.round(n_sources*lens/lens.sum()).astype(int)
            T = [np.linspace(0, 1, k+1) for k in sources_per_seg]
            bdry_pts = np.concatenate([seg.p(t) for seg,t in zip(domain.bdry.segments,T)])
            bdry_normals = np.concatenate([seg.N(t) for seg,t in zip(domain.bdry.segments,T)])
            sources = bdry_pts + fs_d*bdry_normals
        basis = FundamentalBasis(sources, fs_bdry_order)

    # domains with corners
    else:
        int_angles = domain.int_angles
        ratio = np.pi / int_angles
        is_regular = np.abs(ratio - np.round(ratio))/ratio < 1e-15
        is_singular = ~is_regular
        # all regular corners: pure Fourier-Bessel at each corner
        if np.all(is_regular):
            orders = fb_corner_orders(domain, n_basis)
            basis = FourierBesselBasis.from_domain(domain, orders)
        # one singular corner: pure Fourier-Bessel at singular corner
        elif is_singular.sum() == 1:
            orders = fb_corner_orders(domain, n_basis)
            basis = FourierBesselBasis.from_domain(domain, orders)
        # multiple singular corners: 50-50 split of Fourier-Bessel at singular corners, FS near corners
        else:
            n_fs = np.round(fs_frac*n_basis).astype(int)
            n_fb = n_basis - n_fs
            # Fourier-Bessel terms
            fb_orders = fb_corner_orders(domain, n_fb)
            fb_basis = FourierBesselBasis.from_domain(domain, fb_orders)

            # Fundamental solution terms
            sources_per_corner = fs_corner_orders(domain, n_fs, order=fs_corner_order)
            diam = domain.diameter
            fs_basis = FundamentalBasis.by_corners(domain, sources_per_corner, fs_C*diam, fs_sigma, fs_corner_order)

            # combine
            basis = fb_basis + fs_basis
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
    
    def to_normalized(self, quad_pts, quad_wts=None, max_scale=True):
        return NormalizedBasis(self, quad_pts, quad_wts, max_scale=max_scale)
    
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
    involve a Pointset which is different from the desired evaluation set). Prunes only basis terms with exactly
    zero norm. Accepts either a single PointSet or a list of component PointSets (e.g. [bdry_pts, int_pts])
    for norm computation.
    """
    def __init__(self, basis, pts, wts=None, max_scale=False):
        if not isinstance(basis, ParticularBasis):
            raise TypeError("'basis' must be an instance of ParticularBasis")
        self.basis = basis

        if isinstance(pts, PointSet):
            self.component_pts = [pts]
        else:
            self.component_pts = list(pts)
            for p in self.component_pts:
                if not isinstance(p, PointSet):
                    raise TypeError("each element of pts must be a PointSet")

        # wts: legacy scalar-weight path — only valid for single combined PointSet
        if wts is not None:
            self._legacy_quad_wts = np.sqrt(wts)[:, np.newaxis]
        else:
            self._legacy_quad_wts = None

        # rescale each column by max before norm computation
        self.max_scale = max_scale

    def __len__(self):
        return len(self.basis)

    @instance_lru_cache(maxsize=4)
    def _weighted_eval(self, lam, pts):
        """basis._eval_pointset with row-weighting baked in (if pts has weights)."""
        A = self.basis._eval_pointset(lam, pts)
        if hasattr(pts, 'sqrt_wts'):
            return A * pts.sqrt_wts
        return A

    @instance_lru_cache(maxsize=4)
    def _weighted_grad_eval(self, lam, pts):
        """basis._grad_pointset with row-weighting baked in (if pts has weights)."""
        Ag = self.basis._grad_pointset(lam, pts)
        if hasattr(pts, 'sqrt_wts'):
            return Ag * pts.sqrt_wts
        return Ag

    @instance_lru_cache(maxsize=128)
    def norms(self, lam):
        As = [self._weighted_eval(lam, pts) for pts in self.component_pts]
        A = np.vstack(As)
        if self._legacy_quad_wts is not None:
            A = A * self._legacy_quad_wts
        if self.max_scale:
            col_max = np.abs(A.max(axis=0))
            col_max[col_max==0] = 1.0
            A /= col_max
        col_norms = norm(A, axis=0)
        if self.max_scale:
            col_norms *= col_max
        active = col_norms > 0
        return col_norms[active], active

    def _eval_pointset(self, lam, pts):
        A = self.basis._eval_pointset(lam, pts)
        norms, active = self.norms(lam)
        return A[:, active] / norms

    def _grad_pointset(self, lam, pts):
        Ag = self.basis._grad_pointset(lam, pts)
        norms, active = self.norms(lam)
        return Ag[:, active] / norms

    def __call__(self, lam, pts, wts=False):
        if not isinstance(pts, PointSet):
            pts = PointSet(pts)
        norms, active = self.norms(lam)          # warms _weighted_eval for component_pts
        if wts is True and hasattr(pts, 'wts'):
            A_w = self._weighted_eval(lam, pts)  # cache HIT if pts is a component
            return A_w[:, active] / norms
        elif isinstance(wts, np.ndarray):
            A = self.basis._eval_pointset(lam, pts)
            return (A[:, active] / norms) * wts[:, np.newaxis]
        else:
            A = self.basis._eval_pointset(lam, pts)
            return A[:, active] / norms

    def ddiff(self, lam, pts, vecs, wts=False):
        if not isinstance(pts, PointSet):
            pts = PointSet(pts)
        norms, active = self.norms(lam)
        if isinstance(vecs, PointSet):
            vecs = vecs.pts
        vecs = vecs[:, np.newaxis]
        if wts is True and hasattr(pts, 'wts'):
            Ag_w = self._weighted_grad_eval(lam, pts)  # cache HIT if pts is a component
            Ag = Ag_w[:, active] / norms
        elif isinstance(wts, np.ndarray):
            Ag_raw = self.basis._grad_pointset(lam, pts)
            Ag = (Ag_raw[:, active] / norms) * wts[:, np.newaxis]
        else:
            Ag_raw = self.basis._grad_pointset(lam, pts)
            Ag = Ag_raw[:, active] / norms
        return Ag.real * vecs.real + Ag.imag * vecs.imag

    def __str__(self):
        return f"NormalizedBasis({self.basis})"

# Fourier-Bessel bases and helper functions
def fb_corner_fraction(domain, regular_mult=0, singular_mult=1, reentrant_mult=1):
    """computes the score for each corner for placement of Fourier-Bessel particular solutions"""
    # weight by angle measure
    int_angles = domain.int_angles
    scores = int_angles.copy()

    # apply adjustments for regular, singular, reentrant corners
    ratio = np.pi / int_angles
    is_regular = np.abs(ratio - np.round(ratio))/ratio < 1e-15
    # all regular corners: override regular_mult
    if np.all(is_regular): regular_mult = 1
    scores[is_regular] *= regular_mult

    is_singular = ~is_regular
    scores[is_singular] *= singular_mult

    is_reentrant = (int_angles > np.pi)
    scores[is_reentrant] *= reentrant_mult

    # all weights zero => uniform weighting
    if scores.max() == 0:
        return np.ones(len(int_angles))/len(int_angles)
    else:
        # fraction is score[i] / sum(scores)
        return scores/scores.sum()

def fb_corner_orders(domain, n_basis, f=None, min_order=1, exact=False):
    """computes the number of FB terms to place at domain corners for a basis of size n_basis"""
    if f is None:
        f = fb_corner_fraction(domain)
    orders = np.zeros(len(domain.corners), dtype=int)
    orders[f != 0] = min_order
    remaining = n_basis - orders.sum()

    if remaining > 0:
        if not exact:
            orders_plus = np.round(remaining*f).astype(int)
        if exact:
            orders_plus = np.floor(remaining*f).astype(int)
            remainders = remaining*f - orders_plus
            deficit = remaining - orders_plus.sum()
            if deficit > 0:
                priority = np.lexsort((f, remainders))[::-1]
                for i in priority[:deficit]:
                    orders_plus[i] += 1
        orders += orders_plus
    return orders
 
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
    def __init__(self, sources, phi0, phi1, orders, branch_cuts, kind='sin', branch_polylines=None):
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
        self.branch_cuts = np.array(branch_cuts, dtype=float)[mask]

        # optional polyline+ray branch cuts, aligned with sources (None = plain ray).
        # Each entry is a tuple (vertices, beta): exterior polyline vertices [q1,...,qm]
        # and the final ray angle. For these sources the wrapping ray is the initial
        # polyline direction arg(q1 - c).
        keep = np.nonzero(mask)[0]
        if branch_polylines is None:
            self._polylines = [None]*self.n_sources
        else:
            self._polylines = [branch_polylines[j] for j in keep]
        for i, pl in enumerate(self._polylines):
            if pl is not None:
                verts, _beta = pl
                self.branch_cuts[i] = np.angle(complex_form(np.asarray(verts))[0] - self.sources[i])
        self.branch_rays = np.exp(1j*self.branch_cuts)

        if self.orders.shape[0] != self.n_sources:
            raise ValueError('orders must match length of vertices')

        self._set_alphak()
        self._build_index_maps()
        self._setup_polyline_cuts()

    def __str__(self):
        return f"FourierBesselBasis(n_sources={self.n_sources}, n_func={len(self)}, kind={self.kind})"
    
    def __len__(self):
        if self.kind in ['sin','cos']:
            return self.orders.sum()
        elif self.kind == 'sincos':
            return 2*self.orders.sum()
    
    @staticmethod
    def _corner_branch_data(dom, polyline_cuts):
        """Branch cut data for corner sources: validated free rays where possible, and
        polyline+ray cuts for any "surrounded" corner with no straight-ray sightline.
        Returns (branch_cuts, branch_polylines), the latter None when all cuts are plain."""
        from .geometry import corner_branch_cut_polyline
        branch_cuts = dom.branch_cut_rays()
        nan_idx = np.where(np.isnan(branch_cuts))[0]
        if len(nan_idx) == 0:
            return branch_cuts, None
        if not polyline_cuts:
            raise ValueError(
                f"corners {nan_idx.tolist()} have no straight-ray branch cut to infinity; "
                "pass polyline_cuts=True to auto-generate polyline+ray cuts")
        branch_polylines = [None]*len(dom.corners)
        for i in nan_idx:
            verts, beta = corner_branch_cut_polyline(dom, int(i))
            branch_polylines[i] = (verts, beta)
            branch_cuts[i] = np.angle(verts[0] - dom.corners[i])   # finite placeholder
        return branch_cuts, branch_polylines

    @classmethod
    def from_domain(cls, dom, orders, polyline_cuts=True):
        """Builds a Fourier-Bessel basis with source points at the corners of the given
        domain. Branch cuts are placed by `dom.branch_cut_rays()`; corners with no
        straight-ray sightline get an auto-generated polyline+ray cut (set
        `polyline_cuts=False` to instead raise on such corners)."""
        if not isinstance(dom, BaseDomain):
            raise TypeError("'dom' must be a valid Domain object")

        orders = np.asarray(orders, dtype='int')
        sources = dom.corners
        phi0, phi1 = dom.corner_angles
        branch_cuts, branch_polylines = cls._corner_branch_data(dom, polyline_cuts)

        # determine kind of basis
        if dom.bc_type == 'dir': kind = 'sin'
        elif dom.bc_type == 'neu': kind = 'cos'
        else: kind = 'sincos'
        return cls(sources, phi0, phi1, orders, branch_cuts, kind, branch_polylines=branch_polylines)

    @classmethod
    def at_corners(cls, domain, orders, polyline_cuts=True):
        """Builds a corner-adapted Fourier-Bessel basis for the given domain.
        """
        orders = np.asarray(orders, dtype='int')
        sources = domain.corners
        phi0, phi1 = domain.corner_angles
        branch_cuts, branch_polylines = cls._corner_branch_data(domain, polyline_cuts)

        # determine kind of basis
        if domain.bc_type == 'dir': kind = 'sin'
        elif domain.bc_type == 'neu': kind = 'cos'
        else: kind = 'sincos'
        return cls(sources, phi0, phi1, orders, branch_cuts, kind, branch_polylines=branch_polylines)
    
    @classmethod
    def on_boundary(cls, domain, n_per_seg, order=1, spacing='even'):
        """Builds a Fourier-Bessel basis along the boundary of 'domain'"""
        n_per_seg = np.asarray(n_per_seg, dtype='int')
        
        sources = domain.bdry_pts(n_per_seg, kind=spacing).pts
        tangents = domain.bdry_tangents(n_per_seg, kind=spacing).pts
        phi0 = np.angle(tangents)
        phi0[phi0 <= 0] += 2*np.pi
        phi1 = phi0 + np.pi
        phi1[phi1 > 2*np.pi] -= 2*np.pi
        branch_cuts = phi0

        # determine kind of basis
        if domain.bc_type == 'dir': kind = 'sin'
        elif domain.bc_type == 'neu': kind = 'cos'
        else: kind = 'sincos'
        orders = order*np.full(len(sources), order)
        return cls(sources, phi0, phi1, orders, branch_cuts, kind)

    @property
    def n_sources(self):
        return len(self.sources)
    
    @property
    def alpha(self):
        phi = np.angle(self._ray1/self._ray0)
        phi[phi <= 0] += 2*np.pi
        return np.pi/phi

    def _set_alphak(self):
        # write angles as complex rays
        self._ray0 = np.exp(1j*self._phi0)
        self._ray1 = np.exp(1j*self._phi1)

        # compute alpha[i]*k for k in [1,...,orders[i]] for the ith source point
        # wedge angle must lie in (0, 2*pi], since 0 would mean ray1 coincides
        # with ray0 (a degenerate wedge of full angle 2*pi)
        phi = np.angle(self._ray1/self._ray0)
        phi[phi <= 0] += 2*np.pi
        alpha = np.pi/phi
        self.alphak = [alphai*np.arange(1,ki+1) for alphai,ki in zip(alpha,self.orders)]
        self.alphak_vec = np.concatenate(self.alphak)[np.newaxis]

        # branch cut offset relative to ray0, also in (0, 2*pi]
        self._phi_hat = np.angle(self.branch_rays/self._ray0)
        self._phi_hat[self._phi_hat <= 0] += 2*np.pi

        del self._phi0, self._phi1

    def _build_index_maps(self):
        """Build arrays mapping each basis column to its source index and alpha*k value."""
        src_indices, alphak_vals = [], []
        for i, aks in enumerate(self.alphak):
            for ak in aks:
                src_indices.append(i)
                alphak_vals.append(ak)
        self._src_idx = np.array(src_indices, dtype=int)
        self._alphak_col = np.array(alphak_vals)

    def _setup_polyline_cuts(self):
        """Precompute, for each source with a polyline+ray branch cut, the data needed to
        relocate the 2pi sheet discontinuity from the initial ray onto the polyline.
        Sources with plain ray cuts incur no setup and no per-eval cost."""
        self._polyline_srcs = [i for i, pl in enumerate(self._polylines) if pl is not None]
        self._poly_data = {}
        for i in self._polyline_srcs:
            verts, beta = self._polylines[i]
            verts = complex_form(np.asarray(verts))
            c = self.sources[i]
            theta0 = self.branch_cuts[i]
            sign = self._polyline_sign(i, c, verts, theta0, float(beta))
            self._poly_data[i] = (c, verts, float(beta), float(theta0), sign)

    def _wrapped_theta(self, i, z_rel):
        """The plain wrapped angle of complex offsets z_rel (= z - source_i) for source i,
        matching the vectorized formula in _theta."""
        th = np.angle(z_rel/self._ray0[i])
        th = np.where(th <= 0, th + 2*np.pi, th)
        th = np.where(th > self._phi_hat[i], th - 2*np.pi, th)
        return th

    def _correction_polygon(self, c, verts, theta0, beta, L):
        """Polygon [q1,...,qm, B, A] enclosing the region between the initial ray (beyond
        q1) and the polyline+ray cut, capped at radius L."""
        A = verts[0] + L*np.exp(1j*theta0)
        B = verts[-1] + L*np.exp(1j*beta)
        return np.concatenate((verts, [B, A]))

    def _polyline_sign(self, i, c, verts, theta0, beta):
        """Determine the +/-1 sheet-correction sign by enforcing continuity of the
        corrected angle across the initial ray just beyond q1."""
        scale = np.abs(verts[0] - c)
        nu, delta, L = 0.5*scale, 1e-3*scale, 1e3*scale + np.abs(verts - c).max()
        M = verts[0] + nu*np.exp(1j*theta0)            # on the initial ray, beyond q1
        n_hat = 1j*np.exp(1j*theta0)                   # unit normal to the ray
        zp, zm = M + delta*n_hat, M - delta*n_hat
        thp = self._wrapped_theta(i, np.array([zp]) - c)[0]
        thm = self._wrapped_theta(i, np.array([zm]) - c)[0]
        P = self._correction_polygon(c, verts, theta0, beta, L)
        mp_ = _points_in_polygon(np.array([zp]), P)[0]
        mm_ = _points_in_polygon(np.array([zm]), P)[0]
        dmask = int(mp_) - int(mm_)
        if dmask == 0:
            return 1   # degenerate; correction region not separated here
        return int(np.round((thp - thm)/(2*np.pi*dmask)))

    @instance_cache
    def _z(self, pts):
        """Positions of PointSet pts relative to sources"""
        return np.subtract.outer(pts.pts, self.sources)

    @instance_cache
    def _theta(self, pts):
        """Computes the angles of the given PointSet pts with respect to the bases vertices
        with branch cuts as needed."""
        # evaluation points relative to source points
        z = self._z(pts)

        # angles relative to phi0/ray0 for each source point, in (0, 2*pi]
        theta = np.angle(z/self._ray0)
        theta[theta <= 0] += 2*np.pi

        # wrap angles relative to precomputed branch cut offset
        theta[theta > self._phi_hat] -= 2*np.pi # angles past branch cut wrapped-down

        # polyline+ray cuts: relocate the 2pi discontinuity off the initial ray onto the
        # polyline by a +/-2pi sheet correction inside the enclosed region. Skipped
        # entirely when there are no polyline cuts (the common case).
        if self._polyline_srcs:
            zpts = pts.pts
            for i in self._polyline_srcs:
                c, verts, beta, theta0, sign = self._poly_data[i]
                L = 2*np.abs(zpts - c).max() + 2*np.abs(verts - c).max() + 1.0
                P = self._correction_polygon(c, verts, theta0, beta, L)
                mask = _points_in_polygon(zpts, P)
                theta[:, i] -= sign*2*np.pi*mask

        return theta
    
    @instance_cache
    def _r(self, pts):
        """Computes the distances from the PointSet pts to the source points"""
        # evaluation points relative to source points
        z = self._z(pts)
        r = np.abs(z)
        return r

    @instance_cache
    def _r_rep(self, pts):
        return self._r(pts)[:, self._src_idx]

    @instance_cache
    def _sin(self, pts):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet pts"""
        theta = self._theta(pts)
        theta_cols = theta[:, self._src_idx]
        return np.sin(theta_cols * self._alphak_col)

    @instance_cache
    def _cos(self, pts):
        """Computes the cosine terms of Fourier-Bessel functions on the given PointSet pts"""
        theta = self._theta(pts)
        theta_cols = theta[:, self._src_idx]
        return np.cos(theta_cols * self._alphak_col)
    

    def _bessel(self, lam, pts):
        """Computes the Bessel part of the Fourier-Bessel functions on the given PointSet pts"""
        r_rep = self._r_rep(pts)
        bessel = jv(self.alphak_vec, np.sqrt(lam)*r_rep)
        return bessel


    def _besselp(self, lam, pts):
        """Computes the derivatives of the Bessel part of the Fourier-Bessel functions on the given PointSet pts"""
        r_rep = self._r_rep(pts)
        besselp = jvp(self.alphak_vec, np.sqrt(lam)*r_rep)
        return besselp


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
    
    @instance_cache
    def _dr_dz(self, pts):
         # partial derivatives of distance to source points w.r.t. x and y
        z = self._z(pts)
        r = self._r(pts)
        dr_dz = np.repeat(z/r, self.orders,axis=1)
        return dr_dz

    @instance_cache
    def _dtheta_dz(self,pts):
        z = self._z(pts)
        r = self._r(pts)
        dtheta_dz_temp = (-z.imag + 1j*z.real)/(r**2)
        dtheta_dz = np.repeat(dtheta_dz_temp,self.orders,axis=1)
        return dtheta_dz
    

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
    
def fs_bdry_sps(domain, n, order=1, min_per_seg=1):
    """determines sources_per_seg for a FS basis of (approximate) size 'n' along the boundary of the domain"""
    n_sources = np.round(n/(1+2*(order-1)))
    seg_lens = domain.seg_lens
    sources_per_seg = np.round(n_sources*seg_lens/seg_lens.sum()).astype(int)
    sources_per_seg[sources_per_seg < min_per_seg] = min_per_seg
    return sources_per_seg

def fs_corner_fraction(domain, regular_mult=0, singular_mult=1, reentrant_mult=1):
    """computes the score for each corner for placement of FS particular solutions"""
    # weight by angle measure
    int_angles = domain.int_angles
    scores = int_angles.copy()

    # apply adjustments for regular, singular, reentrant corners
    ratio = np.pi / int_angles
    is_regular = np.abs(ratio - np.round(ratio))/ratio < 1e-15
    # all regular corners: override regular_mult
    if np.all(is_regular): regular_mult = 1
    scores[is_regular] *= regular_mult

    is_singular = ~is_regular
    scores[is_singular] *= singular_mult

    is_reentrant = (int_angles > np.pi)
    scores[is_reentrant] *= reentrant_mult

    # all weights zero => uniform weighting
    if scores.max() == 0:
        return np.ones(len(int_angles))/len(int_angles)
    else:
        # fraction is score[i] / sum(scores)
        return scores/scores.sum()

def fs_corner_orders(domain, n, f=None, order=1, min_sources=1):
    """computes the number of FS terms to place at domain corners for a basis of size n"""
    if f is None:
        f = fs_corner_fraction(domain)
    n_sources = np.round(n/(1+2*(order-1)))
    sources_per_corner = np.zeros(len(domain.corners), dtype=int)
    sources_per_corner[f != 0] = min_sources
    remaining = n_sources - sources_per_corner.sum()

    if remaining > 0:
        sources_per_corner += np.round(remaining*f).astype(int)
    
    return sources_per_corner
    
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
    
    @classmethod
    def by_boundary(cls, domain, n_per_seg, d=1, order=1, spacing='even'):
        n_per_seg = np.asarray(n_per_seg, dtype='int')
        bdry_pts = domain.bdry_pts(n_per_seg, kind=spacing).pts
        bdry_normals = domain.bdry_normals(n_per_seg, kind=spacing).pts
        sources = bdry_pts + d*bdry_normals
        return cls(sources, order)
    
    @classmethod
    def by_corners(cls, domain, n_per_corner, C=1, sigma=1, order=1):
        if isinstance(n_per_corner, np.integer):
            n_per_corner = np.full(len(domain.corners), n_per_corner)
        psi = domain.int_angles
        phi0, phi1 = domain.corner_angles
        out_angles = phi0 + psi/2
        rays = -np.exp(1j*out_angles)
        
        J = [np.arange(1,n+1) for n in n_per_corner]
        dists = [C*np.exp(-sigma*(np.sqrt(n)-np.sqrt(j))) for n,j in zip(n_per_corner, J)]
        sources = np.concatenate([corner+d*ray for corner,d,ray in zip(domain.corners,dists,rays)])
        return cls(sources, order)

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

    @instance_cache
    def _z(self, pts):
        """Displacement vectors from each source to each evaluation point."""
        return np.subtract.outer(pts.pts, self.sources)

    @instance_cache
    def _r(self, pts):
        """Distances from each evaluation point to each source."""
        return np.abs(self._z(pts))

    @instance_cache
    def _theta(self, pts):
        """Angles from each evaluation point to each source."""
        return np.angle(self._z(pts))

    @instance_cache
    def _r_cols(self, pts):
        """r values broadcast to basis columns: shape (n_pts, n_basis)."""
        r = self._r(pts)
        return r[:, self._src_idx]

    @instance_cache
    def _angular(self, pts):
        """Cosine and sine angular factors for every basis column."""
        theta = self._theta(pts)
        theta_cols = theta[:, self._src_idx]  # (n_pts, n_basis)
        m_theta = self._m[np.newaxis, :] * theta_cols
        ang = np.where(self._is_sin, np.sin(m_theta), np.cos(m_theta))
        return ang

    @instance_cache
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


    def _bessel(self, lam, pts):
        """Y_m(√λ · r) for every basis column."""
        k = np.sqrt(lam)
        r_cols = self._r_cols(pts)
        return yv(self._m[np.newaxis, :], k * r_cols)


    def _besselp(self, lam, pts):
        """Y_m'(√λ · r) for every basis column (derivative w.r.t. the argument)."""
        k = np.sqrt(lam)
        r_cols = self._r_cols(pts)
        return yvp(self._m[np.newaxis, :], k * r_cols)

    # ------------------------------------------------------------------ #
    #  Core evaluation methods                                            #
    # ------------------------------------------------------------------ #


    def _eval_pointset(self, lam, pts):
        """Evaluate all basis functions at the given points.

        Returns array of shape (n_pts, n_basis).
        """
        return self._bessel(lam, pts) * self._angular(pts)


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
    
    @instance_cache
    def _sin(self, pts):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet ps"""
        mp.mp.dps = self.dps
        theta = self._theta(pts)
        alphak_theta = theta[:, self._src_idx] * self._alphak_col
        n_pts, n_basis = alphak_theta.shape
        sin = mp.matrix(n_pts, n_basis)
        for j in range(n_pts):
            for k in range(n_basis):
                sin[j,k] = mp.sin(alphak_theta[j,k])
        return sin

    @instance_cache
    def _cos(self, pts):
        """Computes the sine terms of Fourier-Bessel functions on the given PointSet ps"""
        mp.mp.dps = self.dps
        theta = self._theta(pts)
        alphak_theta = theta[:, self._src_idx] * self._alphak_col
        n_pts, n_basis = alphak_theta.shape
        cos = mp.matrix(n_pts, n_basis)
        for j in range(n_pts):
            for k in range(n_basis):
                cos[j,k] = mp.cos(alphak_theta[j,k])
        return cos
    

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
    

    def _eval_pointset(self, lam, pts):
        mp.mp.dps = self.dps
        mat = self._eval_pointset_mp(lam, pts)
        arr = np.array(mat.tolist(), dtype='float')
        return arr


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


    def norms(self, lam):
        mp.mp.dps = self.dps
        A = self.basis._eval_pointset_mp(lam, self.quad_pts)
        m, n = A.rows, A.cols
        norms = mp.matrix(n, 1)
        for j in range(n):
            norms[j,0] = mp.norm(A[:,j])
        return norms


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


    def norms(self, lam):
        mp.mp.dps = self.dps
        A = self.basis._eval_pointset_mp(lam, self.quad_pts)
        m, n = A.rows, A.cols
        norms = mp.matrix(n, 1)
        for j in range(n):
            norms[j,0] = mp.norm(A[:,j], p=mp.inf)
        return norms


    def _eval_pointset_mp(self, lam, pts):
        mp.mp.dps = self.dps
        A = self.basis._eval_pointset_mp(lam, pts)
        norms = self.norms(lam)
        mat = mp.matrix(A.rows, A.cols)
        for j in range(A.cols):
            mat[:,j] = A[:,j]/norms[j,0]
        return mat