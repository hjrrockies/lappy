# module imports
from .core import BaseEigensolver, BaseDomain
from .utils import invert_permutation, complex_form
from .opt import bracket_mins, minimize_on_bracket
from .geometry import PointSet, Polygon
from .bases import make_default_basis, ParticularBasis, NormalizedBasis, MultiBasis, FourierBesselBasis
from .cubature import polygon_cubature
from .asymp import weyl_est
from .rellich import build_rellich_data, rellich_gram_basis, rellich_gram_from_cauchy_data, \
    lowdin_transform, orthonormalize_coef
from .cauchy import CauchyData

from .cache import instance_lru_cache
import numpy as np
import scipy.linalg as la
import warnings
from gsvd4py import gsvd, gsvdvals
from tqdm import tqdm

### tolerance defaults
# rtol truncates the regularized pencil: larger discards more near-dependent
# columns. 1e-12 rather than 1e-14 because a basis of particular solutions is
# genuinely redundant -- rect(1,1) drops 27 of 120 columns, GWW1 (eight
# Fourier-Bessel corner blocks) far more -- and retaining directions below the
# level at which the matrix entries are known injects noise into the tension
# curve, which is what produces spurious minima and runaway bracket refinement.
#
# It is a compromise, not an optimum: measured over 12 domains, GWW1 gains 0.7
# digits and chevron 0.2, while mushroom loses 1.4 and L_shape 0.1. The right
# value is domain-dependent and there is no spectral gap to find it from (the
# largest consecutive drop in the singular values is only 0.4-0.7 decades --
# smooth decay, no rank cliff). See benchmarks/suite/run/NOTEBOOK.md.
rtol_default = 1e-12
ttol_default = 1e-3
ltol_default = 1e-8

# MPS functions
def regularize_pencil(A1, A2, reg_type='svd', rtol=rtol_default):
    """Regularizes the pencil for the MPS problem."""
    if not (np.isreal(rtol) and np.isscalar(rtol) and rtol >= 0):
        raise TypeError("'rtol' must be a nonnegative scalar")
    m = A1.shape[0]
    A = np.vstack([A1, A2])

    # svd-based regularization
    if reg_type == 'svd':
        # compute non-pivoted qr
        Q,R = la.qr(A, mode='economic')
        # compute truncated svd
        Z,s,Yt = la.svd(R)
        cutoff = (s >= rtol).sum()
        Z1 = Z[:,:cutoff]
        Y1 = Yt[:cutoff].T
        A = Q@Z1
        return A[:m], A[m:], Y1, s[:cutoff]
    
    # pivoted qr regularization
    elif reg_type == 'qrp':   
        # pivoted qr
        Q,R,P = la.qr(A, mode='economic', pivoting=True)
        # truncate
        cutoff = (np.abs(np.diag(R)) >= rtol).sum()
        A = Q[:,:cutoff]
        R11 = R[:cutoff,:cutoff]
        return A[:m], A[m:], R11, P
    
    else:
        raise ValueError(f"regularization method {reg_type} not one of 'svd', 'qrp', or 'implicit'")

def tensions(A1, A2, reg_type='svd', rtol=rtol_default):
    # regularize
    tola = tolb = None
    if reg_type == 'implicit':
        tola = np.max(A1.shape)*la.norm(A1, ord=1)*rtol
        tolb = np.max(A2.shape)*la.norm(A2, ord=1)*rtol
    elif reg_type:
        A1, A2 = regularize_pencil(A1, A2, reg_type, rtol)[:2]

    # compute generalized cosines and sines
    c, s = gsvdvals(A1, A2, tola=tola, tolb=tolb)
    return np.divide(c, s, out=np.full(c.shape, np.inf), where=(s!=0))[::-1]

def nullspace_coef(A1, A2, mult=1, reg_type='svd', rtol=rtol_default, ttol=ttol_default):
    # regularize
    tola = tolb = None
    if reg_type == 'implicit':
        tola = np.max(A1.shape)*la.norm(A1, ord=1)*rtol
        tolb = np.max(A2.shape)*la.norm(A2, ord=1)*rtol
    elif reg_type:
        A1, A2, M, v = regularize_pencil(A1, A2, reg_type, rtol)
        if reg_type == 'svd': Y1, s1 = M, v
        elif reg_type == 'qrp': R11, Pinv = M, invert_permutation(v)

    # compute GSVD (LAPACK-style)
    D1, D2, R, Q = gsvd(A1, A2, mode='separate', compute_u=False, compute_v=False, tola=tola, tolb=tolb)[:-2]
    c, s = np.max(D1, axis=0), np.max(D2, axis=0)
    idx = np.lexsort((s,-c))[-mult:][::-1]

    # warn if multiplicity is deficient
    if c[idx[-1]]/s[idx[-1]] > ttol:
        warnings.warn(f"Eigenvalue may have deficient multiplicity ({c[idx[-1]]/s[idx[-1]]:.3e}>{ttol:.3e})")

    # solve for coefficient vectors
    Er = np.eye(R.shape[0])[:,idx]
    Xr = Q[:,-R.shape[1]:]@la.solve_triangular(R, Er)

    # post-process if pre-regularization was used (svd/qrp truncate the pencil before the
    # GSVD; implicit/none truncate inside the GSVD itself, so Xr is already in original coords)
    if reg_type not in ('svd', 'qrp'): coef = Xr
    else:
        if reg_type == 'svd':
            coef = Y1@(Xr/(s1.reshape(-1,1)))
        elif reg_type == 'qrp':
            coef = np.zeros((A1.shape[1],mult))
            coef[Pinv] = la.solve_triangular(R11, Xr)
    return coef

def nullspace_eval(A1, A2, mult=1, A_extra=None, reg_type='svd', rtol=rtol_default, ttol=ttol_default):
    m2 = A2.shape[0]
    # handle additional points for evaluation
    if A_extra is not None:
        A2 = np.vstack([A2, A_extra])

    # regularize
    tola = tolb = None
    if reg_type == 'implicit':
        tola = np.max(A1.shape)*la.norm(A1, ord=1)*rtol
        tolb = np.max(A2.shape)*la.norm(A2, ord=1)*rtol
    elif reg_type:
        A1, A2, _, _ = regularize_pencil(A1, A2, reg_type, rtol)

    # compute GSVD
    U, V, C, S = gsvd(A1, A2, mode='econ', compute_right=False, tola=tola, tolb=tolb)
    c, s = np.sqrt(np.diag(C.T@C)), np.sqrt(np.diag(S.T@S))
    sigmas = np.divide(c, s, out=np.full(c.shape, np.inf), where=(s!=0))

    # warn if multiplicity is deficient
    if sigmas[-mult] > ttol:
        warnings.warn(f"Eigenvalue may have deficient multiplicity ({sigmas[-mult]:.3e}>{ttol:.3e})")

    # compute (weighted) nullspace evaluation
    U1, U2 = (U[:,-mult:]*c[-mult:])[:,::-1], (V[:,-mult:]*s[-mult:])[:,::-1]
    if A_extra is not None:
        U2, U_extra = U2[:m2], U2[m2:]
    else:
        U_extra = None

    # re-orthogonalize
    U2, Rhat = la.qr(U2, mode='economic')
    U1 = la.solve_triangular(Rhat, U1.T, trans=1).T
    if A_extra is not None:
        U_extra = la.solve_triangular(Rhat, U_extra.T, trans=1).T

    return tuple([arr for arr in (U1, U2, U_extra) if arr is not None])
    
def make_bdry_vander(basis, bdry_pts, bdry_normals=None, bc_param=0, bdry_wts=None):
    """Builds the MPS boundary matrix A_B(lam) corresponding to the given basis and boundary data."""
    # process boundary condition
    bc_param = np.asarray(bc_param)
    if not np.all(np.isreal(bc_param)):
        raise ValueError("'bc_param' must be real-valued")
    if bc_param.ndim > 1:
        raise ValueError("'bc_param' must be one or zero-dimensional")
    elif bc_param.ndim == 1:
        if len(bc_param) != len(bdry_pts):
            raise ValueError("'bc_param' must match the shape of 'bdry_pts'")

    # process bdry_pts
    if not isinstance(bdry_pts, PointSet):
        bdry_pts = PointSet(bdry_pts)

    # process bdry_wts
    if bdry_wts is None or bdry_wts is True: 
        bdry_wts = hasattr(bdry_pts, 'wts')
    elif not isinstance(bdry_wts, np.ndarray):
        raise TypeError("'bdry_wts' must be None, True/False, or ndarray")

    # dirichlet boundary condition
    if np.all(bc_param == 0):
        def A_B(lam): return basis(lam, bdry_pts, bdry_wts)
        
    # neumann boundary condition
    elif np.all(bc_param == 1):
        def A_B(lam): return basis.ddiff(lam, bdry_pts, bdry_normals, bdry_wts)
        
    # robin boundary condition
    else:
        bc_param = bc_param[:,np.newaxis]
        def A_B(lam):
            dir = basis(lam, bdry_pts, hasattr(bdry_pts, 'wts'))
            neu = basis.ddiff(lam, bdry_pts, bdry_normals, bdry_wts)
            return (1-bc_param)*dir + bc_param*neu
        
    return A_B
    
def make_vander(basis, pts, wts=None):
    # process inputs
    if not isinstance(pts, PointSet):
        pts = PointSet(pts)

    # process wts
    if wts is None or wts is True: 
        wts = hasattr(pts, 'wts')
    elif not isinstance(wts, np.ndarray):
        raise TypeError("'wts' must be None, True/False, or ndarray")
    
    def A(lam):
        return basis(lam, pts, wts)
    return A
    
def make_ddiff_vander(basis, pts, vecs, wts=None):
    # process inputs
    if not isinstance(pts, PointSet):
        pts = PointSet(pts)

    # process wts
    if wts is None or wts is True: 
        wts = hasattr(pts, 'wts')
    elif not isinstance(wts, np.ndarray):
        raise TypeError("'wts' must be None, True/False, or ndarray")
    
    def A(lam):
        return basis.ddiff(lam, pts, vecs, wts)
    return A

class MPSEigensolver(BaseEigensolver):
    def __init__(self, basis, bdry_pts, int_pts, bdry_normals=None, bc_param=0,
                 reg_type='svd', rtol=rtol_default, ttol=ttol_default, ltol=ltol_default,
                 cauchy_data=None):

        self.basis = basis
        self.bdry_pts = bdry_pts
        self.bdry_normals = bdry_normals
        self.int_pts = int_pts
        self.bc_param = bc_param

        # validate
        if not isinstance(basis, ParticularBasis):
            raise TypeError("'basis' must be a ParticularBasis object")

        self.A_B = make_bdry_vander(basis, bdry_pts, bdry_normals, bc_param)
        self.A_I = make_vander(basis, int_pts)

        # regularization and solver tolerances
        self.reg_type = reg_type
        self.rtol = rtol
        self.ttol = ttol
        self.ltol = ltol

        # Boundary-only Cauchy-data / quadrature bundle used for Rellich-identity
        # normalization (docs/rellich.md, docs/rellich_hadamard_mps.pdf). This is
        # opaque, already-materialized geometry+quadrature data (as from
        # rellich.build_rellich_data), mirroring bdry_pts/int_pts -- MPSEigensolver
        # itself has no notion of a domain. None disables eigenfunction_coef's
        # normalization (it falls back to raw coefficients with a warning).
        self._cauchy_data = cauchy_data

    @property
    def cauchy_data(self):
        """The boundary Cauchy-data/quadrature bundle backing Rellich-identity
        normalization (see rellich.build_rellich_data), or None if unavailable.
        Exposed for reuse by code building other boundary-integral quantities
        from the same basis/eigenfunctions (e.g. Hadamard-type shape-derivative
        formulas) -- see lappy.cauchy for the generic Cauchy-data/kernel-assembler
        API this is built on. lappy itself does not implement such formulas."""
        return self._cauchy_data

    @classmethod
    def default_basis(domain, n):
        pass

    @classmethod
    def default_bdry_pts():
        pass

    @classmethod
    def default_int_pts():
        pass

    @classmethod
    def from_domain(cls, domain, lam_max=None, prec=ltol_default, basis=None, mesh=False, weights=False,
                    reg_type='svd', rtol=rtol_default, ttol=ttol_default,
                    rellich=True, rellich_x0=None, rellich_mult=2, rellich_min_per_seg=4,
                    rellich_margin=2.0, rellich_c_lam=1.0, rellich_beta=0.2):
        if not isinstance(domain, BaseDomain):
            raise TypeError("'domain' must be a Domain object")
        if lam_max is None:
            lam_max = weyl_est(6, domain)

        # make basis for the domain
        if basis is None:
            raise NotImplementedError
        elif not isinstance(basis, ParticularBasis):
            raise TypeError("'basis' must be a ParticularBasis object")

        # boundary data
        n_per_seg = pts_per_seg(domain, basis)
        bdry_pts, bdry_normals, bc_param = make_default_bdry_data(domain, basis, weights)

        # interior points
        kind = 'mesh' if mesh else 'random'
        int_pts = make_default_int_pts(domain, kind, weights, len(basis), lam_max, prec)

        # normalize basis
        basis = basis.to_normalized((bdry_pts, int_pts))

        # boundary-only (Rellich identity) normalization data: see docs/rellich.md.
        # Robin boundaries are unsupported here; eigenfunction_coef falls back to raw
        # (un-normalized) coefficients with a warning in that case.
        cauchy_data = None
        if rellich:
            if domain.bc_type == 'rob':
                warnings.warn("Rellich-identity orthonormalization is not supported for "
                              "Robin boundary conditions; eigenfunction_coef will return "
                              "un-normalized coefficients.")
            else:
                cauchy_data = build_rellich_data(domain, basis, lam_max, rellich_x0, rellich_mult,
                                                 rellich_min_per_seg, rellich_margin,
                                                 rellich_c_lam, rellich_beta)

        return cls(basis, bdry_pts, int_pts, bdry_normals, bc_param, reg_type, rtol, ttol, prec,
                   cauchy_data=cauchy_data)
        
    def _get_params(self, reg_type=None, rtol=None, ttol=None, ltol=None):
        """Helper to resolve parameters against instance defaults"""
        return (
            reg_type if reg_type is not None else self.reg_type,
            rtol if rtol is not None else self.rtol,
            ttol if ttol is not None else self.ttol,
            ltol if ltol is not None else self.ltol
        )
    
    @instance_lru_cache(maxsize=256)
    def _tensions_scalar(self, lam, reg_type=None, rtol=None):
        """Evaluate tensions at a single scalar lambda=lam (cached)."""
        reg_type, rtol, _, _ = self._get_params(reg_type, rtol)
        return tensions(self.A_B(lam), self.A_I(lam), reg_type, rtol)

    def tensions(self, lam, reg_type=None, rtol=None, n_workers=1):
        """Evaluate tensions at lambda=lam.

        Parameters
        ----------
        lam : float or np.ndarray
            Spectral parameter. If an array, returns a list of tension arrays,
            one per element. Use n_workers > 1 for parallel evaluation.
        n_workers : int
            Number of threads for parallel array dispatch (default 1 = serial).
        """
        if isinstance(lam, np.ndarray):
            if lam.size == 0:
                return []
            if n_workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    return list(ex.map(
                        lambda l: self._tensions_scalar(float(l), reg_type, rtol), lam))
            return [self._tensions_scalar(float(l), reg_type, rtol) for l in lam]
        return self._tensions_scalar(float(lam), reg_type, rtol)

    def sigma(self, lam, reg_type=None, rtol=None):
        return self.tensions(lam, reg_type, rtol)[0]
    
    @instance_lru_cache(maxsize=64)
    def _cauchy_gram(self, eig):
        """Boundary-only (Rellich identity) L^2(Omega) Gram matrix of the basis at eig.
        See docs/rellich.md; cached per eigenvalue since it is otherwise the same cost
        order as one basis evaluation. NOTE: this is the basis-level (N x N) Gram matrix --
        sandwiching it between two copies of a coefficient vector (as orthonormalize_coef
        does) is the numerically risky pattern eigenfunction_coef no longer uses by default
        (docs/rellich_hadamard_mps.pdf Sec. 3.1); kept for direct use by tests/benchmarks and
        any external Hadamard-type consumer that genuinely needs the full basis Gram."""
        return rellich_gram_basis(self.basis, eig, self._cauchy_data)

    @instance_lru_cache(maxsize=64)
    def _nullspace_coef_raw(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        """Raw (arbitrarily-scaled) GSVD nullspace coefficients -- eigenfunction_coef's
        orthonorm=False path, factored out (and independently cached under a fixed
        calling convention) so orthonorm=True can reliably reuse this cache via a direct
        call rather than depending on eigenfunction_coef's own recursive-call kwarg style."""
        reg_type, rtol, ttol, _ = self._get_params(reg_type, rtol, ttol)
        return nullspace_coef(self.A_B(eig), self.A_I(eig), mult, reg_type, rtol, ttol)

    @instance_lru_cache(maxsize=64)
    def _orthonorm_transform_coef(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        """Löwdin orthonormalization transform D (mult x mult) for eigenfunction_coef's
        orthonorm=True path, built via the safe "evaluate first, sandwich never" Rellich Gram
        (docs/rellich_hadamard_mps.pdf Sec. 3.2): the raw candidate coefficients' OWN Cauchy
        data is evaluated directly at the shared Rellich boundary node set (a single
        un-sandwiched matrix-vector product -- basis(eig,pts)@coef, never coef.T@G_NxN@coef),
        and Löwdin-orthogonalized from the resulting small (mult x mult) Gram. Stays entirely
        within nullspace_coef's own GSVD pencil (no extra GSVD call). Returns (D, G), or
        (None, None) with a warning if no cauchy_data was supplied at construction."""
        if self._cauchy_data is None:
            warnings.warn("Rellich-identity normalization unavailable (no cauchy_data "
                          "supplied at construction); returning un-normalized coefficients.")
            return None, None
        coef = self._nullspace_coef_raw(eig, mult, reg_type, rtol, ttol)
        pts, normals, tangents = self._cauchy_data.pts, self._cauchy_data.normals, self._cauchy_data.tangents
        Phi = self.basis(eig, pts) @ coef
        Phi_N = self.basis.ddiff(eig, pts, normals) @ coef
        Phi_T = self.basis.ddiff(eig, pts, tangents) @ coef
        cd = CauchyData(pts, normals, tangents, self._cauchy_data.wts, Phi, Phi_N, Phi_T)
        G = rellich_gram_from_cauchy_data(cd, eig, self._cauchy_data)
        return lowdin_transform(G), G

    @instance_lru_cache(maxsize=64)
    def eigenfunction_coef(self, eig, mult=1, reg_type=None, rtol=None, ttol=None, orthonorm=True):
        """Coefficient vector(s) (in basis-function space) for the eigenfunction(s)
        at eig. By default (orthonorm=True), the coefficients are rescaled (and,
        for mult>1, rotated) to be orthonormal in the true L^2(Omega) inner product,
        computed via the boundary-only Rellich identity (docs/rellich.md) using the safe
        "evaluate first, sandwich never" transform (_orthonorm_transform_coef). Pass
        orthonorm=False to get the raw (arbitrarily-scaled) GSVD nullspace coefficients
        instead. Falls back to raw coefficients with a warning if no cauchy_data was
        supplied at construction (e.g. from_domain(..., rellich=False), a domain with
        Robin boundary conditions, or manual construction)."""
        coef = self._nullspace_coef_raw(eig, mult, reg_type, rtol, ttol)
        if not orthonorm:
            return coef
        D, G = self._orthonorm_transform_coef(eig, mult, reg_type, rtol, ttol)
        if D is None:
            return coef
        return coef @ D.T

    def eigenfunction(self, eig, mult=1, reg_type=None, rtol=None, ttol=None, orthonorm=True):
        coef = self.eigenfunction_coef(eig, mult, reg_type, rtol, ttol, orthonorm)
        def eigenfunc(pts):
            if isinstance(pts, PointSet):
                shape = (len(pts), coef.shape[1])
            else:
                pts = np.asarray(pts)
                if pts.dtype == 'complex128':
                    shape = (*pts.shape,  coef.shape[1])
                    pts = pts.flatten()
                elif pts.dtype == 'float64':
                    shape = (*pts.shape[:-1],  coef.shape[1])
                    pts = pts.reshape((-1,2))
                pts = PointSet(complex_form(pts))
            return (self.basis._eval_pointset(eig, pts)@coef).reshape(shape)
        return eigenfunc
    
    def eigenfunction_grad(self, eig, mult=1, reg_type=None, rtol=None, ttol=None, orthonorm=True):
        coef = self.eigenfunction_coef(eig, mult, reg_type, rtol, ttol, orthonorm)
        def eigenfunc_grad(pts):
            if isinstance(pts, PointSet):
                shape = (len(pts), coef.shape[1])
            else:
                pts = np.asarray(pts)
                if pts.dtype == 'complex128':
                    shape = (*pts.shape,  coef.shape[1])
                    pts = pts.flatten()
                elif pts.dtype == 'float64':
                    shape = (*pts.shape[:-1],  coef.shape[1])
                    pts = pts.reshape((-1,2))
                pts = PointSet(complex_form(pts))
            return (self.basis._grad_pointset(eig, pts)@coef).reshape(shape)
        return eigenfunc_grad
    
    def eigenfunction_eval_extras(self, eig, mult=1, extra_pts=None, ddiff_pts=None, ddiff_vecs=None,
                                  reg_type=None, rtol=None, ttol=None, orthonorm=False):
        reg_type, rtol, ttol, _ = self._get_params(reg_type, rtol, ttol)
        # convert extra_pts & ddiff_pts to PointSets
        if extra_pts is not None:
            if not isinstance(extra_pts, PointSet):
                extra_pts = PointSet(extra_pts)
            A_extra = self.basis(eig, extra_pts, hasattr(extra_pts, 'wts'))
        else: A_extra = None
        if ddiff_pts is not None:
            if not isinstance(ddiff_pts, PointSet):
                ddiff_pts = PointSet(ddiff_pts)
            A_ddiff = self.basis.ddiff(eig, ddiff_pts, ddiff_vecs, hasattr(ddiff_pts, 'wts'))
        else:
            A_ddiff = None

        # make vandermonde matrices
        A_B, A_I = self.A_B(eig), self.A_I(eig)
        if A_extra is None and A_ddiff is not None:
            A_extra = A_ddiff
        elif A_ddiff is not None:
            A_extra = np.vstack([A_extra, A_ddiff])

        # evaluate nullspace, unpack
        out = nullspace_eval(A_B, A_I, mult, A_extra, reg_type, rtol, ttol)
        if len(out) == 3:
            U_B, U_I, U_extra = out
            if extra_pts is not None and ddiff_pts is not None:
                U_extra, U_ddiff = U_extra[:len(extra_pts)], U_extra[len(extra_pts):]
            elif extra_pts is None and ddiff_pts is not None:
                U_extra, U_ddiff = None, U_extra
            else:
                U_ddiff = None
        else:
            U_B, U_I = out
            U_extra, U_ddiff = None, None

        # unweight for eigenfunction evaluation
        if hasattr(self.bdry_pts, 'wts'):
            U_B /= self.bdry_pts.sqrt_wts
        if hasattr(self.int_pts, 'wts'):
            U_I /= self.int_pts.sqrt_wts
        if hasattr(extra_pts, 'wts'):
            U_extra /= extra_pts.sqrt_wts
        if hasattr(ddiff_pts, 'wts'):
            U_ddiff /= ddiff_pts.sqrt_wts

        if orthonorm:
            D, G = self._orthonorm_transform_eval(eig, mult, reg_type, rtol, ttol)
            if D is not None:
                if U_B is not None: U_B = U_B @ D.T
                if U_I is not None: U_I = U_I @ D.T
                if U_extra is not None: U_extra = U_extra @ D.T
                if U_ddiff is not None: U_ddiff = U_ddiff @ D.T
        return tuple([arr for arr in (U_B, U_I, U_extra, U_ddiff) if arr is not None])

    def _rellich_cauchy_data_eval(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        """Evaluates the raw (unnormalized) candidate eigenfunctions' Cauchy data directly at
        the shared Rellich boundary node set, via the GSVD-eval pipeline (nullspace_eval) --
        never reconstructing a basis coefficient vector at all. Concatenates the node set with
        itself, paired with normals then tangents ("doubling trick"), to get both boundary
        derivatives from a single extra GSVD call. orthonorm=False here is required, not a
        default: this data is what _orthonorm_transform_eval is built FROM, so using
        orthonorm=True would recurse into the transform being constructed."""
        pts = self._cauchy_data.pts
        normals, tangents = self._cauchy_data.normals, self._cauchy_data.tangents
        ddiff_pts = np.concatenate([pts, pts])
        ddiff_vecs = np.concatenate([normals, tangents])
        _, _, U_extra, U_ddiff = self.eigenfunction_eval_extras(
            eig, mult, extra_pts=pts, ddiff_pts=ddiff_pts, ddiff_vecs=ddiff_vecs,
            reg_type=reg_type, rtol=rtol, ttol=ttol, orthonorm=False)
        n = len(pts)
        return CauchyData(pts, normals, tangents, self._cauchy_data.wts,
                          U_extra, U_ddiff[:n], U_ddiff[n:])

    @instance_lru_cache(maxsize=64)
    def _orthonorm_transform_eval(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        """Löwdin orthonormalization transform D (mult x mult) for eigenfunction_eval_extras's
        orthonorm=True path, built via GSVD-eval (nullspace_eval) Cauchy data at the shared
        Rellich boundary node set -- an independent transform from _orthonorm_transform_coef's
        (different GSVD pencil: nullspace_eval decomposes A_I augmented with the Rellich-node
        rows, nullspace_coef does not), since for a degenerate cluster (mult>1) the two pencils'
        raw candidate bases need not be related by the same rotation. Agrees with
        _orthonorm_transform_coef up to a global sign for mult=1 (empirically verified: both
        reach ~1e-10 relative accuracy against an independent cubature reference on the same
        test case). Returns (D, G), or (None, None) with a warning if no cauchy_data was
        supplied at construction."""
        if self._cauchy_data is None:
            warnings.warn("Rellich-identity normalization unavailable (no cauchy_data "
                          "supplied at construction); returning un-normalized values.")
            return None, None
        cd = self._rellich_cauchy_data_eval(eig, mult, reg_type, rtol, ttol)
        G = rellich_gram_from_cauchy_data(cd, eig, self._cauchy_data)
        return lowdin_transform(G), G

    @instance_lru_cache(maxsize=64)
    def eigenfunction_eval(self, eig, mult=1, reg_type=None, rtol=None, ttol=None, orthonorm=False):
        return self.eigenfunction_eval_extras(eig, mult, reg_type=reg_type, rtol=rtol, ttol=ttol,
                                              orthonorm=orthonorm)

    @instance_lru_cache(maxsize=64)
    def eigenfunction_eval_normals(self, eig, mult=1, reg_type=None, rtol=None, ttol=None, orthonorm=False):
        return self.eigenfunction_eval_extras(eig, mult, ddiff_pts=self.bdry_pts, ddiff_vecs=self.bdry_normals,
                                              reg_type=reg_type, rtol=rtol, ttol=ttol, orthonorm=orthonorm)

    def eigenfunction_energies(self, eig, mult=1, reg_type=None, rtol=None, ttol=None, orthonorm=True):
        """computes the energy values for the particular solution basis
        for the eigenfunction(s) corresponding to the given eigenvalue"""
        C = self.eigenfunction_coef(eig, mult, reg_type, rtol, ttol, orthonorm)
        AI = self.A_I(eig)
        col_norms = la.norm(AI, axis=0)
        denoms = la.norm(AI@C, axis=0)
        energies = ((col_norms[:,np.newaxis]*C)**2)/(denoms**2)
        return energies
    
    def _tension_diagnostics(self, lam, reg_type=None, rtol=None):
        """computes diagnostics for tension error estimation"""
        reg_type, rtol, _, _ = self._get_params(reg_type, rtol)
        A1, A2 = self.A_B(lam), self.A_I(lam)
        tola = tolb = None
        if reg_type == 'implicit':
            tola = np.max(A1.shape)*la.norm(A1, ord=1)*rtol
            tolb = np.max(A2.shape)*la.norm(A2, ord=1)*rtol
            A1_reg, A2_reg = A1, A2
        elif reg_type:
            A1_reg, A2_reg = regularize_pencil(A1, A2, reg_type, rtol)[:2]
        else:
            rtol = 0
            A1_reg, A2_reg = A1, A2

        # compute GSVD (LAPACK-style)
        D1, D2, R, Q, k, l = gsvd(A1_reg, A2_reg, mode='separate', compute_u=False, compute_v=False, tola=tola, tolb=tolb)
        c, s = np.max(D1, axis=0), np.max(D2, axis=0)
        idx = np.lexsort((s,-c))[-1]

        # solve for right generalized singular vector
        e = np.eye(R.shape[0])[:,idx]
        x = Q[:,-R.shape[1]:]@la.solve_triangular(R, e)

        # set up diagonistics dict
        out = dict()
        out['c'] = c[idx]
        out['s'] = s[idx]
        out['sigma'] = c[idx]/s[idx]
        out['x'] = x
        out['x_norm'] = la.norm(x)
        out['sigma_cond'] = (out['x_norm']/s[idx])*(1 + out['sigma'])
        AB_svdvals = la.svdvals(A1)
        out['AB_max_svdval'] = AB_svdvals.max()
        out['AB_min_svdval'] = AB_svdvals.min()
        AI_svdvals = la.svdvals(A2)
        out['AI_max_svdval'] = AI_svdvals.max()
        out['AI_min_svdval'] = AI_svdvals.min()
        AB_reg_svdvals = la.svdvals(A1_reg)
        out['AB_reg_max_svdval'] = AB_reg_svdvals.max()
        out['AB_reg_min_svdval'] = AB_reg_svdvals.min()
        AI_reg_svdvals = la.svdvals(A2_reg)
        out['AI_reg_max_svdval'] = AI_reg_svdvals.max()
        out['AI_reg_min_svdval'] = AI_reg_svdvals.min()
        out['n'] = A1.shape[1]
        out['n_reg'] = A1_reg.shape[1]
        out['k'] = k
        out['l'] = l
        out['gsvd_rank'] = k+l
        out['K'] = np.sqrt(out['n_reg'])*max(out['AB_reg_max_svdval'],out['AI_reg_max_svdval'])
        out['err_linalg'] = out['K']*(out['sigma']/out['AB_reg_min_svdval'])*(1 + out['sigma'])*1e-16
        out['err_reg'] = (1 + out['sigma'])*out['x_norm']*rtol

        return out


    def sigma_cond(self, lam, mult=1, reg_type=None, rtol=None):
        """computes the condition number of the smallest generalized singular value(s) of the MPS pencil,
        to detect the accuracy floor."""
        reg_type, rtol, _, _ = self._get_params(reg_type, rtol)
        A1, A2 = self.A_B(lam), self.A_I(lam)
        tola = tolb = None
        if reg_type == 'implicit':
            tola = np.max(A1.shape)*la.norm(A1, ord=1)*rtol
            tolb = np.max(A2.shape)*la.norm(A2, ord=1)*rtol
        elif reg_type:
            A1, A2 = regularize_pencil(A1, A2, reg_type, rtol)[:2]
        # compute GSVD (LAPACK-style)
        D1, D2, R, Q = gsvd(A1, A2, mode='separate', compute_u=False, compute_v=False, tola=tola, tolb=tolb)[:-2]
        c, s = np.max(D1, axis=0), np.max(D2, axis=0)
        idx = np.lexsort((s,-c))[-mult:][::-1]

        # solve for coefficient vectors
        Er = np.eye(R.shape[0])[:,idx]
        Xr = Q[:,-R.shape[1]:]@la.solve_triangular(R, Er)

        # return condition number(s) of smallest generalized singular value(s)
        cond = (la.norm(Xr, axis=0)/s[idx])*(1 + c[idx]/s[idx])
        return cond

    def solve_interval(self, a, b, n_pts, reg_type=None, rtol=None, ttol=None,
                       ltol=None, minsolver='parabolic', n_workers=1, verbose=0):
        """solves for all eigenvalues in [a,b] using MPS"""
        reg_type, rtol, ttol, ltol = self._get_params(reg_type, rtol, ttol, ltol)
        return solve_interval(lambda lam: self.tensions(lam, reg_type, rtol), a, b, n_pts,
                              ltol, ttol, minsolver, n_workers=n_workers, verbose=verbose)
    
    def plot_tensions(self, low, high, nlam, n_angle=1, rtol=None, reg_type=None,
                      ax=None, **plot_kwargs):
        import matplotlib.pyplot as plt
        lamgrid = np.linspace(low, high, nlam+1)
        results = self.tensions(lamgrid, reg_type, rtol, n_workers=10)
        tans = np.array([r[:n_angle] for r in results])
        if ax is None:
            fig = plt.figure()
            plt.plot(lamgrid, tans, **plot_kwargs)
            return fig
        else:
            ax.plot(lamgrid, tans, **plot_kwargs)
            ax.set_xlim(low, high)

    def adapt_rtol(self, a, b, n=15, reg_type=None, rtol_min=1e-14, rtol_max=1e-5):
        """Find the smallest rtol that yields a smooth sigma(λ) curve on [a, b].
        """
        reg_type, _, _, _ = self._get_params(reg_type)
        lamgrid = np.linspace(a, b, n)

        def noise(sigma):
            d1 = np.diff(sigma)
            d2 = np.diff(d1)
            return la.norm(d2)/la.norm(d1)

        # compute tensions, noise
        tensions0 = self.tensions(lamgrid, rtol=rtol_max)
        sigma0 = np.array([t[0] for t in tensions0])
        noise0 = noise(sigma0)

        # shrink rtol until noise jumps
        rtol = rtol_max/10
        tensions1 = self.tensions(lamgrid, rtol=rtol)
        sigma1 = np.array([t[0] for t in tensions1])
        noise1 = noise(sigma1)
        while noise1 < 1.1*noise0 and rtol >= rtol_min:
            rtol = rtol/10
            tensions1 = self.tensions(lamgrid, rtol=rtol)
            sigma1 = np.array([t[0] for t in tensions1])
            noise1 = noise(sigma1)

        return 10*rtol
    
### Minimization Eigsearch Code
def make_lamgrid(a, b, n_pts):
    """Makes a grid with ghost points"""
    if b <= a: raise ValueError("b must be greater than a")

    lamgrid_int = np.linspace(a,b,n_pts)
    lamgrid = np.empty(len(lamgrid_int)+2)
    lamgrid[1:-1] = lamgrid_int
    
    # add ghost points to ensure robust search
    lamgrid[0] = 2*lamgrid[1]-lamgrid[2]
    if lamgrid[0] <= 0: lamgrid[0] = a/2
    lamgrid[-1] = 2*lamgrid[-2]-lamgrid[-3]

    return lamgrid

def sort_merge_brackets(eig_brackets, ltol=ltol_default, verbose=0):
    # sort brackets in increasing order
    sort_idx = np.argsort([lam[1] for lam in eig_brackets])
    eig_brackets = [eig_brackets[i] for i in sort_idx]

    if verbose > 0:
        print(f"\tlen(brackets)={len(eig_brackets)}")

    # process brackets for proximity
    i = 0
    while i < len(eig_brackets)-1:
        brack0 = eig_brackets[i]
        brack1 = eig_brackets[i+1]
        if verbose > 1: print(f"\tchecking [{brack0[0]:.2e},{brack0[2]:.2e}]")
        tol = ltol*brack0[1]
        if brack1[1]-brack0[1] < tol:
            if verbose > 1: print(f"\tmerging [{brack1[0]:.2e},{brack1[2]:.2e}] diff={brack1[1]-brack0[1]} < tol={tol}")
            # merge, using average as eigenvalue
            # use lower bound of first bracket and upper bound of second bracket
            new_brack = np.empty(3, dtype='float')
            new_brack[0] = brack0[0]
            new_brack[1] = (brack0[1]+brack1[1])/2
            new_brack[2] = brack1[2]
            eig_brackets[i] = new_brack
            # delete no-longer needed bracket, post-merger
            del eig_brackets[i+1]
        else:
            i += 1
    if verbose > 0: print(f"\tlen(brackets)={len(eig_brackets)} after merging")
    return eig_brackets

def estimate_multiplicity(tensions, eig, a, b, ttol=ttol_default, verbose=0):
    # compute tensions at eigenvalue and bracket bounds
    t_eig = tensions(eig)
    t_a = tensions(a)
    t_b = tensions(b)

    # truncate to common length (number of converged generalized singular values)
    n = min(len(t_eig), len(t_a), len(t_b))
    t_eig = t_eig[:n]
    t_a = t_a[:n]
    t_b = t_b[:n]

    # check for presence of local min and sufficiently small tension
    is_locmin = (t_eig <= t_a)&(t_eig <= t_b)&((t_eig != t_a)|(t_eig != t_b))
    is_small = t_eig <= ttol
    if verbose > 1:
        print(f"\teig={eig:.16e}")
        print(f"\tis_locmin: {is_locmin[:10].astype(int)}")
        print(f"\tis_small:  {is_small[:10].astype(int)}")

    # multiplicity is the largest k such that is_locmin[j] & is_small[j] for all j < k
    mult = 0
    while mult < n and is_locmin[mult] and is_small[mult]:
        mult += 1
    if verbose > 0:
        print(f"\tmult({eig:.2e}) = {mult}")
    return mult
   
def solve_interval(tensions, a, b, n_pts, ltol=ltol_default, ttol=ttol_default,
                   minsolver='parabolic', bracket_kwargs={}, n_workers=1, verbose=0):
    """Finds eigenvalues from MPS tensions."""
    if minsolver not in ['parabolic','golden','brent']:
        raise ValueError("'minsolver' must be one of 'parabolic', 'golden', or 'brent'")

    # build initial search grid
    lamgrid = make_lamgrid(a, b, n_pts)
    if verbose > 0: print(f"solve_interval on [{a:.5e},{b:.5e}], n_pts={n_pts}")

    # evaluate tensions on the lambda grid
    if verbose > 0: print(f"1. evaluating tensions on lamgrid...")
    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            rows = list(ex.map(lambda lam: tensions(lam)[:2], lamgrid))
        tensiongrid = np.array(rows).T
    elif verbose > 0:
        tensiongrid = np.array([tensions(lam)[:2] for lam in tqdm(lamgrid)]).T
    else:
        tensiongrid = np.array([tensions(lam)[:2] for lam in lamgrid]).T
    fevals = len(lamgrid)

    # tension_logmean = 10.0**(np.log10(tensiongrid[0]).mean())
    # if tension_logmean < ttol:
    #     raise EigensolverFailure(f"Tension grid all small (logmean={tension_logmean}): solver likely needs to be reconfigured.")

    # get brackets containing minima
    # if verbose > 0:
    #     d1 = np.diff(tensiongrid)
    #     d2 = np.diff(d1)
    #     print("noise_est =", la.norm(d2)/la.norm(d1))
    if verbose > 0: print("2. finding eigenvalue brackets...")
    brackets, fe = bracket_mins(lambda lam: tensions(lam)[:2], lamgrid, 
                                tensiongrid, ltol, verbose=verbose-1, **bracket_kwargs)
    fevals += fe

    # minimize on each bracket, filtering for small tension values at minimizer
    if verbose > 0: print("3. minimizing tension on brackets...")
    eig_brackets = []
    for bracket in brackets:
        minimizer, fe = minimize_on_bracket(lambda lam: tensions(lam)[0], bracket, ltol, minsolver, verbose-1)
        fevals += fe
        minima = tensions(minimizer)[0]
        if minima <= ttol:
            # new bracket with "minimizer in the middle"
            if verbose > 1: print(f"eigenvalue accepted lam={minimizer:.5e}")
            lam = bracket[0]
            eig_brackets.append([lam[0], minimizer, lam[2]])
        elif verbose > 1: print(f"tension above threshold: {minima:.1e} > {ttol:.1e}")

    # sort brackets and merge sufficiently close eigenvalues
    if verbose > 0: print("4. sorting & merging eigenvalues...")
    eig_brackets = sort_merge_brackets(eig_brackets, ltol, verbose-1)

    # filter for eigenvalues within search interval
    if verbose > 0: print("5. filtering eigenvalues...")
    eig_brackets = [bracket for bracket in eig_brackets if (bracket[1] >= a and bracket[1] <= b)]
    eigs = [bracket[1] for bracket in eig_brackets]

    # estimate multiplicity for each eigenvalue
    if verbose > 0: print("6. estimating multiplicities...")
    mults = []
    for bracket in eig_brackets:
        a, eig, b = bracket
        mult = estimate_multiplicity(tensions, eig, a, b, ttol, verbose-1)
        mults.append(mult)
    if verbose > 0:
        print(f"***found {len(eigs)} eigenvalues, total_mult={np.sum(mults)}, fevals={fevals}***")
    return np.array(eigs), np.array(mults), fevals

### default MPS collocation points
def pts_per_seg(domain, basis, mult=2, min_per_seg=0):
    if isinstance(basis, NormalizedBasis):
        return pts_per_seg(domain, basis.basis, mult, min_per_seg)
    
    elif isinstance(basis, MultiBasis):
        pps = np.sum([pts_per_seg(domain, basis_, mult, 0) for basis_ in basis.bases], axis=0)
        return np.maximum(pps, min_per_seg).astype('int')
    
    elif isinstance(basis, FourierBesselBasis):
        # get the number of basis functions associated to each corner of the domain
        orders = np.zeros(len(domain.bdry.segments), dtype='int')
        p0 = np.array([seg.p0 for seg in domain.bdry.segments])
        has_basis = np.any(np.isclose(np.subtract.outer(p0, basis.sources), 0), axis=1)
        orders[has_basis] = basis.orders

        # get the adjacent edge lengths to ith vertex into the first and last positions of column i, then drop rows
        seg_lens = domain.seg_lens
        rolled_lens = np.array([np.roll(seg_lens, -j) for j in range(len(seg_lens))])[1:-1]

        # normalize each column by sum of edge lengths
        normalized_lens = rolled_lens/rolled_lens.sum(axis=0)

        # multiply by orders, take ceiling
        pps = np.ceil(mult*orders*normalized_lens)

        # unroll and sum
        pps = np.array([np.roll(pps[i], i+1) for i in range(len(pps))]).sum(axis=0)

        # threshold with min_per_seg
        return np.maximum(pps, min_per_seg).astype('int')
    
    else:
        # place points proportional to segmenth length
        seg_lens = domain.seg_lens
        pps = np.round(mult*len(basis)*seg_lens/seg_lens.sum()).astype(int)
        return np.maximum(pps, min_per_seg).astype('int')

def bdry_jacobi_exponents(domain, order=0):
    """Per-segment Gauss-Jacobi exponents (a, b) from corner angles
    (a/b = pi/angle at the segment's p0/pf corner), shifted down by `order`
    (scalar, or a length-n_segments array for per-segment control).

    order=0 (default) gives the primary eigenfunction exponent. order=1 gives
    the exponent for the eigenfunction's outward-normal derivative -- e.g. for
    Hadamard shape-derivative boundary integrals, which need this regardless
    of the PDE's own boundary condition. Higher/fractional order shifts are
    accepted for other known-quantity integrals (e.g. order=2 for a squared
    normal derivative, as in the classical Dirichlet shape-derivative
    integral integrand (du/dn)^2).

    Note: Gauss-Jacobi quadrature requires each resulting exponent > -1
    (unenforced here, per the same "leave uncapped" choice as the primary
    exponent -- scipy will raise if violated)."""
    int_angles = domain.int_angles
    a = np.pi/int_angles
    b = np.roll(a, -1)   # segment i's pf == segment (i+1)%n's p0

    order = np.broadcast_to(order, len(domain.bdry.segments))
    a = a - order
    b = b - order
    return a, b

def make_default_bdry_data(domain, basis, weights=False, mult=2, min_per_seg=0):
    """Default boundary collocation data for MPSEigensolver.from_domain.

    Plain Gauss-Legendre points, with point counts from pts_per_seg. MPS
    collocation is a pointwise-residual task, not an integration task: the
    GSVD searches over coefficient vectors that are (except at the located
    eigenvalue) not the true eigenfunction, so there's no fixed singular
    exponent to grade a quadrature rule to -- what matters is sample density
    relative to the boundary function's variation, which pts_per_seg already
    handles (it's also basis-aware: a corner's own adjacent segments are
    skipped, since Fourier-Bessel terms centered there vanish identically on
    those two edges by construction and so provide no collocation constraint).

    For accurate integration of a *known* boundary quantity with known corner
    singular behavior (e.g. the L2 norm of an eigenfunction's outward-normal
    derivative, or a shape-derivative sensitivity integral) use
    bdry_jacobi_exponents with domain.bdry_pts(..., kind='jacobi', a=a, b=b)
    directly instead -- a different task from collocation, deliberately kept
    separate here."""
    n_per_seg = pts_per_seg(domain, basis, mult, min_per_seg)
    bdry_pts = domain.bdry_pts(n_per_seg, kind='legendre', weights=weights)
    bdry_normals = domain.bdry_normals(n_per_seg, kind='legendre', weights=weights)
    bc_param = np.concatenate([np.full(n, seg.bc, 'float') for seg, n in zip(domain.bdry.segments, n_per_seg)])
    return bdry_pts, bdry_normals, bc_param

def make_default_int_pts(domain, kind='random', weights=False, npts_rand=50, lam_max=None, prec=1e-8):
    if isinstance(domain, Polygon):
        if kind == 'random':
            return domain.int_pts(method='random', weights=weights, npts_rand=npts_rand)
        elif kind == 'mesh':
            if lam_max is None:
                lam_max = weyl_est(6, domain)
            nodes, weights = polygon_cubature(domain, lam_max, prec)
            if weights:
                int_pts = PointSet(nodes, weights)
            else:
                int_pts = PointSet(nodes)
            return int_pts
    else:
        raise NotImplementedError