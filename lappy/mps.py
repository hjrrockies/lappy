# module imports
from .core import BaseEigenproblem, BaseEigensolver, BaseDomain
from .utils import invert_permutation, complex_form
from .opt import bracket_mins, minimize_on_bracket, discrete_locmin_idx
from .geometry import PointSet, pts_per_seg
from .bases import make_default_basis, ParticularBasis

from functools import cache, lru_cache
import numpy as np
import scipy.linalg as la
import warnings
from pygsvd import gsvd
import matplotlib.pyplot as plt
from scipy.optimize import bracket, minimize_scalar

### tolerance defaults
rtol_default = 1e-14
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
        raise ValueError(f"regularization method {reg_type} not one of 'svd' or 'qrp'")
    
def tensions(A1, A2, reg_type='svd', rtol=rtol_default):
    # regularize
    if reg_type:
        A1, A2 = regularize_pencil(A1, A2, reg_type, rtol)[:2]

    # compute GSVD    
    C,S,_ = gsvd(A1, A2, extras='')
    return np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))[::-1]

def nullspace_coef(A1, A2, mult=1, reg_type='svd', rtol=rtol_default, ttol=ttol_default):
    # regularize
    if reg_type:
        A1, A2, M, v = regularize_pencil(A1, A2, reg_type, rtol)
        if reg_type == 'svd': Y1, s1 = M, v
        elif reg_type == 'qrp': R11, Pinv = M, invert_permutation(v)

    # compute GSVD
    C,S,X = gsvd(A1, A2, extras='')
    sigmas = np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))
        
    # warn if multiplicity is deficient
    if sigmas[-mult] > ttol:
        warnings.warn(f"Eigenvalue may have deficient multiplicity ({sigmas[-mult]:.3e}>{ttol:.3e})")

    # solve for coefficient vectors
    Xr = la.lstsq(X.T, np.eye(X.shape[1])[:,-mult:])[0][:,::-1]
    if not reg_type: coef = Xr
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
    if reg_type:
        A1, A2, _, _ = regularize_pencil(A1, A2, reg_type, rtol)

    # compute GSVD
    C,S,_,U,V = gsvd(A1, A2)
    sigmas = np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))

    # warn if multiplicity is deficient
    if sigmas[-mult] > ttol:
        warnings.warn(f"Eigenvalue may have deficient multiplicity ({sigmas[-mult]:.3e}>{ttol:.3e})")

    # compute (weighted) nullspace evaluation
    U1, U2 = (U[:,-mult:]*C[-mult:])[:,::-1], (V[:,-mult:]*S[-mult:])[:,::-1]
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
        @lru_cache
        def A_B(lam): return basis(lam, bdry_pts, bdry_wts)
        
    # neumann boundary condition
    elif np.all(bc_param == 1):
        @lru_cache
        def A_B(lam): return basis.ddiff(lam, bdry_pts, bdry_normals, bdry_wts)
        
    # robin boundary condition
    else:
        bc_param = bc_param[:,np.newaxis]
        @lru_cache
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
    
    @lru_cache
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
    
    @lru_cache
    def A(lam):
        return basis.ddiff(lam, pts, vecs, wts)
    return A

class MPSEigensolver(BaseEigensolver):
    def __init__(self, basis, bdry_pts, int_pts, bdry_normals=None, bc_param=0, 
                 reg_type='svd', rtol=rtol_default, ttol=ttol_default, ltol=ltol_default):

        self.A_B = make_bdry_vander(basis, bdry_pts, bdry_normals, bc_param)
        self.A_I = make_vander(basis, int_pts)
        self.A_N = make_ddiff_vander(basis, bdry_pts, bdry_normals)

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

    @classmethod
    def from_domain(cls, domain, basis=None, use_mesh=False, use_weights=False, cubature_kind='dunavant', 
                    cubature_deg=4, mesh_kwargs={}, reg_type='svd', rtol=rtol_default, 
                    ltol=ltol_default, ttol=ttol_default):
        raise NotImplementedError("functionality under development")
        if not isinstance(domain, BaseDomain):
            raise TypeError("'eigenprob' must be an Eigenproblem object")
        # make basis for the domain
        if basis is None:
            basis = make_default_basis(domain, ltol=ltol)
        elif not isinstance(basis, ParticularBasis):
            raise TypeError("'basis' must be a ParticularBasis object")

        # boundary data
        n_per_seg = pts_per_seg(domain, basis)
        bdry_pts, bdry_normals, bc_param = domain.bdry_data(n_per_seg, weights=use_weights)

        # interior points
        if not use_mesh:
            int_pts = domain.int_pts('random', npts_rand=2*len(basis), weights=use_weights)
        else:
            int_pts = domain.int_pts('mesh', use_weights, cubature_kind, cubature_deg, mesh_kwargs)

        # normalize basis
        basis = basis.to_normalized(bdry_pts + int_pts)

        return cls(basis, bdry_pts, int_pts, bdry_normals, bc_param, reg_type, rtol, ttol, ltol)
        
    def _get_params(self, reg_type=None, rtol=None, ttol=None, ltol=None):
        """Helper to resolve parameters against instance defaults"""
        return (
            reg_type if reg_type is not None else self.reg_type,
            rtol if rtol is not None else self.rtol,
            ttol if ttol is not None else self.ttol,
            ltol if ltol is not None else self.ltol
        )
    
    @cache
    def tensions(self, lam, reg_type=None, rtol=None):
        reg_type, rtol, _, _ = self._get_params(reg_type, rtol)
        return tensions(self.A_B(lam), self.A_I(lam), reg_type, rtol)
    
    def sigma(self, lam, reg_type=None, rtol=None):
        return self.tensions(lam, reg_type, rtol)[0]
    
    @lru_cache
    def eigenfunction_coef(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        reg_type, rtol, ttol, _ = self._get_params(reg_type, rtol, ttol)
        return nullspace_coef(self.A_B(eig), self.A_I(eig), mult, reg_type, rtol, ttol)
    
    def eigenfunction(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        coef = self.eigenfunction_coef(eig, mult, reg_type, rtol, ttol)
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
    
    def eigenfunction_grad(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        coef = self.eigenfunction_coef(eig, mult, reg_type, rtol, ttol)
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
                                  reg_type=None, rtol=None, ttol=None):
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
        return tuple([arr for arr in (U_B, U_I, U_extra, U_ddiff) if arr is not None])
        
    @lru_cache
    def eigenfunction_eval(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        return self.eigenfunction_eval_extras(eig, mult, reg_type=reg_type, rtol=rtol, ttol=ttol)

    @lru_cache
    def eigenfunction_eval_normals(self, eig, mult=1, reg_type=None, rtol=None, ttol=None):
        return self.eigenfunction_eval_extras(eig, mult, ddiff_pts=self.bdry_pts, ddiff_vecs=self.bdry_normals,
                                              reg_type=reg_type, rtol=rtol, ttol=ttol)

    def solve_interval(self, a, b, n_pts, reg_type=None, rtol=None, ttol=None, 
                       ltol=None, minsolver='parabolic', verbose=0):
        """solves for all eigenvalues in [a,b] using MPS"""
        reg_type, rtol, ttol, ltol = self._get_params(reg_type, rtol, ttol, ltol)
        return solve_interval(lambda lam: self.tensions(lam, reg_type, rtol), a, b, n_pts, 
                              ltol, ttol, minsolver, verbose)
    
    def plot_tensions(self, low, high, nlam, n_angle=1, ax=None, mps_kwargs={}, **plot_kwargs):
        L = np.linspace(low, high, nlam+1)
        tans = np.array([self.tensions(lam, **mps_kwargs)[:n_angle] for lam in L])
        if ax is None:
            fig = plt.figure()
            plt.plot(L, tans, **plot_kwargs)
            return fig
        else:
            ax.plot(L, tans, **plot_kwargs)
            ax.set_xlim(low,high)

    def adaptive_rtol(self, lam, reg_type=None, rtol=None, ltol=None, rtol_max=1e-5):
        """Adjust rtol to reduce tension evaluation noise at the scale of ltol"""
        reg_type, rtol, _, ltol = self._get_params(reg_type, rtol, None, ltol)
        lamgrid = np.linspace(lam*(1-ltol), lam*(1+ltol), 5)
        tensions = np.array([self.tensions(lam, reg_type)[0] for lam in lamgrid])
        locmin_idx = discrete_locmin_idx(tensions)
        while len(locmin_idx) > 1 and rtol < rtol_max:
            rtol = min(rtol_max, 2*rtol) # double rtol
            tensions = np.array([self.tensions(lam, reg_type)[0] for lam in lamgrid])
            locmin_idx = discrete_locmin_idx(tensions)
        return rtol
    
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

    # process brackets for proximity
    i = 0
    while i < len(eig_brackets)-1:
        lam0 = eig_brackets[i]
        lam1 = eig_brackets[i+1]
        tol = ltol*lam0[1]
        if lam1[1]-lam0[1] < tol:
            # merge, using average as eigenvalue
            # use lower bound of first bracket and upper bound of second bracket
            new_brack = np.empty(3, dtype='float')
            new_brack[0] = lam0[0]
            new_brack[1] = (lam0[1]+lam1[1])/2
            new_brack[2] = lam1[2]
            eig_brackets[i] = new_brack
            # delete no-longer needed bracket, post-merger
            del eig_brackets[i+1]
        else:
            i += 1
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

    # multiplicity is largest k such that is_locmin[j] and is_small[j] are both true for all j=0,...,k-1
    mult = np.argmin(is_locmin & is_small)
    return mult
   
def solve_interval(tensions, a, b, n_pts, ltol=ltol_default, ttol=ttol_default, 
                   minsolver='parabolic', bracket_kwargs={}, verbose=0):
    """Finds eigenvalues from MPS tensions."""
    if minsolver not in ['parabolic','golden','brent']:
        raise ValueError("'minsolver' must be one of 'parabolic', 'golden', or 'brent'")

    # build initial search grid
    lamgrid = make_lamgrid(a, b, n_pts)

    # evaluate tensions on the lambda grid
    sigmagrid = np.array([tensions(lam)[:2] for lam in lamgrid]).T
    fevals = len(lamgrid)

    # get brackets containing minima
    brackets, fe = bracket_mins(lambda lam: tensions(lam)[:2], lamgrid, 
                                sigmagrid, ltol, verbose=verbose, **bracket_kwargs)
    fevals += fe

    # minimize on each bracket, filtering for small tension values at minimizer
    eig_brackets = []
    for bracket in brackets:
        minimizer, fe = minimize_on_bracket(lambda lam: tensions(lam)[0], bracket, ltol, minsolver, verbose)
        fevals += fe
        minima = tensions(minimizer)[0]
        if minima < ttol:
            # new bracket with "minimizer in the middle"
            lam = bracket[0]
            eig_brackets.append([lam[0], minimizer, lam[2]])

    # sort brackets and merge sufficiently close eigenvalues
    eig_brackets = sort_merge_brackets(eig_brackets, ltol, verbose)

    # filter for eigenvalues within search interval
    eig_brackets = [bracket for bracket in eig_brackets if (bracket[1] >= a and bracket[1] <= b)]
    eigs = [bracket[1] for bracket in eig_brackets]

    # estimate multiplicity for each eigenvalue
    mults = []
    for bracket in eig_brackets:
        a, eig, b = bracket
        mult = estimate_multiplicity(tensions, eig, a, b, ttol, verbose)
        mults.append(mult)
    return np.array(eigs), np.array(mults), fevals