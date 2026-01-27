import numpy as np
import scipy.linalg as la
import warnings
from pygsvd import gsvd
from .utils import invert_permutation
from .opt import parabolic_gridmin
from .bases import ParticularBasis
from functools import cache, lru_cache

### tolerance defaults
rtol_default = 1e-30
ttol_default = 1e-3
ltol_default = 1e-12

### new code
def make_pencil(A_B, A_I, A_T=None, A_N=None, w_B=None, w_I=None):
    """Makes the pencil for the MPS problem."""
    # process and check shapes
    m1,n1 = A_B.shape
    m2,n2 = A_I.shape
    if n1 != n2:
        raise ValueError("A_B and A_I must have the same number of columns")
    else:
        n = n1
    if A_T is not None:
        if A_T.shape != A_B.shape:
            raise ValueError("A_T must have the same shape as A_B")
        else:
            m = 2*m1
    else:
        m = m1
    if A_N is not None:
        if A_N.shape != A_B.shape:
            raise ValueError("A_T must have the same shape as A_B")

    # no weights
    if w_B is None and w_I is None:
        A = np.vstack([arr for arr in (A_B, A_T, A_N, A_I) if arr is not None])
    # scale by weights
    else:
        if w_B is not None: w_B = w_B.reshape(-1,1)
        else: w_B = 1
        if w_I is not None: w_I = w_I.reshape(-1,1)
        else: w_I = 1
        A = np.vstack([wts*arr for arr, wts in zip((A_B, A_T, A_N, A_I),(w_B, w_B, w_B, w_I)) if arr is not None])
    return A[:m], A[m:]

def regularize_pencil(A1, A2, rtol=rtol_default, reg_method='svd'):
    """Regularizes the pencil for the MPS problem."""
    m = A1.shape[0]
    A = np.vstack([A1, A2])
    if reg_method == 'svd':
        # compute non-pivoted qr
        Q,R = la.qr(A, mode='economic')
        # compute truncated svd
        Z,s,Yt = la.svd(R)
        cutoff = (s >= rtol).sum()
        Z1 = Z[:,:cutoff]
        Y1 = Yt[:cutoff].T
        A = Q@Z1
        return A[:m], A[m:], Y1, s[:cutoff]
    
    elif reg_method == 'qrp':
        # pivoted qr
        Q,R,P = la.qr(A, mode='economic', pivoting=True)
        # truncate
        cutoff = (np.abs(np.diag(R)) >= rtol).sum()
        A = Q[:,:cutoff]
        R11 = R[:cutoff,:cutoff]
        return A[:m], A[m:], R11, P
    
    else: 
        raise ValueError(f"regularization method {reg_method} not one of 'svd' or 'qrp'")

def tensions(A_B, A_I, A_T=None, w_B=None, w_I=None, rtol=rtol_default, reg_method='svd'):
    """Computes the MPS tension values corresponding to the pencil {A_B, A_I}, optionally the pencil
    {[A_B; A_T],A_I}. Also optionally weights the Vandermonde matrices with quadrature weights."""
    # build pencil for GSVD
    A1,A2 = make_pencil(A_B, A_I, A_T, None, w_B, w_I)

    # regularize pencil
    if rtol is not None:
        A1,A2,_,_ = regularize_pencil(A1, A2, rtol, reg_method)

    # compute GSVD
    C,S,_ = gsvd(A1, A2, full_matrices=True, extras='')
    sigmas = np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))
    return sigmas[::-1]

def eigenbasis_coef(A_B, A_I, A_T=None, w_B=None, w_I=None, mult=None, ttol=ttol_default, 
                    rtol=rtol_default, reg_method='svd'):
    """Computes the coefficient vectors of the L^2-orthonormal eigenbasis corresponding to the pencil {A_B, A_I}, 
    optionally the pencil {[A_B; A_T],A_I}. Also optionally weights the Vandermonde matrices with quadrature weights."""
    # build pencil for GSVD
    A1,A2 = make_pencil(A_B, A_I, A_T, None, w_B, w_I)

    # regularize pencil
    if rtol is not None:
        A1,A2,M,v = regularize_pencil(A1, A2, rtol, reg_method)
        if reg_method == 'svd': Y1, s1 = M, v
        elif reg_method == 'qrp': R11, Pinv = M, invert_permutation(v)

    # compute GSVD
    C,S,X = gsvd(A1, A2, extras='')
    sigmas = np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))

    # determine multiplicity if needed
    if mult is None:
        mult = (sigmas<=ttol).sum()
        if mult == 0:
            raise ValueError(f"No eigenbasis up to mtol={ttol}")
        
    # warn if multiplicity is deficient
    if sigmas[-mult] > ttol:
        warnings.warn(f"Eigenvalue may have deficient multiplicity ({sigmas[mult-1]:.3e}>{ttol:.3e})")

    # solve for coefficient vectors
    Xr = la.lstsq(X.T, np.eye(X.shape[0])[:,-mult:])[0]
    if rtol is None: C = Xr
    else:
        if reg_method == 'svd': 
            C = Y1@(Xr/(s1.reshape(-1,1)))
        elif reg_method == 'qrp':
            C = np.zeros((A_B.shape[1],mult))
            C[Pinv] = la.solve_triangular(R11, Xr)
    return C

def eigenbasis_eval(A_B, A_I, A_T=None, A_N=None, w_B=None, w_I=None, mult=None, ttol=ttol_default, 
                    rtol=rtol_default, reg_method='svd'):
    """Computes the evaluation of the L^2-orthonormal eigenbasis corresponding to the pencil {A_B, A_I}, 
    or optionally one of the pencils {[A_B; A_T],A_I}, {A_B, [A_I; A_N]} or {[A_B; A_T],[A_I; A_N]}. 
    Also optionally weights the Vandermonde matrices with quadrature weights."""
    # build pencil for GSVD
    A1,A2 = make_pencil(A_B, A_I, A_T, A_N, w_B, w_I)

    # regularize pencil
    if rtol is not None:
        A1,A2,_,_ = regularize_pencil(A1, A2, rtol, reg_method)

    # compute GSVD
    C,S,_,U,V = gsvd(A1, A2)
    sigmas = np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))

    # determine multiplicity if needed
    if mult is None:
        mult = (sigmas<=ttol).sum()
        if mult == 0:
            raise ValueError(f"No eigenbasis up to mtol={ttol}")
        
    # warn if multiplicity is deficient
    if sigmas[-mult] > ttol:
        warnings.warn(f"Eigenvalue may have deficient multiplicity ({sigmas[mult-1]:.3e}>{ttol:.3e})")

    # compute (weighted) eigenbasis evaluation
    m_B = A_B.shape[0]
    U1, U2 = U[:,-mult:]*C[-mult:], V[:,-mult:]*S[-mult:]
    if A_T is not None:
        U_B, U_T = U1[:m_B], U1[m_B:]
    else:
        U_B, U_T = U1, None
    if A_N is not None:
        U_N, U_I = U2[:m_B], U2[m_B:]
    else:
        U_N, U_I = None, U2

    # re-orthogonalize if needed
    if A_N is not None:
        U_I, Rhat = la.qr(U_I, mode='economic')
        U_B = la.solve_triangular(Rhat, U_B.T, trans=1).T
        U_N = la.solve_triangular(Rhat, U_N.T, trans=1).T
        if U_T is not None: U_T = la.solve_triangular(Rhat, U_T.T, trans=1).T
        
    # unweight if needed
    if w_B is None and w_I is None:
        return tuple([arr for arr in (U_B, U_I, U_T, U_N) if arr is not None])
    else:
        if w_B is not None: w_B = w_B.reshape(-1,1)
        else: w_B = 1
        if w_I is not None: w_I = w_I.reshape(-1,1)
        else: w_I = 1
        return tuple([arr/wts for arr, wts in zip((U_B, U_I, U_T, U_N),(w_B, w_I, w_B, w_B)) if arr is not None])
    
class MPSProblem:
    """Class for MPS problems. Requires a basis, boundary points, and interior points. Can also 
    include tangent and normal derivative data. Optionally includes weights."""
    def __init__(self, basis, bdry_pts, int_pts, bdry_tans=None, bdry_norms=None,
                 rtol=rtol_default, ltol=ltol_default, ttol=ttol_default):
        if not isinstance(basis, ParticularBasis):
            raise TypeError("'basis' must be an instance of ParticularBasis")
        self.basis = basis
        self.bdry_pts = bdry_pts
        self.int_pts = int_pts
        self.bdry_tans = bdry_tans
        self.bdry_norms = bdry_norms
        self.rtol = rtol
        self.ltol = ltol
        self.ttol = ttol

        if hasattr(bdry_pts, 'wts'): self.w_B = bdry_pts._sqrt_wts
        else: self.w_B = None
        if hasattr(int_pts, 'wts'): self.w_I = int_pts._sqrt_wts
        else: self.w_I = None

    @lru_cache
    def eval_basis(self, lam, eval_norms=False):
        A_B = self.basis._eval_pointset(lam, self.bdry_pts)
        A_I = self.basis._eval_pointset(lam, self.int_pts)
        if self.bdry_tans is not None or self.bdry_norms is not None:
            dA = self.basis._grad_pointset(lam, self.bdry_pts)
        if self.bdry_tans is not None:
            dx = self.bdry_tans.x[:,np.newaxis]
            dy = self.bdry_tans.y[:,np.newaxis]
            A_T = dA.real*dx + dA.imag*dy
        else:
            A_T = None
        if eval_norms and self.bdry_norms is not None:
            dx = self.bdry_norms.x[:,np.newaxis]
            dy = self.bdry_norms.y[:,np.newaxis]
            A_N = dA.real*dx + dA.imag*dy
        else:
            A_N = None

        return A_B, A_I, A_T, A_N

    @cache
    def tensions(self, lam, rtol=None, reg_method='svd'):
        if rtol is None:
            rtol = self.rtol
        A_B, A_I, A_T, _ = self.eval_basis(lam)

        return tensions(A_B, A_I, A_T, self.w_B, self.w_I, rtol, reg_method)
    
    @cache
    def eigenbasis_coef(self, lam, mult=None, ttol=None, rtol=None, reg_method='svd'):
        if rtol is None:
            rtol = self.rtol
        if ttol is None:
            ttol = self.ttol
        A_B, A_I, A_T, _ = self.eval_basis(lam)

        return eigenbasis_coef(A_B, A_I, A_T, self.w_B, self.w_I, mult, ttol, rtol, reg_method)
    
    @lru_cache
    def eigenbasis_eval(self, lam, mult=None, ttol=None, rtol=None, reg_method='svd'):
        if rtol is None:
            rtol = self.rtol
        if ttol is None:
            ttol = self.mtol
        A_B, A_I, A_T, A_N = self.eval_basis(lam, eval_norms=True)

        return eigenbasis_eval(A_B, A_I, A_T, A_N, self.w_B, self.w_I, mult, ttol, rtol, reg_method)
    
    def solve_interval(self, a, b, npts, rtol=None, reg_method='svd', ltol=None, ttol=None):
        """Solve the MPS minimization problem on the given interval."""
        if ltol is None:
            ltol = self.ltol
        if ttol is None:
            ttol = self.ttol

        f = lambda lam: self.tensions(lam, rtol, reg_method)
        return solve_interval(f, a, b, npts, ltol, ttol)

### Matrix factorizations
# def qr_preprocess(A_B, A_I, tol=rtol_default):
#     """TO-DO"""
#     A = np.vstack((A_B,A_I))
#     Q,R,P = la.qr(A, mode='economic', pivoting=True)
#     if tol > 0:
#         # drop columns of Q corresponding to small diagonal entries of R
#         r = np.abs(np.diag(R))
#         cutoff = (r>r[0]*tol).sum()
#     else:
#         cutoff = Q.shape[1]
#     return Q[:,:cutoff],R[:cutoff,:cutoff],P

# def svd_preprocess(R, tol=rtol_default):
#     """TO-DO"""
#     U,s,Vh = la.svd(R)
#     if tol == 0:
#         cutoff = R.shape[1]
#     else:
#         cutoff = (s>s[0]*tol).sum()
#     return U[:,:cutoff],s[:cutoff],Vh[:cutoff]

# def subspace_sines(A_B, A_I, tol=rtol_default):
#     """TO-DO"""
#     m = A_B.shape[0]

#     # compute QR factorization of A (optionally rank-revealing if tol>0)
#     Q = qr_preprocess(A_B, A_I, tol)[0]

#     # calculate singular values, which are the sines of subspace angles
#     return la.svd(Q[:m], compute_uv=False)[::-1]

# def subspace_tans(A_B, A_I, tol=0, reg_type='svd'):
#     """TO-DO"""
#     if tol > 0:
#         m = A_B.shape[0]
#         # regularization based on singular values of R from A = QR
#         if reg_type == 'svd':
#             Q,R,_ = qr_preprocess(A_B, A_I, 0)
#             U = svd_preprocess(R, tol)[0]
#             A = Q@U
#             A_B, A_I = A[:m], A[m:]
#         # regularization based on diagonal entries of R from A = QRP
#         elif reg_type == 'qrp':
#             A = qr_preprocess(A_B, A_I, tol)[0]
#             A_B, A_I = A[:m], A[m:]
#         else: 
#             raise ValueError(f"regularization type {reg_type} not one of 'svd' or 'qrp'")

#     # compute generalized singular values, which are the tangents of subspace angles
#     C,S,_ = gsvd(A_B, A_I, extras='')
#     return np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))[::-1]

# def eigenbasis_svd(A_B, A_I, mult=None, mtol=mtol_default, rtol=rtol_default):
#     """TO-DO"""
#     m,n = A_B.shape
#     Q,R,P = qr_preprocess(A_B, A_I, rtol)
#     Pinv = invert_permutation(P)

#     # compute nullspace basis
#     Q_B = Q[:m]
#     _,s,Vh = la.svd(Q_B)
#     s = s[::-1]

#     # determine multiplicity if needed
#     if mult is None:
#         mult = (s<=mtol).sum()
#         if mult == 0:
#             raise ValueError(f"No eigenbasis up to mtol={mtol}")
        
#     # warn if multiplicity is deficient
#     if s[mult-1] > mtol:
#         warnings.warn(f"Eigenvalue may have deficient multiplicity ({s[mult]:.3e}>{mtol:.3e})")

#     # compute C
#     C = np.zeros((n, mult))
#     C[:R.shape[0]] = la.solve_triangular(R, Vh[-mult:].T)
#     return C[Pinv]

# def gsvd_partial(A_B, A_I, mtol=mtol_default, rtol=rtol_default, reg_type='svd'):
#     m,n = A_B.shape
#     if rtol > 0:
#         # regularization based on singular values of R from A = QR
#         if reg_type == 'svd':
#             Q,R,P = qr_preprocess(A_B, A_I, 0)
#             U = svd_preprocess(R, rtol)[0]
#             A = Q@U
#             A_B, A_I = A[:m], A[m:]
#         # regularization based on diagonal entries of R from A = QRP
#         elif reg_type == 'qrp':
#             Q,R,P = qr_preprocess(A_B, A_I, rtol)
#             A = Q
#             A_B, A_I = A[:m], A[m:]
#         else: 
#             raise ValueError(f"regularization type {reg_type} not one of 'svd' or 'qrp'")

#     C,S,X = gsvd(A_B, A_I, full_matrices=True, extras='')
#     return C,S,X

# def eigenbasis_gsvd(A_B, A_I, mult=None, mtol=mtol_default, rtol=rtol_default, reg_type='svd'):
#     """TO-DO"""
#     m,n = A_B.shape
#     if rtol > 0:
#         # regularization based on singular values of R from A = QR
#         if reg_type == 'svd':
#             Q,R,P = qr_preprocess(A_B, A_I, 0)
#             U = svd_preprocess(R, rtol)[0]
#             cutoff = U.shape[1]
#             R = R[:cutoff,:cutoff]
#             A = Q@U
#             A_B, A_I = A[:m], A[m:]
#         # regularization based on diagonal entries of R from A = QRP
#         elif reg_type == 'qrp':
#             Q,R,P = qr_preprocess(A_B, A_I, rtol)
#             A = Q
#             A_B, A_I = A[:m], A[m:]
#         else: 
#             raise ValueError(f"regularization type {reg_type} not one of 'svd' or 'qrp'")

#     C,S,X = gsvd(A_B, A_I, full_matrices=True, extras='')
#     s = np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))[::-1]

#     # determine multiplicity if needed
#     if mult is None:
#         mult = (s<=mtol).sum()
#         if mult == 0:
#             raise ValueError(f"No eigenbasis up to mtol={mtol}")
        
#     # warn if multiplicity is deficient
#     if s[mult-1] > mtol:
#         warnings.warn(f"Eigenvalue may have deficient multiplicity ({s[mult-1]:.3e}>{mtol:.3e})")

#     Xinv = la.pinv(X.T)
#     if rtol == 0:
#         return Xinv[:,-mult:]
#     else:
#         C = np.zeros((n, mult))
#         Pinv = invert_permutation(P)
#         if reg_type == 'svd':
#             C[:R.shape[0]] = la.solve_triangular(R, U@(Xinv[:,-mult:]))
#         elif reg_type == 'qrp':
#             C[:R.shape[0]] = la.solve_triangular(R, Xinv[:,-mult:])
#         return C[Pinv]
    
# def eigenbasis_eval_svd(A_B, A_I, mult=None, mtol=mtol_default, rtol=rtol_default):
#     """TO-DO"""
#     m,n = A_B.shape
#     Q = qr_preprocess(A_B, A_I, rtol)[0]

#     # compute nullspace basis
#     Q_B = Q[:m]
#     _,s,Vh = la.svd(Q_B)
#     s = s[::-1]

#     # determine multiplicity if needed
#     if mult is None:
#         mult = (s<=mtol).sum()
#         if mult == 0:
#             raise ValueError(f"No eigenbasis up to mtol={mtol}")
        
#     # warn if multiplicity is deficient
#     if s[mult-1] > mtol:
#         warnings.warn(f"Eigenvalue may have deficient multiplicity ({s[mult]:.3e}>{mtol:.3e})")

#     # evaluation
#     arr = Q@(Vh[-mult:].T)
#     return arr[:m], arr[m:]

# def eigenbasis_eval_gsvd(A_B, A_I, mult=None, mtol=mtol_default, rtol=rtol_default, reg_type='svd'):
#     """TO-DO"""
#     m,n = A_B.shape
#     if rtol > 0:
#         # regularization based on singular values of R from A = QR
#         if reg_type == 'svd':
#             Q,R,_ = qr_preprocess(A_B, A_I, 0)
#             U = svd_preprocess(R, rtol)[0]
#             A = Q@U
#             A_B, A_I = A[:m], A[m:]
#         # regularization based on diagonal entries of R from A = QRP
#         elif reg_type == 'qrp':
#             Q,R,_ = qr_preprocess(A_B, A_I, rtol)
#             A = Q
#             A_B, A_I = A[:m], A[m:]
#         else: 
#             raise ValueError(f"regularization type {reg_type} not one of 'svd' or 'qrp'")

#     C,S,_,Uhat,Vhat = gsvd(A_B, A_I)
#     s = np.divide(C, S, out=np.full(C.shape, np.inf), where=(S!=0))[::-1]

#     # determine multiplicity if needed
#     if mult is None:
#         mult = (s<=mtol).sum()
#         if mult == 0:
#             raise ValueError(f"No eigenbasis up to mtol={mtol}")
        
#     # warn if multiplicity is deficient
#     if s[mult-1] > mtol:
#         warnings.warn(f"Eigenvalue may have deficient multiplicity ({s[mult]:.3e}>{mtol:.3e})")

#     # evaluation
#     return Uhat[:,-mult:]*C[-mult:], Vhat[:,-mult:]*S[-mult:]

### Minimization Eigsearch Code
def proximity_check(lam, eigs, ltol):
    """Checks if a candidate eigenvalue is within tolerance of previously found eigenvalues"""
    eigs = np.unique(eigs)
    if len(eigs) == 0:
        return True
    elif np.abs(lam-eigs).min() < ltol:
        return False
    else:
        return True
    
def solve_interval(f, a, b, n_pts, xtol=ltol_default, ftol=ttol_default, verbose=0):
    """Finds eigenvalues in the interval [a,b] as minima of 'f' (generally tensions). 
    Searches over a grid with n_pts."""
    if b <= a:
        raise ValueError('b must be greater than a')
    
    # initial search grid 
    lamgrid_int = np.linspace(a,b,n_pts)
    lamgrid = np.empty(len(lamgrid_int)+2)
    lamgrid[1:-1] = lamgrid_int
    
    # add ghost points to ensure robust search
    lamgrid[0] = 2*lamgrid[1]-lamgrid[2]
    if lamgrid[0] <= 0: lamgrid[0] = a/2
    lamgrid[-1] = 2*lamgrid[-2]-lamgrid[-3]

    # evaluate f on the lambda grid
    Y = np.array([f(lam)[:2] for lam in lamgrid]).T
    fevals = len(lamgrid)

    # run recursive gridsearch, subdividing where needed
    minima, fe = parabolic_gridmin(lambda lam: f(lam)[:2], lamgrid, Y, xtol=xtol, verbose=verbose)
    fevals += fe
    if verbose > 0: print(f"fevals={fevals}")
    minima = np.sort(minima)

    # filter out spurious minima and minima too close to known eigenvalues
    eigs = []
    for lam in minima:
        mybool = proximity_check(lam, eigs, xtol)
        y = f(lam)[0]
        if not mybool and verbose > 0:
            print(f"lam={lam:.3f} too close to previously found eigenvalue")
        elif y > ftol and verbose > 0:
            print(f"f(lam={lam}) above threshold, {y:.3e}>{ftol:.3e}")
        elif mybool:
            if verbose > 0:
                print(f"lam={lam:.3f} accepted as eigenvalue")
            eigs.append(lam)
    eigs = np.array(eigs)
    return eigs, fevals

def estimate_multiplicity(f, a, b, tol, npts=5, verbose=False):
    """Estimates the multiplicity of a zero/minimum of f contained within a small bracket (a,b), 
    where f is a vector-valued function which satisfies 0 <= f_1 <= ... <= f_i <= f_i+1 (i.e. the 
    entries are non-decreasing). Multiplicity in this case being the number of components of f 
    which have a local minimum near zero within (a,b). Near zero meaning f_i(x) < tol."""

    # evaluate f on grid
    x = np.linspace(a, b, npts)
    y = [f(val) for val in x]

    # "prune" evaluations to have the same length (e.g. if some singular values do not converge in all cases)
    n = np.min([len(row) for row in y])
    y = np.array([row[:n] for row in y])
    if verbose:
        print(f"checking ({a},{b}) for multiplicity")
        print(y)

    # find absolute minimizer on grid of each component
    idx = np.argmin(y,axis=0)
    # filter for interior mins
    is_loc_min = (idx > 0)&(idx < npts-1)
    if verbose: print(is_loc_min)
    # filter for below tolerance
    is_small = (y[idx,np.arange(n)] < tol)
    if verbose: print(is_small)
    # filter for both
    is_loc_zero = is_loc_min & is_small
    if verbose: print(is_loc_zero)

    return np.argmin(is_loc_zero)

def local_tolerance(f, lam, eig1=None, eig2=None, ftol=ttol_default, h=ltol_default):
    """Gets the local tolerance for subspace angles based on nearby eigenvalues."""
    if eig1 is None:
        tol1 = np.inf
    else:
        y1 = f(eig1)[0]
        y1ph = f(eig1+h)[0]
        slope1 = np.abs((y1ph-y1)/h)
        tol1 = y1 + slope1*np.abs(lam-eig1)
    if eig2 is None:
        tol2 = np.inf
    else:
        y2 = f(eig2)[0]
        y2mh = f(eig2-h)[0]
        slope2 = np.abs((y2-y2mh)/h)
        tol2 = y2 + slope2*np.abs(lam-eig2)
    return min(ftol,tol1,tol2)

def estimate_multiplicity(f, eigs, ftol=ttol_default, h=ltol_default, verbose=0):
    # Determining tolerance requires subspace angle bounds from nearby eigenvalues
    # Get all eigenvalues sorted
    eigs = np.sort(eigs)

    # loop over eigenvalues, set local tolerance for multiplicity check
    # then add to spectrum with multiplicity and local subspace angle tolerance
    mults = []
    fevals = 0
    for i,eig in enumerate(eigs):
        if verbose > 0: print(f"estimating multiplicity of lam={eig:.3e}")
        # get local tolerance parameter
        if i == 0:
            tol = local_tolerance(f, eig, eig2=eigs[i+1], ftol=ftol, h=h)
            fevals += 2
        elif i == len(eigs)-1:
            tol = local_tolerance(f, eig, eig1=eigs[i-1], ftol=ftol, h=h)
            fevals += 2
        else:
            tol = local_tolerance(f, eig, eigs[i-1],eigs[i+1], ftol=ftol, h=h)
            fevals += 4
            
        # multiplicity = number of subspace angles below tolerance
        y = f(eig)
        mult = (y<=tol).sum()
        mults.append(mult)
        if verbose > 1: print(f"y={np.array_str(y,precision=2)}, tol={tol:.3e}, mult={mult}")
    
    return np.array(mults, dtype='int'), fevals