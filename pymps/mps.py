import numpy as np
import scipy.linalg as la
from pygsvd import gsvd
from .utils import invert_permutation

def qr_preprocess(A,tol=0):
    """TO-DO"""
    if tol > 0:
        Q,R,P = la.qr(A, mode='economic', pivoting=True)
        # drop columns of Q corresponding to small diagonal entries of R
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*tol).sum()
    else:
        Q,R = la.qr(A, mode='economic')
        cutoff, P = Q.shape[1], np.arange(Q.shape[1])
    return Q[:,:cutoff],R[:cutoff,:cutoff],P

def svd_preprocess(R,tol):
    """TO-DO"""
    U,s,Vh = la.svd(R)
    cutoff = (s>s[0]*tol).sum()
    return U[:,:cutoff],s[:cutoff],Vh[:cutoff]

def subspace_sines(A,m,tol=0):
    """TO-DO"""
    # compute QR factorization of A (optionally rank-revealing if tol>0)
    Q = qr_preprocess(A,tol)[0]

    # calculate singular values, which are the sines of subspace angles
    return la.svd(Q[:m],compute_uv=False)[::-1]

def subspace_tans(A,m,tol=0,reg_type='svd'):
    """TO-DO"""
    if tol > 0:
        # regularization based on singular values of R from A = QR
        if reg_type == 'svd':
            Q,R,_ = qr_preprocess(A)
            U = svd_preprocess(R,tol)[0]
            A = Q@U
        # regularization based on diagonal entries of R from A = QRP
        elif reg_type == 'qrp':
            A = qr_preprocess(A,tol)[0]
        else: 
            raise ValueError(f"regularization type {reg_type} not one of 'svd' or 'qrp'")

    # compute generalized singular values, which are the tangents of subspace angles
    C,S,_ = gsvd(A[:m],A[m:],extras='')
    return np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]

def nullspace_basis_svd(A,m,mtol,rtol=0):
    """TO-DO"""
    Q,R,P = qr_preprocess(A,rtol)
    Pinv = invert_permutation(P)

    # compute nullspace basis
    _,s,Vh = la.svd(Q[:m])

    # determine multiplicity
    mult = (s<=mtol).sum()
    if mult == 0:
        raise ValueError(f"No nullspace basis up to mtol={mtol}")

    # compute C
    C = np.zeros((A.shape[1],mult))
    C[:R.shape[0]] = la.solve_triangular(R,Vh[-mult:].T)
    return C[Pinv]

def nullspace_basis_gsvd(A,m,mtol,rtol=0,reg_type='svd'):
    """TO-DO"""
    n = A.shape[1]
    if rtol > 0:
        # regularization based on singular values of R from A = QR
        if reg_type == 'svd':
            Q,R,P = qr_preprocess(A)
            U = svd_preprocess(R,rtol)[0]
            A = Q@U
        # regularization based on diagonal entries of R from A = QRP
        elif reg_type == 'qrp':
            Q,R,P = qr_preprocess(A,rtol)
            A = Q
        else: 
            raise ValueError(f"regularization type {reg_type} not one of 'svd' or 'qrp'")

    C,S,X = gsvd(A[:m],A[m:],full_matrices=True,extras='')
    s = np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))

    # determine multiplicity
    mult = (s<mtol).sum()
    if mult == 0:
        raise ValueError(f"No nullspace basis up to mtol={mtol}")

    Xinv = la.pinv(X.T)
    if rtol == 0:
        return Xinv[:,-mult:]
    else:
        C = np.zeros((n,mult))
        Pinv = invert_permutation(P)
        if reg_type == 'svd':
            C[:n] = la.solve_triangular(R,U@(Xinv[:,-mult:]))
        elif reg_type == 'qrp':
            C[:n] = la.solve_triangular(R,Xinv[:,-mult:])
        return C[Pinv]
    
def nullspace_basis_eval_svd(A,m,mtol,rtol=0):
    """TO-DO"""
    Q,R,_ = qr_preprocess(A,rtol)

    # compute nullspace basis
    _,s,Vh = la.svd(Q[:m])

    # determine multiplicity
    mult = (s<=mtol).sum()
    if mult == 0:
        raise ValueError(f"No nullspace basis up to mtol={mtol}")

    # evaluation
    return (Q@(Vh[-mult:].T))

def nullspace_basis_eval_gsvd(A,m,mtol,rtol=0,reg_type='svd'):
    """TO-DO"""
    n = A.shape[1]
    if rtol > 0:
        # regularization based on singular values of R from A = QR
        if reg_type == 'svd':
            Q,R,P = qr_preprocess(A)
            U = svd_preprocess(R,rtol)[0]
            A = Q@U
        # regularization based on diagonal entries of R from A = QRP
        elif reg_type == 'qrp':
            Q,R,P = qr_preprocess(A,rtol)
            A = Q
        else: 
            raise ValueError(f"regularization type {reg_type} not one of 'svd' or 'qrp'")

    C,S,X,Uhat,Vhat = gsvd(A[:m],A[m:])
    s = np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))

    # determine multiplicity
    mult = (s<mtol).sum()
    if mult == 0:
        raise ValueError(f"No nullspace basis up to mtol={mtol}")

    return np.vstack((Uhat[:,-mult:]*C[-mult:],Vhat[:,-mult:]*S[-mult:]))