import numpy as np
from scipy.special import jv
from scipy.spatial.distance import cdist
import scipy.linalg as la
from utils import *

def radii(x,y,x_v,y_v):
    """Computes the radial distance from each point in x,y to each polygon vertex in x_v, y_v.
    For use in evaluating Fourier-Bessel functions in the Method of Particular Solutions.
    """
    return cdist(np.array([x,y]).T,np.array([x_v,y_v]).T)

def thetas(x,y,x_v,y_v):
    """Computes the angles between given points and the polygon edges which are
    counter-clockwise from the vertices (x_i,v_i). For use in evaluating Fourier-Bessel
    functions in the Method of Particular Solutions."""
    theta = np.zeros((len(x_v),len(x)))
    dx_p = np.roll(x_v,-1)-x_v
    dy_p = np.roll(y_v,-1)-y_v
    for i in range(len(x_v)):
        dx, dy = x-x_v[i], y-y_v[i]
        theta[i] = np.arccos(1-cdist([[dx_p[i],dy_p[i]]],np.array([dx,dy]).T,'cosine'))
        reentrant = (dx_p[i]*dy - dy_p[i]*dx)<0
        theta[i][reentrant] = 2*np.pi - theta[i][reentrant]
    return theta.T

def build_A_lam(x_v,y_v,x_b,y_b,x_i,y_i,k):
    """Constructs A(\lambda) for the Method of Particular Solutionsm given vertices
    (x_v,y_v), boundary points (x_b,y_b), interior points (x_i,y_i), and expansion
    orders k = (k1,...,kn)"""
    if isinstance(x_v,list) or isinstance(y_v,list):
        x_v = np.array(x_v)
        y_v = np.array(y_v)

    if isinstance(k,int):
        k = k*np.ones(len(x_v),dtype='int')
    else:
        k = np.array(k,dtype='int')

    n = len(x_v)
    m_b = len(x_b)
    m_i = len(x_i)
    m = m_b + m_i

    # compute alphas
    alpha = np.pi/calc_angles(x_v,y_v)

    # compute radii and angles relative to corners with expansions
    x = np.concatenate((x_b,x_i))
    y = np.concatenate((y_b,y_i))
    mask = k>0
    x_ve, y_ve = x_v[mask], y_v[mask]
    r = radii(x,y,x_ve,y_ve)
    theta = thetas(x,y,x_v,y_v)[:,mask]

    # set up evaluations of Fourier-Bessel
    # first calculate the fourier part (independent of λ!)
    alphak = [alphai*np.arange(1,ki+1) for alphai,ki in zip(alpha,k)]
    fourier = np.empty((m,k.sum()))
    cumk = np.concatenate(([0],np.cumsum(k)))
    for i in range(n):
        if k[i] > 0:
            fourier[:,cumk[i]:cumk[i+1]] = np.sin(np.outer(theta[:,i],alphak[i]))

    # set up evaluations of bessel part
    alphak_vec = np.concatenate(alphak)[np.newaxis]
    r_rep = np.repeat(r,k[mask],axis=1)

    # define A(λ)
    def A_lam(lambda_):
        return jv(alphak_vec,np.sqrt(lambda_)*r_rep)*fourier
    return A_lam

def sigma(lambda_,A_lam,m_b,tol=1e-16):
    """Computes the smallest singular value of the submatrix of the column-pivoted
    and thin QR factorization of A(\lambda) corresponding to the boundary points"""
    A = A_lam(lambda_)
    Q,R,P = la.qr(A, mode='economic', pivoting=True)

    # drop columns of Q corresponding to small diagonal entries of R
    r = np.abs(np.diag(R))
    cutoff = (r>r[0]*tol).sum()

    # calculate and return smallest singular value
    try:
        s = la.svd(Q[:m_b,:cutoff],compute_uv=False)[-1]
    except:
        return np.inf
    return s
