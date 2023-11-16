import numpy as np
import scipy.linalg as la
from .bases import FourierBesselBasis
from .quad import triangular_mesh, tri_quad2
from .utils import *

class PolygonEP:
    """Class for polygonal Dirichlet Laplacian eigenproblems
    Builds a Fouier-Bessel basis which is adapdated to the polygon with given
    vertices and number of expansion orders. Evaluates this basis on a set of

    """
    def __init__(self,vertices,orders,boundary_pts=20,interior_pts=50,
                 boundary_method='even',interior_method='random',
                 quad_nodes=None,quad_weights=None):
        self.basis = FourierBesselBasis(vertices,orders)
        self.vertices = self.basis.vertices
        self.n_vert = self.basis.n_vert
        self.orders = self.basis.orders

        x_v,y_v = self.vertices.T
        # catch integer argument to construct boundary points
        if type(boundary_pts) is int:
            if np.sum(self.orders>0) == 1:
                idx = np.argwhere(self.orders>0)[0]
                skip = [idx-1,idx]
            else:
                skip = None
            boundary_pts = boundary_points(x_v,y_v,boundary_pts,boundary_method,skip)

        # process boundary_pts array to be of shape (m_b,2)
        self.boundary_pts = np.array(boundary_pts)
        if self.boundary_pts.shape[0] == 2:
            self.boundary_pts = self.boundary_pts.T
        if self.boundary_pts.shape[1] != 2 or self.boundary_pts.ndim != 2:
            print(self.boundary_pts.shape)
            raise ValueError('boundary_pts must be a 2-dimensional array '\
                             'of x & y coordinates')

        # catch integer argument to construct interior points
        if type(interior_pts) is int:
            if interior_method == 'random':
                interior_pts = interior_points(x_v,y_v,interior_pts)

        # process interior_pts array to be of shape (m_i,2)
        self.interior_pts = np.array(interior_pts)
        if self.interior_pts.shape[0] == 2:
            self.interior_pts = self.interior_pts.T
        if self.interior_pts.shape[1] != 2 or self.interior_pts.ndim != 2:
            raise ValueError('interior_pts must be a 2-dimensional array '\
                             'of x & y coordinates')

        # set default points for the basis to be boundary_pts and interior_pts
        x_b,y_b = self.boundary_pts.T
        x_i,y_i = self.interior_pts.T
        x = np.concatenate((x_b,x_i))
        y = np.concatenate((y_b,y_i))
        self.basis.set_default_points(x,y)

        # label interior points and which edges the boundary points lie on
        self.points = np.vstack((x,y)).T
        self.edge_idx = edge_indices(self.points,self.vertices)

        # default tolerance parameters
        self.rtol = 1e-40
        self.btol = 1e2

        # nodes and weights for quadrature. Process the nodes to be boundary-first
        if quad_nodes is not None:
            node_edge_idx = edge_indices(quad_nodes,self.vertices)
            sort_idx = np.argsort(node_edge_idx)
            self.nodes = quad_nodes[sort_idx]
            self.node_edge_idx = node_edge_idx[sort_idx]
            self.weights = np.sqrt(quad_weights[sort_idx])[:,np.newaxis]
            self.n_basis = FourierBesselBasis(vertices,orders)
            self.n_basis.set_default_points(*self.nodes.T)

    def sigma(self,lambda_,rtol=None,btol=None,mult_check=False):
        """Compute the smallest singular value of Q_B(\lambda)"""
        # get default tolerances
        if rtol is None: rtol = self.rtol
        if btol is None: btol = self.btol

        A = self.basis(lambda_)
        m_b = self.boundary_pts.shape[0]
        Q,R,p = la.qr(A, mode='economic', pivoting=True)

        # drop columns of Q corresponding to small diagonal entries of R
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*rtol).sum()

        # calculate singular values
        try:
            s = la.svd(Q[:m_b,:cutoff],compute_uv=False)
        except:
            return ValueError('non-finite SVD')

        if mult_check:
            mult = (s<btol*s[-1]).sum()
            return s[-1], mult
        else:
            return s[-1]

    def eigenbasis(self,lambda_,rtol=None,btol=None):
        """Returns a callable function which evaluates the approximate eigenbasis
        corresponding to lambda_"""
        # get default tolerances
        if rtol is None: rtol = self.rtol
        if btol is None: btol = self.btol

        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(lambda_,rtol,btol)

        # return callable function of x,y from the basis
        def func(x,y):
            shape = np.asarray(x).shape
            shape = (*shape,C.shape[1])
            return (self.basis(lambda_,x,y)@C).reshape(shape)
        return func

    def eigenbasis_coef(self,lambda_,rtol=None,btol=None):
        """Compute the coefficients of an eigenbasis, assuming lambda_ is an eigenvalue"""
        # get default tolerances
        if rtol is None: rtol = self.rtol
        if btol is None: btol = self.btol

        A = self.weights*self.n_basis(lambda_)
        m_b = (self.node_edge_idx<self.n_vert).sum()
        Q,R,P = la.qr(A, mode='economic', pivoting=True)
        Pinv = invert_permutation(P)

        # drop columns of Q corresponding to small diagonal entries of R
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*rtol).sum()

        # compute nullspace basis
        try:
            _,s,Vh = la.svd(Q[:m_b,:cutoff])
        except:
            raise ValueError('non-finite SVD')

        # determine multiplicity
        mult = (s<btol*s[-1]).sum()

        # compute C
        C = np.zeros((A.shape[1],mult))
        C[:cutoff] = la.solve_triangular(R[:cutoff,:cutoff],Vh[-mult:].T)
        return C[Pinv]

    def eigenbasis_node_eval(self,lambda_,rtol=None,btol=None):
        """Evaluate the eigenbasis on the quadrature nodes, assuming lambda_
        is an eigenvalue"""
        # get default tolerances
        if rtol is None: rtol = self.rtol
        if btol is None: btol = self.btol

        A = self.weights*self.n_basis(lambda_)
        m_b = (self.node_edge_idx<self.n_vert).sum()
        Q,R,p = la.qr(A, mode='economic', pivoting=True)

        # drop columns of Q corresponding to small diagonal entries of R
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*rtol).sum()

        # compute nullspace basis
        try:
            _,s,Vh = la.svd(Q[:m_b,:cutoff])
        except:
            raise ValueError('non-finite SVD')

        # determine multiplicity
        mult = (s<btol*s[-1]).sum()

        return (Q[:,:cutoff]@(Vh[-mult:].T))/self.weights

rho = (3-5**0.5)/2
def golden_search(f,a,b,tol=1e-12,maxiter=100):
    h = b-a
    u, v = a+rho*h, b-rho*h
    fu, fv = f(u), f(v)
    i = 0
    while (b-a>=tol)&(i<=maxiter):
        i += 1
        if fu < fv:
            b = v
            h = b-a
            v = u
            u = a+rho*h
            fv = fu
            fu = f(u)
        else:
            a = u
            h = b-a
            u = v
            v = b-rho*h
            fu = fv
            fv = f(v)
    if f(a)<f(b): return a
    else: return b
