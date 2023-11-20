import numpy as np
import scipy.linalg as la
from .bases import FourierBesselBasis
from .quad import triangular_mesh, tri_quad2
from .utils import *
from functools import lru_cache
from numpy.polynomial.legendre import leggauss

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

    @lru_cache
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

    @lru_cache
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

    @lru_cache
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

    @lru_cache
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

    def eigenbasis_grad(self,lambda_,x,y,rtol=None,btol=None):
        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(lambda_,rtol,btol)

        # evaluate partial derivatives
        du_dx, du_dy = self.basis.grad(lambda_,x,y)

        return du_dx@C, du_dy@C

    @lru_cache
    def outward_normal_derivatives(self,lambda_,n=20,rtol=None,btol=None):
        # polygon vertices
        x_v,y_v = self.vertices.T

        # Gaussian quadrature nodes
        qnodes = leggauss(n)[0]

        # transform nodes to boundary intervals
        slope_x, int_x = (np.roll(x_v,-1)-x_v)/2, (np.roll(x_v,-1)+x_v)/2
        slope_y, int_y = (np.roll(y_v,-1)-y_v)/2, (np.roll(y_v,-1)+y_v)/2
        x_b = qnodes[np.newaxis]*slope_x[:,np.newaxis] + int_x[:,np.newaxis]
        y_b = qnodes[np.newaxis]*slope_y[:,np.newaxis] + int_y[:,np.newaxis]

        # get basis gradients & reshape
        du_dx, du_dy = self.eigenbasis_grad(lambda_,x_b,y_b,rtol,btol)
        du_dx = du_dx.reshape((self.n_vert,-1,du_dx.shape[1]))
        du_dy = du_dy.reshape((self.n_vert,-1,du_dy.shape[1]))

        # compute outward normal derivatives
        n = calc_normals(x_v,y_v).T[:,:,np.newaxis]
        return n[:,:1]*du_dx + n[:,1:]*du_dy

    @lru_cache
    def _gram_tensors(self,lambda_,n=20,rtol=None,btol=None):
        # polygon vertices
        x_v,y_v = self.vertices.T

        # Gaussian quadrature nodes & weights
        qnodes, qweights = leggauss(n)

        # outward normal derivatives
        du_dn = self.outward_normal_derivatives(lambda_,n,rtol,btol)

        # get edge lengths & weighting arrays
        d = calc_dists(x_v,y_v)[1][:,np.newaxis]
        f = ((d/2)*(qnodes+1)[np.newaxis])[:,:,np.newaxis]
        g = f[:,::-1]

        # compute integrals with quadrature
        I = (du_dn.transpose((0,2,1))@(du_dn*f*qweights[:,np.newaxis]))/(2*d[:,np.newaxis])
        J = (du_dn.transpose((0,2,1))@(du_dn*g*qweights[:,np.newaxis]))/(2*d[:,np.newaxis])

        # get X and Y gram tensors
        dx = np.roll(x_v,-1)-x_v
        dy = np.roll(y_v,-1)-y_v
        X = -np.roll(dy,1)*np.roll(I,1,axis=0).T - dy*J.T
        Y = np.roll(dx,1)*np.roll(I,1,axis=0).T + dx*J.T
        return X,Y

    def dlambda(self,lambda_,dx=None,dy=None,n=20,rtol=None,btol=None):
        # catch direction derivative input errors
        if (dx is None) != (dy is None):
            raise ValueError('dx and dy must both be set, or left unset')
        elif dx is not None:
            dx,dy = np.asarray(dx),np.asarray(dy)
            if (dx.shape[0] != self.n_vert) or (dy.shape[0] != self.n_vert):
                raise ValueError(f'dx and dy must both be length {self.n_vert}')

        # compute multiplicity
        _,mult = self.sigma(lambda_,rtol,btol,mult_check=True)

        # catch repeated eigs
        if (mult > 1) and (dx is None):
            raise ValueError(f'Repeated eigenvalues only have a directional derivative (mult = {mult})')

        # compute gram tensors
        X,Y = self._gram_tensors(lambda_,n,rtol,btol)

        # catch multiplicity mismatch
        if X.shape[1] != mult:
            raise ValueError(f'Multiplicity mismatch: {mult} != {X.shape[1]}')

        if dx is None: # simple eigenvalue gradient
            return X.flatten(), Y.flatten()
        else: # directional derivative
            M = (X@dx)+(Y@dy)
            if mult==1: # simple eigenvalue
                return M[0,0]
            else: # repeated eigenvalue
                return la.eigh(M,eigvals_only=True)

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
