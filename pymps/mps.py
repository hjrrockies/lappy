import numpy as np
import scipy.linalg as la
from .bases import FourierBesselBasis
from .quad import boundary_nodes_polygon, cached_leggauss
from .utils import *
from .opt import minsearch, minsearch2
from functools import cache, lru_cache
from pygsvd import gsvd
from scipy.special import jv
from joblib import Parallel, delayed

def solve_obj(lambda_,sin,r_rep,alphak_vec,m_b):
    """Function for solving for the spectrum"""
    A = jv(alphak_vec,np.sqrt(lambda_)*r_rep)*sin
    C,S,_ = gsvd(A[:m_b],A[m_b:],extras='')
    sigmas = np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]
    return sigmas[:2]

class PolygonEVP:
    """Class for polygonal Dirichlet Laplacian eigenproblems
    """
    def __init__(self,vertices,orders,boundary_pts=20,interior_pts=50,
                boundary_method='even',interior_method='random',
                bdry_nodes=None,bdry_weights=None,int_nodes=None,int_weights=None,
                branch_cuts='middle_out'):
        vertices = np.asarray(vertices)
        if vertices.ndim == 2:
            if vertices.shape[0] == 2:
                vertices = complex_form(vertices[0],vertices[1])
            elif vertices.shape[1] == 2:
                vertices = complex_form(vertices[:,0],vertices[:,1])
        self.basis = FourierBesselBasis(vertices,orders,branch_cuts)
        self.vertices = self.basis.vertices
        self.n_vert = self.basis.n_vert
        self.orders = self.basis.orders

        # catch integer argument to construct boundary points
        if type(boundary_pts) is int:
            if np.sum(self.orders>0) == 1:
                idx = np.argwhere(self.orders>0)[0]
                skip = np.concatenate((idx-1,idx))
            else:
                skip = None
            boundary_pts = boundary_nodes_polygon(vertices,orders=boundary_pts,rule=boundary_method,skip=skip)[0]

        # process boundary_pts array to be in complex form
        self.boundary_pts = np.asarray(boundary_pts)
        if self.boundary_pts.ndim == 2:
            if self.boundary_pts.shape[0] == 2:
                self.boundary_pts = complex_form(self.boundary_pts[0],self.boundary_pts[1])
            elif self.boundary_pts.shape[1] == 2:
                self.boundary_pts = complex_form(self.boundary_pts[:,0],self.boundary_pts[:,1])

        # catch integer argument to construct interior points
        if type(interior_pts) is int:
            if interior_method == 'random':
                interior_pts = interior_points(interior_pts,self.vertices)
            else:
                raise Exception(f"No interior point selection named {interior_method}")

        # process interior_pts array to be in complex form
        self.interior_pts = np.asarray(interior_pts)
        if self.interior_pts.ndim == 2:
            if self.interior_pts.shape[0] == 2:
                self.interior_pts = complex_form(self.interior_pts[0],self.interior_pts[1])
            elif self.boundary_pts.shape[1] == 2:
                self.interior_pts = complex_form(self.interior_pts[:,0],self.interior_pts[:,1])

        # set default points for the basis to be boundary_pts and interior_pts
        self.basis.set_default_points(np.concatenate((self.boundary_pts,self.interior_pts)))

        # default tolerance parameters
        self.rtol = 0
        self.mtol = 1e-4

        # nodes and weights for quadrature
        if bdry_nodes is not None:
            self.bdry_nodes = bdry_nodes
            self.bdry_weights = bdry_weights
            self.int_nodes = int_nodes
            self.int_weights = int_weights
            self.nodes = np.concatenate((bdry_nodes,int_nodes))
            quad_weights = np.concatenate((bdry_weights,int_weights))
            self.weights = np.sqrt(quad_weights)[:,np.newaxis]
            self.n_basis = FourierBesselBasis(vertices,orders,branch_cuts)
            self.n_basis.set_default_points(self.nodes)

        # area and perimiter
        self.area = polygon_area(self.vertices)
        self.perimiter = polygon_perimiter(self.vertices)

        # lower bound for first eigenvalue
        self.eig1_lb = 5.76*np.pi/self.area

        # asymptotic adjustment parameter
        int_angles = interior_angles(self.vertices)
        self.asym_K = (np.pi/int_angles - int_angles/np.pi).sum()/24

        # eigenvalues
        self.eigenvalues = np.array([],dtype='float')

    @cache
    def subspace_sines(self,lambda_,rtol=None,pivot=True):
        """Compute the sines of subspace angles, which are the singular values of Q_B(\\lambda)"""
        if rtol is None: rtol = self.rtol
        A = self.basis(lambda_)
        m_b = self.boundary_pts.shape[0]
        if pivot:
            Q,R,_ = la.qr(A, mode='economic', pivoting=True)
            # drop columns of Q corresponding to small diagonal entries of R
            r = np.abs(np.diag(R))
            cutoff = (r>r[0]*rtol).sum()
        else:
            Q,R = la.qr(A, mode='economic')
            cutoff = Q.shape[1]

        # calculate singular values
        try:
            return la.svd(Q[:m_b,:cutoff],compute_uv=False)[::-1]
        except:
            raise ValueError(f'non-finite SVD for lam={lambda_}')

    @cache
    def weighted_subspace_sines(self,lambda_,rtol=None,pivot=True):
        """Compute the sines of subspace angles, which are the singular values of Q_B(\\lambda)
        for the quadrature-weighted problem"""
        if rtol is None: rtol = self.rtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        if pivot:
            Q,R,_ = la.qr(A, mode='economic', pivoting=True)
            # drop columns of Q corresponding to small diagonal entries of R
            r = np.abs(np.diag(R))
            cutoff = (r>r[0]*rtol).sum()
        else:
            Q,R = la.qr(A, mode='economic')
            cutoff = Q.shape[1]

        # calculate singular values
        try:
            return la.svd(Q[:m_b,:cutoff],compute_uv=False)[::-1]
        except:
            raise ValueError(f'non-finite SVD for lam={lambda_}')

    @cache
    def gsvd_subspace_tans(self,lambda_):
        """Use GSVD to compute the tangents of the subspace angles"""
        A = self.basis(lambda_)
        m_b = self.boundary_pts.shape[0]
        C,S,_ = gsvd(A[:m_b],A[m_b:],extras='')
        return np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]
    
    def _get_obj_args(self):
        """Builds the objective function and arguments for the eigensolver"""
        sin,r_rep = self.basis.sin, self.basis.r_rep
        alphak_vec = self.basis.alphak_vec
        m_b = int(self.boundary_pts.shape[0])
        return sin,r_rep,alphak_vec,m_b

    @cache
    def rgsvd_subspace_tans(self,lambda_,rtol=None):
        """Use regularized GSVD to compute the tangents of the subspace angles"""
        if rtol is None: rtol = self.rtol
        A = self.basis(lambda_)
        m_b = self.boundary_pts.shape[0]

        Q,R = la.qr(A, mode='economic')
        U,s,_ = la.svd(R)
        cutoff = (s>rtol).sum()
        if cutoff == 0:
            cutoff = 1
        U1 = U[:,:cutoff]

        C,S,_ = gsvd(Q[:m_b]@U1,Q[m_b:]@U1,extras='')
        return np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]

    @cache
    def r2gsvd_subspace_tans(self,lambda_,rtol=None):
        """Use regularized GSVD to compute the tangents of the subspace angles"""
        if rtol is None: rtol = self.rtol
        A = self.basis(lambda_)
        m_b = self.boundary_pts.shape[0]

        Q,R,_ = la.qr(A, mode='economic',pivoting=True)
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*rtol).sum()

        C,S,_ = gsvd(Q[:m_b,:cutoff],Q[m_b:,:cutoff],extras='')
        return np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]

    @cache
    def gevd_subspace_tans(self,lambda_):
        """Use GEVD to compute the tangents of the subspace angles"""
        A = self.basis(lambda_)
        m_b = self.boundary_pts.shape[0]
        e = la.eigvals(A[:m_b].T@A[:m_b],A[m_b:].T@A[m_b:])
        return np.sort(np.sqrt(e.real))

    @cache
    def gsvd_weighted_subspace_tans(self,lambda_):
        """Computes the subspace angle tangents for the quadrature-weighted problem
        using the GSVD"""
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        C,S,_ = gsvd(A[:m_b],A[m_b:],extras='')
        return np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]

    @cache
    def rgsvd_weighted_subspace_tans(self,lambda_,rtol=None):
        """Computes the subspace angle tangents for the quadrature-weighted problem
        using the regularized GSVD"""
        if rtol is None: rtol = self.rtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        Q,R = la.qr(A, mode='economic')
        U,s,V = la.svd(R)
        cutoff = (s>rtol).sum()
        if cutoff == 0:
            cutoff = 1
        U1 = U[:,:cutoff]

        C,S,_ = gsvd(Q[:m_b]@U1,Q[m_b:]@U1,extras='')
        return np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]

    @cache
    def r2gsvd_weighted_subspace_tans(self,lambda_,rtol=None):
        """Computes the subspace angle tangents for the quadrature-weighted problem
        using the regularized GSVD"""
        if rtol is None: rtol = self.rtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        Q,R,_ = la.qr(A, mode='economic',pivoting=True)
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*rtol).sum()

        C,S,_ = gsvd(Q[:m_b,:cutoff],Q[m_b:,:cutoff],extras='')
        return np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))[::-1]

    @cache
    def sigma(self,lambda_,rtol=None,pivot=True):
        """Compute the smallest singular value of Q_B(\\lambda)"""
        if rtol is None: rtol = self.rtol
        s = self.subspace_sines(lambda_,rtol,pivot)
        return s[0]

    @cache
    def gsigma(self,lambda_):
        """Compute the smallest generalized singular value of the pencil {A_B,A_I}"""
        s = self.gsvd_subspace_tans(lambda_)
        return s[0]

    @cache
    def rgsigma(self,lambda_,rtol=None):
        """Compute the smallest generalized singular value of the pencil {A_B,A_I}"""
        if rtol is None: rtol = self.rtol
        s = self.rgsvd_subspace_tans(lambda_,rtol=rtol)
        return s[0]

    @cache
    def r2gsigma(self,lambda_,rtol=None):
        """Compute the smallest generalized singular value of the pencil {A_B,A_I}"""
        if rtol is None: rtol = self.rtol
        s = self.r2gsvd_subspace_tans(lambda_,rtol=rtol)
        return s[0]

    @cache
    def gesigma(self,lambda_):
        """Compute the smallest generalized eigenvalue value of the pencil {(A_B^T)A_B,(A_I^T)A_I}"""
        s = self.gevd_subspace_tans(lambda_)
        return s[0]
    
    def solve_eigs_interval(self,a,b,n_levels=None,ppl=10,xtol=1e-8,maxdepth=10,rtol=None,mtol=None,n_jobs=1,verbose=False):
        """Finds all the eigenvalues in an interval [a,b]"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol

        # extract objective function arguments
        fargs = self._get_obj_args()

        # define number of grid points
        if n_levels is None:
            n_levels = int(np.ceil(self.weyl_N(b)-self.weyl_N(a))) + 1
        n = n_levels*ppl

        # single process mode
        if n_jobs == 1:
            minima, fevals = minsearch2(solve_obj,a,b,n,xtol,fargs,maxdepth,verbose)
        # multi-process mode
        else:
            minima, fevals = [],0
            # split search domain into n_jobs pieces
            length = (b-a)/n_jobs
            n = int(np.ceil(n/n_jobs))
            A = length*np.arange(n_jobs) + a
            B = length*np.arange(1,n_jobs+1) + a
            res = Parallel(n_jobs=n_jobs)(delayed(minsearch2)(solve_obj,a_,b_,n,xtol,fargs,maxdepth,verbose) for a_,b_ in zip(A,B))
            for mins,fe in res:
                minima += mins
                fevals += fe
        
        # process to check for multiplicity
        minima = np.sort(minima)
        eigs = []
        for min in minima:
            tans = self.gsvd_subspace_tans(min)
            fevals += 1
            mult = int((tans<mtol).sum())
            eigs += mult*[min]
        eigs = np.sort(eigs)

        # # weyl check
        # weyl_est = self.weyl_N(eigs)
        # increments = weyl_est[1:]-weyl_est[0]
        # diff = increments - np.arange(len(eigs)-1)
        # if np.any(diff>1):
        #     print(f"warning: fewer eigenvalues detected in interval ({a:.3e},{b:.3e}) than expected")
        #     first_exception = np.nonzero(diff>1)[0]
        #     print(f"deficiency detected at lam={eigs[first_exception+1]}")
        # if np.any(diff<-1):
        #     print(f"warning: excess eigenvalues detected in interval ({a:.3e},{b:.3e}) than expected")
        #     first_exception = np.nonzero(diff<-1)[0]
        #     print(f"excess detected at lam={eigs[first_exception+1]}")
        return eigs,fevals
        
    
    def solve_eigs_ordered(self,N,ppl=10,xtol=1e-8,rtol=None,mtol=None,maxdepth=10,n_jobs=1,verbose=False):
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # check for solved eigenvalues
        N_current = len(self.eigenvalues)
        
        # already have enough eigs
        if N_current >= N:
            return self.eigenvalues[:N]
        # need more eigs
        else:
            N_needed = N - N_current
            # set up search domain
            if len(self.eigenvalues) == 0:
                a = self.eig1_lb
            else:
                a = self.eigenvalues.max()+xtol
            b = self.weyl_k(N+1)
            n_levels = N_needed+1
            eigs,fevals = self.solve_eigs_interval(a,b,n_levels,ppl,xtol,maxdepth,rtol,mtol,n_jobs,verbose)
            self.eigenvalues = np.sort(np.concatenate((self.eigenvalues,eigs)))
            return self.eigenvalues[:N], fevals

        # # check if enough eigenvalues have been found
        # if len(self.eigenvalues) >= N:
        #     return self.eigenvalues[:N]
        # # if still deficient, check weyl asymptotics
        # else:
        #     argdiff, diff = self.weyl_check(self.eigenvalues)


    @lru_cache
    def eigenbasis(self,lambda_,rtol=None,mtol=None):
        """Returns a callable function which evaluates the approximate eigenbasis
        corresponding to lambda_"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(lambda_,rtol,mtol)

        # return callable function from the basis
        def func(points,y=None):
            if y is not None:
                points = complex_form(points,y)
            shape = np.asarray(points).shape
            shape = (*shape,C.shape[1])
            return (self.basis(lambda_,points)@C).reshape(shape)
        return func

    @lru_cache
    def eigenbasis_unweighted_coef(self,lambda_,rtol=None,mtol=None):
        """Returns the coefficients of an eigenbasis derived without quadrature weighting"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        A = self.basis(lambda_)
        m_b = len(self.boundary_pts)
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
        mult = (s<mtol).sum()
        if mult == 0:
            raise ValueError(f'lambda = {lambda_:.3e} has multiplicity zero (sigma = {s[-1]:.3e} > {mtol:.3e} = mtol)')

        # compute C
        C = np.zeros((A.shape[1],mult))
        C[:cutoff] = la.solve_triangular(R[:cutoff,:cutoff],Vh[-mult:].T)
        return C[Pinv]

    @lru_cache
    def eigenbasis_coef(self,lambda_,rtol=None,mtol=None):
        """Compute the coefficients of an eigenbasis, assuming lambda_ is an eigenvalue"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
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
        mult = (s<mtol).sum()
        if mult == 0:
            raise ValueError(f'lambda = {lambda_:.3e} has multiplicity zero (sigma = {s[-1]:.3e} > {mtol} = mtol)')

        # compute C
        C = np.zeros((A.shape[1],mult))
        C[:cutoff] = la.solve_triangular(R[:cutoff,:cutoff],Vh[-mult:].T)
        return C[Pinv]

    @lru_cache
    def gsvd_eigenbasis_coef(self,lambda_,mtol=None):
        """Compute the coefficients of an eigenbasis, assuming lambda_ is an eigenvalue
        using the GSVD formulation"""
        if mtol is None: mtol = self.mtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        C,S,X = gsvd(A[:m_b],A[m_b:],extras='')
        s = np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))

        # determine multiplicity
        mult = (s<mtol).sum()
        if mult == 0:
            raise ValueError(f'lambda = {lambda_:.3e} has multiplicity zero (sigma = {s[-1]:.3e} > {mtol} = mtol)')

        # least squares solution for the right singular vectors
        return la.lstsq(X.T,np.eye(X.T.shape[0])[:,-mult:])[0]

    @lru_cache
    def rgsvd_eigenbasis_coef(self,lambda_,rtol=None,mtol=None):
        """Compute the coefficients of an eigenbasis, assuming lambda_ is an eigenvalue
        using the GSVD formulation with regularization"""
        raise NotImplementedError('rgsvd_eigenbasis_coef needs debugging')
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        Q,R,P = la.qr(A, mode='economic', pivoting=True)
        Pinv = invert_permutation(P)
        U,s,V = la.svd(R)
        cutoff = (s>rtol).sum()
        if cutoff == 0:
            cutoff = 1
        U1 = U[:,:cutoff]

        C,S,X = gsvd(Q[:m_b]@U1,Q[m_b:]@U1,extras='')
        s = np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))

        # determine multiplicity
        mult = (s<mtol).sum()
        if mult == 0:
            raise ValueError(f'lambda = {lambda_:.3e} has multiplicity zero (sigma = {s[-1]:.3e} > {mtol} = mtol)')

        # least squares solution for the right singular vectors
        C = np.zeros((A.shape[1],mult))
        v = la.lstsq(X.T,np.eye(X.T.shape[0])[:,-mult:])[0]
        C[:cutoff] = la.solve_triangular(R[:cutoff,:cutoff],v)
        return C[Pinv]

    @lru_cache
    def r2gsvd_eigenbasis_coef(self,lambda_,rtol=None,mtol=None):
        """Compute the coefficients of an eigenbasis, assuming lambda_ is an eigenvalue
        using the GSVD formulation with regularization"""
        # raise NotImplementedError('rgsvd_eigenbasis_coef needs debugging')
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        Q,R,P = la.qr(A, mode='economic', pivoting=True)
        Pinv = invert_permutation(P)
        # drop columns of Q corresponding to small diagonal entries of R
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*rtol).sum()

        C,S,X = gsvd(Q[:m_b,:cutoff],Q[m_b:,:cutoff],extras='')
        s = np.divide(C,S,out=np.full(C.shape,np.inf),where=(S!=0))

        # determine multiplicity
        mult = (s<mtol).sum()
        if mult == 0:
            raise ValueError(f'lambda = {lambda_:.3e} has multiplicity zero (sigma = {s[-1]:.3e} > {mtol} = mtol)')

        # least squares solution for the right singular vectors
        C = np.zeros((A.shape[1],mult))
        v = la.lstsq(X.T,np.eye(X.T.shape[0])[:,-mult:])[0]
        C[:cutoff] = la.solve_triangular(R[:cutoff,:cutoff],v)
        return C[Pinv]

    @lru_cache
    def eigenbasis_node_eval(self,lambda_,rtol=None,mtol=None):
        """Evaluate the eigenbasis on the quadrature nodes, assuming lambda_
        is an eigenvalue"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        A = self.weights*self.n_basis(lambda_)
        m_b = len(self.bdry_nodes)
        Q,R,_ = la.qr(A, mode='economic', pivoting=True)

        # drop columns of Q corresponding to small diagonal entries of R
        r = np.abs(np.diag(R))
        cutoff = (r>r[0]*rtol).sum()

        # compute nullspace basis
        try:
            _,s,Vh = la.svd(Q[:m_b,:cutoff])
        except:
            raise ValueError('non-finite SVD')

        # determine multiplicity
        mult = (s<mtol).sum()
        if mult == 0:
            raise ValueError(f'lambda = {lambda_:.3e} has multiplicity zero (sigma = {s[-1]:.3e} > {mtol} = mtol)')

        return (Q[:,:cutoff]@(Vh[-mult:].T))/self.weights

    def eigenbasis_grad(self,lambda_,points,y=None,rtol=None,mtol=None):
        """Evaluates the gradient of the eigenbasis. Returns in complex form,
        with the real part being the partials w.r.t. x, and the imaginary part
        being the partials w.r.t. to y"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(lambda_,rtol,mtol)

        if y is not None:
            points = complex_form(points,y)
        # evaluate partial derivatives
        du_dz = self.basis.grad(lambda_,points)

        return du_dz@C

    @lru_cache
    def outward_normal_derivatives(self,lambda_,n=20,rtol=None,mtol=None):
        raise NotImplementedError('Needs to be updated for complex arithmetic')
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # polygon vertices
        x_v,y_v = self.vertices.real, self.vertices.imag
        e = polygon_edges(self.vertices)

        # Gaussian quadrature nodes
        qnodes = cached_leggauss(n)[0][np.newaxis]

        # transform nodes to boundary intervals
        boundary_nodes = e[:,np.newaxis]*qnodes + self.vertices[:,np.newaxis]

        # get basis gradients & reshape
        du_dz = self.eigenbasis_grad(lambda_,boundary_nodes.flatten(),rtol,mtol)
        mult = du_dz.shape[1]
        du_dz = du_dz.reshape()

        # compute outward normal derivatives
        n = side_normals(self.vertices)

        return n[:,:1]*du_dx + n[:,1:]*du_dy

    @lru_cache
    def _gram_tensors(self,lambda_,n=20,rtol=None,mtol=None):
        raise NotImplementedError('Needs to be updated for complex arithmetic')
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # polygon vertices
        x_v,y_v = self.vertices.T

        # Gaussian quadrature nodes & weights
        qnodes, qweights = cached_leggauss(n)

        # outward normal derivatives
        du_dn = self.outward_normal_derivatives(lambda_,n,rtol,mtol)

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

    def dlambda(self,lambda_,dx=None,dy=None,n=20,rtol=None,mtol=None):
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # catch direction derivative input errors
        if (dx is None) != (dy is None):
            raise ValueError('dx and dy must both be set, or left unset')
        elif dx is not None:
            dx,dy = np.asarray(dx),np.asarray(dy)
            if (dx.shape[0] != self.n_vert) or (dy.shape[0] != self.n_vert):
                raise ValueError(f'dx and dy must both be length {self.n_vert}')

        # compute multiplicity
        _,mult = self.sigma(lambda_,rtol,mtol,mult_check=True)

        # catch repeated eigs
        if (mult > 1) and (dx is None):
            raise ValueError(f'Repeated eigenvalues only have a directional derivative (mult = {mult})')

        # compute gram tensors
        X,Y = self._gram_tensors(lambda_,n,rtol,mtol)

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
            
    ### Asymptotics
    def weyl_N(self,lam):
        """Two-term Weyl asymptotics for the eigenvalue counting function"""
        A = self.area
        P = self.perimiter
        return (A*lam - P*np.sqrt(lam))/(4*np.pi)

    def weyl_k(self,k):
        """Weyl asymptotic estimate for the kth eigenvalue"""
        A = self.area
        P = self.perimiter
        return ((P+np.sqrt(P**2+16*np.pi*A*k))/(2*A))**2
    
    def adj_weyl_N(self,lam):
        A = self.area
        P = self.perimiter
        K = self.asym_K
        return (A*lam - P*np.sqrt(lam))/(4*np.pi) + K
    
    def adj_weyl_k(self,k):
        A = self.area
        P = self.perimiter
        K = self.asym_K
        return ((P+np.sqrt(P**2+16*np.pi*A*(k-K)))/(2*A))**2
    
    def weyl_check(self,eigs,tol=1):
        weyl_est = self.weyl_N(eigs)
        true_N = np.arange(1,len(eigs)+1)
        diff = weyl_est - true_N
        argdiff = eigs[diff>tol]
        return argdiff, diff

    ### Helper functions for plotting subspace angles
    def plot_sigma(self,low,high,nlam,ax=None,rtol=None,**kwargs):
        if rtol is None: rtol = self.rtol
        if low < 1e-16 : low = 1e-16
        L = np.linspace(low,high,nlam+1)
        sigma = []
        for lam in L:
            sigma.append(self.sigma(lam,rtol))
        if ax is None:
            fig = plt.figure()
            plt.plot(L,sigma,**kwargs)
        else:
            ax.plot(L,sigma,**kwargs)

    def plot_gsigma(self,low,high,nlam,ax=None,**kwargs):
        if low < 1e-16 : low = 1e-16
        L = np.linspace(low,high,nlam+1)
        gsigma = []
        for lam in L:
            gsigma.append(self.gsigma(lam))
        if ax is None:
            fig = plt.figure()
            plt.plot(L,gsigma,**kwargs)
        else:
            ax.plot(L,gsigma,**kwargs)

    def plot_gsvd_subspace_tans(self,low,high,nlam,n_angle,ax=None,**kwargs):
        if low < 1e-16 : low = 1e-16
        L = np.linspace(low,high,nlam+1)
        tans = np.empty((len(L),n_angle))
        for i,lam in enumerate(L):
            tans[i] = self.gsvd_subspace_tans(lam)[:n_angle]
        if ax is None:
            fig = plt.figure()
            plt.plot(L,tans,**kwargs)
        else:
            ax.plot(L,tans,**kwargs)

    def plot_rgsigma(self,low,high,nlam,ax=None,rtol=None,**kwargs):
        if rtol is None: rtol = self.rtol
        if low < 1e-16 : low = 1e-16
        L = np.linspace(low,high,nlam+1)
        rgsigma = []
        for lam in L:
            rgsigma.append(self.rgsigma(lam,rtol=rtol))
        if ax is None:
            fig = plt.figure()
            plt.plot(L,rgsigma,**kwargs)
        else:
            ax.plot(L,rgsigma,**kwargs)

    def plot_r2gsigma(self,low,high,nlam,ax=None,rtol=None,**kwargs):
        if rtol is None: rtol = self.rtol
        if low < 1e-16 : low = 1e-16
        L = np.linspace(low,high,nlam+1)
        rgsigma = []
        for lam in L:
            rgsigma.append(self.r2gsigma(lam,rtol=rtol))
        if ax is None:
            fig = plt.figure()
            plt.plot(L,rgsigma,**kwargs)
        else:
            ax.plot(L,rgsigma,**kwargs)

    def plot_gesigma(self,low,high,nlam,ax=None,**kwargs):
        if low < 1e-16 : low = 1e-16
        L = np.linspace(low,high,nlam+1)
        gesigma = []
        for lam in L:
            gesigma.append(self.gesigma(lam))
        if ax is None:
            fig = plt.figure()
            plt.plot(L,gesigma,**kwargs)
        else:
            ax.plot(L,gesigma,**kwargs)