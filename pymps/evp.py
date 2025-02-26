import numpy as np
from . import mps
from .utils import (complex_form, rand_interior_points, polygon_area, polygon_perimeter, 
                    interior_angles, singular_corner_check, invert_permutation)
from .bases import FourierBesselBasis
from .quad import boundary_nodes_polygon, triangular_mesh, tri_quad
from .opt import gridmin
from functools import cache, lru_cache
import matplotlib.pyplot as plt
import warnings

class PolygonEVP:
    """Class for polygonal Dirichlet Laplacian eigenproblems
    """
    def __init__(self,vertices,basis='auto',boundary_pts='auto',interior_pts='auto',
                 weights='auto',rtol=1e-14,mtol=1e-4,order=40):
        vertices = np.asarray(vertices)
        if vertices.ndim == 2:
            if vertices.shape[0] == 2:
                vertices = complex_form(vertices[0],vertices[1])
            elif vertices.shape[1] == 2:
                vertices = complex_form(vertices[:,0],vertices[:,1])
        self.vertices = vertices

        # set up basis
        if type(basis) is FourierBesselBasis:
            self.basis = basis
        elif basis == 'auto':
            orders = PolygonEVP.set_orders(vertices,order)
            self.basis = FourierBesselBasis(vertices,orders)
        elif type(basis) in [int,list,np.ndarray]:
            self.basis = FourierBesselBasis(vertices,basis)

        # set up weights
        if type(weights) is np.ndarray:
            if len(boundary_pts) + len(interior_pts) != len(weights):
                raise ValueError('provided boundary & interior pts not compatible with provided weights')
            self.weights = weights
        elif weights is None:
            self.weights = None
        elif weights == 'auto':
            if boundary_pts != 'auto' or interior_pts != 'auto':
                raise ValueError("if weights are set to auto, then so must be boundary_pts and interior_pts")
            
            # set up boundary pts and weights
            # if only expanding around one vertex with Fourier-Bessel, discard boundary pts next to this vertex
            if np.sum(self.basis.orders>0) == 1 and type(self.basis) is FourierBesselBasis:
                n_pts = self.basis.orders.max()
                idx = np.nonzero(self.basis.orders)[0][0]
                skip = [idx-1,idx]
            else:
                n_pts = 2*int(np.ceil(self.basis.orders.mean()))
                skip = None
            self.boundary_pts, boundary_wts = boundary_nodes_polygon(vertices,n_pts=n_pts,skip=skip)

            # set up interior pts and weights
            mesh_size = PolygonEVP.set_mesh_size(vertices)
            mesh = triangular_mesh(vertices,mesh_size)
            self.interior_pts, interior_wts = tri_quad(mesh)

            self.weights = np.sqrt(np.concatenate((boundary_wts,interior_wts)))[:,np.newaxis]         

        # only consider boundary_pts and interior_pts args if weights is not auto
        if weights != 'auto':
            # set up boundary points
            if type(boundary_pts) is np.ndarray:
                if boundary_pts.ndim > 1:
                    boundary_pts = complex_form(boundary_pts)
                self.boundary_pts = boundary_pts
            elif type(boundary_pts) is int:
                # if only expanding around one vertex with Fourier-Bessel, discard boundary pts next to this vertex
                if np.sum(self.basis.orders>0) == 1 and type(self.basis) is FourierBesselBasis:
                    idx = np.nonzero(self.basis.orders)[0][0]
                    skip = [idx-1,idx]
                else:
                    skip = None
                self.boundary_pts = boundary_nodes_polygon(vertices,n_pts=boundary_pts,skip=skip)[0]
            elif boundary_pts is None:
                if np.sum(self.basis.orders>0) == 1 and type(self.basis) is FourierBesselBasis:
                    n_pts = self.basis.orders.max()
                    idx = np.nonzero(self.basis.orders)[0][0]
                    skip = [idx-1,idx]
                else:
                    n_pts = 2*int(np.ceil(self.basis.orders.mean()))
                    skip = None
                self.boundary_pts = boundary_nodes_polygon(vertices,n_pts=n_pts,skip=skip)[0]

            # catch integer argument to construct interior points
            if type(interior_pts) is np.ndarray:
                if interior_pts.ndim > 1:
                    interior_pts = complex_form(interior_pts)
                self.interior_pts = interior_pts
            elif type(interior_pts) is int:
                self.interior_pts = rand_interior_points(vertices,interior_pts)

        # set default points for the basis to be boundary_pts and interior_pts
        self.basis.set_default_points(np.concatenate((self.boundary_pts,self.interior_pts)))
        self.m_b = len(self.boundary_pts)

        # set regularization and multiplicity tolerances
        self.rtol = rtol
        self.mtol = mtol

        # area and perimiter
        self.area = polygon_area(vertices)
        self.perimiter = polygon_perimeter(vertices)

        # lower bound for first eigenvalue
        self.eig1_lb = 5.76*np.pi/self.area

        # asymptotic adjustment parameter
        int_angles = interior_angles(self.vertices)
        self.asym_K = (np.pi/int_angles - int_angles/np.pi).sum()/24

        # spectrum tracker
        self.spectrum = Spectrum()

    @property
    def eigs(self):
        return self.spectrum.eigs

    @classmethod
    def set_orders(cls,vertices,order):
        phis = interior_angles(vertices)
        sing = singular_corner_check(phis)
        orders = np.zeros(len(vertices),dtype='int')
        # all corners are regular, expand around vertex with largest angle only
        if not np.any(sing):
            orders[np.argmax(phis)] = order
        else:
            orders[sing] = np.rint(order*(phis[sing]/phis[sing].mean()))
        return orders
    
    @classmethod
    def set_mesh_size(cls,vertices):
        diam = np.abs(np.subtract.outer(vertices,vertices)).max()
        return diam/2

    @cache
    def subspace_sines(self,lam,rtol=None,weights=None):
        """Compute the sines of subspace angles via SVD of Q_B(lam)"""
        if rtol is None: rtol = self.rtol
        if weights is None: weights=self.weights

        if (weights is None) or (weights is False):
            A = self.basis(lam)
        else:
            A = weights*self.basis(lam)

        return mps.subspace_sines(A,self.m_b,rtol)
    
    @cache
    def subspace_tans(self,lam,rtol=None,reg_type='svd',weights=None):
        """Compute the tangents of subspace angles via GSVD of {A_B(lam),A_I(lam)}"""
        if rtol is None: rtol = self.rtol
        if weights is None: weights=self.weights

        if (weights is None) or (weights is False):
            A = self.basis(lam)
        else:
            A = weights*self.basis(lam)

        return mps.subspace_tans(A,self.m_b,rtol,reg_type)
    
    def _proximity_check(self,lam,eigs_tmp,xtol):
        """Checks if a candidate eigenvalue is within tolerance of previously found eigenvalues"""
        eigs_tmp = np.concatenate((np.unique(self.eigs),eigs_tmp))
        if len(eigs_tmp) == 0:
            return True
        elif np.abs(lam-eigs_tmp).min() <= xtol:
            return False
        else:
            return True

    def _local_tolerance(self,lam,eig1,eig2=0,h=1e-8,mps_kwargs={}):
        """Gets the local tolerance for subspace angles based on nearby eigenvalues."""
        slope1 = np.abs((self.subspace_tans(eig1+h,**mps_kwargs)[0]-self.subspace_tans(eig1,**mps_kwargs)[0])/h)
        tol1 = self.subspace_tans(eig1,**mps_kwargs)[0] + slope1*np.abs(lam-eig1)
        if eig2 == 0:
            tol2 = np.inf
        else:
            slope2 = np.abs((self.subspace_tans(eig2,**mps_kwargs)[0]-self.subspace_tans(eig2-h,**mps_kwargs)[0])/h)
            tol2 = self.subspace_tans(eig2,**mps_kwargs)[0] + slope2*np.abs(lam-eig2)
        return min(self.mtol,tol1,tol2)
    
    def solve_eigs_interval(self,a,b,n_pts,xtol=1e-8,mps_kwargs={},verbose=0):
        """Finds eigenvalues in the interval [a,b]. Searches over a grid with n_pts."""
        # initial search grid 
        lamgrid_int = np.linspace(a,b,n_pts)
        lamgrid = np.empty(len(lamgrid_int)+2)
        lamgrid[1:-1] = lamgrid_int
        
        # add ghost points to ensure robust search
        lamgrid[0] = 2*lamgrid[1]-lamgrid[2]
        if lamgrid[0] <= 0: lamgrid[0] = a/2
        lamgrid[-1] = 2*lamgrid[-2]-lamgrid[-3]

        tans = np.array([self.subspace_tans(lam,**mps_kwargs)[:2] for lam in lamgrid]).T
        fevals = len(lamgrid)

        minima,fe = gridmin(lambda lam: self.subspace_tans(lam,**mps_kwargs)[:2],lamgrid,tans,xtol=xtol,verbose=verbose)
        fevals += fe
        if verbose > 0: print(f"fevals={fevals}")  
        minima = np.sort(minima)

        eigs_tmp = []
        # filter out spurious minima and minima too close to known eigenvalues
        for lam in minima:
            if self._proximity_check(lam,eigs_tmp,xtol) and self.subspace_tans(lam,**mps_kwargs)[0]<self.mtol:
                eigs_tmp.append(lam)

        # Determining tolerance requires subspace angle bounds from nearby eigenvalues
        # Get all eigenvalues sorted
        eigs_ = np.concatenate((eigs_tmp,np.unique(self.eigs)))
        sortidx = np.argsort(eigs_)
        eigs_sorted = eigs_[sortidx]

        # inv_sortidx gives the positions of new eigenvalues within the sort
        inv_sortidx = invert_permutation(sortidx)[:len(eigs_tmp)]
        # loop over new eigenvalues, set local tolerance for multiplicity check
        # then add to spectrum with multiplicity and local subspace angle tolerance
        for idx in inv_sortidx:
            if verbose > 0: print(f"estimating multiplicity of lam={eigs_sorted[idx]:.3e}")
            # get local tolerance parameter
            if idx == len(eigs_sorted)-1:
                tol = self._local_tolerance(eigs_sorted[idx],eigs_sorted[idx-1],h=xtol,mps_kwargs=mps_kwargs)
            else:
                tol = self._local_tolerance(eigs_sorted[idx],eigs_sorted[idx-1],eigs_sorted[idx+1],xtol,mps_kwargs)
                
            # multiplicity = number of subspace tangents below tolerance
            tans = self.subspace_tans(eigs_sorted[idx],**mps_kwargs)
            mult = (tans<=tol).sum()
            if verbose > 1: print(f"tans={np.array_str(tans,precision=2)}, tol={tol:.3e}, mult={mult}")

            # add eigenvalue to spectrum
            self.spectrum.add_eig(eigs_sorted[idx],mult,tol)
        
        eigs = self.eigs
        return eigs[(eigs>=a-xtol)&(eigs<=b+xtol)]

    def solve_eigs_ordered(self,k,ppl=10,xtol=1e-8,mps_kwargs={},maxiter=10,verbose=0):
        """Find the first k eigenvalues of the domain, up to xtol. Checks for multiplicity."""
        eig_count = len(self.eigs)
        # return first k eigs if already known
        if len(self.eigs) >= k:
            return self.eigs[:k]
        else:
            deficit = k - eig_count
            if eig_count == 0:
                a = self.eig1_lb
            else:
                a = self.eigs[-1]
            b = self.weyl_k(k+1)
            n_pts = ppl*deficit
            self.solve_eigs_interval(a,b,n_pts,xtol,mps_kwargs,verbose)

        # # run weyl check
        # deltas = self.weyl_check(len(self.eigs))
        # i = 0
        # while np.any(deltas<0) and i < maxiter:
        #     idx = np.nonzero(deltas<0)[0][0]
        #     warnings.warn(f"weyl check failed at lam={self.eigs[idx]}, delta={deltas[idx]}")
        #     self.solve_eigs_interval(self.eigs[idx-1]+xtol,self.eigs[idx]-xtol,10*ppl,xtol,mps_kwargs,verbose)
        #     i += 1
        #     deltas = self.weyl_check(len(self.eigs))

        return self.eigs[:k]

    @cache
    def eigenbasis_coef(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
        """Computes the coefficient vectors of the eigenbasis"""
        if mtol is None: mtol = self.spectrum.mtols[eig]
        if rtol is None: rtol = self.rtol
        if weights is None: weights=self.weights

        if (weights is None) or (weights is False):
            A = self.basis(eig)
        else:
            A = weights*self.basis(eig)

        if solver == 'gsvd':
            return mps.nullspace_basis_gsvd(A,self.m_b,mtol,rtol,reg_type)
        elif solver == 'svd':
            return mps.nullspace_basis_svd(A,self.m_b,mtol,rtol)
        
    @cache
    def eigenbasis(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
        """Returns a callable function which evaluates the approximate eigenbasis
        corresponding to lam"""
        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig,mtol,rtol,solver,reg_type,weights)

        # return callable function from the basis
        def func(points,y=None):
            if y is not None:
                points = complex_form(points,y)
            shape = np.asarray(points).shape
            shape = (*shape,C.shape[1])
            return (self.basis(eig,points)@C).reshape(shape)
        return func

    @cache
    def eigenbasis_grad(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
        """Returns a callable function which evaluates the approximate eigenbasis
        corresponding to lam. Returns in complex form, with the real part being the 
        partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig,mtol,rtol,solver,reg_type,weights)

        # return callable function from the basis
        def func(points,y=None):
            if y is not None:
                points = complex_form(points,y)
            shape = np.asarray(points).shape
            shape = (*shape,C.shape[1])
            return (self.basis.grad(eig,points)@C).reshape(shape)
        return func
    
    @cache
    def eval_eigenbasis(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
        """Evaluate the eigenbasis on the collocation points"""
        if mtol is None: mtol = self.spectrum.mtols[eig]
        if rtol is None: rtol = self.rtol
        if weights is None: weights=self.weights

        if (weights is None) or (weights is False):
            A = self.basis(eig)
            weights = 1
        else:
            A = weights*self.basis(eig)

        if solver == 'gsvd':
             return mps.nullspace_basis_eval_gsvd(A,self.m_b,mtol,rtol,reg_type)/weights
        elif solver == 'svd':
            return mps.nullspace_basis_eval_svd(A,self.m_b,mtol,rtol)/weights
    
    @cache
    def eval_eigenbasis_grad(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
        """Evaluates the gradient of the eigenbasis on the collocation points. Returns in complex form, 
        with the real part being the partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig,mtol,rtol,solver,reg_type,weights)

        return self.basis.grad(eig)@C

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
    
    def weyl_check(self,k):
        weyl_est = self.adj_weyl_N(self.eigs[:k])
        true_N = np.arange(1,k+1)
        deltas = true_N - weyl_est
        return deltas

    ### Helper functions for plotting
    def plot_subspace_sines(self,low,high,nlam,n_angle,ax=None,mps_kwargs={},**kwargs):
        if low < self.eig1_lb/2 : low = self.eig1_lb/2
        L = np.linspace(low,high,nlam+1)
        sines = np.empty((len(L),n_angle))
        for i,lam in enumerate(L):
            sines[i] = self.subspace_sines(lam,**mps_kwargs)[:n_angle]
        if ax is None:
            fig = plt.figure()
            plt.plot(L,sines,**kwargs)
        else:
            ax.plot(L,sines,**kwargs)

    def plot_subspace_tans(self,low,high,nlam,n_angle,ax=None,mps_kwargs={},**kwargs):
        if low < self.eig1_lb/2 : low = self.eig1_lb/2
        L = np.linspace(low,high,nlam+1)
        tans = np.empty((len(L),n_angle))
        for i,lam in enumerate(L):
            tans[i] = self.subspace_tans(lam,**mps_kwargs)[:n_angle]
        if ax is None:
            fig = plt.figure()
            plt.plot(L,tans,**kwargs)
        else:
            ax.plot(L,tans,**kwargs)

class Eigenvalue:
    def __init__(self,val,mult,mtol=None):
        self._val = float(val)
        self._mult = int(mult)
        if mtol is not None:
            self._mtol = float(mtol)
        else:
            self._mtol = None

    @property
    def val(self):
        return self._val
    
    @property
    def mult(self):
        return self._mult
    
    @property
    def mtol(self):
        return self._mtol
    
    @mtol.setter
    def mtol(self,val):
        self._mtol = float(val)
    
    def __eq__(self,other):
        if type(other) is Eigenvalue:
            return (self._val == other._val) and (self._mult == other._mult)
        else:
            return False
        
    def __hash__(self):
        return hash((self._val,self._mult))

    def __repr__(self):
        return f"Eigenvalue({self.val},mult={self.mult},mtol={self.mtol})"


class Spectrum:
    """Class for tracking the Laplacian spectrum of a domain"""
    def __init__(self):
        self._eigs = set()

    def add_eig(self,eig,mult=1,mtol=None):
        if eig in self.eigs:
            raise ValueError(f"{eig} already in spectrum (with mult = {self.mults[eig]}, mtol = {self.mtols[eig]})")
        else:
            self._eigs.add(Eigenvalue(eig,mult,mtol))

    def remove_eig(self,eig,mult=None):
        if mult is None:
            mult = self.mults[eig]
        self._eigs.remove(Eigenvalue(eig,mult))

    @property
    def sorted(self):
        return sorted(self._eigs,key=lambda eig: eig.val)

    @property
    def eigs(self):
        vals = np.concatenate(([],*(eig._mult*[eig._val] for eig in self._eigs)))
        return np.sort(vals.flatten())

    @eigs.deleter
    def eigs(self):
        del self._eigs

    @property
    def mults(self):
        return {eig.val:eig.mult for eig in self._eigs}
    
    @property
    def mtols(self):
        return {eig.val:eig.mtol for eig in self._eigs}

    def __repr__(self):
        return f"Spectrum:{repr(self._eigs)}"

    def __eq__(self,other):
        if type(other) is Spectrum:
            return self._eigs == other._eigs