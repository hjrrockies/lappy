import numpy as np
import scipy.linalg as la
from . import mps
from .utils import (complex_form, rand_interior_points, polygon_area, polygon_perimeter, edge_lengths,
                    polygon_edges, interior_angles, singular_corner_check, invert_permutation, plot_polygon,
                    Interval, MultiInterval)
from .bases import ParticularBasis, FourierBesselBasis
from .quad import boundary_nodes_polygon, triangular_mesh, tri_quad, cached_leggauss
from .geometry import PointSet, Domain
from functools import cache, lru_cache
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import points

# class PolygonEVP:
#     """Class for polygonal Dirichlet Laplacian eigenproblems
#     """
#     def __init__(self,vertices,basis='auto',boundary_pts='auto',interior_pts='auto',
#                  weights='auto',rtol=1e-14,mtol=1e-4,order=10):
#         vertices = np.asarray(vertices)
#         if vertices.ndim == 2:
#             if vertices.shape[0] == 2:
#                 vertices = complex_form(vertices[0],vertices[1])
#             elif vertices.shape[1] == 2:
#                 vertices = complex_form(vertices[:,0],vertices[:,1])
#         self.vertices = vertices

#         # set up basis
#         if type(basis) is FourierBesselBasis:
#             self.basis = basis
#         elif basis == 'auto':
#             orders = PolygonEVP.set_orders(vertices,order)
#             self.basis = FourierBesselBasis(vertices,orders)
#         elif type(basis) in [int,list,np.ndarray]:
#             self.basis = FourierBesselBasis(vertices,basis)

#         # set up weights
#         if type(weights) is np.ndarray:
#             if len(boundary_pts) + len(interior_pts) != len(weights):
#                 raise ValueError('provided boundary & interior pts not compatible with provided weights')
#             self.weights = weights.reshape((-1,1))
#         elif weights is None:
#             self.weights = None
#         elif weights == 'auto':
#             if interior_pts != 'auto' or not (type(boundary_pts) is int or boundary_pts == 'auto'):
#                 raise ValueError("if weights are set to auto, interior_pts must be auto and boundary pts may be int array")
            
#             # set up boundary pts and weights
#             # if only expanding around one vertex with Fourier-Bessel, discard boundary pts next to this vertex
#             if np.sum(self.basis.orders>0) == 1 and type(self.basis) is FourierBesselBasis:
#                 if boundary_pts == 'auto':
#                     n_pts = self.basis.orders.max()
#                     # triangles need more points!
#                     if (len(self.vertices)) == 3:
#                         n_pts = 2*n_pts
#                 else:
#                     n_pts = boundary_pts
#                 idx = np.nonzero(self.basis.orders)[0][0]
#                 mask = np.full(len(vertices),True)
#                 mask[idx-1] = False
#                 mask[idx] = False
                
#             else:
#                 mask = np.full(len(vertices),True)
#                 if boundary_pts == 'auto':
#                     n_pts = 2*int(np.ceil(self.basis.orders.mean()))
#                 else:
#                     n_pts = boundary_pts
#             lens = edge_lengths(vertices)
#             n_pts_vec = np.rint((n_pts-5)*(lens/lens[mask].mean()))+5
#             n_pts_vec[~mask] = 0
#             self.boundary_pts, boundary_wts = boundary_nodes_polygon(vertices,n_pts=n_pts_vec.astype(int))

#             # set up interior pts and weights
#             mesh_size = PolygonEVP.set_mesh_size(vertices)
#             mesh = triangular_mesh(vertices,mesh_size)
#             self.interior_pts, interior_wts = tri_quad(mesh)

#             self.weights = np.sqrt(np.concatenate((boundary_wts,interior_wts))).reshape((-1,1))

#         # only consider boundary_pts and interior_pts args if weights is not auto
#         weights_auto = False
#         if type(weights) is str:
#             weights_auto = (weights == 'auto')
#         if not weights_auto:
#             # set up boundary points
#             if type(boundary_pts) is np.ndarray:
#                 if boundary_pts.ndim > 1:
#                     boundary_pts = complex_form(boundary_pts)
#                 self.boundary_pts = boundary_pts
#             elif type(boundary_pts) is int:
#                 # if only expanding around one vertex with Fourier-Bessel, discard boundary pts next to this vertex
#                 if np.sum(self.basis.orders>0) == 1 and type(self.basis) is FourierBesselBasis:
#                     idx = np.nonzero(self.basis.orders)[0][0]
#                     skip = [idx-1,idx]
#                 else:
#                     skip = None
#                 self.boundary_pts = boundary_nodes_polygon(vertices,n_pts=boundary_pts,skip=skip)[0]
#             elif boundary_pts is None:
#                 if np.sum(self.basis.orders>0) == 1 and type(self.basis) is FourierBesselBasis:
#                     n_pts = self.basis.orders.max()
#                     idx = np.nonzero(self.basis.orders)[0][0]
#                     mask = np.full(len(vertices),True)
#                     mask[idx-1] = False
#                     mask[idx] = False
#                 else:
#                     n_pts = 2*int(np.ceil(self.basis.orders.mean()))
#                     mask = np.full(len(vertices),True)
#                 lens = edge_lengths(vertices)
#                 n_pts_vec = np.rint((n_pts-5)*(lens/lens[mask].mean()))+5
#                 n_pts_vec[~mask] = 0
#                 self.boundary_pts = boundary_nodes_polygon(vertices,n_pts=n_pts_vec.astype(int))[0]

#             # catch integer argument to construct interior points
#             if type(interior_pts) is np.ndarray:
#                 if interior_pts.ndim > 1:
#                     interior_pts = complex_form(interior_pts)
#                 self.interior_pts = interior_pts
#             elif type(interior_pts) is int:
#                 self.interior_pts = rand_interior_points(vertices,interior_pts)

#         # set default points for the basis to be boundary_pts and interior_pts
#         self.basis.set_default_points(np.concatenate((self.boundary_pts,self.interior_pts)))
#         self.m_b = len(self.boundary_pts)

#         # set regularization and multiplicity tolerances
#         self.rtol = rtol
#         self.mtol = mtol

#         # area and perimiter
#         self.area = polygon_area(vertices)
#         self.perimiter = polygon_perimeter(vertices)

#         # lower bound for first eigenvalue
#         self.eig1_lb = 5.76*np.pi/self.area

#         # asymptotic adjustment parameter
#         int_angles = interior_angles(self.vertices)
#         self.asym_K = (np.pi/int_angles - int_angles/np.pi).sum()/24

#         # spectrum tracker
#         self.spectrum = Spectrum()

#         # shapely polygon
#         self.polygon = Polygon(np.array([self.vertices.real,self.vertices.imag]).T)

#         # automatically boost rtol if needed
#         ntest = 5
#         maxtry = 10
#         i = 0
#         while True:
#             if i == maxtry:
#                 break
#             x = (1+np.random.rand(ntest))*self.eig1_lb
#             test_evals = [self.subspace_tans(x_)[0] for x_ in x]
#             logmean = np.log10(test_evals).mean()
#             if logmean >= -1:
#                 break
#             else:
#                 self.rtol *= 10
#                 i += 1

#     @property
#     def eigs(self):
#         return self.spectrum.eigs
    
#     @property
#     def mults(self):
#         return self.spectrum.mults
    
#     @property
#     def mtols(self):
#         return self.spectrum.mtols

#     @classmethod
#     def set_orders(cls,vertices,order):
#         phis = interior_angles(vertices)
#         sing = singular_corner_check(phis)
#         orders = np.zeros(len(vertices),dtype='int')
#         # all corners are regular, expand around vertex with largest angle only
#         if not np.any(sing):
#             orders[np.argmax(phis)] = order
#         else:
#             orders[sing] = np.rint((order-3)*(phis[sing]/phis[sing].mean()))+3
#         return orders
    
#     @classmethod
#     def set_mesh_size(cls,vertices):
#         diam = np.abs(np.subtract.outer(vertices,vertices)).max()
#         return diam/2

#     @cache
#     def subspace_sines(self,lam,rtol=None,weights=None):
#         """Compute the sines of subspace angles via SVD of Q_B(lam)"""
#         if rtol is None: rtol = self.rtol
#         if weights is None: weights=self.weights

#         if (weights is None) or (weights is False):
#             A = self.basis(lam)
#         else:
#             A = weights*self.basis(lam)

#         return mps.subspace_sines(A,self.m_b,rtol)
    
#     @cache
#     def subspace_tans(self,lam,rtol=None,reg_type='svd',weights=None):
#         """Compute the tangents of subspace angles via GSVD of {A_B(lam),A_I(lam)}"""
#         if rtol is None: rtol = self.rtol
#         if weights is None: weights=self.weights

#         if (weights is None) or (weights is False):
#             A = self.basis(lam)
#         else:
#             A = weights*self.basis(lam)

#         return mps.subspace_tans(A,self.m_b,rtol,reg_type)
    
#     def _proximity_check(self,lam,eigs_tmp,xtol):
#         """Checks if a candidate eigenvalue is within tolerance of previously found eigenvalues"""
#         eigs_tmp = np.concatenate((np.unique(self.eigs),eigs_tmp))
#         if len(eigs_tmp) == 0:
#             return True
#         elif np.abs(lam-eigs_tmp).min() <= xtol:
#             return False
#         else:
#             return True

#     def _local_tolerance(self,lam,eig1,eig2=0,h=1e-12,mps_kwargs={}):
#         """Gets the local tolerance for subspace angles based on nearby eigenvalues."""
#         slope1 = np.abs((self.subspace_tans(eig1+h,**mps_kwargs)[0]-self.subspace_tans(eig1,**mps_kwargs)[0])/h)
#         tol1 = self.subspace_tans(eig1,**mps_kwargs)[0] + slope1*np.abs(lam-eig1)
#         if eig2 == 0:
#             tol2 = np.inf
#         else:
#             slope2 = np.abs((self.subspace_tans(eig2,**mps_kwargs)[0]-self.subspace_tans(eig2-h,**mps_kwargs)[0])/h)
#             tol2 = self.subspace_tans(eig2,**mps_kwargs)[0] + slope2*np.abs(lam-eig2)
#         return min(self.mtol,tol1,tol2)
    
#     def solve_eigs_interval(self,a,b,n_pts,xtol=1e-12,mps_kwargs={},verbose=0):
#         """Finds eigenvalues in the interval [a,b]. Searches over a grid with n_pts."""
#         # initial search grid 
#         lamgrid_int = np.linspace(a,b,n_pts)
#         lamgrid = np.empty(len(lamgrid_int)+2)
#         lamgrid[1:-1] = lamgrid_int
        
#         # add ghost points to ensure robust search
#         lamgrid[0] = 2*lamgrid[1]-lamgrid[2]
#         if lamgrid[0] <= 0: lamgrid[0] = a/2
#         lamgrid[-1] = 2*lamgrid[-2]-lamgrid[-3]

#         tans = np.array([self.subspace_tans(lam,**mps_kwargs)[:2] for lam in lamgrid]).T
#         fevals = len(lamgrid)

#         minima,fe = gridmin(lambda lam: self.subspace_tans(lam,**mps_kwargs)[:2],lamgrid,tans,xtol=xtol,verbose=verbose)
#         fevals += fe
#         if verbose > 0: print(f"fevals={fevals}")  
#         minima = np.sort(minima)

#         eigs_tmp = []
#         # filter out spurious minima and minima too close to known eigenvalues
#         for lam in minima:
#             mybool = self._proximity_check(lam,eigs_tmp,10*xtol)
#             tan = self.subspace_tans(lam,**mps_kwargs)[0]
#             if not mybool and verbose > 0:
#                 print(f"lam={lam:.3f} too close to previously found eigenvalue")
#             elif tan > self.mtol and verbose > 0:
#                 print(f"lam={lam} above threshold, {tan:.3e}>{self.mtol:.3e}")
#             else:
#                 if verbose > 0:
#                     print(f"lam={lam:.3f} accepted as eigenvalue")
#                 eigs_tmp.append(lam)

#         # Determining tolerance requires subspace angle bounds from nearby eigenvalues
#         # Get all eigenvalues sorted
#         eigs_ = np.concatenate((eigs_tmp,np.unique(self.eigs)))
#         sortidx = np.argsort(eigs_)
#         eigs_sorted = eigs_[sortidx]

#         # inv_sortidx gives the positions of new eigenvalues within the sort
#         inv_sortidx = invert_permutation(sortidx)[:len(eigs_tmp)]
#         # loop over new eigenvalues, set local tolerance for multiplicity check
#         # then add to spectrum with multiplicity and local subspace angle tolerance
#         for idx in inv_sortidx:
#             if verbose > 0: print(f"estimating multiplicity of lam={eigs_sorted[idx]:.3e}")
#             # get local tolerance parameter
#             if idx == len(eigs_sorted)-1:
#                 tol = self._local_tolerance(eigs_sorted[idx],eigs_sorted[idx-1],h=xtol,mps_kwargs=mps_kwargs)
#                 fevals += 2
#             else:
#                 tol = self._local_tolerance(eigs_sorted[idx],eigs_sorted[idx-1],eigs_sorted[idx+1],xtol,mps_kwargs)
#                 fevals += 1
                
#             # multiplicity = number of subspace tangents below tolerance
#             tans = self.subspace_tans(eigs_sorted[idx],**mps_kwargs)
#             mult = (tans<=tol).sum()
#             if verbose > 1: print(f"tans={np.array_str(tans,precision=2)}, tol={tol:.3e}, mult={mult}")

#             # add eigenvalue to spectrum
#             self.spectrum.add_eig(eigs_sorted[idx],mult,tol)
        
#         eigs = self.eigs
#         return eigs[(eigs>=a-xtol)&(eigs<=b+xtol)], fevals

#     def solve_eigs_ordered(self,k,ppl=10,xtol=1e-12,mps_kwargs={},maxiter=10,verbose=0):
#         """Find the first k eigenvalues of the domain, up to xtol. Checks for multiplicity."""
#         eig_count = len(self.eigs)
#         # return first k eigs if already known
#         if len(self.eigs) >= k:
#             return self.eigs[:k],0
#         else:
#             deficit = k - eig_count
#             if eig_count == 0:
#                 a = self.eig1_lb
#             else:
#                 a = self.eigs[-1]
#             b = self.weyl_k(k+1)
#             n_pts = ppl*deficit
#             _,fevals = self.solve_eigs_interval(a,b,n_pts,xtol,mps_kwargs,verbose)

#         # # run weyl check
#         # deltas = self.weyl_check(len(self.eigs))[:k]
#         # i,j = 0, 0
#         # ppl_new = 2*ppl
#         # while np.any(deltas<-0.5) and i < maxiter:
#         #     idx = np.nonzero(deltas<-0.5)[0][0]
#         #     warnings.warn(f"weyl check failed at lam={self.eigs[idx]}, delta={deltas[idx]}")
#         #     if idx == idx_old:
#         #         ppl_new = (j+2)*ppl
#         #         j += 1
#         #     else:
#         #         j = 0
#         #     _,fe = self.solve_eigs_interval(self.eigs[idx-1],self.eigs[idx],ppl_new,xtol,mps_kwargs,verbose)
#         #     fevals += fe
#         #     i += 1
#         #     deltas = self.weyl_check(len(self.eigs))[:k]
#         #     idx_old = idx
#         # if verbose > 0: print("deltas =",deltas)

#         # # extend search
#         # i = 0
#         # while len(self.eigs) < k and i < maxiter:
#         #     warnings.warn(f"extending search for eigenvalues to [{self.weyl_k(k+1+i)},{self.weyl_k(k+2+i)}]")
#         #     _,fe = self.solve_eigs_interval(self.weyl_k(k+1+i),self.weyl_k(k+2+i),ppl,xtol,mps_kwargs,verbose)
#         #     fevals += fe
#         return self.eigs[:k], fevals

#     @cache
#     def eigenbasis_coef(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#         """Computes the coefficient vectors of the eigenbasis"""
#         if mtol is None: mtol = self.mtols[eig]
#         if rtol is None: rtol = self.rtol
#         if weights is None: weights=self.weights

#         if (weights is None) or (weights is False):
#             A = self.basis(eig)
#         else:
#             A = weights*self.basis(eig)

#         if solver == 'gsvd':
#             return mps.nullspace_basis_gsvd(A,self.m_b,mtol,rtol,reg_type)
#         elif solver == 'svd':
#             return mps.nullspace_basis_svd(A,self.m_b,mtol,rtol)
        
#     def eigenbasis(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#         """Returns a callable function which evaluates the approximate eigenbasis
#         corresponding to lam"""
#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig,mtol,rtol,solver,reg_type,weights)

#         # return callable function from the basis
#         def func(points,y=None):
#             if y is not None:
#                 points = complex_form(points,y)
#             shape = np.asarray(points).shape
#             shape = (*shape,C.shape[1])
#             return (self.basis(eig,points)@C).reshape(shape)
#         return func

#     def eigenbasis_grad(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#         """Returns a callable function which evaluates the approximate eigenbasis
#         corresponding to lam. Returns in complex form, with the real part being the 
#         partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig,mtol,rtol,solver,reg_type,weights)

#         # return callable function from the basis
#         def func(points,y=None):
#             if y is not None:
#                 points = complex_form(points,y)
#             shape = np.asarray(points).shape
#             shape = (*shape,C.shape[1])
#             return (self.basis.grad(eig,points)@C).reshape(shape)
#         return func
    
#     @cache
#     def eval_eigenbasis(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#         """Evaluate the eigenbasis on the collocation points"""
#         if mtol is None: mtol = self.mtols[eig]
#         if rtol is None: rtol = self.rtol
#         if weights is None: weights=self.weights

#         if (weights is None) or (weights is False):
#             A = self.basis(eig)
#             weights = 1
#         else:
#             A = weights*self.basis(eig)

#         if solver == 'gsvd':
#              return mps.nullspace_basis_eval_gsvd(A,self.m_b,mtol,rtol,reg_type)/weights
#         elif solver == 'svd':
#             return mps.nullspace_basis_eval_svd(A,self.m_b,mtol,rtol)/weights
    
#     @cache
#     def eval_eigenbasis_grad(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#         """Evaluates the gradient of the eigenbasis on the collocation points. Returns in complex form, 
#         with the real part being the partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig,mtol,rtol,solver,reg_type,weights)

#         return self.basis.grad(eig)@C

#     # @lru_cache
#     # def outward_normal_derivatives(self,eig,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#     #     if mtol is None: mtol = self.mtols[eig]
#     #     if rtol is None: rtol = self.rtol
#     #     if weights is None: weights=self.weights

#     #     # get basis gradients
#     #     du_dz = self.eval_eigenbasis_grad(eig,mtol,rtol,solver,reg_type,weights)

#     #     return du_dz[:self.m_b]

#     @lru_cache
#     def _gram_tensors(self,eig,n_pts=20,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#         if mtol is None: mtol = self.mtols[eig]
#         if rtol is None: rtol = self.rtol

#         # Gaussian quadrature nodes & weights
#         bdry_pts = boundary_nodes_polygon(self.vertices,n_pts)[0]

#         # outward normal derivatives
#         du_dn = np.abs(self.eigenbasis_grad(eig,mtol,rtol,solver,reg_type,weights)(bdry_pts)).T
#         dU_dn = du_dn[np.newaxis]*du_dn[:,np.newaxis]
#         dU_dn = dU_dn.reshape((*dU_dn.shape[:2],len(self.vertices),-1))

#         # get edge lengths & weighting arrays
#         d = edge_lengths(self.vertices)
#         qnodes, qweights = cached_leggauss(n_pts)
#         f = np.repeat([qnodes],len(self.vertices),axis=0)*d[:,np.newaxis]
#         g = f[:,::-1]
#         wts = np.repeat([qweights],len(self.vertices),axis=0)*d[:,np.newaxis]

#         # compute integrals with quadrature
#         I = (f*dU_dn*wts).sum(axis=-1)/d**2
#         J = (g*dU_dn*wts).sum(axis=-1)/d**2

#         # get X and Y gram tensors
#         dz = polygon_edges(self.vertices)
#         dx,dy = dz.real,dz.imag
#         X = -np.roll(dy,1)*np.roll(I,1,axis=-1) - dy*J
#         Y = np.roll(dx,1)*np.roll(I,1,axis=-1) + dx*J
#         return X,Y

#     def eig_grad(self,eig,dx=None,dy=None,n_pts=20,mtol=None,rtol=None,solver='gsvd',reg_type='svd',weights=None):
#         if mtol is None: mtol = self.mtols[eig]
#         if rtol is None: rtol = self.rtol

#         # catch direction derivative input errors
#         if (dx is None) != (dy is None):
#             raise ValueError('dx and dy must both be set, or left unset')
#         elif dx is not None:
#             dx,dy = np.asarray(dx),np.asarray(dy)
#             n_vert = len(self.vertices)
#             if (dx.shape[0] != n_vert) or (dy.shape[0] != n_vert):
#                 raise ValueError(f'dx and dy must both be length {n_vert}')

#         # compute multiplicity
#         mult = self.spectrum.mults[eig]

#         # catch repeated eigs
#         if (mult > 1) and (dx is None):
#             raise ValueError(f'Repeated eigenvalues only have a directional derivative (mult = {mult})')

#         # compute gram tensors
#         X,Y = self._gram_tensors(eig,n_pts,mtol,rtol,solver,reg_type,weights)

#         # catch multiplicity mismatch
#         if X.shape[1] != mult:
#             raise ValueError(f'Multiplicity mismatch: {mult} != {X.shape[1]}')

#         if dx is None: # simple eigenvalue gradient
#             return X.flatten() + 1j*Y.flatten()
#         else: # directional derivative
#             M = (X@dx)+(Y@dy)
#             if mult==1: # simple eigenvalue
#                 return M[0,0]
#             else: # repeated eigenvalue
#                 return la.eigh(M,eigvals_only=True)
            
#     ### Asymptotics
#     def weyl_N(self,lam):
#         """Two-term Weyl asymptotics for the eigenvalue counting function"""
#         A = self.area
#         P = self.perimiter
#         return (A*lam - P*np.sqrt(lam))/(4*np.pi)

#     def weyl_k(self,k):
#         """Weyl asymptotic estimate for the kth eigenvalue"""
#         A = self.area
#         P = self.perimiter
#         return ((P+np.sqrt(P**2+16*np.pi*A*k))/(2*A))**2
    
#     def adj_weyl_N(self,lam):
#         A = self.area
#         P = self.perimiter
#         K = self.asym_K
#         return (A*lam - P*np.sqrt(lam))/(4*np.pi) + K
    
#     def adj_weyl_k(self,k):
#         A = self.area
#         P = self.perimiter
#         K = self.asym_K
#         return ((P+np.sqrt(P**2+16*np.pi*A*(k-K)))/(2*A))**2
    
#     def weyl_check(self,k):
#         weyl_est = self.adj_weyl_N(self.eigs[:k])
#         true_N = np.arange(1,k+1)
#         deltas = true_N - weyl_est
#         return deltas

#     ### Helper functions for plotting
#     def plot_subspace_sines(self,low,high,nlam,n_angle,ax=None,mps_kwargs={},**kwargs):
#         if low < self.eig1_lb/2 : low = self.eig1_lb/2
#         L = np.linspace(low,high,nlam+1)
#         sines = np.empty((len(L),n_angle))
#         for i,lam in enumerate(L):
#             sines[i] = self.subspace_sines(lam,**mps_kwargs)[:n_angle]
#         if ax is None:
#             fig = plt.figure()
#             plt.plot(L,sines,**kwargs)
#             return fig
#         else:
#             ax.plot(L,sines,**kwargs)

#     def plot_subspace_tans(self,low,high,nlam,n_angle,ax=None,mps_kwargs={},**kwargs):
#         if low < self.eig1_lb/2 : low = self.eig1_lb/2
#         L = np.linspace(low,high,nlam+1)
#         tans = np.empty((len(L),n_angle))
#         for i,lam in enumerate(L):
#             tans[i] = self.subspace_tans(lam,**mps_kwargs)[:n_angle]
#         if ax is None:
#             fig = plt.figure()
#             plt.plot(L,tans,**kwargs)
#             return fig
#         else:
#             ax.plot(L,tans,**kwargs)

#     def _set_eigenplot(self,n=300):
#         # plotting points
#         x = np.linspace(self.vertices.real.min()+1e-16,self.vertices.real.max()-1e-16,n)
#         y = np.linspace(self.vertices.imag.min()+1e-16,self.vertices.imag.max()-1e-16,n)
#         X,Y = np.meshgrid(x,y,indexing='ij')
#         XY = np.array([X.flatten(),Y.flatten()]).T
#         self.plotmask = np.nonzero(self.polygon.contains(points(XY)))[0]
#         self.plotZ = X.flatten()[self.plotmask] + 1j*Y.flatten()[self.plotmask]
#         xe = np.linspace(x.min() - (x[1] - x[0]) / 2, x.max() + (x[1] - x[0]) / 2, n + 1)
#         ye = np.linspace(y.min() - (y[1] - y[0]) / 2, y.max() + (y[1] - y[0]) / 2, n + 1)
#         self.plotX, self.plotY = np.meshgrid(xe, ye,indexing='ij')  # Edge grid
#         self.Zshape = X.shape

#     def plot_eigenfunction(self,eig,axs=None):
#         if not hasattr(self,'plotZ'):
#             self._set_eigenplot()
#         mult = self.mults[eig]
#         U = np.full((np.prod(self.Zshape),mult),np.nan)
#         U[self.plotmask] = self.eigenbasis(eig)(self.plotZ)
#         if mult == 1:
#             if axs:
#                 axs.pcolormesh(self.plotX,self.plotY,U[:,0].reshape(self.Zshape))
#                 plot_polygon(self.vertices,ax=axs,c='k')
#                 axs.set_aspect('equal')
#             else:
#                 fig = plt.figure()
#                 plt.pcolormesh(self.plotX,self.plotY,U[:,0].reshape(self.Zshape))
#                 plot_polygon(self.vertices,ax=plt.gca(),c='k')
#                 plt.gca().set_aspect('equal')
#                 return fig
#         else:
#             if axs:
#                 for i,ax in enumerate(axs):
#                     ax.pcolormesh(self.plotX,self.plotY,U[:,i].reshape(self.Zshape))
#             else:
#                 if mult > 3:
#                     fig,axs = plt.subplots(int(np.ceil(mult/3)),3,figsize=(12,4*int(np.ceil(mult/3))))
#                 else:
#                     fig,axs = plt.subplots(1,mult,figsize=(4*mult,4))
#                 axs = axs.flatten()
#                 for i in range(mult):
#                     axs[i].pcolormesh(self.plotX,self.plotY,U[:,i].reshape(self.Zshape))
#                     axs[i].set_aspect('equal')
#                     plot_polygon(self.vertices,ax=axs[i],c='k')
#                 return fig
            
#     def plot_eigenfunctions(self,eigs):
#         for eig in eigs:
#             fig = self.plot_eigenfunction(eig)
#             fig.suptitle(f"Eigenfunction(s) for $\\lambda = {eig:.3f}$")
#             plt.show()
        
class Eigenproblem:
    """Class for planar eigenproblems. Assumes the Dirichlet problem, solved using MPS."""
    def __init__(self, domain, basis, bdry_pts, int_pts, bdry_nodes=None, int_nodes=None):

        # get domain
        if not isinstance(domain, Domain):
            raise ValueError("'domain' must be an instance of Domain")
        self._domain = domain

        # lower bound on Dirichlet spectrum
        self.eig_lbound = 5.76*np.pi/self._domain.area

        # spectrum tracker
        self.spectrum = Spectrum()

        # basis
        self.basis = basis

        # boundary and interior pointset properties
        # self.allow_update_points = True
        # self.allow_update_nodes = True
        self.bdry_pts = bdry_pts
        self.int_pts = int_pts
        if bdry_nodes is not None or int_nodes is not None:
            if bdry_nodes is not None and int_nodes is not None:
                self.bdry_nodes = bdry_nodes
                self.int_nodes = int_nodes
            else:
                raise ValueError("bdry_nodes and int_nodes must both be provided or both must be None.")

        # tolerances
        self.rtol = mps.rtol_default
        self.etol = mps.etol_default
        self.mtol = mps.mtol_default

    @property
    def domain(self):
        return self._domain
    
    @property
    def eigs(self):
        return self.spectrum.eigs
    
    @property
    def mults(self):
        return self.spectrum.mults
    
    @property
    def eig_seq(self):
        return self.spectrum.eig_seq
    
    # basis for MPS solve. Lets user define a basis if desired, or defaults to automatic basis generation
    @property
    def basis(self):
        return self._basis
        
    @basis.setter
    def basis(self, new_basis):
        if not isinstance(new_basis, ParticularBasis):
            raise TypeError("'new_basis' must be an instance of ParticularBasis")
        self._basis = new_basis
        self.allow_update_basis = False

    @basis.deleter
    def basis(self):
        self._basis = None
        self.allow_update_basis = True

    # def make_default_basis(self, n_eigs=10, etol=mps.etol_default):
    #     """Builds a default basis set for the domain which is good for computing the first n_eigs eigenvalues up to etol
    #     relative accuracy.
        
    #     Since the Eigenproblem class does not assume a particular kind of geometry, this function dispatches to 
    #     make_basis in lappy.bases which handles different kinds of domain geometry.
    #     """
    #     self._basis = make_default_basis(self.domain, n_eigs, etol) # skips property setter to leave allow_update_basis unchanged

    # MPS pointsets for boundary and interior. Lets user define points if desired, defaults to automatic generation
    # bdry_pts and int_pts are used for the "cheap" search for the eigenvalues. Ideally these pointsets are as small as
    # possible to accurately resolve the subspace angle curves.
    @property
    def bdry_pts(self):
        return self._bdry_pts
    
    @bdry_pts.setter
    def bdry_pts(self, new_pts):
        if not isinstance(new_pts, PointSet):
            new_pts = PointSet(new_pts)
        if not np.all(self._domain.bdry_contains(new_pts)):
            raise ValueError("provided points must be on the boundary of the domain")
        else:
            self._bdry_pts = new_pts
            # self.allow_update_points = False

    @bdry_pts.deleter
    def bdry_pts(self):
        del self._bdry_pts
        # if not hasattr(self,"_int_pts"):
            # self.allow_update_points = True

    @property
    def int_pts(self):
        return self._int_pts
    
    @int_pts.setter
    def int_pts(self, new_pts):
        if not isinstance(new_pts, PointSet):
            new_pts = PointSet(new_pts)
        if not np.all(self._domain.contains(new_pts)):
            raise ValueError("provided points must be inside the domain")
        else:
            self._int_pts = new_pts
            # self.allow_update_points = False

    @int_pts.deleter
    def int_pts(self):
        del self._int_pts
        # if not hasattr(self,"_bdry_pts"):
            # self.allow_update_points = True

    # bdry_nodes and int_nodes are used for eigenfunction computation, as well as for eigenperturbation information.
    # They need to have corresponding quadrature weights bdry_wts and int_wts for estimation of integrals.
    # Generally they are higher-resolution sets than bdry_pts and int_pts.
    @property
    def bdry_nodes(self):
        return self._bdry_nodes
    
    @bdry_nodes.setter
    def bdry_nodes(self, new_pts, new_wts=None):
        if not isinstance(new_pts, PointSet):
            new_pts = PointSet(new_pts, new_wts)
        if not np.all(self._domain.bdry_contains(new_pts)):
            raise ValueError("provided points must be on the boundary of the domain")
        else:
            self._bdry_nodes = new_pts
            # self.allow_update_nodes = False

    @bdry_nodes.deleter
    def bdry_nodes(self):
        del self._bdry_nodes
        # if not hasattr(self,"_int_nodes"):
        #     self.allow_update_nodes = True

    @property
    def int_nodes(self):
        return self._int_nodes
    
    @int_nodes.setter
    def int_nodes(self, new_pts, new_wts=None):
        if not isinstance(new_pts, PointSet):
            new_pts = PointSet(new_pts, new_wts)
        if not np.all(self._domain.contains(new_pts)):
            raise ValueError("provided points must be inside the domain")
        else:
            self._int_nodes = new_pts
            # self.allow_update_nodes = False

    @int_nodes.deleter
    def int_nodes(self):
        del self._int_nodes
        # if not hasattr(self,"_bdry_nodes"):
        #     self.allow_update_nodes = True

    ### MPS Functions ###
    def _eval_basis(self, lam, use_nodes=False):
        """Evaluates the eigenproblem basis on the domain's collocation points or quadrature nodes"""
        if use_nodes is True:
            A_B = self._basis._eval_pointset(lam, self._bdry_nodes)
            A_I = self._basis._eval_pointset(lam, self._int_nodes)
            if hasattr(self._bdry_nodes,"_wts"):
                A_B = self._bdry_nodes._sqrt_wts*A_B
            if hasattr(self._int_nodes,"_wts"):
                A_I = self._int_nodes._sqrt_wts*A_I
        # otherwise use collocation points
        else:
            A_B = self._basis._eval_pointset(lam, self._bdry_pts)
            A_I = self._basis._eval_pointset(lam, self._int_pts)
        return A_B, A_I

    # Subspace angles via SVD and GSVD
    @cache
    def subspace_sines(self, lam, rtol=None, use_nodes=False):
        """Compute the sines of subspace angles via SVD of Q_B(lam)"""
        if rtol is None: rtol = self.rtol
        A_B, A_I = self._eval_basis(lam, use_nodes)
        return mps.subspace_sines(A_B, A_I, rtol)
    
    @cache
    def subspace_tans(self, lam, rtol=None, reg_type='svd', use_nodes=False):
        """Compute the tangents of subspace angles via GSVD of {A_B(lam),A_I(lam)}"""
        if rtol is None: rtol = self.rtol
        A_B, A_I = self._eval_basis(lam, use_nodes)
        return mps.subspace_tans(A_B, A_I, rtol, reg_type)
    
    @cache
    def tensions(self, lam, rtol=None, reg_method='svd', use_nodes=False, use_tangents=False):
        """Computes tensions using the GSVD of the pencil {A_B, A_I} or {[A_B; A_T], A_I}, optionally
        with weights."""
        if rtol is None: rtol = self.rtol
        # switch between sparse collocation points and dense quadrature nodes
        if use_nodes:
            bdry_pts = self._bdry_nodes
            int_pts = self._int_nodes
            if use_tangents: bdry_tans = self._bdry_node_tans
        else:
            bdry_pts = self._bdry_pts
            int_pts = self._int_pts
            if use_tangents: bdry_tans = self._bdry_pt_tans

        # evaluate on boundary and interior
        A_B = self._basis._eval_pointset(lam, bdry_pts)
        A_I = self._basis._eval_pointset(lam, int_pts)

        # evaluate boundary tangent derivatives if needed
        if use_tangents:
            dA = self._basis._grad_pointset(lam, bdry_pts)
            A_T = bdry_tans.x*dA.real + bdry_tans.y*dA.imag
        else:
            A_T = None
        
        # set weights
        if hasattr(bdry_pts, 'wts'): w_B = bdry_pts._sqrt_wts
        else: w_B = None
        if hasattr(int_pts, 'wts'): w_I = int_pts._sqrt_wts
        else: w_I = None

        # evaluate MPS tensions
        return mps.tensions(A_B, A_I, A_T, w_B, w_I, rtol, reg_method)
        
    # Eigenbases
    @cache
    def eigenbasis_coef(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
        """Computes the coefficient vectors of an eigenbasis for the given eigenvalue"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # process inputs
        if isinstance(eig, Eigenvalue):
            eig, mult = eig._val, eig._mult
        elif isinstance(eig, (float, np.float64)):
            if eig in self.eig_seq and mult is None:
                mult = self.mults[eig]

        A_B, A_I = self._eval_basis(eig, use_nodes)
        if solver == 'gsvd':
            return mps.eigenbasis_gsvd(A_B, A_I, mult, mtol, rtol, reg_type)
        elif solver == 'svd':
            return mps.eigenbasis_svd(A_B, A_I, mult, mtol, rtol)
        
    def eigenbasis(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
        """Returns a callable function which evaluates the approximate eigenbasis
        corresponding to lam"""
        # process inputs
        if isinstance(eig, Eigenvalue):
            eig, mult = eig._val, eig._mult
        elif isinstance(eig, (float, np.float64)):
            if eig in self.eig_seq and mult is None:
                mult = self.mults[eig]

        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, use_nodes)

        # return callable function for the basis
        def func(pts):
            if isinstance(pts, PointSet):
                shape = (len(pts), C.shape[1])
            else:
                pts = np.asarray(pts)
                if pts.dtype == 'complex128':
                    shape = (*pts.shape,  C.shape[1])
                    pts = pts.flatten()
                elif pts.dtype == 'float64':
                    shape = (*pts.shape[:-1],  C.shape[1])
                    pts = pts.reshape((-1,2))
                pts = PointSet(complex_form(pts))
            return (self._basis._eval_pointset(eig, pts)@C).reshape(shape)
        return func

    def eigenbasis_grad(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
        """Returns a callable function which evaluates the approximate eigenbasis gradients (in x and y)
        corresponding to eig. Returns in complex form, with the real part being the 
        partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
        # process inputs
        if isinstance(eig, Eigenvalue):
            eig, mult = eig._val, eig._mult
        elif isinstance(eig, (float, np.float64)):
            if eig in self.eig_seq and mult is None:
                mult = self.mults[eig]

        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, use_nodes)

        # return callable function from the basis
        def func(pts):
            if isinstance(pts, PointSet):
                shape = (len(pts), C.shape[1])
            else:
                pts = np.asarray(pts)
                if pts.dtype == 'complex128':
                    shape = (*pts.shape,  C.shape[1])
                    pts = pts.flatten()
                elif pts.dtype == 'float64':
                    shape = (*pts.shape[:-1],  C.shape[1])
                    pts = pts.reshape((-1,2))
                pts = PointSet(complex_form(pts))
            return (self._basis._grad_pointset(eig, pts)@C).reshape(shape)
        return func
    
    def eigenbasis_normals(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
        """Returns a callable function which evaluates the approximate eigenbasis outward-normal derivatives
        along the boundary of the domain"""

        # process inputs
        if isinstance(eig, Eigenvalue):
            eig, mult = eig._val, eig._mult
        elif isinstance(eig, (float, np.float64)):
            if eig in self.eig_seq and mult is None:
                mult = self.mults[eig]

        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, use_nodes)

        # return callable function from the basis
        def func(pts):
            if isinstance(pts, PointSet):
                shape = (len(pts), C.shape[1])
            else:
                pts = np.asarray(pts)
                if pts.dtype == 'complex128':
                    shape = (*pts.shape,  C.shape[1])
                    pts = pts.flatten()
                elif pts.dtype == 'float64':
                    shape = (*pts.shape[:-1],  C.shape[1])
                    pts = pts.reshape((-1,2))
                pts = PointSet(complex_form(pts))
            normals = self._domain.bdry_normals(pts)[:,np.newaxis]
            grad = self._basis.grad(eig, pts)@C
            return (grad.real*normals.real + grad.imag*normals.imag).reshape(shape)
        return func
    
    @lru_cache
    def eigenbasis_eval(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', on_nodes=True):
        """Evaluate the eigenbasis on the collocation points or the quadrature nodes"""
        if rtol is None: rtol = self.rtol
        if mtol is None: mtol = self.mtol
        # process inputs
        if isinstance(eig, Eigenvalue):
            eig, mult = eig._val, eig._mult
        elif isinstance(eig, (float, np.float64)):
            if eig in self.eig_seq and mult is None:
                mult = self.mults[eig]

        A_B, A_I = self._eval_basis(eig, on_nodes)

        if solver == 'gsvd':
            U_B, U_I = mps.eigenbasis_eval_gsvd(A_B, A_I, mult, mtol, rtol, reg_type)
        elif solver == 'svd':
            U_B, U_I = mps.eigenbasis_eval_svd(A_B, A_I, mult, mtol, rtol)

        if on_nodes:
            if hasattr(self._bdry_nodes, "_wts"):
                U_B = U_B/self._bdry_nodes._sqrt_wts
            if hasattr(self._int_nodes, "_wts"):
                U_I = U_I/self._int_nodes._sqrt_wts

        return U_B, U_I
        
    @lru_cache
    def eigenbasis_grad_eval(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', on_nodes=True):
        """Evaluates the gradient of the eigenbasis on the collocation points or quadrature nodes. Returns in complex form, 
        with the real part being the partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
        # process inputs
        if isinstance(eig, Eigenvalue):
            eig, mult = eig._val, eig._mult
        elif isinstance(eig, (float, np.float64)):
            if eig in self.eig_seq and mult is None:
                mult = self.mults[eig]

        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, on_nodes)

        if on_nodes:
            return self._basis._grad_pointset(eig, self._bdry_nodes)@C, self._basis.grad(eig, self._int_nodes)@C
        else:
            return self._basis._grad_pointset(eig, self._bdry_pts)@C, self._basis.grad(eig, self._int_pts)@C
        
    @lru_cache
    def eigenbasis_normals_eval(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', on_nodes=True):
        """Evaluates the outward-normal derivatives of the eigenbasis on the collocation points or quadrature nodes."""
        # process inputs
        if isinstance(eig, Eigenvalue):
            eig, mult = eig._val, eig._mult
        elif isinstance(eig, (float, np.float64)):
            if eig in self.eig_seq and mult is None:
                mult = self.mults[eig]

        # get eigenbasis coefficient matrix
        C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, on_nodes)

        if on_nodes:
            grad = self._basis._grad_pointset(eig, self._bdry_nodes)@C
            normals = self._domain.bdry_normals(self._bdry_nodes)[:,np.newaxis]
        else:
            grad = self._basis._grad_pointset(eig, self._bdry_pts)@C
            normals = self._domain.bdry_normals(self._bdry_pts)[:,np.newaxis]
        return grad.real*normals.real + grad.imag*normals.imag
    
    ### Eigensolvers
    def solve_eigs_interval(self, a, b, n_pts, ftol=None, xtol=None, mps_kwargs={}, use_nodes=False, verbose=0):
        if ftol is None: ftol = self.mtol
        if xtol is None: xtol = self.etol

        if a < self.eig_lbound: a = self.eig_lbound/2

        f = lambda lam: self.subspace_tans(lam, use_nodes=use_nodes, **mps_kwargs)
        eigs, fevals = mps.solve_eigs_interval(lambda lam: f(lam)[:2], a, b, n_pts, ftol, xtol, verbose)
        mults = []
        for eig in eigs:
            mult = mps.estimate_multiplicity(f, eig-2*xtol, eig+2*xtol, ftol, verbose=bool(verbose))
            mults.append(mult)
            fevals += 5

        return eigs, np.array(mults), fevals
    
    def solve_eigs_ordered(self, k, ftol=None, xtol=None, mps_kwargs={}, use_nodes=False, verbose=0):
        """solves for the first k eigenvalues"""
        a = 0.9*self.eig_lbound
        b = self.weyl_k(k+5) # conservative estimate
        n_pts = 10*(k+5)
        eigs, mults, fevals = self.solve_eigs_interval(a, b, n_pts, ftol=ftol, xtol=xtol, mps_kwargs=mps_kwargs, 
                                               use_nodes=use_nodes, verbose=verbose)
        for eig, mult in zip(eigs, mults):
            try:
                self.spectrum.add_eig(Eigenvalue(eig, mult))
            except:
                pass

        return self.eig_seq[:k]

    ### Helper functions for plotting
    def plot_subspace_sines(self, low, high, nlam, n_angle, ax=None, mps_kwargs={}, **plot_kwargs):
        if low < self.eig_lbound/2 : low = self.eig_lbound/2
        L = np.linspace(low, high, nlam+1)
        sines = np.array([self.subspace_sines(lam, **mps_kwargs)[:n_angle] for lam in L])
        if ax is None:
            fig = plt.figure()
            plt.plot(L, sines, **plot_kwargs)
            return fig
        else:
            ax.plot(L, sines, **plot_kwargs)

    def plot_subspace_tans(self, low, high, nlam, n_angle, ax=None, mps_kwargs={}, **plot_kwargs):
        if low < self.eig_lbound/2 : low = self.eig_lbound/2
        L = np.linspace(low, high, nlam+1)
        tans = np.array([self.subspace_tans(lam, **mps_kwargs)[:n_angle] for lam in L])
        if ax is None:
            fig = plt.figure()
            plt.plot(L, tans, **plot_kwargs)
            return fig
        else:
            ax.plot(L, tans, **plot_kwargs)
            ax.set_xlim(low,high)

    ### Asymptotics
    def weyl_N(self,lam):
        """Two-term Weyl asymptotics for the eigenvalue counting function"""
        A = self.domain.area
        P = self.domain.perimeter
        return (A*lam - P*np.sqrt(lam))/(4*np.pi)

    def weyl_k(self,k):
        """Weyl asymptotic estimate for the kth eigenvalue"""
        A = self.domain.area
        P = self.domain.perimeter
        return ((P+np.sqrt(P**2+16*np.pi*A*k))/(2*A))**2
            
class Eigenvalue:
    def __init__(self, val, mult=1):
        self._val = float(val)
        self._mult = int(mult)

    @property
    def val(self):
        return self._val
    
    @property
    def mult(self):
        return self._mult
    
    def __eq__(self, other):
        if type(other) is Eigenvalue:
            return (self._val == other._val) and (self._mult == other._mult)
        else:
            return False
        
    def __hash__(self):
        return hash((self._val, self._mult))

    def __repr__(self):
        return f"Eigenvalue({self._val},mult={self._mult})"

class Spectrum:
    """Class for tracking the Laplacian spectrum of a domain"""
    def __init__(self, eigs=None, mults=None):
        self._eigs = set()
        if eigs is not None:
            if mults is None:
                for eig in eigs:
                    self.add_eig(eig)
            elif len(eigs) == len(mults):
                for eig, mult in zip(eigs, mults):
                    self.add_eig(eig, mult)
            else:
                raise ValueError("'eigs' and 'mults' must have the same length")

    def add_eig(self, eig, mult=None):
        if isinstance(eig, Eigenvalue):
            if eig in self._eigs:
                raise ValueError(f"{eig} already in spectrum.")
            else:
                self._eigs.add(eig)
        else:
            if eig in self.eig_seq:
                raise ValueError(f"{eig} already in spectrum (with mult = {self.mults[eig]})")
            else:
                self._eigs.add(Eigenvalue(eig, mult))

    def remove_eig(self, eig, mult=None):
        if isinstance(eig, Eigenvalue):
            self._eigs.remove(eig)
        else:
            if mult is None:
                mult = self.mults[eig]
            self._eigs.remove(Eigenvalue(eig,mult))

    @property
    def eigs(self):
        return frozenset(self._eigs)

    @property
    def sorted(self):
        return sorted(self._eigs,key=lambda eig: eig.val)

    @property
    def eig_seq(self):
        vals = np.concatenate(([],*(eig._mult*[eig._val] for eig in self._eigs)))
        return np.sort(vals.flatten())

    @property
    def mults(self):
        return {eig.val:eig.mult for eig in self._eigs}

    def __str__(self):
        return f"Spectrum:{str(self._eigs)}"

    def __eq__(self, other):
        if isinstance(other, Spectrum):
            return self._eigs == other._eigs
        else:
            return False
        
    def __add__(self, other):
        if isinstance(other, Eigenvalue):
            self.add_eig(other)
            return self
        elif isinstance(other, Spectrum):
            return Spectrum(self._eigs.union(other._eigs))