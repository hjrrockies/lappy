import numpy as np
import scipy.linalg as la
from .core import BaseEigenproblem, BaseEigensolver
from .mps import MPSEigensolver
from .utils import complex_form, dir_lbound, dir_weyl_N, dir_weyl_k
from .bases import ParticularBasis
from .geometry import PointSet, Domain
from functools import cache, lru_cache
import matplotlib.pyplot as plt

# eigenproblem class
class Eigenproblem(BaseEigenproblem):
    """Class for planar dirichlet laplacian eigenproblems."""
    def __init__(self, domain, eval_solver=None, evec_solver=None):
        super().__init__(domain)
        self.eval_solver = eval_solver
        self.evec_solver = evec_solver

    @property
    def eval_solver(self):
        return self._eval_solver
    
    @eval_solver.setter
    def eval_solver(self, solver):
        if isinstance(solver, BaseEigensolver) or solver is None:
            self._eval_solver = solver
        elif solver == 'mps':
            self._eval_solver = MPSEigensolver.from_domain(self.domain)
        else:
            raise TypeError("'solver' must be a valid Eigensolver or None")

    @eval_solver.deleter
    def eval_solver(self):
        del self._eval_solver
        
    @property
    def evec_solver(self):
        return self._evec_solver
    
    @evec_solver.setter
    def evec_solver(self, solver):
        if isinstance(solver, BaseEigensolver) or solver is None:
            self._evec_solver = solver
        elif solver == 'mps':
            self._evec_solver = MPSEigensolver.from_domain(self.domain, weights=True)
        else:
            raise TypeError("'solver' must be a valid Eigensolver or None")
        
    @evec_solver.deleter
    def evec_solver(self):
        del self._evec_solver

    def _get_eval_solver(self, solver):
        if solver is not None:
            if not isinstance(solver, BaseEigensolver):
                raise TypeError("'solver' must be a valid Eigensolver")
        elif self.eval_solver is not None:
            solver = self.eval_solver
        else:
            raise ValueError("eigenproblem has no eval_solver")
        return solver

    def _get_evec_solver(self, solver):
        if solver is not None:
            if not isinstance(solver, BaseEigensolver):
                raise TypeError("'solver' must be a valid Eigensolver")
        elif self.evec_solver is not None:
            solver = self.evec_solver
        else:
            raise ValueError("eigenproblem has no evec_solver")
        return solver
        
    def solve(self, n_eigs, solver=None, **solver_kwargs):
        """Solves for the first n_eigs (counting multiplicities) eigenvalues of the domain."""
        # dirichlet case
        if self.bc_type == 'dir':
            return self._solve_dirichlet(n_eigs, solver, **solver_kwargs)
        elif self.bc_type == 'neu':
            return self._solve_neumann(n_eigs, solver, **solver_kwargs)
        else:
            raise NotImplementedError("'solve' not implemented for mixed or robin boundary conditions")

    def _solve_dirichlet(self, n_eigs, solver=None, **solver_kwargs):
        ppl = 10
        # first pass using weyl bound for n_eigs+1
        k = n_eigs+1
        a = dir_lbound(self.domain.area)
        b = dir_weyl_k(k, self.domain.area, self.domain.perimeter)
        eigs, mults, fevals = self.solve_interval(a, b, ppl*k, solver, **solver_kwargs)
        while mults.sum() < n_eigs:
            a = dir_weyl_k(k, self.domain.area, self.domain.perimeter)
            b = dir_weyl_k(k+1, self.domain.area, self.domain.perimeter)
            eigs_, mults_, fe = self.solve_interval(a, b, ppl, solver, **solver_kwargs)
            eigs = np.concatenate((eigs, eigs_))
            mults = np.concatenate((mults, mults_))
            fevals += fe
            k += 1
        mults = mults.astype('int')
        vals = np.concatenate(([],*(mult*[eig] for eig, mult in zip(eigs, mults))))
        return np.sort(vals.flatten())[:n_eigs], fevals

    def _solve_neumann(self, n_eigs, solver, **solver_kwargs):
        raise NotImplementedError

    def solve_interval(self, a, b, n_pts, solver=None, **solver_kwargs):
        """Solves for all eigenvalues in the interval [a,b] using the specified solver."""
        solver = self._get_eval_solver(solver)
        return solver.solve_interval(a, b, n_pts, **solver_kwargs)
        
    def eigenfunction(self, eig, mult=1, solver=None, **solver_kwargs):
        solver = self._get_evec_solver(solver)
        return solver.eigenfunction(eig, mult, **solver_kwargs)
        
    def eigenfunction_grad(self, eig, mult=1, solver=None, **solver_kwargs):
        solver = self._get_evec_solver(solver)
        return solver.eigenfunction_grad(eig, mult, **solver_kwargs)
        
    def eigenfunction_coef(self, eig, mult=1, solver=None, **solver_kwargs):
        solver = self._get_evec_solver(solver)
        return solver.eigenfunction_coef(eig, mult, **solver_kwargs)
        
    def eigenfunction_eval(self, eig, mult=1, solver=None, **solver_kwargs):
        solver = self._get_evec_solver(solver)
        return solver.eigenfunction_eval(eig, mult, **solver_kwargs)
        
    def eigenfunction_eval_extras(self, eig, mult=1, extra_pts=None, solver=None, **solver_kwargs):
        solver = self._get_evec_solver(solver)
        return solver.eigenfunction_eval_extras(eig, mult, extra_pts, **solver_kwargs)
        
    def eigenfunction_eval_normals(self, eig, mult=1, solver=None, **solver_kwargs):
        solver = self._get_evec_solver(solver)
        return solver.eigenfunction_eval_normals(eig, mult, **solver_kwargs)

# class Eigenproblem:
#     """Class for planar eigenproblems. Assumes the Dirichlet problem, solved using MPS."""
#     def __init__(self, domain, basis, bdry_pts=None, int_pts=None, bdry_nodes=None, int_nodes=None):

#         # get domain
#         if not isinstance(domain, Domain):
#             raise ValueError("'domain' must be an instance of Domain")
#         self._domain = domain

#         # lower bound on Dirichlet spectrum
#         self.eig_lbound = 5.76*np.pi/self._domain.area

#         # spectrum tracker
#         self.spectrum = Spectrum()

#         # basis
#         self.basis = basis

#         # boundary and interior pointset properties
#         # self.allow_update_points = True
#         # self.allow_update_nodes = True
#         self.bdry_pts = bdry_pts
#         self.int_pts = int_pts
#         if bdry_nodes is not None or int_nodes is not None:
#             if bdry_nodes is not None and int_nodes is not None:
#                 self.bdry_nodes = bdry_nodes
#                 self.int_nodes = int_nodes
#             else:
#                 raise ValueError("bdry_nodes and int_nodes must both be provided or both must be None.")

#         # tolerances
#         self.rtol = rtol_default
#         self.ltol = ltol_default
#         self.ttol = ttol_default

#     @property
#     def domain(self):
#         return self._domain
    
#     @property
#     def eigs(self):
#         return self.spectrum.eigs
    
#     @property
#     def mults(self):
#         return self.spectrum.mults
    
#     @property
#     def eig_seq(self):
#         return self.spectrum.eig_seq

#     @staticmethod
#     def make_mps_pencil(basis, bdry_pts, int_pts, bdry_tangents=None, bdry_normals=None):
#         """Makes the pencil for the MPS problem for eigenvalues and eigenfunctions. Optionally uses
#         boundary """
#         if not isinstance(basis, ParticularBasis):
#             raise TypeError("'basis' must be an instance of ParticularBasis")
        
#         # boundary and interior points
#         if hasattr(bdry_pts, '_wts'):
#             def A_B(lam): return bdry_pts._sqrt_wts*basis._eval_pointset(lam, bdry_pts)
#         else:
#             def A_B(lam): return basis._eval_pointset(lam, bdry_pts)
#         if hasattr(int_pts, '_wts'):
#             def A_I(lam): return int_pts._sqrt_wts*basis._eval_pointset(lam, int_pts)
#         else:
#             def A_I(lam): return basis._eval_pointset(lam, int_pts)

#         # handle boundary tangents
#         if bdry_tangents is not None:
#             if hasattr(bdry_pts,'_wts'):
#                 def A_T(lam):
#                     dA = basis._grad_pointset(lam, bdry_pts)
#                     return bdry_pts._sqrt_wts*(bdry_tangents.x*dA.real + bdry_tangents.y*dA.imag)
#             else:
#                 def A_T(lam):
#                     dA = basis._grad_pointset(lam, bdry_pts)
#                     return bdry_tangents.x*dA.real + bdry_tangents.y*dA.imag
#             def A1(lam): return np.vstack((A_B(lam), A_T(lam)))
#         else: A1 = A_B

#         # handle boundary normals
#         if bdry_normals is not None:
#             if hasattr(bdry_pts,'_wts'):
#                 def A_N(lam):
#                     dA = basis._grad_pointset(lam, bdry_pts)
#                     return bdry_pts._sqrt_wts*(bdry_normals.x*dA.real + bdry_normals.y*dA.imag)
#             else:
#                 def A_N(lam):
#                     dA = basis._grad_pointset(lam, bdry_pts)
#                     return bdry_normals.x*dA.real + bdry_normals.y*dA.imag
#             def A2(lam): return np.vstack((A_I(lam), A_N(lam)))
#         else: A2 = A_I

#         # return pencil
#         return A1, A2

    
#     # basis for MPS solve. Lets user define a basis if desired, or defaults to automatic basis generation
#     @property
#     def basis(self):
#         return self._basis
        
#     @basis.setter
#     def basis(self, new_basis):
#         if not isinstance(new_basis, ParticularBasis):
#             raise TypeError("'new_basis' must be an instance of ParticularBasis")
#         self._basis = new_basis
#         self.allow_update_basis = False

#     @basis.deleter
#     def basis(self):
#         self._basis = None
#         self.allow_update_basis = True

#     # def make_default_basis(self, n_eigs=10, etol=mps.etol_default):
#     #     """Builds a default basis set for the domain which is good for computing the first n_eigs eigenvalues up to etol
#     #     relative accuracy.
        
#     #     Since the Eigenproblem class does not assume a particular kind of geometry, this function dispatches to 
#     #     make_basis in lappy.bases which handles different kinds of domain geometry.
#     #     """
#     #     self._basis = make_default_basis(self.domain, n_eigs, etol) # skips property setter to leave allow_update_basis unchanged

#     # MPS pointsets for boundary and interior. Lets user define points if desired, defaults to automatic generation
#     # bdry_pts and int_pts are used for the "cheap" search for the eigenvalues. Ideally these pointsets are as small as
#     # possible to accurately resolve the subspace angle curves.
#     @property
#     def bdry_pts(self):
#         return self._bdry_pts
    
#     @bdry_pts.setter
#     def bdry_pts(self, new_pts):
#         if not isinstance(new_pts, PointSet):
#             new_pts = PointSet(new_pts)
#         if not np.all(self._domain.bdry_contains(new_pts)):
#             raise ValueError("provided points must be on the boundary of the domain")
#         else:
#             self._bdry_pts = new_pts
#             # self.allow_update_points = False

#     @bdry_pts.deleter
#     def bdry_pts(self):
#         del self._bdry_pts
#         # if not hasattr(self,"_int_pts"):
#             # self.allow_update_points = True

#     @property
#     def int_pts(self):
#         return self._int_pts
    
#     @int_pts.setter
#     def int_pts(self, new_pts):
#         if not isinstance(new_pts, PointSet):
#             new_pts = PointSet(new_pts)
#         if not np.all(self._domain.contains(new_pts)):
#             raise ValueError("provided points must be inside the domain")
#         else:
#             self._int_pts = new_pts
#             # self.allow_update_points = False

#     @int_pts.deleter
#     def int_pts(self):
#         del self._int_pts
#         # if not hasattr(self,"_bdry_pts"):
#             # self.allow_update_points = True

#     # bdry_nodes and int_nodes are used for eigenfunction computation, as well as for eigenperturbation information.
#     # They need to have corresponding quadrature weights bdry_wts and int_wts for estimation of integrals.
#     # Generally they are higher-resolution sets than bdry_pts and int_pts.
#     @property
#     def bdry_nodes(self):
#         return self._bdry_nodes
    
#     @bdry_nodes.setter
#     def bdry_nodes(self, new_pts, new_wts=None):
#         if not isinstance(new_pts, PointSet):
#             new_pts = PointSet(new_pts, new_wts)
#         if not np.all(self._domain.bdry_contains(new_pts)):
#             raise ValueError("provided points must be on the boundary of the domain")
#         else:
#             self._bdry_nodes = new_pts
#             # self.allow_update_nodes = False

#     @bdry_nodes.deleter
#     def bdry_nodes(self):
#         del self._bdry_nodes
#         # if not hasattr(self,"_int_nodes"):
#         #     self.allow_update_nodes = True

#     @property
#     def int_nodes(self):
#         return self._int_nodes
    
#     @int_nodes.setter
#     def int_nodes(self, new_pts, new_wts=None):
#         if not isinstance(new_pts, PointSet):
#             new_pts = PointSet(new_pts, new_wts)
#         if not np.all(self._domain.contains(new_pts)):
#             raise ValueError("provided points must be inside the domain")
#         else:
#             self._int_nodes = new_pts
#             # self.allow_update_nodes = False

#     @int_nodes.deleter
#     def int_nodes(self):
#         del self._int_nodes
#         # if not hasattr(self,"_bdry_nodes"):
#         #     self.allow_update_nodes = True

#     ### MPS Functions ###
#     @cache
#     def _eval_basis(self, lam, use_nodes=False, use_tangents=False, use_normals=False):
#         """todo"""
#         # switch between sparse collocation points and dense quadrature nodes
#         if use_nodes:
#             bdry_pts = self._bdry_nodes
#             int_pts = self._int_nodes
#             if use_tangents: bdry_tans = self._bdry_node_tans
#             if use_normals: bdry_norms = self._bdry_node_norms
#         else:
#             bdry_pts = self._bdry_pts
#             int_pts = self._int_pts
#             if use_tangents: bdry_tans = self._bdry_pt_tans
#             if use_normals: bdry_norms = self._bdry_pt_norms

#         # evaluate on boundary and interior
#         A_B = self._basis._eval_pointset(lam, bdry_pts)
#         A_I = self._basis._eval_pointset(lam, int_pts)

#         # evaluate boundary tangent derivatives if needed
#         if use_tangents or use_normals:
#             dA = self._basis._grad_pointset(lam, bdry_pts)
#         if use_tangents:
#             A_T = bdry_tans.x*dA.real + bdry_tans.y*dA.imag
#         else:
#             A_T = None
#         if use_normals:
#             A_N = bdry_norms.x*dA.real + bdry_norms.y*dA.imag
#         else:
#             A_N = None
        
#         # set weights
#         if hasattr(bdry_pts, 'wts'): w_B = bdry_pts._sqrt_wts
#         else: w_B = None
#         if hasattr(int_pts, 'wts'): w_I = int_pts._sqrt_wts
#         else: w_I = None

#         return A_B, A_I, A_T, A_N, w_B, w_I
    
#     @cache
#     def tensions(self, lam, rtol=None, reg_method='svd', use_nodes=False, use_tangents=False):
#         """Computes tensions using the GSVD of the pencil {A_B, A_I} or {[A_B; A_T], A_I}, optionally
#         with weights."""
#         if rtol is None: rtol = self.rtol
#         A_B, A_I, A_T, _, w_B, w_I = self._eval_basis(lam, use_nodes, use_tangents)

#         # evaluate MPS tensions
#         return tensions(A_B, A_I, A_T, w_B, w_I, rtol, reg_method)
    
#     @lru_cache
#     def eigenbasis_coef(self, eig, mult=None, ttol=None, rtol=None, 
#                         reg_method='svd', use_nodes=False, use_tangents=False):
#         """todo"""
#         if rtol is None: rtol = self.rtol
#         if ttol is None: ttol = self.ttol
#         A_B, A_I, A_T, _, w_B, w_I = self._eval_basis(eig, use_nodes, use_tangents)

#         # compute eigenbasis coef
#         return eigenbasis_coef(A_B, A_I, A_T, w_B, w_I, mult, ttol, rtol, reg_method)

#     @lru_cache
#     def eigenbasis_eval(self, eig, mult=None, ttol=None, rtol=None, reg_method='svd', 
#                         use_nodes=True, use_tangents=False, use_normals=False):
#         """todo"""
#         if rtol is None: rtol = self.rtol
#         if ttol is None: ttol = self.ttol
#         A_B, A_I, A_T, A_N, w_B, w_I = self._eval_basis(eig, use_nodes, use_tangents, use_normals)

#         # evaluate eigenbasis
#         return eigenbasis_eval(A_B, A_I, A_T, A_N, w_B, w_I, mult, ttol, rtol, reg_method)
    
#     def eigenbasis(self, eig, mult=None, ttol=None, rtol=None, reg_method='svd', 
#                         use_nodes=False, use_tangents=False):
#         """todo"""
#         if rtol is None: rtol = self.rtol
#         if ttol is None: ttol = self.ttol
#         C = self.eigenbasis_coef(eig, mult, ttol, rtol, reg_method, use_nodes, use_tangents)

#         def func(pts):
#             if isinstance(pts, PointSet):
#                 shape = (len(pts), C.shape[1])
#             else:
#                 pts = np.asarray(pts)
#                 if pts.dtype == 'complex128':
#                     shape = (*pts.shape,  C.shape[1])
#                     pts = pts.flatten()
#                 elif pts.dtype == 'float64':
#                     shape = (*pts.shape[:-1],  C.shape[1])
#                     pts = pts.reshape((-1,2))
#                 pts = PointSet(complex_form(pts))
#             return (self._basis._eval_pointset(eig, pts)@C).reshape(shape)
#         return func



#     def _eval_basis(self, lam, use_nodes=False):
#         """Evaluates the eigenproblem basis on the domain's collocation points or quadrature nodes"""
#         if use_nodes is True:
#             A_B = self._basis._eval_pointset(lam, self._bdry_nodes)
#             A_I = self._basis._eval_pointset(lam, self._int_nodes)
#             if hasattr(self._bdry_nodes,"_wts"):
#                 A_B = self._bdry_nodes._sqrt_wts*A_B
#             if hasattr(self._int_nodes,"_wts"):
#                 A_I = self._int_nodes._sqrt_wts*A_I
#         # otherwise use collocation points
#         else:
#             A_B = self._basis._eval_pointset(lam, self._bdry_pts)
#             A_I = self._basis._eval_pointset(lam, self._int_pts)
#         return A_B, A_I

#     # Subspace angles via SVD and GSVD
#     @cache
#     def subspace_sines(self, lam, rtol=None, use_nodes=False):
#         """Compute the sines of subspace angles via SVD of Q_B(lam)"""
#         if rtol is None: rtol = self.rtol
#         A_B, A_I = self._eval_basis(lam, use_nodes)
#         return mps.subspace_sines(A_B, A_I, rtol)
    
#     @cache
#     def subspace_tans(self, lam, rtol=None, reg_type='svd', use_nodes=False):
#         """Compute the tangents of subspace angles via GSVD of {A_B(lam),A_I(lam)}"""
#         if rtol is None: rtol = self.rtol
#         A_B, A_I = self._eval_basis(lam, use_nodes)
#         return mps.subspace_tans(A_B, A_I, rtol, reg_type)
        
#     # Eigenbases
#     @cache
#     def eigenbasis_coef(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
#         """Computes the coefficient vectors of an eigenbasis for the given eigenvalue"""
#         if rtol is None: rtol = self.rtol
#         if mtol is None: mtol = self.mtol
#         # process inputs
#         if isinstance(eig, Eigenvalue):
#             eig, mult = eig._val, eig._mult
#         elif isinstance(eig, (float, np.float64)):
#             if eig in self.eig_seq and mult is None:
#                 mult = self.mults[eig]

#         A_B, A_I = self._eval_basis(eig, use_nodes)
#         if solver == 'gsvd':
#             return mps.eigenbasis_gsvd(A_B, A_I, mult, mtol, rtol, reg_type)
#         elif solver == 'svd':
#             return mps.eigenbasis_svd(A_B, A_I, mult, mtol, rtol)
        
#     def eigenbasis(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
#         """Returns a callable function which evaluates the approximate eigenbasis
#         corresponding to lam"""
#         # process inputs
#         if isinstance(eig, Eigenvalue):
#             eig, mult = eig._val, eig._mult
#         elif isinstance(eig, (float, np.float64)):
#             if eig in self.eig_seq and mult is None:
#                 mult = self.mults[eig]

#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, use_nodes)

#         # return callable function for the basis
#         def func(pts):
#             if isinstance(pts, PointSet):
#                 shape = (len(pts), C.shape[1])
#             else:
#                 pts = np.asarray(pts)
#                 if pts.dtype == 'complex128':
#                     shape = (*pts.shape,  C.shape[1])
#                     pts = pts.flatten()
#                 elif pts.dtype == 'float64':
#                     shape = (*pts.shape[:-1],  C.shape[1])
#                     pts = pts.reshape((-1,2))
#                 pts = PointSet(complex_form(pts))
#             return (self._basis._eval_pointset(eig, pts)@C).reshape(shape)
#         return func

#     def eigenbasis_grad(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
#         """Returns a callable function which evaluates the approximate eigenbasis gradients (in x and y)
#         corresponding to eig. Returns in complex form, with the real part being the 
#         partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
#         # process inputs
#         if isinstance(eig, Eigenvalue):
#             eig, mult = eig._val, eig._mult
#         elif isinstance(eig, (float, np.float64)):
#             if eig in self.eig_seq and mult is None:
#                 mult = self.mults[eig]

#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, use_nodes)

#         # return callable function from the basis
#         def func(pts):
#             if isinstance(pts, PointSet):
#                 shape = (len(pts), C.shape[1])
#             else:
#                 pts = np.asarray(pts)
#                 if pts.dtype == 'complex128':
#                     shape = (*pts.shape,  C.shape[1])
#                     pts = pts.flatten()
#                 elif pts.dtype == 'float64':
#                     shape = (*pts.shape[:-1],  C.shape[1])
#                     pts = pts.reshape((-1,2))
#                 pts = PointSet(complex_form(pts))
#             return (self._basis._grad_pointset(eig, pts)@C).reshape(shape)
#         return func
    
#     def eigenbasis_normals(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', use_nodes=True):
#         """Returns a callable function which evaluates the approximate eigenbasis outward-normal derivatives
#         along the boundary of the domain"""

#         # process inputs
#         if isinstance(eig, Eigenvalue):
#             eig, mult = eig._val, eig._mult
#         elif isinstance(eig, (float, np.float64)):
#             if eig in self.eig_seq and mult is None:
#                 mult = self.mults[eig]

#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, use_nodes)

#         # return callable function from the basis
#         def func(pts):
#             if isinstance(pts, PointSet):
#                 shape = (len(pts), C.shape[1])
#             else:
#                 pts = np.asarray(pts)
#                 if pts.dtype == 'complex128':
#                     shape = (*pts.shape,  C.shape[1])
#                     pts = pts.flatten()
#                 elif pts.dtype == 'float64':
#                     shape = (*pts.shape[:-1],  C.shape[1])
#                     pts = pts.reshape((-1,2))
#                 pts = PointSet(complex_form(pts))
#             normals = self._domain.bdry_normals(pts)[:,np.newaxis]
#             grad = self._basis.grad(eig, pts)@C
#             return (grad.real*normals.real + grad.imag*normals.imag).reshape(shape)
#         return func
    
#     @lru_cache
#     def eigenbasis_eval(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', on_nodes=True):
#         """Evaluate the eigenbasis on the collocation points or the quadrature nodes"""
#         if rtol is None: rtol = self.rtol
#         if mtol is None: mtol = self.mtol
#         # process inputs
#         if isinstance(eig, Eigenvalue):
#             eig, mult = eig._val, eig._mult
#         elif isinstance(eig, (float, np.float64)):
#             if eig in self.eig_seq and mult is None:
#                 mult = self.mults[eig]

#         A_B, A_I = self._eval_basis(eig, on_nodes)

#         if solver == 'gsvd':
#             U_B, U_I = mps.eigenbasis_eval_gsvd(A_B, A_I, mult, mtol, rtol, reg_type)
#         elif solver == 'svd':
#             U_B, U_I = mps.eigenbasis_eval_svd(A_B, A_I, mult, mtol, rtol)

#         if on_nodes:
#             if hasattr(self._bdry_nodes, "_wts"):
#                 U_B = U_B/self._bdry_nodes._sqrt_wts
#             if hasattr(self._int_nodes, "_wts"):
#                 U_I = U_I/self._int_nodes._sqrt_wts

#         return U_B, U_I
        
#     @lru_cache
#     def eigenbasis_grad_eval(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', on_nodes=True):
#         """Evaluates the gradient of the eigenbasis on the collocation points or quadrature nodes. Returns in complex form, 
#         with the real part being the partials w.r.t. x, and the imaginary part being the partials w.r.t. to y"""
#         # process inputs
#         if isinstance(eig, Eigenvalue):
#             eig, mult = eig._val, eig._mult
#         elif isinstance(eig, (float, np.float64)):
#             if eig in self.eig_seq and mult is None:
#                 mult = self.mults[eig]

#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, on_nodes)

#         if on_nodes:
#             return self._basis._grad_pointset(eig, self._bdry_nodes)@C, self._basis.grad(eig, self._int_nodes)@C
#         else:
#             return self._basis._grad_pointset(eig, self._bdry_pts)@C, self._basis.grad(eig, self._int_pts)@C
        
#     @lru_cache
#     def eigenbasis_normals_eval(self, eig, mult=None, mtol=None, rtol=None, solver='gsvd', reg_type='svd', on_nodes=True):
#         """Evaluates the outward-normal derivatives of the eigenbasis on the collocation points or quadrature nodes."""
#         # process inputs
#         if isinstance(eig, Eigenvalue):
#             eig, mult = eig._val, eig._mult
#         elif isinstance(eig, (float, np.float64)):
#             if eig in self.eig_seq and mult is None:
#                 mult = self.mults[eig]

#         # get eigenbasis coefficient matrix
#         C = self.eigenbasis_coef(eig, mult, mtol, rtol, solver, reg_type, on_nodes)

#         if on_nodes:
#             grad = self._basis._grad_pointset(eig, self._bdry_nodes)@C
#             normals = self._domain.bdry_normals(self._bdry_nodes)[:,np.newaxis]
#         else:
#             grad = self._basis._grad_pointset(eig, self._bdry_pts)@C
#             normals = self._domain.bdry_normals(self._bdry_pts)[:,np.newaxis]
#         return grad.real*normals.real + grad.imag*normals.imag
    
#     ### Eigensolvers
#     def solve_eigs_interval(self, a, b, n_pts, ftol=None, xtol=None, mps_kwargs={}, use_nodes=False, verbose=0):
#         if ftol is None: ftol = self.mtol
#         if xtol is None: xtol = self.etol

#         if a < self.eig_lbound: a = self.eig_lbound/2

#         f = lambda lam: self.subspace_tans(lam, use_nodes=use_nodes, **mps_kwargs)
#         eigs, fevals = mps.solve_eigs_interval(lambda lam: f(lam)[:2], a, b, n_pts, ftol, xtol, verbose)
#         mults = []
#         for eig in eigs:
#             mult = mps.estimate_multiplicity(f, eig-2*xtol, eig+2*xtol, ftol, verbose=bool(verbose))
#             mults.append(mult)
#             fevals += 5

#         return eigs, np.array(mults), fevals
    
#     def solve_eigs_ordered(self, k, ftol=None, xtol=None, mps_kwargs={}, use_nodes=False, verbose=0):
#         """solves for the first k eigenvalues"""
#         a = 0.9*self.eig_lbound
#         b = self.weyl_k(k+5) # conservative estimate
#         n_pts = 10*(k+5)
#         eigs, mults, fevals = self.solve_eigs_interval(a, b, n_pts, ftol=ftol, xtol=xtol, mps_kwargs=mps_kwargs, 
#                                                use_nodes=use_nodes, verbose=verbose)
#         for eig, mult in zip(eigs, mults):
#             try:
#                 self.spectrum.add_eig(Eigenvalue(eig, mult))
#             except:
#                 pass

#         return self.eig_seq[:k]

#     ### Helper functions for plotting
#     def plot_subspace_sines(self, low, high, nlam, n_angle, ax=None, mps_kwargs={}, **plot_kwargs):
#         if low < self.eig_lbound/2 : low = self.eig_lbound/2
#         L = np.linspace(low, high, nlam+1)
#         sines = np.array([self.subspace_sines(lam, **mps_kwargs)[:n_angle] for lam in L])
#         if ax is None:
#             fig = plt.figure()
#             plt.plot(L, sines, **plot_kwargs)
#             return fig
#         else:
#             ax.plot(L, sines, **plot_kwargs)

#     def plot_subspace_tans(self, low, high, nlam, n_angle, ax=None, mps_kwargs={}, **plot_kwargs):
#         if low < self.eig_lbound/2 : low = self.eig_lbound/2
#         L = np.linspace(low, high, nlam+1)
#         tans = np.array([self.subspace_tans(lam, **mps_kwargs)[:n_angle] for lam in L])
#         if ax is None:
#             fig = plt.figure()
#             plt.plot(L, tans, **plot_kwargs)
#             return fig
#         else:
#             ax.plot(L, tans, **plot_kwargs)
#             ax.set_xlim(low,high)

#     ### Asymptotics
#     def weyl_N(self,lam):
#         """Two-term Weyl asymptotics for the eigenvalue counting function"""
#         A = self.domain.area
#         P = self.domain.perimeter
#         return (A*lam - P*np.sqrt(lam))/(4*np.pi)

#     def weyl_k(self,k):
#         """Weyl asymptotic estimate for the kth eigenvalue"""
#         A = self.domain.area
#         P = self.domain.perimeter
#         return ((P+np.sqrt(P**2+16*np.pi*A*k))/(2*A))**2
            
# class Eigenvalue:
#     def __init__(self, val, mult=1):
#         self.val = float(val)
#         self.mult = int(mult)
    
#     def __eq__(self, other):
#         if type(other) is Eigenvalue:
#             return (self.val == other.val) and (self.mult == other.mult)
#         else:
#             return False
        
#     def __hash__(self):
#         return hash((self.val, self.mult))

#     def __repr__(self):
#         return f"Eigenvalue({self.val},mult={self.mult})"

# class Spectrum:
#     """Class for tracking the Laplacian spectrum of a domain"""
#     def __init__(self, eigs=None, mults=None):
#         self._eigs = set()
#         if eigs is not None:
#             if mults is None:
#                 for eig in eigs:
#                     self.add_eig(eig)
#             elif len(eigs) == len(mults):
#                 for eig, mult in zip(eigs, mults):
#                     self.add_eig(eig, mult)
#             else:
#                 raise ValueError("'eigs' and 'mults' must have the same length")

#     def add_eig(self, eig, mult=None):
#         if isinstance(eig, Eigenvalue):
#             if eig in self._eigs:
#                 raise ValueError(f"{eig} already in spectrum.")
#             else:
#                 self._eigs.add(eig)
#         else:
#             if eig in self.eig_seq:
#                 raise ValueError(f"{eig} already in spectrum (with mult = {self.mults[eig]})")
#             else:
#                 self._eigs.add(Eigenvalue(eig, mult))

#     def remove_eig(self, eig, mult=None):
#         if isinstance(eig, Eigenvalue):
#             self._eigs.remove(eig)
#         else:
#             if mult is None:
#                 mult = self.mults[eig]
#             self._eigs.remove(Eigenvalue(eig,mult))

#     @property
#     def eigs(self):
#         return frozenset(self._eigs)

#     @property
#     def sorted(self):
#         return sorted(self._eigs,key=lambda eig: eig.val)

#     @property
#     def eig_seq(self):
#         vals = np.concatenate(([],*(eig._mult*[eig._val] for eig in self._eigs)))
#         return np.sort(vals.flatten())

#     @property
#     def mults(self):
#         return {eig.val:eig.mult for eig in self._eigs}

#     def __str__(self):
#         return f"Spectrum:{str(self._eigs)}"

#     def __eq__(self, other):
#         if isinstance(other, Spectrum):
#             return self._eigs == other._eigs
#         else:
#             return False
        
#     def __add__(self, other):
#         if isinstance(other, Eigenvalue):
#             self.add_eig(other)
#             return self
#         elif isinstance(other, Spectrum):
#             return Spectrum(self._eigs.union(other._eigs))