import numpy as np
from .core import BaseEigenproblem, BaseEigensolver
from .mps import MPSEigensolver
from .utils import complex_form
from .bounds import faber_krahn as _faber_krahn
from .asymp import weyl_est as _weyl_est
from .bases import ParticularBasis
from .geometry import PointSet, Domain

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
        a = _faber_krahn(area=self.domain.area)
        b = _weyl_est(k, area=self.domain.area, perim=self.domain.perimeter, bc_type='dir')
        eigs, mults, fevals = self.solve_interval(a, b, ppl*k, solver, **solver_kwargs)
        while mults.sum() < n_eigs:
            a = _weyl_est(k, area=self.domain.area, perim=self.domain.perimeter, bc_type='dir')
            b = _weyl_est(k+1, area=self.domain.area, perim=self.domain.perimeter, bc_type='dir')
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
