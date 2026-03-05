import numpy as np
from .core import BaseEigenproblem, BaseEigensolver
from .mps import MPSEigensolver
from .utils import complex_form
from .bounds import faber_krahn as _faber_krahn
from .asymp import weyl_est as _weyl_est, weyl_count_check as _weyl_count_check
from .bases import ParticularBasis
from .geometry import PointSet, Domain

def _expand_mults(eigs, mults):
    """Expand eigenvalue array by integer multiplicities → flat sorted ndarray."""
    mults = np.asarray(mults, dtype=int)
    return np.sort(np.repeat(eigs, mults))


def _merge_eigs(existing, new, ltol):
    """Merge two sorted eigenvalue arrays, deduplicating with solver-scale tolerance."""
    if len(existing) == 0:
        return new
    if len(new) == 0:
        return existing
    combined = np.sort(np.concatenate([existing, new]))
    keep = np.concatenate(
        [[True], np.diff(combined) > ltol * np.maximum(combined[:-1], 1.0)]
    )
    return combined[keep]


def _find_deficient_gaps(eigs_for_check, a, b, domain):
    """Return list of (sub_a, sub_b) subintervals where Weyl count is deficient."""
    if len(eigs_for_check) == 0:
        return []
    check = _weyl_count_check(eigs_for_check, domain)
    gaps = []
    for j in range(len(check)):
        if check[j] <= -1 and (j == 0 or check[j] < check[j - 1]):
            sub_a = a if j == 0 else eigs_for_check[j - 1]
            sub_b = eigs_for_check[j]
            sub_a = max(sub_a, a)
            sub_b = min(sub_b, b)
            if sub_b > sub_a:
                gaps.append((sub_a, sub_b))
    return gaps


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

    def solve(self, k, ppl=5, solver=None, **solver_kwargs):
        """Solve for the first k eigenvalues (counting multiplicity).

        Parameters
        ----------
        k : int
            Number of eigenvalues to find.
        ppl : int, optional
            Grid points per Weyl-level in the initial interval scan (default 10).
        solver : BaseEigensolver, optional
            Override the instance's eval_solver for this call.
        **solver_kwargs
            Forwarded to every solver.solve_interval() call.

        Returns
        -------
        numpy.ndarray, shape (k,)
            Sorted eigenvalues, repeated by multiplicity.
        """
        bc = self.domain.bc_type
        if bc not in ('dir', 'neu'):
            raise NotImplementedError(
                f"solve() not implemented for bc_type={bc!r}; "
                "only 'dir' and 'neu' are supported"
            )
        return self._solve_dir_neu(k, ppl, solver, **solver_kwargs)

    def _solve_dir_neu(self, k, ppl, solver=None, **solver_kwargs):
        solver = self._get_eval_solver(solver)
        bc = self.domain.bc_type

        ltol = solver_kwargs.get('ltol', getattr(solver, 'ltol', 1e-8))

        if bc == 'dir':
            a = _faber_krahn(area=self.domain.area)
            k_search = k
        else:  # 'neu'
            a = ltol
            k_search = k - 1

        if k_search <= 0:
            return np.zeros(k)

        m = k_search + 1
        b = _weyl_est(m, self.domain)
        raw_eigs, raw_mults, _ = solver.solve_interval(a, b, ppl * m, **solver_kwargs)
        eigs_flat = _expand_mults(raw_eigs, raw_mults)

        while len(eigs_flat) < k_search:
            eigs_for_check = (
                np.concatenate([[0.0], eigs_flat]) if bc == 'neu' else eigs_flat
            )
            deficient_gaps = _find_deficient_gaps(eigs_for_check, a, b, self.domain)

            if deficient_gaps:
                prev_count = len(eigs_flat)
                for sub_a, sub_b in deficient_gaps:
                    n_sub = max(ppl, round(2 * ppl * m * (sub_b - sub_a) / (b - a)))
                    new_eigs, new_mults, _ = solver.solve_interval(
                        sub_a, sub_b, n_sub, **solver_kwargs
                    )
                    eigs_flat = _merge_eigs(eigs_flat, _expand_mults(new_eigs, new_mults), ltol)

                if len(eigs_flat) > prev_count:
                    continue

            m += 1
            b_new = _weyl_est(m + 1, self.domain)
            new_eigs, new_mults, _ = solver.solve_interval(b, b_new, ppl, **solver_kwargs)
            eigs_flat = _merge_eigs(eigs_flat, _expand_mults(new_eigs, new_mults), ltol)
            b = b_new

        if bc == 'neu':
            eigs_flat = np.concatenate([[0.0], eigs_flat])
        return np.sort(eigs_flat)[:k]

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
