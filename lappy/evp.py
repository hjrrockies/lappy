from .core import BaseEigenproblem, BaseEigensolver
from . import mps
from .mps import MPSEigensolver
from .utils import complex_form
from .bounds import faber_krahn as _faber_krahn

# Relative slack applied to a sharp lower bound before using it as a search
# endpoint. Far below any accuracy of interest; costs one extra grid point.
_WINDOW_PAD = 1e-6
from .asymp import weyl_est as _weyl_est, weyl_count_check as _weyl_count_check
from .bases import ParticularBasis
from .geometry import PointSet, Domain

import numpy as np

def _expand_mults(eigs, mults):
    """Expand eigenvalue array by integer multiplicities → flat sorted ndarray."""
    mults = np.asarray(mults, dtype=int)
    return np.sort(np.repeat(eigs, mults))


def _merge_eigs(existing, new, ltol):
    """Merge two sorted eigenvalue arrays, preserving multiplicity.

    For each cluster of values within ltol, keeps max(count_in_existing,
    count_in_new) copies, so a degenerate eigenvalue found in both runs is
    not collapsed to a simple one.
    """
    if len(existing) == 0:
        return new
    if len(new) == 0:
        return existing

    vals = np.concatenate([existing, new])
    from_new = np.concatenate([np.zeros(len(existing), bool), np.ones(len(new), bool)])
    order = np.argsort(vals, kind='stable')
    vals = vals[order]
    from_new = from_new[order]

    scale = np.maximum(vals[:-1], 1.0)
    cluster_id = np.cumsum(np.concatenate([[True], np.diff(vals) > ltol * scale]))

    keep = np.zeros(len(vals), bool)
    for cid in np.unique(cluster_id):
        mask = cluster_id == cid
        n_exist = int(np.sum(~from_new[mask]))
        n_new_c = int(np.sum(from_new[mask]))
        count = max(n_exist, n_new_c)
        keep[np.where(mask)[0][:count]] = True

    return vals[keep]


def _find_deficient_gaps(eigs_for_check, a, b, domain, thresh=1):
    """Return list of (sub_a, sub_b) subintervals where Weyl count is deficient."""
    if len(eigs_for_check) == 0:
        return []
    check = _weyl_count_check(eigs_for_check, domain)
    gaps = []
    for j in range(len(check)):
        if check[j] <= -thresh and (j == 0 or check[j] < check[j - 1]):
            sub_a = a if j == 0 else eigs_for_check[j - 1]
            sub_b = eigs_for_check[j]
            sub_a = max(sub_a, a)
            sub_b = min(sub_b, b)
            if sub_b > sub_a:
                gaps.append((sub_a, sub_b))
    return gaps


# eigenproblem class
class Eigenproblem(BaseEigenproblem):
    """Class for planar dirichlet laplacian eigenproblems.

    `precision` is one dial with the same meaning at both stages: it sizes the basis (through
    `basis_plan`, which derives the whole construction from geometry, `lam_max` and that target)
    AND becomes the eigenvalue search's `ltol`, so the minimization is solved to matching depth.
    Asking for a basis good to `p` and then stopping the search at `1e-8` -- the old default --
    wasted the basis; asking the search for more than the basis can support wasted the search.

    Leaving `precision` as None falls back to `mps.ltol_default`. Either way no solver is built
    until a solve needs one, so constructing an `Eigenproblem` stays cheap.

    Note what `precision` is not: a guarantee. Measured over 75 cells, the achieved certified
    accuracy runs a median ~1 digit *better* than requested, but the sharp-cornered domains
    (chevrons, thin isoceles triangles) fall short of it -- see
    `benchmarks/basis_lab/PLAN_LAB.md`. `plan.capped` and `plan.shortfall` say so when the planner
    knows it cannot deliver; there is no mechanism yet that certifies the request was met.
    """
    def __init__(self, domain, eval_solver=None, evec_solver=None, precision=None):
        super().__init__(domain)
        self.precision = precision
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
        elif solver == 'fem':
            from .fem import FEMEigensolver
            self._eval_solver = FEMEigensolver(self.domain)
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
        """Resolve the solver, building a default one on demand if none was supplied.

        Lazy rather than built in `__init__`, because constructing a solver costs a basis, a
        collocation set and a boundary quadrature, and an `Eigenproblem` is cheap to make for
        reasons that never reach a solve. Once built it is cached, so two `solve` calls share it.

        This is what makes CLAUDE.md principle 1's three lines run: before, `from_domain` raised
        `NotImplementedError` for `basis=None`, so `Eigenproblem(domain).solve(n)` could not work
        and every caller hand-built a solver.
        """
        if solver is not None:
            if not isinstance(solver, BaseEigensolver):
                raise TypeError("'solver' must be a valid Eigensolver")
            return solver
        if self.eval_solver is None:
            prec = self.precision if self.precision is not None else mps.ltol_default
            self.eval_solver = MPSEigensolver.from_domain(self.domain, prec=prec)
        return self.eval_solver

    def _get_evec_solver(self, solver):
        """As `_get_eval_solver`, but falls back to the eval solver rather than building a second
        one. `MPSEigensolver` already returns eigenfunctions -- the separate `evec_solver` slot
        exists for the case where eigenvalues and eigenfunctions want different configurations
        (e.g. an FEM sweep for the values), not because one solver cannot do both."""
        if solver is not None:
            if not isinstance(solver, BaseEigensolver):
                raise TypeError("'solver' must be a valid Eigensolver")
            return solver
        if self.evec_solver is not None:
            return self.evec_solver
        return self._get_eval_solver(None)

    def check_precision(self, eigs, solver=None):
        """Was the requested `precision` actually achieved at `eigs`? Measured, not predicted.

        Returns a dict with `target`, `achieved` (the worst block's contribution to the
        Moler--Payne relative bound), `digits`, `met`, and the per-block breakdown so a caller can
        see *where* it fell short.

        This exists because nothing else answers the question. `plan.capped`/`plan.shortfall` report
        only what the planner knew in advance; a request that quietly falls short -- which happens
        on the sharp-cornered domains -- was silent. The quantity here is the same one
        `refine_plan` optimizes, and it agrees with a full Moler--Payne certification to 0.16 digits
        across 150 measured cells (`benchmarks/basis_lab/PLAN_LAB.md`) at roughly 1% of the cost,
        because `||u||_L2 = 1` by construction for orthonormalized coefficients and the boundary sup
        is one basis evaluation per arc.

        Returns `None` if this problem's basis did not come from `basis_plan` (a hand-built basis
        has no per-arc structure to attribute a residual to).
        """
        from . import basis_plan
        solver = self._get_eval_solver(solver)
        plan = basis_plan.plan_of(getattr(solver, 'basis', None))
        if plan is None:
            return None
        arc, cor = basis_plan.residual_by_arc(plan, self.domain, solver, eigs)
        achieved = float(max(arc.max(initial=0.0), cor.max(initial=0.0)))
        target = plan.target
        return dict(target=target, achieved=achieved,
                    digits=(-np.log10(achieved) if achieved > 0 else np.inf),
                    met=achieved <= target,
                    arc_residuals=arc, corner_residuals=cor,
                    n_basis=len(solver.basis), capped=plan.capped,
                    shortfall=plan.shortfall or None)

    def refine_basis(self, eigs, verbose=0):
        """Grow the basis where `check_precision` says it falls short, and rebuild the solver.

        Costs a second solve on the caller's part (the eigenvalues move slightly once the basis
        changes), which is why it is opt-in rather than folded into `solve`. In a shape-optimization
        loop this is paid once and amortized over every iterate afterwards; for a one-off solve it
        roughly doubles the work.

        Returns the refined `BasisPlan`, or None if the basis did not come from `basis_plan`.
        """
        from . import basis_plan
        solver = self._get_eval_solver(None)
        plan = basis_plan.plan_of(getattr(solver, 'basis', None))
        if plan is None:
            return None
        prec = self.precision if self.precision is not None else mps.ltol_default

        def factory(basis):
            return MPSEigensolver.from_domain(self.domain, basis=basis, prec=prec)

        refined = basis_plan.refine_plan(plan, self.domain, factory, eigs, verbose=verbose)
        if refined.n_total != plan.n_total:
            self.eval_solver = factory(basis_plan.realize(refined, self.domain))
        return refined

    def solve(self, k, ppl=5, solver=None, max_rescue=10, verbose=0, **solver_kwargs):
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
        bc_type = self.domain.bc_type
        if bc_type not in ('dir', 'neu'):
            raise NotImplementedError(
                f"solve() not implemented for bc_type={bc_type!r}; "
                "only 'dir' and 'neu' are supported"
            )
        return self._solve_dir_neu(k, ppl, solver, max_rescue, verbose, **solver_kwargs)

    def _solve_dir_neu(self, k, ppl, solver=None, max_rescue=10, verbose=0, **solver_kwargs):
        solver = self._get_eval_solver(solver)
        bc_type = self.domain.bc_type

        ltol = solver_kwargs.get('ltol', getattr(solver, 'ltol', 1e-8))
        if 'ltol' in solver_kwargs.keys():
            ltol = solver_kwargs['ltol']

        if bc_type == 'dir':
            # Nudge the lower edge strictly below the bound. Faber-Krahn is
            # *sharp* -- equality holds exactly for the disk -- so for a disk
            # (and to within rounding for near-circular domains) the raw bound
            # coincides with lambda_1. The bracketing search finds minima via
            # opt.discrete_locmin_idx, which ignores the endpoints of the grid
            # by construction, so a minimum sitting on the lower edge cannot be
            # found at any basis size. The unit disk returned modes 2..k+1:
            # every value correct to ~14 digits, every certificate valid, and
            # nothing in the pipeline able to detect the omission.
            a = _faber_krahn(area=self.domain.area) * (1.0 - _WINDOW_PAD)
            k_search = k
        else:  # 'neu'
            a = ltol
            k_search = k - 1

        if k_search <= 0:
            return np.zeros(k)
        
        if verbose > 0: print(f"solving for {k} eigenvalues, bc_type={bc_type}")

        m = k_search + 1
        b = _weyl_est(m, self.domain)
        raw_eigs, raw_mults, fevals = solver.solve_interval(a, b, ppl * m, verbose=verbose, **solver_kwargs)
        eigs_flat = _expand_mults(raw_eigs, raw_mults)

        i = 0
        thresh = 1
        while len(eigs_flat) < k_search and i < max_rescue:
            if verbose > 0: print(f"{len(eigs_flat)} eigs found, continuing search")
            eigs_for_check = (
                np.concatenate([[0.0], eigs_flat]) if bc_type == 'neu' else eigs_flat
            )
            deficient_gaps = _find_deficient_gaps(eigs_for_check, a, b, self.domain, thresh)

            if deficient_gaps:
                if verbose > 0: print(f"weyl_count_check flagged for missing eigs")

                prev_count = len(eigs_flat)
                for sub_a, sub_b in deficient_gaps:
                    if verbose > 0: print(f"refining search on [{sub_a:.2e},{sub_b:.2e}]")
                    n_sub = max(ppl, round(2 * ppl * m * (sub_b - sub_a) / (b - a)))
                    new_eigs, new_mults, fe = solver.solve_interval(
                        sub_a, sub_b, n_sub, verbose=verbose, **solver_kwargs
                    )
                    fevals += fe
                    eigs_flat = _merge_eigs(eigs_flat, _expand_mults(new_eigs, new_mults), ltol)
                    if verbose > 0: print(f"{len(eigs_flat)} eigs found")
                if len(eigs_flat) > prev_count:
                    continue

            m += 1
            b_new = _weyl_est(m + 1, self.domain)
            if verbose > 0: print(f"extending search to lam={b_new:.2e}")
            new_eigs, new_mults, fe = solver.solve_interval(b, b_new, ppl, verbose=verbose, **solver_kwargs)
            fevals += fe
            eigs_flat = _merge_eigs(eigs_flat, _expand_mults(new_eigs, new_mults), ltol)
            b = b_new
            i += 1
            thresh *= 0.9
        
        if bc_type == 'neu':
            eigs_flat = np.concatenate([[0.0], eigs_flat])

        if len(eigs_flat) < k:
            if verbose > 0: print(f"search terminated after too many rescues, {len(eigs_flat)} found")
            return np.sort(eigs_flat)
        else:
            if verbose > 0: print(f"search concluded, {k} eigs found, fevals={fevals}")
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

    def eigenfunction_energies(self, eig, mult=1, solver=None, **solver_kwargs):
        solver = self._get_evec_solver(solver)
        return solver.eigenfunction_energies(eig, mult, **solver_kwargs)
