from .core import BaseEigenproblem, BaseEigensolver, EigensolverFailure
from . import mps
from .mps import MPSEigensolver
from .utils import complex_form
from .bounds import faber_krahn as _faber_krahn

# Relative slack applied to a sharp lower bound before using it as a search
# endpoint. Far below any accuracy of interest; costs one extra grid point.
_WINDOW_PAD = 1e-6
from .asymp import (weyl_est as _weyl_est, weyl_count as _weyl_count,
                    weyl_count_check as _weyl_count_check)
from .bases import ParticularBasis
from .geometry import PointSet, Domain
from .opt import minimize_on_bracket

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

    def solve(self, k, ppl=20, solver=None, max_rescue=10, verbose=0, **solver_kwargs):
        """Solve for the first k eigenvalues (counting multiplicity).

        Returns the FIRST k, which is a stronger promise than k of them, and one that was being
        broken. At the previous default of `ppl=5` the initial scan stepped over a tension
        minimum and returned k accurate eigenvalues with one missing from the bottom, silently
        shifting every index above it -- the count right, the values right, no warning. Swept
        over ten suite polygons x k in 2..10 at precision 1e-10:

            ppl=5    9 of 90 cells wrong        ppl=10   2 of 90        ppl=20   0 of 90

        `right_trapezoid` dropped lam_3 = 44.9484877814 at k=10 while being correct at k=9 and
        k=11; `H_shape` dropped lam_5 = 14.30522996 at k=7 and k=9, needing ppl=20 because its
        sigma well there is narrow (1.5e-10 at the mode, 3.4e-03 just 0.025 away) inside a tight
        cluster. The cost is +43% on the initial grid, which buys not better digits -- those
        were already right -- but the right *set*.

        There is no cheap audit standing behind this, and that is a real limitation rather than
        an oversight. `_find_deficient_gaps` cannot serve: measured per-gap Weyl expected counts
        overlap completely between correct and incorrect results (correct cells reach 2.87
        expected modes in one gap, incorrect ones span 2.27-2.67), because multiplicity
        confounds the two-term count at these wavenumbers. Grid resolution is doing the work,
        so `ppl=20` is validated rather than proven, and a domain with a tighter cluster than
        H_shape's could still defeat it. A validated detector is open in docs/todo.md.

        FOR A LOOP, USE `track` INSTEAD. It follows one mode by value from a nearby starting
        point, which is both ~2.5x faster and immune to this failure mode, since no index is
        involved. `solve` is the cold start.

        Parameters
        ----------
        k : int
            Number of eigenvalues to find.
        ppl : int, optional
            Grid points per Weyl-level in the initial interval scan (default 20). Lowering it
            risks stepping over a bracket; see above.
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

    def _mean_spacing(self, lam):
        """Local mean eigenvalue spacing at `lam`, from the two-term Weyl law.

        Taken as `weyl_est(N+1) - weyl_est(N)` rather than differentiating by hand, so the
        Dirichlet/Neumann sign convention lives in one place. `N` is clamped away from zero
        because `weyl_count` goes negative below the first eigenvalue on some domains.
        """
        n = max(float(_weyl_count(lam, self.domain)), 0.5)
        return float(_weyl_est(n + 1, self.domain) - _weyl_est(n, self.domain))

    def track(self, lam_prev, mult=1, solver=None, window=None, n_pts=9, solver_kwargs=None,
              verbose=0):
        """Follow ONE eigenvalue from a nearby starting value. No global scan.

        This is the inner-loop call. `solve(k)` re-scans the whole spectrum from the
        Faber--Krahn bound every time -- 2-8 s for four eigenvalues, against ~10 ms of solver
        construction -- which a shape-optimization loop does not need, because it already knows
        where `lambda` was at the previous iterate. Tracking also sidesteps `solve(k)`'s
        set-selection problem entirely: it follows a mode by VALUE, so it cannot silently hand
        back a different index (see `tests/test_mode_completeness.py`).

        The intended loop is plan-once / realize-per-iterate:

            plan = basis_plan.plan_basis(dom0, lam_max, target)
            lam  = Eigenproblem(dom0, precision=p).solve(1)[0]      # cold start, once
            for dom in family:
                solver = MPSEigensolver.from_domain(dom, basis=basis_plan.realize(plan, dom))
                lam = Eigenproblem(dom, eval_solver=solver, precision=p).track(lam)

        `window` is the HALF-width of the scan, defaulting to a third of the local Weyl mean
        spacing so it cannot reach the neighbouring eigenvalue. Pass a float to override.

        RAISES `EigensolverFailure` if the discrete minimum lands on the edge of the scan
        window, because then the window -- not the tension -- chose the answer. That guard is not
        hypothetical: `benchmarks/basis_lab/plan_lab._lam_near` records it catching a fixed
        +-2e-3 window that the eigenvalue had simply outrun, producing a reference wrong by 16%
        which then read as every basis being wrong by an identical amount. The same failure
        appears twice in that directory's notebook. Widen `window`, or step the shape more
        finely so consecutive iterates stay close.

        Also raises if the tension at the located minimum is above `ttol`: a minimum of sigma is
        not an eigenvalue unless sigma is actually small there, and a loop that has stepped off
        its mode should be told, not handed the nearest dip. With `mult > 1` the check is applied
        to `sigma[mult-1]`, so a cluster that has lost a member is caught too.

        Returns the eigenvalue as a float.
        """
        solver = self._get_eval_solver(solver)
        solver_kwargs = solver_kwargs or {}
        lam_prev = float(lam_prev)
        if lam_prev <= 0:
            raise ValueError(f"'lam_prev' must be positive (got {lam_prev})")
        if n_pts < 3:
            raise ValueError(f"'n_pts' must be at least 3 (got {n_pts})")

        h = float(window) if window is not None else self._mean_spacing(lam_prev)/3.0
        if not (h > 0 and np.isfinite(h)):
            raise EigensolverFailure(f'could not size a scan window at lam={lam_prev:.9g} '
                                     f'(got half-width {h}); pass `window` explicitly')

        sig = lambda l: float(np.atleast_1d(solver.sigma(float(l), **solver_kwargs))[0])  # noqa: E731
        xs = np.linspace(lam_prev - h, lam_prev + h, n_pts)
        if xs[0] <= 0:
            xs = np.linspace(lam_prev/2, lam_prev + h, n_pts)
        # The scan points are independent, so evaluate them together: `tensions` takes an array
        # and threads it. This is the loop's hot path, and serially it was the whole cost of a
        # `track` -- the parabolic refinement that follows needs only a handful more sigmas.
        try:
            rows = solver.tensions(xs, n_workers=mps.n_workers_default, **solver_kwargs)
            ys = np.array([float(np.atleast_1d(r)[0]) for r in rows])
        except TypeError:
            # A solver whose `tensions` does not take arrays or `n_workers` (anything not an
            # MPSEigensolver) still works, one point at a time.
            ys = np.array([sig(x) for x in xs])
        i = int(np.argmin(ys))
        if verbose > 0:
            print(f'track: window [{xs[0]:.6g}, {xs[-1]:.6g}] about {lam_prev:.9g}, '
                  f'min at index {i} of {n_pts}')
        if i == 0 or i == n_pts - 1:
            raise EigensolverFailure(
                f'tension minimum is at the edge of the scan window around lam={lam_prev:.9g} '
                f'(index {i} of {n_pts}, window half-width {h:.4g}). The window, not the basis, '
                f'would set the answer. Widen `window`, or take smaller steps in the shape so '
                f'consecutive iterates stay closer.')

        lam, _ = minimize_on_bracket(sig, ((xs[i-1], xs[i], xs[i+1]), (ys[i-1], ys[i], ys[i+1])),
                                     1e-15)
        lam = float(lam)

        # `tensions`, not `sigma`: the latter is only the smallest one, so it cannot see whether
        # a cluster of size `mult` is present.
        t = np.atleast_1d(solver.tensions(lam, **solver_kwargs))
        if mult > len(t):
            raise EigensolverFailure(f'mult={mult} exceeds the {len(t)} tensions the pencil '
                                     f'returns at lam={lam:.9g}')
        ttol = getattr(solver, 'ttol', mps.ttol_default)
        if t[mult-1] > ttol:
            raise EigensolverFailure(
                f'tracked a minimum at lam={lam:.9g} but sigma[{mult-1}]={t[mult-1]:.3e} is above '
                f'ttol={ttol:.1e}, so it is not an eigenvalue of multiplicity {mult}. The loop has '
                f'probably stepped off the mode it was following, or the cluster has split.')
        return lam

    def track_set(self, lams_prev, window=None, n_pts=None, ppl=20, solver=None,
                  solver_kwargs=None, verbose=0):
        """Follow a SET of eigenvalues with ONE windowed scan. The inner-loop call for k > 1.

        `track` follows one mode. A loop wanting `lam_1..lam_K` has had to call it K times and
        then repair the result, and near a degeneracy that repair is the common path rather than
        the exceptional one. Measured on a rectangle family walked down to a relative gap of
        6e-09, tracking each of three eigenvalues separately: **1 success in 46 steps**, against
        19 refusals and 25 detected collapses -- every one of which falls back to a full
        `solve`, which is the 2-8 s call `track` exists to avoid. The speedup was simply not
        being had where these problems spend their time.

        Worse, the successes are not all right. Two seeds converging toward one mode can both
        land near it and be accepted: the pair is then reported split when it is not, or with a
        value that is neither member. Measured at the endpoint of an N=4 `lam_2/lam_1` run, where
        the true pair is split by 4.3e-09: the per-value path returned a `lam_2` wrong by 1.1e-08
        relative -- past every guard, because the guards check the tension (which is small: the
        value IS near an eigenvalue) and check coincidence only within `rtol=1e-9`, which a
        4.3e-09 split just clears. The reported ratio came out on the impossible side of 5/2.

        ONE SCAN CANNOT DO EITHER. It brackets every tension minimum in the window at once, so
        two seeds cannot converge onto one mode -- there is no per-seed convergence to do. And it
        reads multiplicity from the tension spectrum, so a pair too tight for the grid comes back
        as one eigenvalue of multiplicity 2, i.e. two EQUAL values. That is the honest answer at
        that resolution, and it is the one the cluster model wants: `Lambda = 0`, with any
        orthonormal basis of the eigenspace serving.

        `lams_prev` is the previous iterate's values, ascending, repeated by multiplicity.
        Returns `(lams, mults)`: `lams` has the same length as `lams_prev`, and `mults` gives the
        multiplicity each value was found with.

        RAISES `EigensolverFailure` rather than guessing, on the three ways this goes wrong: the
        window turned up fewer modes than were asked for, a located minimum sits on the window
        edge (so the window chose it), or a returned value is further from its seed than the
        window half-width (so the set has shifted rather than moved).
        """
        solver = self._get_eval_solver(solver)
        solver_kwargs = dict(solver_kwargs or {})
        lams_prev = np.atleast_1d(np.asarray(lams_prev, dtype=float))
        if lams_prev.ndim != 1 or lams_prev.size == 0:
            raise ValueError(f'lams_prev must be a non-empty 1-D array, got shape '
                             f'{lams_prev.shape}')
        if np.any(lams_prev <= 0):
            raise ValueError(f'lams_prev must be positive, got min {lams_prev.min():.6g}')
        lams_prev = np.sort(lams_prev)
        K = lams_prev.size

        spacing = self._mean_spacing(float(lams_prev.mean()))
        h = float(window) if window is not None else spacing/3.0
        if not (h > 0 and np.isfinite(h)):
            raise EigensolverFailure(f'could not size a scan window about '
                                     f'{lams_prev.mean():.9g} (half-width {h}); pass `window`')
        a = max(float(lams_prev[0]) - h, float(lams_prev[0])*0.5)
        b = float(lams_prev[-1]) + h

        if n_pts is None:
            # the same points-per-Weyl-level currency `solve` uses, so one knob governs both
            n_pts = int(np.clip(round(ppl*(b - a)/spacing), 9, 400))
        if verbose > 0:
            print(f'track_set: [{a:.9g}, {b:.9g}] about {K} seeds, n_pts={n_pts}')

        eigs, mults, _ = solver.solve_interval(a, b, int(n_pts), verbose=verbose-1,
                                               **solver_kwargs)
        eigs = np.asarray(eigs, dtype=float)
        if eigs.size == 0:
            raise EigensolverFailure(
                f'no eigenvalue found in [{a:.9g}, {b:.9g}] while tracking {K} values about '
                f'{lams_prev.mean():.9g}. The set has moved further than the window, or the '
                f'basis no longer resolves this shape.')

        edge = max(1e-12, 1e-6*(b - a))
        on_edge = eigs[(eigs - a < edge) | (b - eigs < edge)]
        if on_edge.size:
            raise EigensolverFailure(
                f'a tension minimum sits on the edge of the scan window [{a:.9g}, {b:.9g}] '
                f'(at {on_edge[0]:.9g}), so the window and not the tension chose it. Widen '
                f'`window`, or step the shape more finely.')

        found = _expand_mults(eigs, mults)
        if found.size < K:
            raise EigensolverFailure(
                f'tracking {K} eigenvalues but the window [{a:.9g}, {b:.9g}] accounts for only '
                f'{found.size} (values {np.array2string(eigs, precision=9)}, multiplicities '
                f'{np.asarray(mults)}). A mode has left the window or been missed.')

        lams = found[:K]
        moved = np.abs(lams - lams_prev)
        if np.any(moved > h):
            i = int(np.argmax(moved))
            raise EigensolverFailure(
                f'tracked value {lams[i]:.9g} is {moved[i]:.3g} from its seed '
                f'{lams_prev[i]:.9g}, beyond the window half-width {h:.3g}. The set has probably '
                f'shifted by an index rather than moved.')

        out_mults = np.ones(K, dtype=int)
        j = 0
        for e, m in zip(eigs, np.asarray(mults, dtype=int)):
            for _ in range(int(m)):
                if j < K:
                    out_mults[j] = int(m)
                    j += 1
        return lams, out_mults

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
