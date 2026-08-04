"""Shared helpers for building high-precision reference Dirichlet eigenvalues.

Deliberately bypasses ``MPSEigensolver.from_domain`` (not trusted in its
current state) and all Rellich/boundary-integral machinery (``cauchy_data``
stays ``None`` -- we only need eigenvalues + tensions, not normalized
eigenfunctions). The solver is assembled by hand from the same free functions
``from_domain`` would call internally: ``bases.make_default_basis``,
``mps.pts_per_seg``, ``domain.bdry_pts``, ``mps.make_default_int_pts``.

All domains here are pure Dirichlet, so boundary normals are never built --
they only matter for Neumann/Robin boundary rows (``bc_param != 0``).
"""
import numpy as np

from lappy.cache import clear_instance_caches
from lappy import geometry, bases, mps, opt, bounds, asymp
from lappy import MPSEigensolver, Eigenproblem


def build_solver(domain, n_basis, rtol=1e-14, ttol=1e-3, bdry_mult=2, int_npts=None,
                  **basis_kwargs):
    """Manually assemble an MPSEigensolver for `domain` at basis size `n_basis`.

    No bdry_normals, no cauchy_data -- Dirichlet eigenvalues only.
    """
    basis = bases.make_default_basis(domain, n_basis, **basis_kwargs)

    n_per_seg = mps.pts_per_seg(domain, basis, mult=bdry_mult)
    bdry_pts = domain.bdry_pts(n_per_seg)

    if int_npts is None:
        int_npts = n_basis
    # domain.int_pts (not mps.make_default_int_pts, which only handles
    # Polygon) works generically for any Domain, curved boundaries included.
    int_pts = domain.int_pts(method='random', npts_rand=int_npts)

    basis = basis.to_normalized((bdry_pts, int_pts))
    solver = MPSEigensolver(basis, bdry_pts, int_pts, rtol=rtol, ttol=ttol)
    return solver


def plot_tension_curve(solver, a, b, n=300, rtol_candidates=(1e-14, 1e-12, 1e-10, 1e-8, 1e-6),
                        n_angle=1, outfile='tension_curve.png'):
    """Plot sigma(lambda) over [a, b] for several candidate rtol values, one
    subplot per candidate, and save to `outfile`.

    Meant for one-time-per-domain visual inspection (read the saved image)
    to hand-pick rtol -- not for automated on-the-fly adaptation (that's
    future work; explicitly not `solver.adapt_rtol` per current guidance).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rtol_candidates), 1, figsize=(8, 2.5 * len(rtol_candidates)),
                              sharex=True)
    if len(rtol_candidates) == 1:
        axes = [axes]
    for ax, rtol in zip(axes, rtol_candidates):
        solver.plot_tensions(a, b, n, n_angle=n_angle, rtol=rtol, ax=ax)
        ax.set_ylabel(f'rtol={rtol:.0e}')
        ax.set_yscale('log')
    axes[-1].set_xlabel('lambda')
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def polish_eigs(solver, eigs, ltol=1e-14, bracket_rel_width=None):
    """Golden-search polish each eigenvalue estimate against solver.sigma,
    returning refined (eigs, tensions).

    `bracket_rel_width` sets how wide a net to cast around each coarse
    estimate before polishing (defaults to `ltol` itself, matching the old
    behavior). If the coarse estimate came from a looser upstream tolerance
    (e.g. manual_solve's `minimize_tol`), pass that tolerance here instead --
    a bracket narrower than the coarse estimate's actual uncertainty can
    miss the true root entirely."""
    if bracket_rel_width is None:
        bracket_rel_width = ltol
    eigs_polished = []
    tensions = []
    for eig in eigs:
        lo, hi = eig * (1 - bracket_rel_width), eig * (1 + bracket_rel_width)
        # golden_search needs a nondegenerate bracket; widen slightly if eig==0
        if lo == hi:
            lo, hi = eig - bracket_rel_width, eig + bracket_rel_width
        peig = opt.golden_search(solver.sigma, lo, hi, tol=ltol * max(abs(eig), 1.0))[0]
        eigs_polished.append(peig)
        tensions.append(solver.sigma(peig))
        # Required, not optional. golden_search evaluates ~100 distinct lambdas
        # per eigenvalue and the symmetry path keeps one solver per sector alive
        # for certification afterwards. Removing this line is enough on its own
        # to get reg_ngon_6 SIGKILLed at n_basis=320 (verified by A/B); with it,
        # the same run finishes at 12.5 certified digits.
        #
        # I previously removed it after measuring "no effect" -- but that
        # measurement instrumented `manual_solve` only and never executed this
        # loop. Scope the benchmark to the code you are drawing conclusions
        # about.
        clear_instance_caches(solver)
    return np.array(eigs_polished), np.array(tensions)


def manual_solve(solver, a, b, n_pts, bracket_xtol=1e-5, minimize_tol=1e-12,
                 max_recurse=8, clear_every=32,
                  ttol=None, n_workers=1, verbose=0):
    """Find eigenvalues in [a, b] by calling opt.bracket_mins /
    opt.minimize_on_bracket / mps.estimate_multiplicity directly, instead of
    going through MPSEigensolver.solve_interval (which threads a single
    shared `ltol` through bracket-width, per-bracket minimizer convergence,
    AND the merge-distance test -- loosening it to dodge a bracket_mins hang
    would also blunt eigenvalue precision and risk merging genuinely
    distinct close eigenvalues). Here the three roles are decoupled:

    - `bracket_xtol` (loose, e.g. 1e-5): only controls when bracket_mins'
      recursive refinement stops subdividing -- just enough to avoid the
      pathological hang for near-degenerate clusters (e.g. reg_ngon's
      dihedral double eigenvalues), not meant to be precise.
    - `minimize_tol` (tight, e.g. 1e-12): per-bracket minimizer convergence,
      independent of the loose bracketing width -- this is what actually
      controls the precision of each returned eigenvalue location.
    - merging: rather than a separate distance threshold, every adjacent
      pair of accepted brackets is tested with mps.estimate_multiplicity
      itself (already ttol-based library logic, unaffected by either
      tolerance above) by treating them as one candidate degenerate
      eigenvalue and checking whether >=2 tension indices are
      simultaneously small there. This reuses the same degeneracy
      definition the codebase already applies for multiplicity, rather than
      an arbitrary width cutoff that could conflate two distinct
      close-but-not-degenerate eigenvalues.

    Returns (eigs, mults, fevals) like MPSEigensolver.solve_interval, but
    with eigenvalue *location* precision governed by `minimize_tol`, not by
    whatever `bracket_xtol` was needed to avoid a hang.
    """
    ttol = solver.ttol if ttol is None else ttol
    lamgrid = mps.make_lamgrid(a, b, n_pts)

    # Periodic cache clearing during the search.
    #
    # Measured *for this loop specifically*: with opt.bracket_mins' max_recurse
    # default in place, peak RSS is 170MB without clearing and 166MB with it --
    # i.e. the recursion cap is what tamed the search, not this. Kept on anyway
    # because it costs nothing measurable.
    #
    # Note the clearing in `polish_eigs` is a different story and is genuinely
    # required; see the comment there. Do not generalise from this measurement
    # to that one -- I did, and it cost a SIGKILL.
    _neval = [0]

    def tensions2(lam):
        _neval[0] += 1
        if clear_every and _neval[0] % clear_every == 0:
            clear_instance_caches(solver)
        return solver.tensions(lam)[:2]

    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            rows = list(ex.map(tensions2, lamgrid))
        tensiongrid = np.array(rows).T
    else:
        tensiongrid = np.array([tensions2(lam) for lam in lamgrid]).T
    fevals = len(lamgrid)

    # max_recurse guards against runaway refinement where sigma is numerical
    # noise: opt.bracket_mins' "too many local minima" check fires only at
    # depth 0, so deeper levels can branch without bound. Eight levels is
    # ~256x the initial grid spacing, far finer than bracket_xtol needs, and
    # polish_eigs re-refines each bracket properly afterwards.
    brackets, fe = opt.bracket_mins(tensions2, lamgrid, tensiongrid, xtol=bracket_xtol,
                                     verbose=verbose, max_recurse=max_recurse)
    fevals += fe

    eig_brackets = []
    for bracket in brackets:
        minimizer, fe = opt.minimize_on_bracket(solver.sigma, bracket, xtol=minimize_tol,
                                                 verbose=verbose)
        fevals += fe
        minima = solver.sigma(minimizer)
        if minima <= ttol:
            lam = bracket[0]
            eig_brackets.append([lam[0], minimizer, lam[2]])

    eig_brackets.sort(key=lambda br: br[1])

    # Iteratively merge adjacent brackets using estimate_multiplicity as the
    # degeneracy test (not a distance threshold), until no more merges occur.
    changed = True
    while changed and len(eig_brackets) > 1:
        changed = False
        merged = []
        i = 0
        while i < len(eig_brackets):
            if i + 1 < len(eig_brackets):
                a0, e0, b0 = eig_brackets[i]
                a1, e1, b1 = eig_brackets[i + 1]
                cand_a, cand_b = min(a0, a1), max(b0, b1)
                cand_eig = (e0 + e1) / 2
                mult = mps.estimate_multiplicity(solver.tensions, cand_eig, cand_a, cand_b, ttol)
                if mult >= 2:
                    merged.append([cand_a, cand_eig, cand_b])
                    i += 2
                    changed = True
                    continue
            merged.append(eig_brackets[i])
            i += 1
        eig_brackets = merged

    eig_brackets = [br for br in eig_brackets if a <= br[1] <= b]
    eigs = np.array([br[1] for br in eig_brackets])
    mults = np.array([mps.estimate_multiplicity(solver.tensions, br[1], br[0], br[2], ttol)
                       for br in eig_brackets], dtype=int)
    return eigs, mults, fevals


def diagnose(solver, eig, rtol=None, verbose=True):
    """Wrap solver._tension_diagnostics + solver.sigma_cond into a compact
    summary: is this well resolution-limited (basis too poor -- low
    n_reg/n, sigma above ttol nearby), conditioning-limited (accuracy
    floor -- large sigma_cond/err_linalg despite good n_reg), or clean
    (n_reg close to n, sigma_cond modest, err_linalg/err_reg small relative
    to sigma)? Returns the raw diagnostics dict with a few derived fields
    added."""
    d = solver._tension_diagnostics(eig, rtol=rtol)
    d['n_reg_frac'] = d['n_reg'] / d['n'] if d['n'] else np.nan
    d['err_total'] = d['err_linalg'] + d['err_reg']
    if verbose:
        print(f"diagnose(eig={eig:.15f}):")
        print(f"  sigma={d['sigma']:.3e}  sigma_cond={d['sigma_cond']:.3e}")
        print(f"  n={d['n']}  n_reg={d['n_reg']} ({d['n_reg_frac']:.1%})  gsvd_rank={d['gsvd_rank']}")
        print(f"  err_linalg={d['err_linalg']:.3e}  err_reg={d['err_reg']:.3e}"
              f"  err_total={d['err_total']:.3e}")
        verdict = 'clean'
        if d['n_reg_frac'] < 0.9:
            verdict = 'resolution-limited (regularization is truncating a lot of the basis -- ' \
                      'try denser collocation and/or a tighter rtol)'
        elif d['err_total'] > 10 * d['sigma'] or d['sigma_cond'] > 1e6:
            verdict = 'conditioning-limited (accuracy floor -- more basis size alone will not help)'
        print(f"  verdict: {verdict}")
    return d


def cross_check(domain, basis, eig, bdry_mult=2, int_npts=50, rtol=1e-14, n_trials=3, verbose=True):
    """Build fresh, independent bdry_pts/int_pts draws (same already-built
    `basis`, no re-normalization) and report sigma(eig) on each -- large
    variation across independent collocation samples flags overfitting to
    one particular draw rather than a genuine eigenvalue."""
    sigmas = []
    for trial in range(n_trials):
        n_per_seg = mps.pts_per_seg(domain, basis, mult=bdry_mult)
        bdry_pts = domain.bdry_pts(n_per_seg)
        int_pts = domain.int_pts(method='random', npts_rand=int_npts)
        trial_solver = MPSEigensolver(basis, bdry_pts, int_pts, rtol=rtol)
        s = trial_solver.sigma(eig)
        sigmas.append(s)
        if verbose:
            print(f"  cross_check trial {trial}: sigma={s:.3e}")
    sigmas = np.array(sigmas)
    if verbose:
        print(f"  spread: max/min = {sigmas.max()/max(sigmas.min(), 1e-300):.2f}")
    return sigmas


def lambda_window(domain, n_eigs, pad=1e-6):
    """Same [a, b] window Eigenproblem.solve would use internally, for
    feeding into manual_solve/solver.solve_interval directly.

    The lower edge is nudged below the Faber--Krahn bound. Faber--Krahn is
    *sharp*, attained exactly by the disk, so for a disk (and nearly so for
    near-circular domains) the raw bound coincides with lambda_1 to machine
    precision. `bracket_mins`/`discrete_locmin_idx` ignore the endpoints of the
    grid by construction, so a minimum sitting exactly on the lower edge is
    unfindable: the disk silently returned modes 2..11, every value correct and
    certified, the list simply missing its ground state. Only the closed-form
    comparison caught it.

    A relative nudge of 1e-6 is far below any accuracy we care about and costs
    one extra grid point.
    """
    a = bounds.faber_krahn(domain) * (1.0 - pad)
    b = asymp.weyl_est(n_eigs + 1, domain)
    return a, b


def solve_domain(domain, n_basis, n_eigs, ppl=10, ttol=1e-3, rtol=1e-14, verbose=1,
                  **basis_kwargs):
    """Build a solver, solve for the first n_eigs Dirichlet eigenvalues, and
    polish them via golden search on the tension. Returns (eigs, tensions)."""
    solver = build_solver(domain, n_basis, rtol=rtol, ttol=ttol, **basis_kwargs)
    eigprob = Eigenproblem(domain, solver)
    eigs = eigprob.solve(n_eigs, ppl=ppl, verbose=verbose)
    return polish_eigs(solver, eigs)


def escalate_and_solve(domain, n_basis_list, n_eigs, target_tension=1e-12, ppl=10,
                        ttol=1e-3, rtol=1e-14, verbose=1, **basis_kwargs):
    """Try increasing basis sizes from n_basis_list, stopping as soon as every
    eigenvalue's tension clears `target_tension`, or after exhausting the list.

    Returns (n_basis_used, eigs, tensions).
    """
    eigs = tensions = None
    n_basis_used = None
    for n_basis in n_basis_list:
        print(f"--- trying n_basis={n_basis} ---")
        eigs, tensions = solve_domain(domain, n_basis, n_eigs, ppl=ppl, ttol=ttol,
                                       rtol=rtol, verbose=verbose, **basis_kwargs)
        n_basis_used = n_basis
        print(f"tensions: {tensions}")
        # np.all() on an empty array is vacuously True -- guard against a
        # failed/incomplete solve (fewer than n_eigs found) being mistaken
        # for early success.
        if len(tensions) >= n_eigs and np.all(tensions <= target_tension):
            break
    return n_basis_used, eigs, tensions


def solve_domain_v2(domain, n_basis, n_eigs, bracket_xtol=1e-5, minimize_tol=1e-12,
                     polish_bracket_rel_width=1e-9, ttol=1e-3, rtol=1e-14, n_pts_per_eig=11,
                     n_workers=4, verbose=0, **basis_kwargs):
    """Corrected standard pipeline: build_solver + manual_solve (decoupled
    bracket/minimize/merge tolerances, see manual_solve's docstring) +
    polish_eigs with a bracket width that actually matches manual_solve's
    minimize_tol precision guarantee.

    Supersedes solve_domain/Eigenproblem.solve for two reasons found in
    this session: (1) Eigenproblem.solve's underlying solve_interval uses
    ltol_default=1e-8 for per-bracket minimizer convergence, but the old
    polish_eigs then searched only within eig*(1+-1e-14) around that coarse
    estimate -- a window ~1e6x narrower than the coarse estimate's actual
    uncertainty, so polishing silently did ~nothing for many domains
    (confirmed: ellipse(2,1) at n_basis=240 jumped from 7-11 digits to
    13-14+ digits with no basis change at all, just this fix). (2) manual_solve
    sidesteps solve_interval's bracket_mins hang for near-degenerate
    clusters (see manual_solve's docstring).

    Returns (eigs, mults, tensions)."""
    solver = build_solver(domain, n_basis, rtol=rtol, ttol=ttol, **basis_kwargs)
    a, b = lambda_window(domain, n_eigs)
    n_pts = max(n_pts_per_eig * n_eigs, 50)
    eigs, mults, fevals = manual_solve(solver, a, b, n_pts, bracket_xtol=bracket_xtol,
                                        minimize_tol=minimize_tol, ttol=ttol,
                                        n_workers=n_workers, verbose=verbose)
    eigs, tensions = polish_eigs(solver, eigs, ltol=1e-14, bracket_rel_width=polish_bracket_rel_width)
    return eigs, mults, tensions


def escalate_and_solve_v2(domain, n_basis_list, n_eigs, target_tension=1e-12, **kwargs):
    """escalate_and_solve, but using solve_domain_v2 (manual_solve +
    correctly-widened polish) instead of the old Eigenproblem-based
    solve_domain. Returns (n_basis_used, eigs, mults, tensions)."""
    eigs = mults = tensions = None
    n_basis_used = None
    for n_basis in n_basis_list:
        print(f"--- trying n_basis={n_basis} ---")
        eigs, mults, tensions = solve_domain_v2(domain, n_basis, n_eigs, **kwargs)
        n_basis_used = n_basis
        print(f"tensions: {tensions}")
        if len(tensions) >= n_eigs and np.all(tensions <= target_tension):
            break
    return n_basis_used, eigs, mults, tensions


def report(name, eigs, tensions):
    """Print a table of eigenvalues/tensions/implied-digit-accuracy, and a
    ready-to-paste np.array literal for lappy/reference.py."""
    print(f"\n=== {name} ===")
    print(f"{'eig':>22}  {'tension':>12}  {'~digits':>8}")
    for eig, t in zip(eigs, tensions):
        # heuristic: tension ~ 1e-N  =>  error <~ 1e-(N-1)
        digits = -np.log10(t) - 1 if t > 0 else np.inf
        print(f"{eig:22.15f}  {t:12.3e}  {digits:8.1f}")

    literal = ", ".join(f"{e:.15f}" for e in eigs)
    print(f"\nnp.array([{literal}])")
