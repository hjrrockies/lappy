"""Solve + certify ONE suite domain, write one JSON result. Subprocess-safe.

    python -m benchmarks.suite.runner <key> [--n-basis N] [--n-eigs K]
                                            [--tag LABEL] [--no-sym]

Deliberately one domain per process: a hung or diverging domain must never take
the overnight run down with it. The driver (`sweep.py`) enforces the timeout.

Writes ``run/results/<key>__<tag>.json``. Never prints large arrays -- the
caller reads the JSON.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'reference'))

# --- memory discipline -----------------------------------------------------
# A single solve is small (~170MB peak: solver 152MB, certification 169MB).
# The danger is multiplicative. The symmetry path builds one solver PER SECTOR
# (4 for a D2 group), and manual_solve forks `n_workers` processes per sector,
# each copy-on-writing the cached basis matrices as it touches them. At
# n_workers=4 that is 16 live copies, which took one runner to 2.35GB and the
# machine to 13GB of swap -- where it thrashed instead of computing.
#
# Two guards, both here so every entry point inherits them:
#   1. Pin BLAS to one thread. With per-lambda process parallelism already in
#      manual_solve, threaded BLAS only oversubscribes (load 17 on 10 cores)
#      and adds a thread-local workspace per fork.
#   2. A hard RSS ceiling, checked from a watchdog thread; the process aborts
#      itself rather than pushing the machine into swap.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

MEM_LIMIT_MB = int(os.environ.get('LAPPY_RUN_MEM_MB', '6000'))


def _cap_address_space(limit_mb=MEM_LIMIT_MB):
    """Hard-cap this process's address space so a runaway allocation raises
    MemoryError instead of driving the machine into swap.

    An observational watchdog is not enough on macOS: memory is compressed, so
    `ru_maxrss` reports the *resident* size while the real footprint can be an
    order of magnitude larger. A `rect_thin` run was killed by an RSS watchdog
    at 4.7GB while Activity Monitor showed a 59.8GB footprint and 40GB of swap
    -- by which point the damage was done.

    RLIMIT_AS makes the allocation itself fail, at the point of the bad malloc,
    with a traceback pointing at the culprit. That turns an unbounded swap
    event into an ordinary recorded failure for one domain.
    """
    import resource
    n = limit_mb * 1024 * 1024
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = n if hard in (resource.RLIM_INFINITY, -1) else min(n, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
        return cap
    except (ValueError, OSError) as exc:
        sys.stderr.write(f'could not cap address space: {exc}\n')
        return None

RUN = os.path.join(HERE, 'run')
RESULTS = os.path.join(RUN, 'results')


def weyl_count(domain, lam):
    """Two-term Weyl count with polygonal corner correction (completeness check)."""
    n = (domain.area * lam / (4 * np.pi)
         - domain.perimeter * np.sqrt(lam) / (4 * np.pi))
    try:
        gam = np.asarray(domain.int_angles, dtype=float)
        n += np.sum((np.pi ** 2 - gam ** 2) / (24 * np.pi * gam))
    except Exception:
        pass
    return float(n)


def solve_and_certify(entry, n_basis, n_eigs, use_sym=False, n_workers=1,
                      max_recurse=None, n_pts_per_eig=11, int_npts=None,
                      bdry_mult=2, basis_kwargs=None, fs_placement='default',
                      fs_frac=0.5, fs_d=1.0, preflight_pts=300):
    """Returns a result dict. Raises on failure; the caller records that."""
    from lappy import bases, mps, MPSEigensolver
    from lappy.symmetry import domain_symmetry
    from common import build_solver, lambda_window   # manual_solve/polish_eigs retired
    from symsolve import solve_sym
    from certify import certify_sym, certify_solver

    dom = entry.domain()
    grp = entry.group() if use_sym else None
    bkw = basis_kwargs or {}

    def _build_basis(d, nb):
        """Basis for this run.

        `default` is `make_default_basis`. `boundary` replaces its
        corner-clustered `FundamentalBasis.by_corners` block with
        `by_boundary`, which distributes sources along an offset boundary.
        The default basis puts *every* function at the corners, so a domain
        whose singular corners are far apart has no representation of the
        region between them; distributing the sources is worth ~1.4 digits on
        parallelogram_p65 and halves its seed spread. See FINDINGS.md section 8.
        """
        if fs_placement == 'default':
            return bases.make_default_basis(d, nb, **bkw)
        n_fs = int(round(fs_frac * nb))
        fb = bases.FourierBesselBasis.from_domain(
            d, bases.fb_corner_orders(d, nb - n_fs))
        n_seg = len(d.bdry.segments)
        per_seg = np.full(n_seg, max(n_fs // n_seg, 1), dtype=int)
        return fb + bases.FundamentalBasis.by_boundary(d, per_seg, d=fs_d)

    if grp is not None:
        # int_npts defaults inside build_sym_solver to n_basis//|G|, i.e.
        # about ONE interior point per basis column, while the boundary gets
        # 2x oversampling. The interior block is what rules out the trivial
        # solution, so at ratio 1 it is barely determined and the answer
        # depends on which points happened to be drawn -- the suspected cause
        # of the seed variance on iso_right_tri and GWW.
        eigs, sectors, tens, solvers = solve_sym(
            dom, grp, n_basis, n_eigs, return_solvers=True, verbose=0,
            **({} if max_recurse is None else {'max_recurse': max_recurse}),
            n_pts_per_eig=n_pts_per_eig,
            int_npts=int_npts, bdry_mult=bdry_mult, **bkw)
        recs = certify_sym(solvers, dom, eigs, sectors, verbose=False)
        method = f'symmetry({grp.name}, |G|={grp.order})'
        mults = None
    else:
        if fs_placement == 'default':
            solver = build_solver(dom, n_basis, bdry_mult=bdry_mult,
                                  int_npts=int_npts or max(2 * n_basis, 500),
                                  **bkw)
        else:
            basis = _build_basis(dom, n_basis)
            n_per_seg = mps.pts_per_seg(dom, basis, mult=bdry_mult)
            bp = dom.bdry_pts(n_per_seg)
            ip = dom.int_pts(method='random',
                             npts_rand=int_npts or max(2 * n_basis, 500))
            basis = basis.to_normalized((bp, ip))
            solver = MPSEigensolver(basis, bp, ip, rtol=1e-14, ttol=1e-3)
        a, b = lambda_window(dom, n_eigs)

        # Pre-flight: characterize the tension curve BEFORE searching. An
        # ill-posed instance breaks the minimizer no matter how it is tuned, so
        # the minimizer should never see one. Records the verdict either way,
        # and supplies the abort threshold the search runs under.
        from benchmarks.suite import preflight as pf
        lam_scan, sig_scan = pf.scan(solver, a, b, n_pts=preflight_pts)
        pre = pf.metrics(dom, lam_scan, sig_scan, a, b)
        pre.update(key=entry.key, n_basis=n_basis, rtol=solver.rtol)
        noisy = pf.is_noisy(pre)

        # solve_interval with ltol=1e-14, so the minimizer extracts the best
        # answer the instance supports and any shortfall is attributable to the
        # instance (taxonomy #1-#3) rather than to the search (#4).
        e, mults, _ = solver.solve_interval(
            a, b, max(n_pts_per_eig * n_eigs, 50), ltol=1e-14,
            bracket_kwargs={'max_minima': pf.max_minima_for(pre)},
            n_workers=n_workers)
        eigs = np.asarray(e)[:n_eigs]
        tens = np.array([solver.sigma(x) for x in eigs])
        mults = np.asarray(mults)[:n_eigs]
        sectors = None
        recs = certify_solver(solver, dom, eigs, mult=mults, verbose=False)
        method = 'full domain'

    eigs = np.asarray(eigs, dtype=float)
    eps = np.array([float(r['eps']) for r in recs])
    n_distinct = len(eigs)

    # Normalize the two solve paths to "list with multiplicity". The
    # full-domain path returns *distinct* eigenvalues with a separate `mult`
    # array; the symmetry path lists a degenerate pair once per sector, so it
    # is already expanded. Comparing an unexpanded list elementwise against a
    # reference table that counts multiplicity silently misaligns everything
    # after the first degeneracy -- which is how the unit square first came
    # back with 13 certified digits and 0.2 digits against its closed form.
    if mults is not None and any(m > 1 for m in mults):
        rep = np.repeat(np.arange(len(eigs)), mults)[:n_eigs]
        eigs, eps = eigs[rep], eps[rep]
        tens = np.asarray(tens)[rep]
        recs = [recs[i] for i in rep]
    # certified digits: -log10 of the certified *relative* error
    digits = -np.log10(np.maximum(eps, 1e-300))

    out = dict(
        key=entry.key, method=method, n_basis=int(n_basis),
        n_eigs=int(n_eigs), n_found=int(len(eigs)),
        n_distinct=int(n_distinct),
        eigs=[float(x) for x in eigs],
        tensions=[float(x) for x in tens],
        eps=[float(x) for x in eps],
        abs_bound=[float(r['abs_bound']) for r in recs],
        digits=[float(x) for x in digits],
        min_digits=float(digits.min()) if len(digits) else float('nan'),
        sectors=[list(map(int, s)) for s in sectors] if sectors else None,
        mult=[int(m) for m in mults] if mults is not None else None,
        weyl_count_at_last=weyl_count(dom, float(eigs[-1])) if len(eigs) else None,
        preflight=(pre if not use_sym else None),
        preflight_noisy=(bool(noisy) if not use_sym else None),
    )

    # Strongest possible check: direct error against a closed form.
    if entry.truth_fn is not None:
        try:
            ref = np.asarray(entry.truth_fn(n_eigs), dtype=float)
            k = min(len(ref), len(eigs))
            rel = np.abs(eigs[:k] - ref[:k]) / np.abs(ref[:k])
            out['analytic_rel_err'] = [float(x) for x in rel]
            out['analytic_min_digits'] = float(
                -np.log10(max(rel.max(), 1e-300)))
        except Exception as exc:
            out['analytic_error'] = f'{type(exc).__name__}: {exc}'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--n-basis', type=int, default=None)
    ap.add_argument('--n-eigs', type=int, default=None)
    ap.add_argument('--tag', default='base',
                    help='label distinguishing configs of the same domain')
    # Symmetry reduction is OFF by default. lappy targets generic planar
    # domains -- in shape optimization, symmetry is measure-zero and the group
    # is not known in advance -- so the symmetry path is a special case most
    # real inputs never take, and benchmarking it as the default measured the
    # wrong thing. It is also frequently worse: at n_basis=120, generic vs
    # symmetry gives iso_right_tri 14.4 vs 7.1 true digits (and 43s vs 171s),
    # iso_tri_h05 8.7 vs 2.7, reg_ngon_8 9.5 vs 8.0, against a worst case of
    # -0.3 elsewhere. Pass --sym to opt back in.
    ap.add_argument('--sym', dest='use_sym', action='store_true', default=False,
                    help='opt in to symmetry-sector reduction (off by default)')
    ap.add_argument('--no-sym', dest='use_sym', action='store_false',
                    help='deprecated; full-domain is now the default')
    ap.add_argument('--workers', type=int, default=1)
    # Default None: defer to common.manual_solve, whose own default is a
    # generous backstop (30) with the noise-floor test doing the real work.
    # This used to hard-code 8, which silently overrode that and left
    # near-degenerate pairs unresolved (rect_near_deg_1e5: 7 distinct
    # eigenvalues instead of 11) even after the stopping rule was fixed.
    ap.add_argument('--max-recurse', type=int, default=None)
    ap.add_argument('--int-npts', type=int, default=None,
                    help='interior collocation points (per sector); '
                         'default is ~1 per basis column')
    ap.add_argument('--bdry-mult', type=int, default=2)
    ap.add_argument('--preflight-pts', type=int, default=300)
    ap.add_argument('--fs-placement', default='default',
                    choices=('default', 'boundary'),
                    help="'boundary' distributes fundamental sources along "
                         'an offset boundary instead of clustering them at '
                         'corners (FINDINGS.md section 8)')
    ap.add_argument('--fs-d', type=float, default=1.0)
    ap.add_argument('--fs-frac', type=float, default=None,
                    help='fraction of the basis given to fundamental solutions;\n'
                         'make_default_basis default is 0.5')
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds numpy global RNG; interior collocation '
                         'points are drawn randomly, so results are not '
                         'reproducible without this')
    ap.add_argument('--pts-per-eig', type=int, default=11)
    args = ap.parse_args(argv)

    from benchmarks.suite.domains import SUITE
    entry = SUITE[args.key]
    n_basis = args.n_basis or entry.n_basis
    n_eigs = args.n_eigs or entry.n_eigs

    _cap_address_space()
    # Interior collocation points come from domain.int_pts(method='random'),
    # which uses numpy's GLOBAL RNG (lappy.geometry:1609). Without seeding,
    # every run draws a different sample and the answer moves: iso_right_tri
    # returned 4.9, 4.0 and 2.5 certified digits on three successive runs of
    # identical code. Reference values have to be reproducible, so seed it and
    # record the seed. The spread across seeds is also a genuine diagnostic --
    # a domain whose accuracy depends strongly on the interior sample is
    # telling you the system is under-determined, not that it is unlucky.
    np.random.seed(args.seed)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f'{args.key}__{args.tag}.json')
    t0 = time.time()
    try:
        out = solve_and_certify(entry, n_basis, n_eigs,
                                use_sym=args.use_sym, n_workers=args.workers,
                                max_recurse=args.max_recurse,
                                n_pts_per_eig=args.pts_per_eig,
                                int_npts=args.int_npts,
                                bdry_mult=args.bdry_mult,
                                basis_kwargs=({'fs_frac': args.fs_frac}
                                              if args.fs_frac is not None
                                              and args.fs_placement == 'default'
                                              else None),
                                fs_placement=args.fs_placement,
                                fs_frac=(args.fs_frac
                                         if args.fs_frac is not None else 0.5),
                                fs_d=args.fs_d,
                                preflight_pts=args.preflight_pts)
        out['ok'] = True
    except Exception as exc:
        out = dict(key=args.key, n_basis=n_basis, n_eigs=n_eigs, ok=False,
                   error=f'{type(exc).__name__}: {exc}',
                   traceback=traceback.format_exc()[-3000:])
    out['tag'] = args.tag
    out['seed'] = args.seed
    out['int_npts'] = args.int_npts
    out['bdry_mult'] = args.bdry_mult
    out['fs_frac'] = args.fs_frac
    out['fs_placement'] = args.fs_placement
    out['seconds'] = time.time() - t0
    out['use_sym'] = args.use_sym
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)

    # one compact line for the caller; never dump arrays
    if out['ok']:
        extra = ''
        if 'analytic_min_digits' in out:
            extra = f" analytic={out['analytic_min_digits']:.1f}"
        print(f"{args.key} tag={args.tag} nb={n_basis} "
              f"certified_digits={out['min_digits']:.1f}{extra} "
              f"found={out['n_found']}/{n_eigs} {out['seconds']:.0f}s")
    else:
        print(f"{args.key} tag={args.tag} nb={n_basis} FAILED {out['error'][:120]}")
    return 0 if out['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
