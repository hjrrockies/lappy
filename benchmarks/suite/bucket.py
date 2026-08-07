"""Bucket ONE domain, carefully. Pre-flight first, then solve, then judge.

    python -m benchmarks.suite.bucket <key> [--n-basis N] [--rtol R]
                                            [--int-npts M] [--preflight-only]

Buckets:
  1  8+ digits with make_default_basis at some n_basis
  2  under 8 digits, but no solver failure and no missing eigenvalues
  3  solver failure and/or missing eigenvalues

The bar is true error where a closed form exists, certified otherwise; both are
reported. Missing eigenvalues are the failure that certification cannot see --
every value can carry a valid Moler--Payne bound while the list is wrong in
every entry after a gap -- so for analytic domains they are checked
element-by-element against the closed form.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'reference'))
from benchmarks.suite import guards
guards.pin_blas()

OUT = os.path.join(HERE, 'run', 'buckets.jsonl')


def compare_to_truth(eigs, mults, ref, rtol=1e-6):
    """Element-wise check against a closed form, expanding by multiplicity.

    Returns (digits, missing, extra). `missing` lists exact eigenvalues with no
    computed counterpart -- the silent failure mode.
    """
    eigs = np.asarray(eigs, float)
    if mults is not None and len(mults) and any(m > 1 for m in mults):
        rep = np.repeat(np.arange(len(eigs)), np.asarray(mults, int))
        listed = eigs[rep]
    else:
        listed = eigs
    k = min(len(listed), len(ref))
    missing = [float(x) for x in ref
               if len(eigs) == 0 or np.min(np.abs(eigs - x)) > rtol * abs(x)]
    if k == 0:
        return float('nan'), missing, []
    rel = np.abs(listed[:k] - ref[:k]) / np.abs(ref[:k])
    return float(-np.log10(max(rel.max(), 1e-300))), missing, []


def make_basis(dom, n_basis, fs_placement='default', fs_d=1.0, fs_frac=0.5):
    """Basis for a trial. `default` is `make_default_basis`; `boundary` replaces its
    corner-clustered fundamental block with sources on an offset boundary.

    The offset `fs_d` is the lever, not the fraction: on parallelogram_p65 the tension at
    lam_1 goes 1.4e-09 (default) -> 8.9e-11 (d=1.0) -> 1.8e-12 (d=0.2-0.4), while fs_frac
    0.5 vs 0.75 is indistinguishable. FINDINGS.md section 8 tested only d=1.0."""
    from lappy import bases
    if fs_placement == 'default':
        return bases.make_default_basis(dom, n_basis)
    # A corner-free domain has no Fourier-Bessel block to keep: make_default_basis gives it
    # pure fundamental solutions, and the only question is the offset. That is exactly where
    # the hard-coded fs_d=1.0 does most damage -- stadium keeps 78 of 324 columns at d=1.0 and
    # 320 of 320 at d=0.1.
    # A source that lands INSIDE Omega is not a legitimate particular solution: it has a pole
    # in the domain, so it solves Helmholtz nowhere near it, the MPS premise fails, and
    # Moler-Payne's hypothesis (u exact in Omega) is void with it. Normal-offset placement does
    # this routinely on a strongly reentrant domain -- chevron_2_3's 305-degree corner leaves a
    # 55-degree exterior wedge, and a perpendicular offset from one arm lands in the other for
    # ANY d (24 of 240 sources at d=0.4, still 8 at d=0.05). Symptom: the tension background
    # collapses to ~3e-07 across the whole window, ttol stops discriminating, the search accepts
    # noise and misses real eigenvalues. Filtering restores a 3e+07 contrast.
    if len(dom.corners) == 0:
        return bases.FundamentalBasis.by_boundary(
            dom, bases.fs_bdry_sps(dom, n_basis, order=1), d=fs_d, order=1)
    n_fs = int(round(fs_frac*n_basis))
    fb = bases.FourierBesselBasis.from_domain(dom, bases.fb_corner_orders(dom, n_basis - n_fs))
    per_seg = bases.fs_bdry_sps(dom, n_fs, order=1)
    return fb + _fs_outside(dom, per_seg, fs_d)


def _fs_outside(dom, per_seg, d):
    """`by_boundary` sources, with any that land inside the domain dropped (see above)."""
    from matplotlib.path import Path
    b = dom.bdry_pts(per_seg)
    nrm = dom.bdry_normals(per_seg)
    src = b.pts + d*nrm.pts
    poly = np.concatenate([sg.p(np.linspace(0, 1, 600)) for sg in dom.bdry.segments])
    inside = Path(np.column_stack([poly.real, poly.imag])).contains_points(
        np.column_stack([src.real, src.imag]))
    if inside.any():
        print(f'BASIS      dropped {int(inside.sum())}/{len(src)} sources that landed inside')
    return bases.FundamentalBasis(src[~inside], 1)


def run(key, n_basis=None, rtol=None, int_npts=None, bdry_mult=2,
        preflight_pts=300, n_eigs=None, preflight_only=False, tag='',
        orthonorm=True, fs_placement='default', fs_d=1.0, fs_frac=0.5,
        pts_per_eig=11):
    from benchmarks.suite.domains import SUITE
    from benchmarks.suite import preflight as pf
    from common import build_solver, lambda_window
    from certify import certify_solver, summarize_l2
    from lappy.core import EigensolverFailure

    entry = SUITE[key]
    n_basis = n_basis or entry.n_basis
    n_eigs = n_eigs or entry.n_eigs

    # Construction can fail before any solve: spiral_t25's basis raises
    # `corner_branch_cut_polyline: no valid polyline cut found` because coils
    # bury corners with no sightline to infinity. That is a bucket-3 failure
    # like any other, and has to be recorded rather than killing the process
    # and leaving a hole in the table.
    try:
        dom = entry.domain()
        a, b = lambda_window(dom, n_eigs)
        # `orthonorm` attaches the corner-adapted boundary quadrature, so the
        # eigenfunctions come out L^2-orthonormal and certification takes its
        # norm from the boundary rather than interior cubature. `b` is the top
        # of the search window, which is exactly the `lam_max` the node set has
        # to be sized for.
        solver = build_solver(dom, n_basis, rtol=rtol, bdry_mult=bdry_mult,
                              int_npts=int_npts or max(2 * n_basis, 500),
                              lam_max=b, orthonorm=orthonorm,
                              basis=make_basis(dom, n_basis, fs_placement, fs_d, fs_frac))
    except Exception as ex:
        rec = dict(key=key, n_basis=n_basis, bucket=3, tag=tag,
                   error=f'construction: {type(ex).__name__}: {ex}')
        print(f'BUILD      FAILED {type(ex).__name__}: {ex}')
        print('BUCKET     3')
        _record(rec)
        return rec

    # --- pre-flight, no search -------------------------------------------
    t0 = time.time()
    lam, sig = pf.scan(solver, a, b, n_pts=preflight_pts)
    m = pf.metrics(dom, lam, sig, a, b)
    m.update(key=key, n_basis=n_basis, rtol=solver.rtol, int_npts=int_npts)
    name = f'{key}{("__" + tag) if tag else ""}.png'
    plot = pf.plot_curve(lam, sig,
                         f'{key}  n_basis={n_basis}  rtol={solver.rtol:.0e}  '
                         f'minima={m["n_minima"]} vs Weyl {m["n_expected"]:.1f}',
                         os.path.join(pf.CURVES, name))
    noisy = pf.is_noisy(m)
    print(f'PREFLIGHT  {pf.summary(m)}')
    print(f'           verdict={"NOISY" if noisy else "clean"}  plot={plot}  '
          f'({time.time()-t0:.0f}s)')
    if preflight_only:
        return dict(key=key, preflight=m, noisy=noisy, plot=plot)

    # --- solve -------------------------------------------------------------
    rec = dict(key=key, n_basis=n_basis, rtol=solver.rtol, int_npts=int_npts,
               preflight=m, noisy=bool(noisy), plot=plot, tag=tag,
               orthonorm=bool(orthonorm), pts_per_eig=int(pts_per_eig),
               fs_placement=fs_placement,
               fs_d=float(fs_d), fs_frac=float(fs_frac),
               bq_nodes=(len(solver.bdry_quad.pts) if solver.bdry_quad else None),
               bq_precision=(float(solver.bdry_quad.precision)
                             if solver.bdry_quad else None))
    t0 = time.time()
    try:
        # Grid density for the initial bracket scan. A BETTER basis makes this MORE
        # important, not less: its minima are deeper and narrower, so a grid that found
        # them with a poor basis can step straight over them. chevron_2_3 at n_basis=480
        # with distributed sources skipped a true eigenvalue at 226.6204 that the
        # n_basis=160 default basis found, and certification cannot see the gap.
        e, mults, _ = solver.solve_interval(
            a, b, max(pts_per_eig * n_eigs, 50), ltol=1e-14,
            bracket_kwargs={'max_minima': pf.max_minima_for(m)})
        eigs = np.asarray(e)[:n_eigs]
        mults = np.asarray(mults)[:n_eigs]
    except EigensolverFailure as ex:
        rec.update(bucket=3, error=f'EigensolverFailure: {ex}',
                   seconds=time.time() - t0)
        print(f'SOLVE      ABORTED by guard: {ex}')
        _record(rec)
        return rec
    except Exception as ex:
        rec.update(bucket=3, error=f'{type(ex).__name__}: {ex}',
                   seconds=time.time() - t0)
        print(f'SOLVE      FAILED {type(ex).__name__}: {ex}')
        _record(rec)
        return rec

    # --- certify + judge ---------------------------------------------------
    try:
        recs = certify_solver(solver, dom, eigs, mult=mults, verbose=False)
        cert = float(-np.log10(max(max(r['eps'] for r in recs), 1e-300)))
    except Exception as ex:
        rec.update(bucket=3, error=f'certify: {type(ex).__name__}: {ex}',
                   n_found=len(eigs), seconds=time.time() - t0)
        print(f'CERTIFY    FAILED {type(ex).__name__}: {ex}')
        _record(rec)
        return rec

    total = int(np.sum(mults)) if mults is not None and len(mults) else len(eigs)
    rec.update(n_distinct=len(eigs), n_listed=total, certified=cert,
               mult=[int(x) for x in mults],
               eigs=[float(x) for x in eigs], seconds=time.time() - t0,
               **summarize_l2(recs))

    true_dig, missing = None, []
    if entry.truth_fn is not None:
        ref = np.asarray(entry.truth_fn(n_eigs), float)
        true_dig, missing, _ = compare_to_truth(eigs, mults, ref)
        rec.update(true_digits=true_dig, missing=missing)

    bar = true_dig if true_dig is not None else cert
    if missing or total < n_eigs:
        bucket = 3
    elif bar >= 8.0:
        bucket = 1
    else:
        bucket = 2
    rec['bucket'] = bucket

    td = f'{true_dig:.1f}' if true_dig is not None else '--'
    print(f'SOLVE      distinct={len(eigs)} listed={total}/{n_eigs}  '
          f'certified={cert:.1f}  true={td}  ({rec["seconds"]:.0f}s)')
    if rec.get('l2_spread_max') is not None:
        print(f'NORM       {"/".join(rec["l2_methods"])}  '
              f'x0-spread<={rec["l2_spread_max"]:.1e}  '
              f'offdiag<={rec["gram_offdiag_max"]:.1e}  '
              f'bq_nodes={rec["bq_nodes"]}  bq_prec={rec["bq_precision"]:.1e}')
    else:
        print(f'NORM       {"/".join(rec["l2_methods"])}')
    if missing:
        print(f'           MISSING {len(missing)}: '
              f'{[f"{x:.6f}" for x in missing[:4]]}')
    print(f'BUCKET     {bucket}')
    _record(rec)
    return rec


def _record(rec):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--n-basis', type=int, default=None)
    ap.add_argument('--rtol', type=float, default=None)
    ap.add_argument('--int-npts', type=int, default=None)
    ap.add_argument('--bdry-mult', type=int, default=2)
    ap.add_argument('--preflight-pts', type=int, default=300)
    ap.add_argument('--n-eigs', type=int, default=None)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--pts-per-eig', type=int, default=11,
                    help='grid points per eigenvalue for the bracket scan')
    ap.add_argument('--fs-placement', default='default', choices=('default','boundary'))
    ap.add_argument('--fs-d', type=float, default=1.0)
    ap.add_argument('--fs-frac', type=float, default=0.5)
    ap.add_argument('--no-orthonorm', dest='orthonorm', action='store_false',
                    default=True,
                    help='certify from interior cubature instead of the '
                         'boundary (the pre-orthonormalization path)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--swap-mb', type=float, default=None,
                    help='abort if system swap grows this far past baseline')
    ap.add_argument('--timeout', type=float, default=None,
                    help='abort after this many seconds')
    args = ap.parse_args(argv)
    # Abort rather than take the machine down: swap growth and wall time are
    # the two ways a solve has gone wrong in this run. See guards.py.
    guards.install(swap_mb=args.swap_mb, timeout_s=args.timeout, label=args.key)
    np.random.seed(args.seed)
    run(args.key, args.n_basis, args.rtol, args.int_npts, args.bdry_mult,
        args.preflight_pts, args.n_eigs, args.preflight_only, args.tag,
        orthonorm=args.orthonorm, fs_placement=args.fs_placement,
        fs_d=args.fs_d, fs_frac=args.fs_frac, pts_per_eig=args.pts_per_eig)
    return 0


if __name__ == '__main__':
    sys.exit(main())
