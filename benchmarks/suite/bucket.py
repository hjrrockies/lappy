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
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

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


def run(key, n_basis=None, rtol=None, int_npts=None, bdry_mult=2,
        preflight_pts=300, n_eigs=None, preflight_only=False, tag=''):
    from benchmarks.suite.domains import SUITE
    from benchmarks.suite import preflight as pf
    from common import build_solver, lambda_window
    from certify import certify_solver
    from lappy.core import EigensolverFailure

    entry = SUITE[key]
    dom = entry.domain()
    n_basis = n_basis or entry.n_basis
    n_eigs = n_eigs or entry.n_eigs

    solver = build_solver(dom, n_basis, rtol=rtol, bdry_mult=bdry_mult,
                          int_npts=int_npts or max(2 * n_basis, 500))
    a, b = lambda_window(dom, n_eigs)

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
               preflight=m, noisy=bool(noisy), plot=plot, tag=tag)
    t0 = time.time()
    try:
        e, mults, _ = solver.solve_interval(
            a, b, max(11 * n_eigs, 50), ltol=1e-14,
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
               eigs=[float(x) for x in eigs], seconds=time.time() - t0)

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
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    np.random.seed(args.seed)
    run(args.key, args.n_basis, args.rtol, args.int_npts, args.bdry_mult,
        args.preflight_pts, args.n_eigs, args.preflight_only, args.tag)
    return 0


if __name__ == '__main__':
    sys.exit(main())
