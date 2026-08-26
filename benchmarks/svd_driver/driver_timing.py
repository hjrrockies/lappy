"""Is `gesvd` fast enough to be the DEFAULT SVD driver in `regularize_pencil`?

    uv run python -m benchmarks.svd_driver.driver_timing [--domains square reg_ngon_8 ...]

THE QUESTION. `regularize_pencil` now tries LAPACK's `gesdd` and falls back to `gesvd` on
`LinAlgError` (see `mps._svd_gesdd_then_gesvd`). `gesvd` is the more robust of the two -- it
converges on inputs `gesdd` gives up on -- so if it is not meaningfully slower at the sizes this
module actually uses, the fallback should simply become the default and the failure mode goes away
rather than being recovered from.

"Meaningfully slower" has to be measured against the SOLVE, not against the other driver. A 2x
slowdown in the SVD is irrelevant if the SVD is 3 per cent of a solve, and decisive if it is 40.
So this reports three things:

  1. the per-call cost of each driver on REAL `R` matrices captured from real solves;
  2. what fraction of a full `solve()` is spent inside the SVD at all;
  3. that the two drivers agree, so switching is not a numerical change.

WHY REAL MATRICES. `R` here is the triangular factor of an economic QR of the stacked pencil, so
it is square of side `n_total` and is very far from a random matrix -- it is graded, with singular
values spanning the whole dynamic range down to the `rtol` cutoff. `gesdd`'s advantage over
`gesvd` depends on the spectrum, so timing random Gaussians would answer a different question.

THE DRIVERS ARE INTERLEAVED AND MEDIANS ARE REPORTED, because this is a laptop that may be running
other jobs; alternating them makes any contention hit both roughly equally, and the median throws
out the worst of it.

THE ANSWER: NO -- KEEP `gesdd` WITH THE FALLBACK. The verdict depends on `n`, and it flips inside
the range this module actually uses. Measured at `prec=1e-13`, `k=4`:

    domain          n     svd/solve    gesdd      gesvd     ratio    per-solve cost
    square        104        7.6%      0.70ms     0.94ms    1.34x        +2.6%
    reg_ngon_5    140        8.1%      1.16ms     1.69ms    1.46x        +3.7%
    L_shape       191       15.4%      2.01ms     3.51ms    1.75x       +11.5%
    reg_ngon_8    240        8.0%      2.83ms     6.67ms    2.35x       +10.9%
    GWW1          319       14.2%      5.32ms    11.67ms    2.19x       +17.0%
    ngon12        372        8.1%      7.26ms    22.34ms    3.08x       +16.8%
    H_shape       432        8.0%     10.15ms    36.00ms    3.55x       +20.3%
    ngon16        512        9.4%     20.29ms   327.25ms   16.13x      +141.7%

At `n = 104` making `gesvd` the default would cost under 3 per cent of a solve and would be worth
it for the simplicity. At `n = 512` it costs 2.4x the whole solve. The `douse` scope runs to
N = 16, which is exactly the bad end, so the blanket default is not viable.

WHY IT DEGRADES SO SHARPLY, and it is not contention or a threading artifact -- both were checked.
The cost is entirely in the singular VECTORS. On the same 512x512 matrix, 9 repeats:

    full SVD        gesdd  20.00 ms    gesvd 340.34 ms    17.02x
    values only     gesdd  12.84 ms    gesvd  11.94 ms     0.93x

With `compute_uv=False` `gesvd` is marginally FASTER. That is the whole algorithmic difference:
`gesdd` is divide-and-conquer and accumulates vectors cheaply, `gesvd` builds them by QR
iteration, and the gap widens with `n`. `regularize_pencil` needs `Z` and `Yt`, so it cannot take
the cheap half.

(The matrices are numerically singular by construction -- `cond ~ 1e16` on that 512x512 -- which
is what the `rtol` cutoff exists for, and also why `gesdd` occasionally fails on one.)

SO THE FALLBACK IS THE RIGHT SHAPE: `gesdd` on the ~99.9 per cent of calls that succeed, `gesvd`
on the one in ~1500 solves that does not. Paying 17x once, rarely, is free; paying it always is
not. The two drivers agree on the singular values to 3.6e-15 relative, so a run that switches
drivers mid-flight sees a perturbation about six orders below `douse`'s `replan_tol` of 1e-9 --
there is no discontinuity in the objective to worry about.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import scipy.linalg as la

from lappy import Eigenproblem
from lappy import mps as _mps
from lappy.asymp import weyl_est

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
DOMAINS = ['square', 'reg_ngon_5', 'reg_ngon_8', 'L_shape', 'chevron_1_15', 'GWW1',
           'H_shape', 'ngon12', 'ngon16']


def _build(name):
    """A suite domain by key, or `ngon<N>` for a regular N-gon built directly.

    The suite stops at `reg_ngon_8` (n = 240), and the `douse` scope runs to N = 16 (n = 560),
    which is where the driver ratio matters most -- it GROWS with `n`, so extrapolating from the
    suite would understate the cost at exactly the sizes the decision is about.
    """
    if name.startswith('ngon'):
        from lappy.geometry import reg_ngon
        return reg_ngon(int(name[4:]))
    from benchmarks.suite.domains import SUITE
    return SUITE[name].build()


def capture(name, k=4, prec=1e-13, max_keep=6):
    """Run a real `solve` and capture the `R` matrices the SVD was actually handed.

    Also accumulates the time spent inside `la.svd`, which is what turns a per-call number into a
    fraction of a solve.
    """
    dom = _build(name)
    kept, spent, calls = [], 0.0, 0
    real_svd = la.svd

    def timed_svd(a, *args, **kwargs):
        nonlocal spent, calls
        calls += 1
        if len(kept) < max_keep and calls % 7 == 1:
            kept.append(np.array(a, copy=True))
        t = time.perf_counter()
        try:
            return real_svd(a, *args, **kwargs)
        finally:
            spent += time.perf_counter() - t

    _mps.la.svd = timed_svd
    try:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            evp = Eigenproblem(dom, precision=prec)
            evp.solve(k)
        wall = time.perf_counter() - t0
    finally:
        _mps.la.svd = real_svd
    return kept, spent, wall, calls


def time_drivers(mats, repeats=5):
    """Interleaved medians for both drivers, plus the worst singular-value disagreement."""
    fast, slow, worst = [], [], 0.0
    for R in mats:
        tf, ts = [], []
        for _ in range(repeats):
            t = time.perf_counter(); s_f = la.svd(R)[1]; tf.append(time.perf_counter() - t)
            t = time.perf_counter(); s_s = la.svd(R, lapack_driver='gesvd')[1]
            ts.append(time.perf_counter() - t)
        fast.append(np.median(tf))
        slow.append(np.median(ts))
        scale = max(float(s_f[0]), 1e-300)
        worst = max(worst, float(np.max(np.abs(s_f - s_s))/scale))
    return float(np.median(fast)), float(np.median(slow)), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='*', default=DOMAINS)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()

    rows = []
    print('  domain           n     calls   svd/solve    gesdd      gesvd    ratio   agree')
    print('  ' + '-'*77)
    for name in args.domains:
        try:
            mats, spent, wall, calls = capture(name, args.k)
            if not mats:
                print(f'  {name:14s} (no SVD calls captured)')
                continue
            f, s, worst = time_drivers(mats, args.repeats)
            n = mats[0].shape[0]
            frac = spent/wall if wall > 0 else float('nan')
            # What the solve would have cost with gesvd throughout: the measured SVD time scaled
            # by the per-call ratio, plus everything else unchanged.
            proj = (wall - spent) + spent*(s/f if f > 0 else 1.0)
            rows.append(dict(domain=name, n=int(n), calls=calls, svd_frac=frac,
                             gesdd=f, gesvd=s, ratio=(s/f if f > 0 else None),
                             wall=wall, projected_wall=proj, worst_rel_disagreement=worst))
            print(f'  {name:14s} {n:5d} {calls:7d}   {100*frac:6.1f}%  {1e3*f:8.2f}ms '
                  f'{1e3*s:8.2f}ms  {s/f:6.2f}x  {worst:7.1e}')
        except Exception as exc:                                       # noqa: BLE001
            print(f'  {name:14s} FAILED {type(exc).__name__}: {exc}')

    if rows:
        print()
        print('  projected effect of making gesvd the default, per solve:')
        print('  domain              measured    projected    change')
        print('  ' + '-'*50)
        for r in rows:
            d = 100*(r['projected_wall']/r['wall'] - 1)
            print(f"  {r['domain']:14s} {r['wall']:9.2f}s {r['projected_wall']:11.2f}s "
                  f"{d:+8.1f}%")
        agree = max(r['worst_rel_disagreement'] for r in rows)
        print(f'\n  worst relative disagreement in singular values, any domain: {agree:.2e}')
        # MERGED, not overwritten -- this gets run on subsets (`--domains ngon16`) and an
        # earlier version dropped every row not in the last invocation.
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, 'driver_timing.json')
        merged = {}
        if os.path.exists(path):
            for r in json.load(open(path)):
                merged[r['domain']] = r
        for r in rows:
            merged[r['domain']] = r
        with open(path, 'w') as fh:
            json.dump(sorted(merged.values(), key=lambda r: r['n']), fh, indent=1)
        print(f'  written to {path} ({len(merged)} domain(s) on record)')


if __name__ == '__main__':
    main()
