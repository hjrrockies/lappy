"""Where does the memory go, and what holds it? Byte attribution per cache, per iterate.

    uv run python -m benchmarks.memory.cache_probe [--iters 60] [--n-eigs 6] [--gc-disable]

WHY THIS EXISTS. A `douse` grid at N=6 with five workers took a 16GB machine from 0 to 15.4GB of
swap in forty minutes on 2026-08-26, and the same shape of run kernel-panicked it at N=12. Worker
RSS sawtoothed upward, 1.0 -> 2.7 GB. Three mechanisms were proposed; this measures which of them
actually holds the bytes, BEFORE anything is changed. The project has twice fixed a wrongly
diagnosed cause, so the fix is gated on this.

WHAT IT REPRODUCES. `douse` builds a FRESH solver per iterate (`douse/evaluate.py:153`) and then,
for each eigenvalue, calls `eigenfunction_coef` and `eigfun_cauchy_data` against
`solver.hadamard_quad` (`douse/derivative.py:101-103`). That second call is the one under
suspicion: it wraps raw arrays in new `PointSet`s (`eigfun_integrals.py:506-508`), and `PointSet`
has no `__hash__`, so identity-keyed caches see a new key every time. The loop below is that loop,
written in `lappy` primitives only so this benchmark does not invert the dependency on `douse`.

READ `swap_used_mb`, NOT `ru_maxrss`. `benchmarks/suite/guards.py` explains why at length: macOS
compresses memory, so a 59.8GB runaway once read as 4.7GB resident. Both are recorded here; only
one of them is trustworthy.

THE THREE DISCRIMINATORS
  * mechanism B (the `instance_lru_cache` self-cycle defers reclamation) -- run with
    `--gc-disable`. If memory climbs monotonically with the collector off and sawtooths with it on,
    the cycle is what makes the sawtooth.
  * mechanism A (identity-keyed caches, fresh keys) -- read `entries` per cache. The design intent
    (`bases.py:283-284`) is ~2 point sets. A count that grows with the eigenvalue count, or with
    the iterate, is the leak.
  * mechanism C (entry counts bound large values) -- read `bytes` for `_raw_eval`/`_raw_grad_eval`.
"""
import argparse
import gc
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suite import guards                                             # noqa: E402

from lappy import basis_plan as BP                                   # noqa: E402
from lappy import geometry as geo                                    # noqa: E402
from lappy.asymp import weyl_est                                     # noqa: E402
from lappy.eigfun_integrals import eigfun_cauchy_data                # noqa: E402
from lappy.evp import Eigenproblem                                   # noqa: E402
from lappy.mps import MPSEigensolver                                 # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
PERIM = 4.0


def _arrays_under(obj, max_depth=4):
    """Every ndarray reachable from `obj` within `max_depth` hops of `gc.get_referents`.

    `functools.lru_cache` is a C object that exposes `cache_info()` but not its values, so the
    only way to weigh what it holds is to walk the object graph. Verified to recover exactly the
    cached arrays for a plain `@lru_cache` function. That this is necessary at all is an argument
    for an explicit cache object: bytes held should be a property you can ask for.
    """
    # The root itself counts. `instance_cache` stores bare ndarrays as dict values, so a walker
    # that only inspects REFERENTS reports 0 bytes for every one of them -- which it did, on the
    # first run of this probe, and would have exonerated exactly the caches under suspicion.
    if isinstance(obj, np.ndarray):
        return [obj]
    seen, out, frontier = {id(obj)}, [], [(obj, 0)]
    while frontier:
        o, d = frontier.pop()
        if d > max_depth:
            continue
        for ref in gc.get_referents(o):
            if id(ref) in seen:
                continue
            seen.add(id(ref))
            if isinstance(ref, np.ndarray):
                out.append(ref)
            else:
                frontier.append((ref, d + 1))
    return out


def _nbytes(arrs):
    """Sum `nbytes`, charging each distinct base buffer once (a view pins the whole thing)."""
    seen, total = set(), 0
    for a in arrs:
        root = a
        while root.base is not None:
            root = root.base
        if not isinstance(root, np.ndarray) or id(root) in seen:
            continue
        seen.add(id(root))
        total += root.nbytes
    return total


def cache_report(obj, recurse=('basis', 'bases', 'solver'), _seen=None):
    """`{cache_name: {'entries': n, 'bytes': b}}` for `obj` and the objects it holds.

    Mirrors `lappy.cache.clear_instance_caches`'s recursion set, so what this measures is exactly
    what that function would drop.
    """
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return {}
    _seen.add(id(obj))
    rep = {}
    for key, val in getattr(obj, '__dict__', {}).items():
        if not key.startswith('_icache_') or key.endswith('_lock'):
            continue
        name = key[len('_icache_'):]
        if isinstance(val, dict):                       # instance_cache: a plain dict
            entries, arrs = len(val), []
            for v in val.values():
                arrs.extend(_arrays_under(v, max_depth=2))
        else:                                           # instance_lru_cache: an lru_cache object
            try:
                entries = val.cache_info().currsize
            except AttributeError:
                continue
            arrs = _arrays_under(val, max_depth=4)
        cur = rep.setdefault(name, {'entries': 0, 'bytes': 0})
        cur['entries'] += entries
        cur['bytes'] += _nbytes(arrs)
    for attr in recurse:
        child = getattr(obj, attr, None)
        if child is None:
            continue
        for c in (child if isinstance(child, (list, tuple)) else [child]):
            for k, v in cache_report(c, recurse, _seen).items():
                cur = rep.setdefault(k, {'entries': 0, 'bytes': 0})
                cur['entries'] += v['entries']
                cur['bytes'] += v['bytes']
    return rep


def hexagon(t, N=6, perim=PERIM):
    """A regular N-gon of fixed perimeter, perturbed by `t`. Stands in for a `douse` iterate."""
    v = perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)
    rng = np.random.default_rng(12345)
    v = v + t*(rng.normal(size=N) + 1j*rng.normal(size=N))*np.abs(v).mean()
    v = v*perim/np.abs(np.diff(np.concatenate([v, v[:1]]))).sum()
    return geo.Polygon(v, bc='dir')


def one_iterate(plan, t, n_eigs, lam_window, prec, seed):
    """Exactly what `douse` does per iterate: fresh solver, track the set, then the gradient path.

    Returns `(lams, report)`. The solver is dropped on return, so anything still held afterwards
    is held by something other than this frame.
    """
    dom = hexagon(t)
    solver = MPSEigensolver.from_domain(dom, lam_max=weyl_est(lam_window, dom),
                                        basis=BP.realize(plan, dom), prec=prec)
    evp = Eigenproblem(dom, eval_solver=solver, precision=prec)
    if seed is None:
        lams = np.asarray(evp.solve(n_eigs), dtype=float)
    else:
        cand, _mults = evp.track_set(np.asarray(seed[:n_eigs], dtype=float))
        lams = np.asarray(cand, dtype=float)
    lams = np.sort(lams)[:n_eigs]

    # THE SUSPECT PATH. `douse/derivative.py:101-103`, once per eigenvalue.
    bq = solver.hadamard_quad
    for lam in lams:
        coef = solver.eigenfunction_coef(float(lam), mult=1)
        eigfun_cauchy_data(solver.basis, float(lam), coef, bq)

    return lams, cache_report(solver)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--n-eigs', type=int, default=6)
    ap.add_argument('--prec', type=float, default=1e-13)
    ap.add_argument('--target', type=float, default=1e-14)
    ap.add_argument('--gc-disable', action='store_true',
                    help='isolates mechanism B: with the collector off, a cycle cannot be '
                         'reclaimed at all, so growth becomes monotone instead of sawtoothing')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    guards.install(label='cache_probe')
    os.makedirs(OUT, exist_ok=True)
    tag = args.tag or ('gcoff' if args.gc_disable else 'gcon')
    path = os.path.join(OUT, f'cache_probe_{tag}.jsonl')

    dom0 = hexagon(0.0)
    plan = BP.plan_basis(dom0, weyl_est(max(8, 2*args.n_eigs + 4), dom0), target=args.target)
    print(f'plan: n_total={plan.n_total} (cap {plan.cfg.n_cap})', flush=True)

    if args.gc_disable:
        gc.collect()
        gc.disable()
        print('GC DISABLED -- monotone growth here means a cycle is holding it', flush=True)

    swap0 = guards.swap_used_mb()
    seed, rows = None, []
    try:
        for i in range(args.iters):
            t0 = time.perf_counter()
            lams, rep = one_iterate(plan, 0.004*i, args.n_eigs,
                                    max(8, 2*args.n_eigs + 4), args.prec, seed)
            seed = lams
            row = dict(i=i, seconds=round(time.perf_counter() - t0, 2),
                       swap_mb=round(guards.swap_used_mb() - swap0, 1),
                       maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576, 1),
                       gc_count=list(gc.get_count()),
                       gen2=gc.get_stats()[2]['collections'],
                       cache_bytes=sum(v['bytes'] for v in rep.values()),
                       cache_entries=sum(v['entries'] for v in rep.values()),
                       caches={k: v for k, v in sorted(rep.items())},
                       lam1=float(lams[0]))
            rows.append(row)
            with open(path, 'a') as fh:
                fh.write(json.dumps(row) + '\n')
            if i % 5 == 0 or i == args.iters - 1:
                print(f"  {i:3d}: {row['seconds']:5.1f}s  swap +{row['swap_mb']:7.1f}MB  "
                      f"maxrss {row['maxrss_mb']:6.1f}MB  cache "
                      f"{row['cache_bytes']/1048576:7.1f}MB / {row['cache_entries']:4d} entries  "
                      f"gen2={row['gen2']}", flush=True)
    finally:
        if args.gc_disable:
            gc.enable()

    print(f'\nwrote {path}')
    if rows:
        last = rows[-1]['caches']
        print('\nbytes held at the last iterate, by cache (this is the attribution):')
        for k, v in sorted(last.items(), key=lambda kv: -kv[1]['bytes']):
            if v['bytes'] or v['entries']:
                print(f"  {k:52s} {v['entries']:5d} entries {v['bytes']/1048576:9.2f} MB")


if __name__ == '__main__':
    main()
