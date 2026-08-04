"""Production run: recompute every reference domain with certified error bars.

For each domain: solve (symmetry-reduced where a symmetry exists, full-domain
otherwise), verify completeness against the two-term Weyl count, certify each
eigenvalue with Moler--Payne, and emit a ready-to-paste table.

Usage:  python produce.py [domain_key ...]      (default: all)
Results are appended to ``results_certified.json`` so runs can be resumed.
"""
import json
import os
import sys
import time

import numpy as np

from lappy import geometry as G, reference, asymp
from lappy.symmetry import domain_symmetry
from common import solve_domain_v2, build_solver, manual_solve, polish_eigs, lambda_window
from symsolve import solve_sym
from certify import certify_sym, certify_solver

# The domain list lives in the curated suite (benchmarks/suite/) rather than
# here, so that "which domains do we test on and why" is answered in one place.
# These scripts are run with this directory as cwd and use flat imports, so the
# repo root has to go on the path explicitly before the package import works.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from benchmarks.suite.domains import for_reference_production  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_certified.json')

# key -> (builder, symmetry name+params, n_basis, n_eigs)
DOMAINS = for_reference_production()

# Differences from the pre-suite list: the redundant iso_tri heights h=2, 8 and
# 20 are gone (near-duplicates of h=0.5/4/16 on every axis; h=1 stays because it
# is the right isoceles triangle, so its MPS values can be checked against a
# closed form), and the new suite entries (parallelogram, right_trapezoid, the
# mushroom neck sweep, stadium_L2) are in. chevron(1,1.25) and the spirals are
# excluded as status='open'; pass include_open=True to attempt them anyway.


def weyl_count(domain, lam):
    """Two-term Weyl count with the polygonal corner correction.

    ``N(lam) ~ |Omega| lam /(4 pi) - |dOmega| sqrt(lam)/(4 pi)
               + sum_j (pi^2 - gamma_j^2)/(24 pi gamma_j)``

    Used only as a sanity check on *how many* eigenvalues we should have
    found below a given cut -- it is asymptotic, so a discrepancy of one or
    two at these small ``lam`` means nothing on its own, but a discrepancy
    that grows with the index means a missed or spurious eigenvalue.
    """
    n = domain.area * lam / (4 * np.pi) - domain.perimeter * np.sqrt(lam) / (4 * np.pi)
    try:
        gam = np.asarray(domain.int_angles, dtype=float)
        n += np.sum((np.pi ** 2 - gam ** 2) / (24 * np.pi * gam))
    except Exception:
        pass
    return n


def run_one(key, verbose=True):
    build, sym, n_basis, n_eigs = DOMAINS[key]
    dom = build()
    t0 = time.time()

    if sym is not None:
        grp = domain_symmetry(sym[0], **sym[1])
    else:
        grp = None

    if grp is not None:
        eigs, sectors, tens, solvers = solve_sym(dom, grp, n_basis, n_eigs,
                                                 return_solvers=True, verbose=verbose)
        recs = certify_sym(solvers, dom, eigs, sectors, verbose=False)
        method = f'symmetry({grp.name}, |G|={grp.order})'
    else:
        solver = build_solver(dom, n_basis, int_npts=max(2 * n_basis, 500))
        a, b = lambda_window(dom, n_eigs)
        e, mults, _ = manual_solve(solver, a, b, max(11 * n_eigs, 50), n_workers=4)
        eigs, tens = polish_eigs(solver, e, ltol=1e-14, bracket_rel_width=1e-9)
        eigs, tens, mults = eigs[:n_eigs], tens[:n_eigs], mults[:n_eigs]
        sectors = None
        recs = certify_solver(solver, dom, eigs, mult=mults, verbose=False)
        method = 'full domain'

    # completeness: how many eigenvalues Weyl expects below the last one found
    n_weyl = weyl_count(dom, eigs[-1]) if len(eigs) else float('nan')

    out = dict(key=key, method=method, n_basis=n_basis,
               eigs=[float(x) for x in eigs],
               tensions=[float(x) for x in tens],
               eps=[float(r['eps']) for r in recs],
               abs_bound=[float(r['abs_bound']) for r in recs],
               sectors=[list(s) for s in sectors] if sectors else None,
               weyl_count_at_last=float(n_weyl), n_found=len(eigs),
               seconds=time.time() - t0)

    if verbose:
        print(f'\n=== {key}  [{method}, n_basis={n_basis}] ===')
        print(f"{'#':>2} {'eigenvalue':>24} {'tension':>10} {'certified rel err':>18} "
              f"{'digits':>7}  sector")
        for i, (l, t, r) in enumerate(zip(eigs, tens, recs)):
            sec = sectors[i] if sectors else ''
            print(f'{i:2d} {l:24.15f} {t:10.2e} {r["eps"]:18.3e} '
                  f'{r["digits"]:7.1f}  {sec}')
        print(f'found {len(eigs)}, Weyl predicts ~{n_weyl:.1f} below lam={eigs[-1]:.4f}')
        print(f'worst certified relative error: {max(r["eps"] for r in recs):.2e}')
        print(f'({out["seconds"]:.1f} s)')
        print('np.array([' + ', '.join(f'{x:.15f}' for x in eigs) + '])')
    return out


def main(keys):
    results = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            results = json.load(fh)
    for key in keys:
        try:
            results[key] = run_one(key)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f'!! {key} FAILED: {type(exc).__name__}: {exc}')
            results[key] = dict(key=key, error=f'{type(exc).__name__}: {exc}')
        with open(OUT, 'w') as fh:
            json.dump(results, fh, indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main(sys.argv[1:] or list(DOMAINS))
