"""How high in k does the polygon path honestly go?

    uv run python -m benchmarks.envelope.k_sweep [--kmax 24] [domain ...]

WHY. `docs/polygon_contract.md` validates the spectral window at `lam_max = weyl_est(6, domain)`
with `solve(k)` swept to k=10, and says plainly that "lambda_20 and beyond untested", that
`n_cap = 240` rests on a rank-saturation measurement from TWO domains its own notes called
possibly coincidental, and that "a loop wanting lambda_1...lambda_20 will meet `n_cap`, and
nothing has measured what happens there". A shape-optimization loop over `lam_k/lam_1` needs to
know where that ceiling actually is, because the ceiling IS the answer to "which k are
reachable".

WHAT IS MEASURED. For each (domain, k): the achieved precision from `Eigenproblem.check_precision`
-- the same Moler--Payne-style quantity `refine_plan` optimizes, which agrees with a full
certification to 0.16 digits at ~1% of the cost -- plus `n_total`, whether the planner capped,
and the wall time. The window is scaled with `k` rather than left at 6, since a 6-eigenvalue
window is simply wrong for k=20.

The deliverable is a table of digits against k, and the largest k at which each domain still
clears the ~10 digits a shape-optimization inner loop needs.

RESUMABLE. One JSON line per (domain, k) cell appended to `run/k_sweep.jsonl`; re-running skips
cells already present. This takes hours and a laptop sleeps.
"""
import argparse
import json
import os
import time
import traceback
import warnings

import numpy as np

from lappy import Eigenproblem
from lappy.asymp import weyl_est

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
LEDGER = os.path.join(OUT, 'k_sweep.jsonl')

# Polygons only: `mps.default_basis_for` raises for curved domains, and the contract's validated
# envelope is the polygon path. Ordered easy -> hard so a truncated run is still informative.
DOMAINS = ['square', 'rect_thin', 'reg_ngon_8', 'L_shape', 'GWW1', 'H_shape',
           'cut_square_r025', 'chevron_1_15', 'iso_tri_h16']
KS = [4, 8, 12, 16, 20, 24]
TARGET_DIGITS = 10.0          # what a shape-optimization inner loop needs


def _done():
    if not os.path.exists(LEDGER):
        return set()
    out = set()
    with open(LEDGER) as f:
        for line in f:
            try:
                r = json.loads(line)
                out.add((r['domain'], r['k']))
            except (ValueError, KeyError):
                pass
    return out


def _append(rec):
    os.makedirs(OUT, exist_ok=True)
    with open(LEDGER, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def measure(key, k, precision=1e-12):
    """One cell. Returns a record; never raises."""
    from benchmarks.suite.domains import SUITE
    rec = dict(domain=key, k=k, precision=precision)
    t0 = time.perf_counter()
    try:
        dom = SUITE[key].build()
        # scale the window with k -- weyl_est(6) is simply the wrong window for k=20
        lam_max = weyl_est(k + 4, dom)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            evp = Eigenproblem(dom, precision=precision)
            eigs = np.asarray(evp.solve(k), dtype=float)
            chk = evp.check_precision(eigs)
            basis = evp.eval_solver.basis
        from lappy.basis_plan import plan_of
        plan = plan_of(basis)
        rec.update(ok=True, n_eigs=len(eigs), lam_max=float(lam_max),
                   lam_last=float(eigs[-1]) if len(eigs) else None,
                   digits=float(chk['digits']), achieved=float(chk['achieved']),
                   met=bool(chk['met']), n_total=(int(plan.n_total) if plan else len(basis)),
                   capped=(bool(plan.capped) if plan else None),
                   shortfall=(str(plan.shortfall) if plan and plan.shortfall else None))
    except Exception as exc:
        rec.update(ok=False, error=f'{type(exc).__name__}: {exc}',
                   trace=traceback.format_exc()[-600:])
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('domains', nargs='*', default=None)
    ap.add_argument('--kmax', type=int, default=24)
    ap.add_argument('--precision', type=float, default=1e-12)
    args = ap.parse_args()

    keys = args.domains or DOMAINS
    ks = [k for k in KS if k <= args.kmax]
    done = _done()
    print(f'{len(keys)} domains x {len(ks)} k values; {len(done)} cells already in the ledger\n')
    print(f'{"domain":>18} {"k":>3} {"digits":>7} {"n":>5} {"capped":>7} {"sec":>7}')
    for key in keys:
        for k in ks:
            if (key, k) in done:
                continue
            rec = measure(key, k, args.precision)
            _append(rec)
            if rec['ok']:
                flag = '' if rec['digits'] >= TARGET_DIGITS else '   <- below target'
                print(f'{key:>18} {k:>3} {rec["digits"]:>7.1f} {rec["n_total"]:>5} '
                      f'{str(rec["capped"]):>7} {rec["seconds"]:>7.1f}{flag}', flush=True)
            else:
                print(f'{key:>18} {k:>3}   FAILED  {rec["error"][:60]}  '
                      f'{rec["seconds"]:>7.1f}', flush=True)


if __name__ == '__main__':
    main()
