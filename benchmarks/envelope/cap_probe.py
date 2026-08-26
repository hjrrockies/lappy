"""Does raising `n_cap` actually buy digits, and on which domains?

    uv run python -m benchmarks.envelope.cap_probe [--k 4] [--caps 240 400 600] [domain ...]

WHY, AND WHAT THE k-SWEEP LEFT AMBIGUOUS. `k_sweep` established that accuracy is flat in `k` and
that the four suite polygons which report `capped=True` are the same four stuck below 10 digits.
That reads as "the cap is the ceiling", and it is the reason `n_cap` sits at the top of
`LAPPY_WISHLIST.md` -- but the correlation has two mechanisms inside it, and only one of them is
the cap.

`_apply_cap` thins SOURCES to fit the cap, and refuses when the Fourier-Bessel budget alone
already exceeds it. In the refusal branch it returns the arcs UNTHINNED -- so the plan reports
`capped=True`, emits a shortfall message, and then serves the full budget anyway. Measured at
`lam_max = weyl_est(6, domain)`, `target=1e-14`:

    domain          n_total   FB + src      what the cap actually did
    chevron_1_15      237     83 + 154      thinned 328 -> 154 sources; cap BINDS
    iso_tri_h16       238     39 + 199      thinned 271 -> 199 sources; cap BINDS
    GWW1              333    242 + 91       nothing removed; FB alone is 242 > 240
    H_shape           452    380 + 72       nothing removed; FB alone is 380 > 240

So two of the four were never capped in any operative sense; their `n_total` is 39% and 88% OVER
the nominal ceiling. Raising `n_cap` cannot help those two, because nothing was taken away from
them. This probe measures the digits at several caps and reports which domains move, which
separates "the cap is the ceiling" from "this geometry is hard" one domain at a time.

The other half of the same question is whether the two genuinely capped domains RECOVER when the
cap is lifted. If they do, `n_cap` is a real ceiling and raising it is the fix `douse` wants. If
they do not, the source layer had already saturated and the cap was costing nothing, which would
retire the wish-list item rather than satisfy it.
"""
import argparse
import json
import os
import time
import traceback
import warnings
from dataclasses import replace

import numpy as np

from lappy import Eigenproblem, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
LEDGER = os.path.join(OUT, 'cap_probe.jsonl')
DOMAINS = ['chevron_1_15', 'iso_tri_h16', 'GWW1', 'H_shape', 'square', 'reg_ngon_8']


def _done():
    out = set()
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            try:
                r = json.loads(line)
                out.add((r['domain'], r['k'], r['n_cap'], float(r.get('prec', -1.0))))
            except (ValueError, KeyError):
                pass
    return out


def probe(name, k, n_cap, target=1e-14, prec=1e-13):
    from benchmarks.suite.domains import SUITE
    rec = dict(domain=name, k=k, n_cap=n_cap, prec=float(prec))
    t0 = time.perf_counter()
    try:
        dom = SUITE[name].build()
        lam_max = weyl_est(max(6, k + 2), dom)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cfg = replace(BP.PlanConfig(), n_cap=n_cap)
            plan = BP.plan_basis(dom, lam_max, target=target, cfg=cfg)
            n_fb = int(sum(c.M for c in plan.corners))
            n_fs = int(sum(a.n_src for a in plan.arcs))
            # `prec` DOMINATES the achieved accuracy and omitting it invalidated the first run
            # of this probe. On the unit square the same 108-column basis gives 8.84 digits at
            # the default and 13.28 at `prec=1e-13`, measured against the exactly known spectrum.
            # `douse.SpectrumEvaluator` uses 1e-13, and this wish list is written from `douse`'s
            # point of view, so 1e-13 is the operating point here.
            solver = MPSEigensolver.from_domain(dom, lam_max=lam_max,
                                                basis=BP.realize(plan, dom), prec=prec)
            evp = Eigenproblem(dom, eval_solver=solver, precision=prec)
            eigs = np.asarray(evp.solve(k), dtype=float)
            chk = evp.check_precision(eigs)
        rec.update(ok=True, n_total=int(plan.n_total), n_fb=n_fb, n_src=n_fs,
                   capped=bool(plan.capped), digits=float(chk['digits']),
                   n_eigs=len(eigs),
                   shortfall=(str(plan.shortfall) if plan.shortfall else None))
    except Exception as exc:
        rec.update(ok=False, error=f'{type(exc).__name__}: {exc}',
                   trace=traceback.format_exc()[-400:])
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('domains', nargs='*', default=None)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--caps', type=int, nargs='*', default=[240, 400, 600])
    ap.add_argument('--prec', type=float, default=1e-13)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    names = args.domains or DOMAINS
    done = _done()
    for name in names:
        for cap in args.caps:
            if (name, args.k, cap, float(args.prec)) in done:
                continue
            rec = probe(name, args.k, cap, prec=args.prec)
            with open(LEDGER, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
            if rec['ok']:
                print(f"  {name:14s} cap={cap:4d}: n_total={rec['n_total']:4d} "
                      f"(FB {rec['n_fb']} + src {rec['n_src']})  digits={rec['digits']:5.2f}  "
                      f"{rec['seconds']:6.1f}s", flush=True)
            else:
                print(f"  {name:14s} cap={cap:4d}: {rec['error'][:90]}", flush=True)

    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    rows = [r for r in rows if r.get('ok') and r['k'] == args.k
            and float(r.get('prec', -1.0)) == float(args.prec)]
    print(f'\ndigits at k={args.k}, by cap:')
    caps = sorted({r['n_cap'] for r in rows})
    print('  domain          ' + ''.join(f'{c:>16d}' for c in caps))
    for name in names:
        cells = []
        for c in caps:
            m = [r for r in rows if r['domain'] == name and r['n_cap'] == c]
            cells.append(f"{m[0]['digits']:6.2f} (n={m[0]['n_total']:3d})" if m else ' '*16)
        print(f'  {name:14s}  ' + ''.join(f'{x:>16s}' for x in cells))


if __name__ == '__main__':
    main()
