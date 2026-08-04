"""Diagnostic experiments for the reference run. Not part of the suite proper.

    python -m benchmarks.suite.experiments rank_vs_p [--n-basis 160]
    python -m benchmarks.suite.experiments rank_curve <key> [--sizes 60,120,240]

`rank_vs_p` is the decisive test of the "precision-bound near-dependence"
hypothesis (see NOTEBOOK.md). It walks the `disk_sector` family, which sweeps
the corner exponent `p = pi/gamma` while holding *everything else* fixed --
same one curved edge, same two straight edges, same symmetry, closed-form
spectrum -- and reports, at fixed basis size:

  n_reg / n   numerical rank of the collocation pencil after regularization
  sigma       tension at the (exactly known) first eigenvalue
  err         true relative error against `reference.sector_eigs`

If the hypothesis holds, `n_reg/n` should fall and `err` should rise
monotonically with `p`, with no other variable moving. If `n_reg/n` stays flat
while `err` rises, the problem is approximation power, not conditioning, and
the whole line of attack is wrong.

Results are appended to `run/experiments.jsonl` so they survive a context reset.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'reference'))

OUT = os.path.join(HERE, 'run', 'experiments.jsonl')


def _record(rec):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


def probe(domain, n_basis, lam, rtol=1e-14):
    """Numerical rank and tension of the pencil at a given lambda."""
    from common import build_solver
    solver = build_solver(domain, n_basis)
    sigma = float(solver.sigma(lam))
    A_B, A_I = solver.A_B(lam), solver.A_I(lam)
    n_cols = A_B.shape[1]
    # numerical rank of the stacked collocation matrix, the quantity
    # regularize_pencil truncates on
    s = np.linalg.svd(np.vstack([A_B, A_I]), compute_uv=False)
    n_reg = int(np.count_nonzero(s > rtol * s[0]))
    return dict(sigma=sigma, n_cols=int(n_cols), n_reg=n_reg,
                rank_frac=n_reg / n_cols,
                cond=float(s[0] / max(s[-1], 1e-300)))


def cmd_rank_vs_p(args):
    """Sweep the corner exponent against exact truth, everything else fixed."""
    from lappy import geometry as G, reference as R

    # gamma chosen so p is never pi/integer: the corner must stay singular.
    thetas = [('slit_p0.50', 2 * np.pi - 0.05),
              ('reflex_p0.67', 3 * np.pi / 2),
              ('p1.4', np.pi / 1.4),
              ('p2.3', np.pi / 2.3),
              ('p3.7', np.pi / 3.7),
              ('p6.5', np.pi / 6.5),
              ('p9.1', np.pi / 9.1),
              ('p13.3', np.pi / 13.3)]
    print(f'{"case":14s} {"p":>6} {"n_reg/n":>10} {"cond":>9} '
          f'{"sigma":>10} {"err_lam1":>10}')
    for name, theta in thetas:
        dom = G.disk_sector(1, theta)
        lam_exact = float(R.sector_eigs(1, 1, theta)[0])
        try:
            pr = probe(dom, args.n_basis, lam_exact)
        except Exception as exc:
            print(f'{name:14s} FAILED {type(exc).__name__}: {exc}')
            continue
        p = np.pi / theta
        rec = dict(experiment='rank_vs_p', case=name, theta=theta, p=p,
                   n_basis=args.n_basis, lam_exact=lam_exact, **pr)
        _record(rec)
        print(f'{name:14s} {p:6.2f} {pr["n_reg"]:4d}/{pr["n_cols"]:<4d} '
              f'{pr["cond"]:9.2e} {pr["sigma"]:10.2e}')
    return 0


def cmd_rank_curve(args):
    """How does numerical rank grow with nominal basis size for one domain?

    If rank saturates while n_basis keeps growing, extra basis functions are
    numerically redundant and escalating n_basis cannot help.
    """
    from benchmarks.suite.domains import SUITE
    entry = SUITE[args.key]
    dom = entry.domain()
    sizes = [int(s) for s in args.sizes.split(',')]

    if entry.truth_fn is not None:
        lam = float(entry.truth_fn(1)[0])
    else:
        from common import lambda_window
        a, b = lambda_window(dom, 1)
        lam = 0.5 * (a + b)

    print(f'{args.key}  lam={lam:.6f}')
    print(f'{"n_basis":>8} {"n_reg":>7} {"n_cols":>7} {"frac":>6} '
          f'{"cond":>9} {"sigma":>10}')
    for nb in sizes:
        try:
            pr = probe(dom, nb, lam)
        except Exception as exc:
            print(f'{nb:8d} FAILED {type(exc).__name__}: {exc}')
            continue
        _record(dict(experiment='rank_curve', key=args.key, n_basis=nb,
                     lam=lam, **pr))
        print(f'{nb:8d} {pr["n_reg"]:7d} {pr["n_cols"]:7d} '
              f'{pr["rank_frac"]:6.2f} {pr["cond"]:9.2e} {pr["sigma"]:10.2e}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('rank_vs_p')
    a.add_argument('--n-basis', type=int, default=160)
    a.set_defaults(func=cmd_rank_vs_p)

    b = sub.add_parser('rank_curve')
    b.add_argument('key')
    b.add_argument('--sizes', default='60,120,240,320')
    b.set_defaults(func=cmd_rank_curve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
