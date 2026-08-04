"""CLI driver: symmetry-reduced solve for one benchmark domain.

    python run_sym.py <domain> [key=val ...]

e.g.  python run_sym.py chevron h1=1 h2=1.5 n_basis=160 n_eigs=10
"""
import sys
import numpy as np

from lappy import geometry as G, reference
from lappy.symmetry import domain_symmetry
from symsolve import solve_sym, report_sym

BUILD = {
    'chevron':    lambda h1=1.0, h2=2.0: G.chevron(h1, h2),
    'iso_tri':    lambda h=1.0: G.iso_tri(h),
    'mushroom':   lambda a=1.0, b=1.0, r=1.5: G.mushroom(a, b, r),
    'L_shape':    lambda: G.L_shape(),
    'cut_square': lambda r=0.5: G.cut_square(r),
    'H_shape':    lambda: G.H_shape(),
    'ellipse':    lambda a=2.0, b=1.0: G.ellipse(a, b),
    'stadium':    lambda L=1.0, H=1.0: G.stadium(L, H),
    'reg_ngon':   lambda n=6: G.reg_ngon(int(n)),
}

REF = {
    'chevron':    lambda k, h1=1.0, h2=2.0: reference.chevron_eigs(k, h1, h2),
    'iso_tri':    lambda k, h=1.0: reference.iso_tri_eigs(k, h),
    'mushroom':   lambda k, **kw: reference.mushroom_eigs(k),
    'L_shape':    lambda k: reference.L_shape_eigs(k),
    'cut_square': lambda k, r=0.5: reference.cut_square_eigs(k, r),
    'H_shape':    lambda k: reference.H_shape_eigs(k),
    'ellipse':    lambda k, a=2.0, b=1.0: reference.ellipse_eigs(k, a, b),
    'reg_ngon':   lambda k, n=6: reference.reg_ngon_eigs(k, int(n)),
}


def main(argv):
    name = argv[0]
    opts = {}
    for tok in argv[1:]:
        k, v = tok.split('=')
        opts[k] = float(v)

    n_basis = int(opts.pop('n_basis', 120))
    n_eigs = int(opts.pop('n_eigs', 10))
    seed = int(opts.pop('seed', 0))
    np.random.seed(seed)

    dom = BUILD[name](**opts)
    grp = domain_symmetry(name, **opts)
    if grp is None:
        raise SystemExit(f'{name} has no registered symmetry')

    try:
        ref = REF[name](n_eigs, **opts)
    except Exception as exc:                       # no table, or unlisted params
        print(f'(no reference table: {exc})')
        ref = None

    e, s, t = solve_sym(dom, grp, n_basis, n_eigs, verbose=1)
    label = f'{name}({",".join(f"{k}={v:g}" for k, v in opts.items())}) sym n_basis={n_basis}'
    report_sym(label, e, s, t, ref=ref)


if __name__ == '__main__':
    main(sys.argv[1:])
