"""Reference Dirichlet eigenvalues for mushroom(a, b, r): a half-disk cap on
a rectangular stem. Two 270-degree reentrant corners at the cap/stem
junction (multiple singular corners) -- make_default_basis takes the mixed
FB+FS path.

DIGIT CEILING: ~11.3-12.5 digits at n_basis=240, all 12 eigenvalues found.
WHY it's this good: fixed by solve_domain_v2's decoupled bracket/minimize/
polish tolerances (see ellipse.py's docstring and TUNING_LOG.md for the
underlying pipeline-bug fix) -- no basis/collocation changes needed."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, report

N_EIGS = 10
N_BASIS_LIST = [60, 120, 240]


def run(a=1.0, b=1.0, r=1.5, n_eigs=N_EIGS):
    dom = geometry.mushroom(a, b, r)
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'mushroom(a={a}, b={b}, r={r})  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    run()
