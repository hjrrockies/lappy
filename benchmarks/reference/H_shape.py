"""Reference Dirichlet eigenvalues for H_shape(): fixed 12-vertex H-shaped
domain with 4 reentrant (270-degree) corners -- the most corner-heavy of the
non-excluded domains in this set. make_default_basis takes the mixed FB+FS
path (multiple singular corners).

DIGIT CEILING: ~7.8-8.2 digits at n_basis=320 for 9 of 10 eigenvalues, plus
one mode (lambda=19.739208802178766, suspiciously close to 2*pi^2 -- likely
a mode that vanishes on the connecting web and is effectively an exact
rectangle eigenvalue) at 13.3 digits. WHY not fully at target: the
pipeline-bug fix alone (see ellipse.py, TUNING_LOG.md) barely moved most
modes (6.7-7.2 digits unchanged at the old n_basis=240) -- genuinely
resolution-limited, confirmed via diagnose() showing n_reg/n ~68-70%
regardless of basis size. A moderate basis bump (240->320) helped
modestly; going further hits the same cubic-cost GSVD wall documented in
ellipse.py."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, report

N_EIGS = 10
N_BASIS_LIST = [60, 120, 240, 320]


def run(n_eigs=N_EIGS):
    dom = geometry.H_shape()
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'H_shape()  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    run()
