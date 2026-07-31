"""Reference Dirichlet eigenvalues for cut_square(r): unit square with one
corner cut by a circular arc. All remaining corners are regular right angles
(no singular corners at all) -- make_default_basis takes the pure
all-regular Fourier-Bessel path.

DIGIT CEILING: r=0.25: 6.4-9.0 digits at n_basis=320; r=0.5: 9.1-13.1
digits at n_basis=320 (essentially at target). WHY: the pipeline-bug fix
(see ellipse.py, TUNING_LOG.md) alone did NOT move these -- confirmed via
solve_domain_v2 at the old n_basis=240 giving the same tensions as before,
and diagnose() showing n_reg/n ~62-64% regardless of a 6x collocation
density increase (bdry_mult, int_npts) -- i.e. genuinely resolution-limited
by basis size, not a pipeline or sampling artifact. A moderate basis bump
(240->320, well under the n_basis=480 cubic-cost wall) recovered most of
the gap, especially for r=0.5."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, report

N_EIGS = 10
RS = [0.25, 0.5]
N_BASIS_LIST = [60, 120, 240, 320]


def run(r, n_eigs=N_EIGS):
    dom = geometry.cut_square(r)
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'cut_square(r={r})  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    for r in RS:
        run(r)
