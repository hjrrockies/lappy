"""Reference Dirichlet eigenvalues for chevron(h1, h2): a sharp reentrant
corner whose severity grows with h2/h1. All 4 corners are singular, so
make_default_basis takes the mixed FB+FS path. Existing
lappy.reference.chevron_eigs only covers h1=1,h2=2; this covers the fuller
family used in the (now-archived) benchmark_suite registry."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, report

N_EIGS = 10
# DIGIT CEILING: 5.2-6.3 digits for h1=1,h2=1.5; 6.1-7.1 for h1=1,h2=2.0;
# 3.7-4.6 for h1=2,h2=3.0; 4.2-5.0 for h1=2,h2=4.0 -- all at n_basis=160,
# still well short of the 10-12 digit target. WHY: chevron(1,1.5) has TWO
# corners at only ~11.3 degrees each (not just the 270-degree reentrant
# one); the default fb_corner_fraction weighting gives them just 3.1% each
# of the FB budget (weight ~ angle magnitude, so the 270-degree corner
# dominates at 75%) -> FB orders [3,43,3,11] at n_basis=120. Tried and
# ruled out as fixes (see TUNING_LOG.md for detail):
#   1. Reweighting fb_corner_orders/fs_corner_orders toward the sharp
#      corners (bypassing make_default_basis) -- made things slower, not
#      better: raw order-count at an already-sharp corner produces huge
#      individual Bessel orders (order=21 at 11.3 degrees means exponents
#      up to ~334) without filling in any missing intermediate resolution.
#   2. 3x denser collocation (bdry_mult, int_npts) with default order
#      allocation -- no change (diagnose() shows n_reg/n ~79% either way).
#   3. Moderate n_basis increase (120->160) -- real but slow improvement
#      (~0.7-1.2 extra digits for a 33% basis increase), consistent with
#      needing much more basis (400+) to reach 10 digits, which is known to
#      hit the slow-Bessel-evaluation wall (n_basis=240 didn't finish in
#      15+ minutes for these corners in the first pass).
#
# h1=1,h2=1.25 excluded outright: its two acute corners are only ~6.3
# degrees (even sharper). Solves either fail to converge (n_basis=60:
# 0/10 eigenvalues found) or become extremely slow (n_basis=120: didn't
# finish in 15+ minutes) -- similar in spirit to why spiral() was excluded
# from the whole benchmark set.
PARAMS = [(1.0, 1.5), (1.0, 2.0), (2.0, 3.0), (2.0, 4.0)]
N_BASIS_LIST = [60, 120, 160]


def run(h1, h2, n_eigs=N_EIGS):
    dom = geometry.chevron(h1, h2)
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'chevron(h1={h1}, h2={h2})  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    for h1, h2 in PARAMS:
        run(h1, h2)
