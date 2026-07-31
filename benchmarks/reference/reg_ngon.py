"""Reference Dirichlet eigenvalues for regular n-gons (unit circumradius),
N=5..8. All corners share the same interior angle (n-2)*pi/n, which is a
non-integer submultiple of pi for these N -- so every corner is "singular"
and make_default_basis takes its FB+FS mixed-basis path (default fs_frac=0.5)
rather than pure Fourier-Bessel.

DIGIT CEILING (per N):
  N=5 @ n_basis=120: 11.7-12.3 digits, all 6 unique eigenvalues (mults
    [1,2,2,1,2,2]).
  N=6 @ n_basis=240: 11.8-13.0 digits, all 8 unique eigenvalues.
  N=7 @ n_basis=120: 10.2-11.7 digits, all 7 unique eigenvalues.
  N=8 @ n_basis=120: 8.8-10.3 digits for 8 of 9 eigenvalues; one outlier at
    lambda=29.5368 stuck at only 2.8 digits (see reference.py's
    reg_ngon_eigs docstring and TUNING_LOG.md).
All comfortably improved from the pre-pipeline-fix pass (see
ellipse.py/TUNING_LOG.md for the underlying fix): manual_solve (used here
via solve_domain_v2/escalate_and_solve_v2) also sidesteps the old
solve_interval-based bracket_mins hang for near-degenerate dihedral pairs
by using decoupled bracket/minimize/merge tolerances (see manual_solve's
docstring in common.py) -- N=7 in particular could not be tried past
n_basis=120 in the old pipeline at all.

WHY N=5,7,8 stay at n_basis=120 (only N=6 got pushed to 240): N=6 was
individually confirmed to finish cleanly and improve at 240. N=8 at 240
was tried and killed after ~10 min (very slow, likely the same
sharp/crowded-cluster-needs-more-order issue as chevron, given N=8's
crowded eigenvalue region near lambda=29.54). N=5,7 at 240 were not
retried under the new (manual_solve-based) pipeline, which no longer has
the specific hang mechanism that blocked N=5 in the old pipeline -- worth
retrying in a future pass but not done here given time.

`diagnose()` on the reg_ngon(N=8) outlier: n_reg/n=89/131 (67.9%) at
rtol=1e-14, unchanged by a denser collocation test (bdry_mult=4,
int_npts=300) or by rtol sweeps from 1e-13 to 1e-10 (n_reg drops as
expected but sigma stays ~4e-4 regardless) -- this specific mode's well is
genuinely shallow at n_basis=120, not a regularization or sampling
artifact."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, report

N_EIGS = 10
NS = [5, 6, 7, 8]
N_BASIS_LIST = {5: [60, 120], 6: [60, 120, 240], 7: [60, 120], 8: [60, 120]}


def run(N, n_eigs=N_EIGS):
    dom = geometry.reg_ngon(N)
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST[N], n_eigs)
    report(f'reg_ngon(N={N})  [n_basis={n_basis}]', eigs, tensions)
    print('mults', mults)
    return eigs, tensions


if __name__ == "__main__":
    for N in NS:
        run(N)
