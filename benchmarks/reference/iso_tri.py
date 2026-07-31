"""Reference Dirichlet eigenvalues for iso_tri(h): isosceles triangle, base 2,
height h. Aspect-ratio stress test -- for h=1 all corners are regular
(45/90/45 degrees), pure FB path; for other h the corners are singular
(non-integer pi/angle ratio) and make_default_basis mixes in FS terms. Large
h means an increasingly sharp apex angle (elongated triangle), a tough case
for basis/collocation conditioning. Existing lappy.reference.iso_tri_eigs
only covers h=20; this covers the fuller family.

DIGIT CEILING: h=0.5: 10.8-12.3 digits; h=1.0: 13.0-13.5; h=2.0: 12.0-13.2;
h=4.0: 11.8-12.9; h=8.0: 11.3-12.1 -- all at n_basis=120, all comfortably
at or above the 10-digit target. WHY it's this good: fixed by
solve_domain_v2's decoupled bracket/minimize/polish tolerances (see
ellipse.py's docstring and TUNING_LOG.md for the underlying pipeline-bug
fix) -- no basis/collocation changes needed for h=0.5..8.

h=16.0 is the exception: manual_solve got unusually slow at n_basis=120
for this height specifically (unlike h=0.5..8, which all solved in ~1-2
min each) -- likely the same "sharp corner needs high FB order -> slow
Bessel evaluation" pattern seen in chevron, since h=16 has the sharpest
apex angle in this set. Killed after ~10+ min rather than chased further;
h=16 keeps its old-pipeline value (7.8-10.1 digits, not re-verified with
the fixed polish step) as a placeholder pending a dedicated pass."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, escalate_and_solve, report

N_EIGS = 10
HS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_BASIS_LIST = [60, 120]

# h=16 not re-verified with the new pipeline (got very slow, see module
# docstring) -- kept on the old pipeline at its original basis size.
HS_OLD_PIPELINE = [16.0]


def run(h, n_eigs=N_EIGS):
    dom = geometry.iso_tri(h)
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'iso_tri(h={h})  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


def run_old_pipeline(h, n_eigs=N_EIGS):
    dom = geometry.iso_tri(h)
    n_basis, eigs, tensions = escalate_and_solve(dom, N_BASIS_LIST, n_eigs)
    report(f'iso_tri(h={h})  [n_basis={n_basis}, old pipeline]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    for h in HS:
        run(h)
    for h in HS_OLD_PIPELINE:
        run_old_pipeline(h)
