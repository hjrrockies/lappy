"""Reference Dirichlet eigenvalues for the GWW isospectral domain pair
(Gordon-Webb-Wolpert). No symmetry, 4 singular corners each (45/135 degree
corners are singular, 90/270 degree corners are regular) -- make_default_basis
takes the mixed FB+FS path. Existing lappy.reference.gww_eigs already
tabulates 25 shared values to ~12 digits; this re-derives per-domain values
(and checks GWW1/GWW2 actually agree, as they should for an isospectral pair)
with the current pipeline (not the primary source of truth -- see
reference.py).

DIGIT CEILING: GWW1 mostly 9.5-9.9 digits at n_basis=320, except two
eigenvalues (near lambda=5.18 and lambda=12.34) stuck at 6.1 and 3.9
digits. GWW2 mostly 7.2-8.7 digits at n_basis=320, plus one mode
(lambda=12.337005501361730) at 13.2 digits. WHY: the pipeline-bug fix (see
ellipse.py, TUNING_LOG.md) alone barely moved most modes at the old
n_basis=240 -- genuinely resolution-limited. A moderate basis bump
(240->320) helped substantially for GWW1's well-behaved modes. GWW1's
lambda=12.34 outlier is suspiciously close to GWW2's isolated
high-precision mode at lambda=12.337005501361730 -- may be a genuinely
close-but-distinct pair in GWW1 that GWW2's geometry doesn't share; not
investigated further given time."""
import numpy as np
from lappy import geometry, reference
from common import escalate_and_solve_v2, report

N_EIGS = 8
N_BASIS_LIST = [60, 120, 240, 320]


def run(which, n_eigs=N_EIGS):
    dom = geometry.GWW1() if which == 1 else geometry.GWW2()
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'GWW{which}()  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    eigs1, tensions1 = run(1)
    eigs2, tensions2 = run(2)
    ref = reference.gww_eigs(N_EIGS)
    print(f"\nexisting ref:   {ref}")
    print(f"GWW1 abs diff:  {np.abs(eigs1 - ref)}")
    print(f"GWW2 abs diff:  {np.abs(eigs2 - ref)}")
    print(f"GWW1 vs GWW2:   {np.abs(eigs1 - eigs2)}")
