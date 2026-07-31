"""Reference Dirichlet eigenvalues for the classic L-shaped domain (single
270-degree reentrant corner). make_default_basis takes the pure
Fourier-Bessel path (exactly one singular corner). Existing
lappy.reference.L_shape_eigs already tabulates 25 values to ~14 digits; this
re-derives the first 10 with the current pipeline as a cross-check/refresh
(not the primary source of truth for this domain -- see reference.py).

DIGIT CEILING: ~12.9-13.3 digits at n_basis=240, all 10 eigenvalues,
matching the existing 14-digit reference table to within that precision.
WHY it's this good: fixed by solve_domain_v2's decoupled bracket/minimize/
polish tolerances (see ellipse.py's docstring and TUNING_LOG.md for the
underlying pipeline-bug fix) -- no basis/collocation changes needed."""
import numpy as np
from lappy import geometry, reference
from common import escalate_and_solve_v2, report

N_EIGS = 10
N_BASIS_LIST = [60, 120, 240]


def run(n_eigs=N_EIGS):
    dom = geometry.L_shape()
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'L_shape()  [n_basis={n_basis}]', eigs, tensions)
    ref = reference.L_shape_eigs(n_eigs)
    print(f"existing ref: {ref}")
    print(f"abs diff:     {np.abs(eigs - ref)}")
    return eigs, tensions


if __name__ == "__main__":
    run()
