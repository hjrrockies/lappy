"""Reference Dirichlet eigenvalues for ellipse(a, b): smooth boundary, no
corners -- make_default_basis falls back to a pure boundary FundamentalBasis,
exercising the corner-free MPS path.

DIGIT CEILING: ~13.3-14.4 digits at n_basis=240, all 3 (a,b) pairs, all 10
eigenvalues each -- essentially machine precision. WHY it's this good: the
old pipeline (Eigenproblem.solve -> solve_interval -> old polish_eigs) had
a bug -- solve_interval's coarse eigenvalue location is only accurate to
ltol_default=1e-8 (relative), but the old polish_eigs then searched for the
true root only within eig*(1+-1e-14), a window ~1e6x narrower than that
uncertainty, so it could not actually refine anything. Fixed by
solve_domain_v2 (manual_solve with a tight, decoupled minimize_tol=1e-12,
then polish_eigs with a correctly-widened bracket) -- see TUNING_LOG.md.
No basis/collocation changes were needed; n_basis=480 was tried (both solo
and batched with the other domains in this directory) and never completed
a single tension-grid pass within a ~40 minute budget -- a plain
cubic-in-problem-size GSVD cost wall unrelated to precision, not chased
further since 240 already reaches machine precision."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, report

N_EIGS = 10
PARAMS = [(2.0, 1.0), (3.0, 1.0), (4.0, 1.0)]
N_BASIS_LIST = [60, 120, 240]


def run(a, b, n_eigs=N_EIGS):
    dom = geometry.ellipse(a, b)
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'ellipse(a={a}, b={b})  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    for a, b in PARAMS:
        run(a, b)
