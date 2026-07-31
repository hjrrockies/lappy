"""Reference Dirichlet eigenvalues for the Bunimovich stadium: two straight
sides + two semicircular caps, tangent (C^1, no corners) at the junctions --
make_default_basis falls back to a pure boundary FundamentalBasis, like
ellipse.py, but with a non-analytic (piecewise) boundary.

NOTE: unlike ellipse/reg_ngon/etc., this domain's tension does NOT improve
with basis size -- it plateaus around 1e-4 (only ~2-3 accurate digits) from
n_basis=60 all the way to n_basis=480, and is insensitive to fs_d, fs_bdry_order,
bdry_mult, and int_npts (all tried; see git history/session notes). Tension
diagnostics (`solver._tension_diagnostics`) show `n_reg` (post-regularization
rank) stuck well below `n` (nominal basis size) at the default rtol=1e-14 --
the boundary-source Vandermonde becomes numerically rank-deficient long before
the requested basis size is reached, most likely from the curvature
discontinuity at the line/arc tangent points (a known weak spot for smooth
boundary-source bases, similar in spirit to why spiral() was excluded, though
milder). Values below are reported as the best achieved with make_default_basis
tuning, NOT wired into lappy/reference.py (too imprecise to serve as a
reference).

Re-confirmed in the deep precision push (TUNING_LOG.md): `diagnose()`
shows `n_reg=77/124 (62.1%)` at n_basis=120, completely unchanged by a 6x
collocation density increase (bdry_mult=6, int_npts=600) -- `sigma` barely
moves (2.15e-04 -> 1.76e-04). Quantitatively confirms this is intrinsic
basis rank-deficiency, not a sampling problem, consistent with the
original diagnosis above. Not chased further -- also note this is the same
"n_reg stuck around 60-70% of n" pattern later found to be common across
several other domains (chevron, reg_ngon N=8, cut_square) in this
directory, though stadium's case is more severe (dominates the achievable
precision at only 2-3 digits, vs. those domains where it's a secondary
effect on top of otherwise-reasonable resolution)."""
import numpy as np
from lappy import geometry
from common import escalate_and_solve_v2, report

N_EIGS = 10
PARAMS = [(1.0, 1.0), (2.0, 1.0)]
N_BASIS_LIST = [60, 120, 240]


def run(L, H, n_eigs=N_EIGS):
    dom = geometry.stadium(L, H)
    n_basis, eigs, mults, tensions = escalate_and_solve_v2(dom, N_BASIS_LIST, n_eigs)
    report(f'stadium(L={L}, H={H})  [n_basis={n_basis}]', eigs, tensions)
    return eigs, tensions


if __name__ == "__main__":
    for L, H in PARAMS:
        run(L, H)
