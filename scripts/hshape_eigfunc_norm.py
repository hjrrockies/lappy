"""Cross-check: an L^2-orthonormalized eigenfunction against independent interior cubature.

Previously print-only; now asserts. The point is that the two norms share no code -- one is a
boundary integral via the Rellich identity with corner-adapted quadrature, the other an
interior cubature over a mesh -- so agreement is evidence rather than self-consistency.

H_shape is the interesting case: four reentrant corners, so no single x0 can zero more than one
of them, which is precisely what the corner quadrature exists to handle.

Run: python -m scripts.hshape_eigfunc_norm
"""
import numpy as np
import scipy.linalg as la

from lappy import bases, cubature, geometry, bounds
from lappy.mps import MPSEigensolver, weyl_est
from lappy.eigfun_integrals import EigfunData, gram, verify_gram


def check(dom, name, n_basis=200, cub_tol=1e-8, spread_tol=1e-8, verify_tol=1e-13):
    basis = bases.make_default_basis(dom, n_basis)
    solver = MPSEigensolver.from_domain(dom, basis=basis)
    lo, hi = bounds.faber_krahn(dom), weyl_est(2, dom)
    out = solver.solve_interval(lo, hi, 20)
    eigs = np.atleast_1d(np.asarray(out[0] if isinstance(out, tuple) else out)).ravel()
    eig = float(eigs[0])

    coef = solver.eigenfunction_coef(eig, mult=1)
    nodes, weights = cubature.polygon_cubature(dom, eig, 1e-12)
    u = (solver.basis(eig, nodes) @ coef)[:, 0]
    cub = la.norm(u*np.sqrt(weights))

    bq = solver.bdry_quad
    U = solver.basis(eig, bq.pts) @ coef
    U_N = solver.basis.ddiff(eig, bq.pts, bq.normals) @ coef
    U_T = solver.basis.ddiff(eig, bq.pts, bq.tangents) @ coef
    ed = EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, U, U_N, U_T)
    spread = max(abs(gram(ed, eig, bq, x0)[0, 0] - 1.0)
                 for x0 in (0.31 + 0.17j, -0.4 + 0.23j, 0.8 - 0.6j))

    # The quadrature's OWN error, isolated: recompute on a refined rule. The x0-spread cannot
    # do this -- it also carries the eigenvalue's error, since lam enters the identity through
    # 1/(2 lam) and through the Cauchy data, so an inexact lam makes u not quite an
    # eigenfunction and the x0-dependence reappears. On L_shape the two differ by three orders
    # (quadrature 1.3e-15, spread 3.3e-12 at a lam off by ~1.6e-9 relative), which is why the
    # spread is reported as a diagnostic and `verify` is what gets asserted.
    _, _, verify = verify_gram(solver.basis, eig, coef, bq, dom)

    print(f"{name:10s} lam={eig:.10f}  nodes={len(bq.pts):4d}  "
          f"cubature norm={cub:.12f}  |1-cub|={abs(cub-1):.2e}  "
          f"quadrature={verify:.2e}  x0-spread={spread:.2e}")
    assert abs(cub - 1.0) < cub_tol, f"{name}: cubature norm {cub} differs from 1 by >{cub_tol}"
    assert verify < verify_tol, (f"{name}: refinement moves the Gram by {verify:.2e} "
                                 f"(>{verify_tol}) -- the quadrature is short")
    assert spread < spread_tol, f"{name}: x0-spread {spread:.2e} indicates quadrature error"


# This comment used to read: "a cross-method comparison is bounded by the WEAKER of the two
# methods, and on H_shape that is not the boundary rule" -- the 1.4e-8 cubature discrepancy was
# blamed on the cubature. `verify_gram` says otherwise. Refining the boundary rule alone moves
# H_shape's Gram by 1.0e-8, matching that discrepancy: the boundary rule WAS the weaker method
# here, and the x0-spread (4.1e-11) understated its error by three orders. The spread cannot see
# this because refining is a different question from moving x0.
#
# Raising `smooth_safety` to 2 takes it to 4.6e-10, after which it plateaus at 3-4e-10 for any
# safety factor and any refinement depth -- a floor that is not the smooth panels and is not yet
# explained. All three bars below are MEASURED at the seed set in __main__, not aspirations.
#
#     domain    cubature   quadrature (verify)   x0-spread
#     L_shape    5.9e-11        1.4e-13           3.3e-12
#     H_shape    1.4e-08        1.0e-08           4.1e-11
TOLS = {'L_shape': dict(cub_tol=1e-10, spread_tol=1e-11, verify_tol=1e-12),
        'H_shape': dict(cub_tol=1e-7, spread_tol=1e-9, verify_tol=5e-8)}

if __name__ == '__main__':
    # Seed: interior collocation points are drawn from numpy's global RNG, so an unseeded run
    # asserts against a moving target. Measured over repeated runs, L_shape's x0-spread ranges
    # 1.2e-14 to 6.3e-12 across draws -- straddling the 1e-12 bar below, which made this script
    # pass or fail by luck. Same lesson as benchmarks/suite/runner.py's --seed.
    np.random.seed(0)
    check(geometry.L_shape(), 'L_shape', n_basis=160, **TOLS['L_shape'])
    check(geometry.H_shape(), 'H_shape', n_basis=200, **TOLS['H_shape'])
    print("OK")
