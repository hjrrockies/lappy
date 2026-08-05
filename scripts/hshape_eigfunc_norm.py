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
from lappy.eigfun_integrals import EigfunData, gram


def check(dom, name, n_basis=200, cub_tol=1e-8, spread_tol=1e-8):
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

    print(f"{name:10s} lam={eig:.10f}  nodes={len(bq.pts):4d}  "
          f"cubature norm={cub:.12f}  |1-cub|={abs(cub-1):.2e}  x0-spread={spread:.2e}")
    assert abs(cub - 1.0) < cub_tol, f"{name}: cubature norm {cub} differs from 1 by >{cub_tol}"
    assert spread < spread_tol, f"{name}: x0-spread {spread:.2e} indicates quadrature error"


# A cross-method comparison is bounded by the WEAKER of the two methods, and on H_shape that is
# not the boundary rule: its x0-spread is 5.8e-11 (self-consistent to that level) while the
# cubature comparison sits at 1.4e-8. H_shape is hard for everything -- the reference run
# certified it to only 9.66 digits -- so the interior cubature and the eigenfunction's own
# residual dominate. The spread tolerance is therefore the sharper claim of the two.
TOLS = {'L_shape': dict(cub_tol=1e-10, spread_tol=1e-12),
        'H_shape': dict(cub_tol=1e-7, spread_tol=1e-9)}

if __name__ == '__main__':
    check(geometry.L_shape(), 'L_shape', n_basis=160, **TOLS['L_shape'])
    check(geometry.H_shape(), 'H_shape', n_basis=200, **TOLS['H_shape'])
    print("OK")
