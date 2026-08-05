"""Wall-clock cost of one orthonormalized eigenfunction on H_shape (four reentrant corners).

Relevant because lappy is meant for the inner loop of a shape-optimization search
(CLAUDE.md principle 4): the boundary quadrature is built ONCE per solve and reused for every
lambda, so what matters per eigenfunction is the Cauchy-data evaluation, not the node set.

Run: python -m scripts.hshape_eigfunc_timing
"""
import time

from lappy import bases, bounds, geometry, asymp
from lappy.mps import MPSEigensolver
from lappy.eigfun_integrals import boundary_quadrature

dom = geometry.H_shape()
basis = bases.make_default_basis(dom, 400, fs_frac=.5, fs_C=0.5)

t0 = time.time()
solver = MPSEigensolver.from_domain(dom, basis=basis)
t_build = time.time() - t0
bq = solver.bdry_quad
print(f"quadrature build (once per solve): {t_build:8.3f} s   "
      f"{len(bq.pts)} nodes, {len(bq.panels)} panels, precision {bq.precision:.1e}")

eigs, mults, fevals = solver.solve_interval(bounds.faber_krahn(dom),
                                            asymp.weyl_est(2, dom), 20)

t0 = time.time()
solver.eigenfunction_coef(eigs[0])
t_first = time.time() - t0
t0 = time.time()
solver.eigenfunction_coef(eigs[0])          # cached
t_cached = time.time() - t0
print(f"first orthonormalized eigenfunction: {t_first:8.3f} s")
print(f"same one again (cached):             {t_cached:8.5f} s")
