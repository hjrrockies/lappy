from lappy import *
import time

dom = geometry.H_shape()
basis = bases.make_default_basis(dom, 400, fs_frac=.5, fs_C=0.5)
pps = mps.pts_per_seg(dom, basis, mult=3)
bdry_pts = dom.bdry_pts(pps)
int_pts = dom.int_pts(npts_rand=len(basis))
rellich_data = rellich.build_rellich_data(dom, basis)
normed_basis = NormalizedBasis(basis, (bdry_pts, int_pts))
solver = MPSEigensolver(normed_basis, bdry_pts, int_pts, cauchy_data=rellich_data)

eigs, mults, fevals = solver.solve_interval(bounds.faber_krahn(dom), asymp.weyl_est(2, dom), 20)

start = time.time()
solver.eigenfunction_coef(eigs[0])
print("wall clock for eigfunc coef:", time.time()-start)