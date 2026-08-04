from lappy import *
import numpy as np
import matplotlib.pyplot as plt

rect = geometry.rect(1,1+1e-5)
basis = bases.make_default_basis(rect, 120)
pps = mps.pts_per_seg(rect, basis)
bdry_pts = rect.bdry_pts(pps)
int_pts = rect.int_pts(npts_rand=len(basis))
normed_basis = NormalizedBasis(basis, (bdry_pts, int_pts))
solver = MPSEigensolver(basis, bdry_pts, int_pts, rtol=1e-12, ltol=1e-12)

a = bounds.faber_krahn(rect)
b = asymp.weyl_est(10, rect)

eigs, mults, fevals = solver.solve_interval(a,b,200)
eigs_true = reference.rect_eigs(10, 1,1+1e-5)
err = np.abs(eigs_true-eigs[:10])
print("relative eig errors:", err/eigs_true)