# GWW domains: Dirichlet eigenvalues and error estimates
from lappy import *
import numpy as np
import matplotlib.pyplot as plt

# get domain (Dirichlet and Neumann boundary)
gww = geometry.GWW1()
gww_neu = geometry.GWW1(bc='neu')

# solver parameters
n_eigs = 25
ltol = 5e-16

# basis parameters
n_basis = 120

# bdry and int pts
n_per_seg = n_basis
bdry_pts, bdry_normals = gww.bdry_data(n_per_seg)[:2]
int_pts = gww.int_pts(npts_rand=n_basis)
print(f"len(basis) = {n_basis}")
print(f"len(bdry_pts)={len(bdry_pts)}, len(int_pts)={len(int_pts)}")

# bdry and int nodes
bdry_nodes = gww.bdry_pts(n_per_seg, weights=True)
int_nodes = gww.int_pts('mesh', mesh_size=0.25)
print(f"len(bdry_nodes)={len(bdry_nodes)}, len(int_nodes)={len(int_nodes)}")

# estimate Poisson extension operator norm
# C(Omega) <= 1/sqrt(inradius(Omega)*mu_1(Omega))
# where mu_1(Omega) is the first nonzero Neumann eigenvalue
print("Estimating Poisson extension operator norm")
basis_neu = FourierBesselBasis.from_domain(gww_neu, [0,n_basis,0, n_basis,0,0,n_basis,n_basis]).to_normalized(bdry_pts + int_pts)
solver_neu = MPSEigensolver(basis_neu, bdry_pts, int_pts, bdry_normals, 1, ltol=ltol, rtol=1e-8)
# solver_neu.plot_tensions(ltol, 1, 100, 3)
# plt.show()

eigprob_neu = Eigenproblem(gww_neu, solver_neu)
eigs_neu = eigprob_neu.solve(2, ppl=10, verbose=2)
print(f"First nonzero Neumann eig:",eigs_neu[1])
poisson_const_bound = 1/np.sqrt(gww_neu.inradius*eigs_neu[1])
print(f"bound on C(Omega):", poisson_const_bound)

# solve for Dirichlet eigs
print("Solving for Dirichlet eigenvalues")
basis = FourierBesselBasis.from_domain(gww, [0,n_basis,0, n_basis,0,0,n_basis,n_basis]).to_normalized(bdry_pts + int_pts)
nbasis = FourierBesselBasis.from_domain(gww, [0,n_basis,0, n_basis,0,0,n_basis,n_basis]).to_normalized(bdry_nodes + int_nodes)
eval_solver = MPSEigensolver(basis, bdry_pts, int_pts, ltol=ltol, rtol=1e-10)
# eval_solver.plot_tensions(bounds.faber_krahn(gww), asymp.weyl_est(25, gww), 1000, 3)
# plt.show()
evec_solver = MPSEigensolver(nbasis, bdry_nodes, int_nodes, ltol=ltol, rtol=1e-10)
eigprob = Eigenproblem(gww, eval_solver, evec_solver)
eigs = eigprob.solve(n_eigs, ppl=50, verbose=2)

# polish eigs using evec_solver, compute tension (without regularization)
print("Polishing eigenvalues, estimating errors via tension")
eigs_polished = []
tensions = []
relerr_est = []
for eig in eigs:
    peig = opt.golden_search(evec_solver.sigma, eig*(1-2*ltol), eig*(1+2*ltol), ltol*eig)[0]
    tension = evec_solver.sigma(peig)
    eigs_polished.append(peig)
    tensions.append(tension)
    relerr_est.append(tension*poisson_const_bound)
eigs_polished = np.array(eigs_polished)
tensions = np.array(tensions)
relerr_est = np.array(relerr_est)
print("eigs:",eigs_polished)
print("tensions:",tensions)
print("relerr_est:",relerr_est)

# save results
np.savez(f'gww_eigs_{n_eigs}', eigs=eigs_polished, tensions=tensions, relerr_est=relerr_est)
