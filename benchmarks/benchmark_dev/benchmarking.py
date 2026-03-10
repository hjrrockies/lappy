from lappy import *
import numpy as np

def estimate_peon(dom, basis, bdry_pts, int_pts, bdry_normals, ltol, rtol, ppl, verbose=0):
    """estimates a bound on the Poisson extension operator norm based on the first
    nonzero Neumann eigenvalue and the inradius of the domain"""
    solver = MPSEigensolver(basis, bdry_pts, int_pts, bdry_normals, 1, ltol=ltol, rtol=rtol)
    eigprob = Eigenproblem(dom, solver)
    eigs_neu = eigprob.solve(2, ppl=ppl, verbose=verbose)
    peon_bound = 1/np.sqrt(dom.inradius*eigs_neu[1])
    return peon_bound

def precise_eigs(n_eigs, dom, basis, bdry_pts, int_pts, bdry_nodes, int_nodes, ltol, rtol, ppl, verbose=0):
    """get high-precision estimates of dirichlet eigenvalues"""
    eval_solver = MPSEigensolver(basis, bdry_pts, int_pts, ltol=ltol, rtol=rtol)
    evec_solver = MPSEigensolver(basis, bdry_nodes, int_nodes, ltol=ltol, rtol=rtol)
    eigprob = Eigenproblem(dom, eval_solver, evec_solver)
    eigs = eigprob.solve(n_eigs, ppl=ppl, verbose=verbose)

    eigs_polished = []
    tensions = []
    for eig in eigs:
        peig = opt.golden_search(evec_solver.sigma, eig*(1-ltol), eig*(1+ltol), ltol*eig)[0]
        tension = evec_solver.sigma(peig)
        eigs_polished.append(peig)
        tensions.append(tension)
    eigs_polished = np.array(eigs_polished)
    tensions = np.array(tensions)

    return eigs_polished, tensions