"""
Tests distribution of basis functions at corners
"""
from lappy import *
from .benchmarking import build_eigprob
import numpy as np
import os

def test_right_trap(n_basis, verbose=False):
    """Tests basis point distribution across two corners"""

    # setup function
    def right_trap_evp(h1, h2, n1, n2):
        """eigenproblem on right trapezoid"""
        # domain and basis
        dom = geometry.right_trapezoid(h1, h2)
        basis = FourierBesselBasis.from_domain(dom, [n1, 0, 0, n2])

        # build bdry and int pts
        pts_per_seg = geometry.pts_per_seg(dom, basis, mult=3)
        bdry_pts = dom.bdry_pts(pts_per_seg)
        np.random.seed(0)
        int_pts = dom.int_pts(npts_rand=100)
        basis = NormalizedBasis(basis, (bdry_pts, int_pts))

        # solver and evp
        solver = MPSEigensolver(basis, bdry_pts, int_pts, ltol=1e-15, ttol=1e-1)
        lam_fk = bounds.faber_krahn(dom)
        solver.rtol = solver.adapt_rtol(lam_fk,1.25*lam_fk,rtol_max=1e-10)
        eigprob = Eigenproblem(dom, solver)
        return eigprob

    # loop over side lengths
    H1 = np.linspace(1, 1.75, 16)[1:]
    H2 = 2-H1
    # loop over bases
    N1 = np.arange(0, n_basis+1, 5)
    N2 = n_basis-N1
    eigs = np.empty((len(H1), len(N1)))
    sigmas = np.empty((len(H1), len(N1)))
    R = np.empty((len(H1), len(N1)))
    for i, (h1, h2) in enumerate(zip(H1, H2)):
        if verbose: print("h1 =", h1)
        for j, (n1, n2) in enumerate(zip(N1, N2)):
            if verbose: print("n1 =", n1)
            eigprob = right_trap_evp(h1, h2, n1, n2)
            try:
                eigs[i,j] = eigprob.solve(1, n_workers=10)[0]
                sigmas[i,j] = eigprob.eval_solver.sigma(eigs[i,j])
                R[i,j] = eigprob.eval_solver.rtol
            except EigensolverFailure as ef:
                print(ef)
                eigs[i,j] = np.nan
                sigmas[i,j] = np.nan
                R[i,j] = np.nan

    return H1, H2, N1, N2, eigs, sigmas, R

def test_chevron(n_basis, rtol=None, verbose=False):
    """Tests basis point distribution across two singular corners"""

    # setup function
    def chevron_evp(h1, h2, n1, n2):
        """eigenproblem on chevron"""
        # domain and basis
        dom = geometry.chevron(h1, h2)
        basis = FourierBesselBasis.from_domain(dom, [0, n1, 0, n2])

        # build bdry and int pts
        pts_per_seg = geometry.pts_per_seg(dom, basis, mult=3)
        bdry_pts = dom.bdry_pts(pts_per_seg)
        np.random.seed(0)
        int_pts = dom.int_pts(npts_rand=100)
        basis = NormalizedBasis(basis, (bdry_pts, int_pts))

        # solver and evp
        solver = MPSEigensolver(basis, bdry_pts, int_pts, ltol=1e-15, ttol=1e-1)
        lam_fk = bounds.faber_krahn(dom)
        if rtol is None:
            solver.rtol = solver.adapt_rtol(lam_fk,1.25*lam_fk,rtol_max=1e-10)
        else:
            solver.rtol = rtol
        eigprob = Eigenproblem(dom, solver)
        return eigprob

    # extend peak, keeping corners regular
    M = np.arange(5,20)
    H1 = np.ones(len(M))
    H2 = np.tan(np.pi*(M+4)/(4*M))
    # loop over bases
    N1 = np.arange(0, n_basis+1, 5)
    N2 = n_basis-N1
    eigs = np.empty((len(H1), len(N1)))
    sigmas = np.empty((len(H1), len(N1)))
    R = np.empty((len(H1), len(N1)))
    for i, (h1, h2) in enumerate(zip(H1,H2)):
        if verbose: print("h2 =", h2)
        for j, (n1, n2) in enumerate(zip(N1, N2)):
            if verbose: print("n1 =", n1)
            eigprob = chevron_evp(h1, h2, n1, n2)
            try:
                eigs_ = eigprob.solve(1, n_workers=10)
                if len(eigs_) > 0:
                    eigs[i,j] = eigprob.solve(1, n_workers=10)[0]
                    sigmas[i,j] = eigprob.eval_solver.sigma(eigs[i,j])
                    R[i,j] = eigprob.eval_solver.rtol
                else:
                    print("found no eigenvalues")
                    eigs[i,j] = np.nan
                    sigmas[i,j] = np.nan
                    R[i,j] = np.nan
            except EigensolverFailure as ef:
                print(ef)
                eigs[i,j] = np.nan
                sigmas[i,j] = np.nan
                R[i,j] = np.nan

    return H1, H2, N1, N2, eigs, sigmas, R

def test_fb_strats(dom, N, rtol=None, verbose=False):
    eigs = np.empty((4,len(N)))
    sigmas = np.empty((4,len(N)))
    R = np.empty((4,len(N)))
    # loop over strategies and basis sizes
    for i, strat in enumerate(['uniform', 'weighted', 'singular_uniform', 'singular_weighted']):
        if verbose: print(strat)
        for j, n_basis in enumerate(N):
            if verbose: print(n_basis)
            eigprob = build_eigprob(dom, n_basis, strat, rtol=rtol, n_bdry_mult=3, n_int=100)
            # if verbose:
            #     basis = eigprob.eval_solver.basis.basis
            #     print(basis.orders)
            try:
                eigs_ = eigprob.solve(1, n_workers=10, ttol=1e-1)
                if len(eigs_) > 0:
                    eigs[i,j] = eigs_[0]
                    sigmas[i,j] = eigprob.eval_solver.sigma(eigs[i,j])
                    R[i,j] = eigprob.eval_solver.rtol
                else:
                    print("found no eigenvalues")
                    eigs[i,j] = np.nan
                    sigmas[i,j] = np.nan  
                    R[i,j] = np.nan
            except EigensolverFailure as ef:
                print(ef)
                eigs[i,j] = np.nan
                sigmas[i,j] = np.nan  
                R[i,j] = np.nan
    return eigs, sigmas, R
    

if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)
    # # trapeizoid tests
    # print("running trapezoid tests")
    # n_basis = 120
    # H1, H2, N1, N2, eigs, sigmas, R = test_right_trap(n_basis, True)
    # np.savez(os.path.join(outdir, "fb_corner_trap.npz"), H1=H1, H2=H2, N1=N1, N2=N2,
    #          eigs=eigs, sigmas=sigmas, R=R)

    # # L_shaped domain
    # print("running L-shaped domain tests")
    # N = np.arange(10, 81, 5)
    # eigs, sigmas, R = test_fb_strats(geometry.L_shape(), N, verbose=True)
    # np.savez(os.path.join(outdir, "fb_corner_lshape.npz"), N=N, eigs=eigs, sigmas=sigmas, R=R)

    # # GWW domains
    # print("running GWW domain tests")
    # N = np.arange(20,201,20)
    # eigs, sigmas, R = test_fb_strats(geometry.GWW1(), N, verbose=True, rtol=1e-12)
    # np.savez(os.path.join(outdir, "fb_corner_gww1.npz"), N=N, eigs=eigs, sigmas=sigmas, R=R)
    # eigs, sigmas, R = test_fb_strats(geometry.GWW2(), N, verbose=True, rtol=1e-12)
    # np.savez(os.path.join(outdir, "fb_corner_gww2.npz"), N=N, eigs=eigs, sigmas=sigmas, R=R)

    # # chevron with two regular corners
    print("running chevron domain tests")
    # N = np.arange(10,81,5)
    # n = 7
    # h2 = np.tan(np.pi*(n+4)/(4*n))
    # eigs, sigmas, R = test_fb_strats(geometry.chevron(1,h2), N, verbose=True)
    # np.savez(os.path.join(outdir, f"fb_corner_chev_{n}.npz"), N=N, eigs=eigs, sigmas=sigmas, R=R)
    n_basis = 80
    H1, H2, N1, N2, eigs, sigmas, R = test_chevron(n_basis, rtol=1e-8, verbose=True)
    np.savez(os.path.join(outdir, "fb_corner_chev.npz"), H1=H1, H2=H2, N1=N1, N2=N2,
             eigs=eigs, sigmas=sigmas, R=R)

