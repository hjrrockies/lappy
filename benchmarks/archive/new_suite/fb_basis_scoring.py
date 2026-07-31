from lappy import *
import numpy as np
import os

def build_eigprob(domain, n_basis, regular_mult=1, singular_mult=2, reentrant_mult=1,
                  rtol=None, ltol=5e-16, n_bdry_mult=2, n_int=50, normalize=True):
    """sets up eigenproblem object"""
    fb_fractions = bases.fb_corner_fraction(domain, regular_mult, singular_mult, reentrant_mult)
    orders = np.round(n_basis*fb_fractions).astype(int)
    basis = FourierBesselBasis.from_domain(domain, orders)

    # build bdry_pts and int_pts, normalize basis
    pts_per_seg = geometry.pts_per_seg(domain, basis, mult=n_bdry_mult)
    bdry_pts = domain.bdry_pts(pts_per_seg)
    np.random.seed(0)
    int_pts = domain.int_pts(npts_rand=n_int)
    if normalize:
        basis = NormalizedBasis(basis, (bdry_pts, int_pts))

    # build solver
    solver = MPSEigensolver(basis, bdry_pts, int_pts, ltol=ltol)
    
    # handle rtol
    if rtol is None:
        lam_fk = bounds.faber_krahn(domain)
        solver.rtol = solver.adapt_rtol(lam_fk, 3*lam_fk, 15)
    else:
        solver.rtol = rtol

    # build eigenproblem
    eigprob = Eigenproblem(domain, solver)
    return eigprob

def test_singular_mult(domain, N, M, regular_mult=1, reentrant_mult=1, rtol=None, verbose=False):
    eigs = np.empty((len(N), len(M)))
    sigmas = np.empty_like(eigs)
    rtols = np.empty_like(eigs)
    for i, n_basis in enumerate(N):
        if verbose: print("n_basis =", n_basis)
        for j, singular_mult in enumerate(M):
            if verbose: print("\tsingular_mult =", singular_mult)
            eigprob = build_eigprob(domain, n_basis, regular_mult, singular_mult, reentrant_mult, rtol)
            try:
                _eigs = eigprob.solve(1)
                if len(_eigs) > 0:
                    eigs[i,j] = _eigs[0]
                    sigmas[i,j] = eigprob.eval_solver.sigma(eigs[i,j])
                    rtols[i,j] = eigprob.eval_solver.rtol
                else:
                    raise EigensolverFailure("not enough eigs found")
            except EigensolverFailure as ef:
                print(ef)
                eigs[i,j] = np.nan
                sigmas[i,j] = np.nan
                rtols[i,j] = np.nan
    return eigs, sigmas, rtols

if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)
    # isosceles triangle tests
    # print("running isosceles triangle tests")
    # phi = np.pi/5
    # h = np.tan(phi)
    # iso_tri = geometry.iso_tri(h)
    # N = np.arange(10, 70, 4)
    # M = np.logspace(0, 2, 20)
    # eigs, sigmas, rtols = test_singular_mult(iso_tri, N, M, rtol=1e-14, verbose=True)
    # np.savez(os.path.join(outdir, "fb_scoring_iso_tri.npz"), N=N, M=M, eigs=eigs, sigmas=sigmas, rtols=rtols)

    # l-shape domain test 1
    # print("running L-shaped domain test")
    # N = np.arange(30, 90, 4)
    # M = np.logspace(0, 2, 20)
    # eigs, sigmas, rtols = test_singular_mult(geometry.L_shape(), N, M, rtol=1e-14, verbose=True)
    # np.savez(os.path.join(outdir, "fb_scoring_lshape.npz"), N=N, M=M, eigs=eigs, sigmas=sigmas, rtols=rtols)

    # gww domain test
    print("running GWW domain test")
    N = np.arange(40, 100, 4)
    M = np.logspace(0, 2, 20)
    eigs, sigmas, rtols = test_singular_mult(geometry.GWW1(), N, M, rtol=1e-12, verbose=True)
    np.savez(os.path.join(outdir, "fb_scoring_gww1.npz"), N=N, M=M, eigs=eigs, sigmas=sigmas, rtols=rtols)
    eigs, sigmas, rtols = test_singular_mult(geometry.GWW2(), N, M, rtol=1e-12, verbose=True)
    np.savez(os.path.join(outdir, "fb_scoring_gww2.npz"), N=N, M=M, eigs=eigs, sigmas=sigmas, rtols=rtols)

    # parallelogram test
    print("running parallelogram test")
    N = np.arange(40, 100, 4)
    M = np.logspace(0, 2, 20)
    eigs, sigmas, rtols = test_singular_mult(geometry.GWW1(), N, M, rtol=1e-12, verbose=True)
    np.savez(os.path.join(outdir, "fb_scoring_gww1.npz"), N=N, M=M, eigs=eigs, sigmas=sigmas, rtols=rtols)



