def gww(n_eigs, n_basis, ltol, rtol, ppl):
    # GWW domain: Dirichlet eigenvalues and error estimates
    from lappy import geometry, FourierBesselBasis
    from benchmarking import estimate_peon, precise_eigs

    # get domain (Dirichlet and Neumann boundary)
    gww = geometry.GWW1()
    gww_neu = geometry.GWW1(bc='neu')

    # basis info
    orders = [0, n_basis, 0, n_basis, 0, 0, n_basis, n_basis]

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
    basis_neu = FourierBesselBasis.from_domain(gww_neu, orders).to_normalized(bdry_pts + int_pts)
    peon_bound = estimate_peon(gww_neu, basis_neu, bdry_pts, int_pts, bdry_normals, ltol, 1e-8, ppl, verbose=2)
    print(f"estimated bound on C(Omega):", peon_bound)

    # solve for Dirichlet eigs
    print("Solving for Dirichlet eigenvalues")
    basis = FourierBesselBasis.from_domain(gww, orders).to_normalized(bdry_pts + int_pts)
    eigs, tensions = precise_eigs(n_eigs, gww, basis, bdry_pts, int_pts, bdry_nodes, int_nodes, ltol, rtol, ppl, 2)

    # compute estimated relative error bound
    relerr_est = tensions*peon_bound

    print(eigs)
    print(relerr_est)

    return eigs, tensions, relerr_est, peon_bound
    

if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="GWW domain dirichlet eigenvalues")
    parser.add_argument("n_eigs", type=int, help="Number of eigenvalues to compute")
    parser.add_argument("n_basis", type=int, help="Number of basis functions to use")
    parser.add_argument("--ltol", type=float, help="Eigenvalue relative tolerance", default=5e-16)
    parser.add_argument("--rtol", type=float, help="Regularization tolerance", default=1e-14)
    parser.add_argument("--ppl", type=int, help="points per level", default=10)
    parser.add_argument("--outfile", type=str, help="results outfile", default="")

    args = parser.parse_args()
    n_eigs = args.n_eigs
    n_basis = args.n_basis
    ltol = args.ltol
    rtol = args.rtol
    ppl = args.ppl
    if args.outfile == "":
        outfile = f"gww_e{n_eigs}_n{n_basis}"
    else:
        outfile = args.outfile
    
    eigs, tensions, relerr_est, peon_bound = gww(n_eigs, n_basis, ltol, rtol, ppl)
    # save results
    np.savez(outfile, eigs=eigs, tensions=tensions, relerr_est=relerr_est, peon_bound=peon_bound)
