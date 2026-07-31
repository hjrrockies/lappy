def L_shape(n_eigs, n_basis, ltol, rtol, ppl):
    # L-shaped domain: Dirichlet eigenvalues and error estimates
    from lappy import geometry, FourierBesselBasis, NormalizedBasis
    from benchmarking import estimate_peon, precise_eigs

    # get domain (Dirichlet and Neumann boundary)
    ldom = geometry.L_shape()
    ldom_neu = geometry.L_shape(bc='neu')

    # basis info
    orders = [n_basis, 0, 0, 0, 0, 0]

    # bdry and int pts
    n_per_seg = n_basis
    bdry_pts, bdry_normals = ldom.bdry_data(n_per_seg)[:2]
    int_pts = ldom.int_pts(npts_rand=n_basis)
    print(f"len(basis) = {n_basis}")
    print(f"len(bdry_pts)={len(bdry_pts)}, len(int_pts)={len(int_pts)}")

    # bdry and int nodes
    bdry_nodes = ldom.bdry_pts(n_per_seg, weights=True)
    int_nodes = ldom.int_pts('mesh', mesh_size=0.25, weights=True)
    print(f"len(bdry_nodes)={len(bdry_nodes)}, len(int_nodes)={len(int_nodes)}")

    # estimate Poisson extension operator norm
    # C(Omega) <= 1/sqrt(inradius(Omega)*mu_1(Omega))
    # where mu_1(Omega) is the first nonzero Neumann eigenvalue
    print("Estimating Poisson extension operator norm")
    basis_neu = FourierBesselBasis.from_domain(ldom_neu, orders).to_normalized((bdry_pts, int_pts))
    peon_bound = estimate_peon(ldom_neu, basis_neu, bdry_pts, int_pts, bdry_normals, ltol, rtol, ppl, verbose=2)
    print(f"estimated bound on C(Omega):", peon_bound)

    # solve for Dirichlet eigs
    print("Solving for Dirichlet eigenvalues")
    basis = FourierBesselBasis.from_domain(ldom, orders).to_normalized((bdry_pts, int_pts))
    eigs, tensions = precise_eigs(n_eigs, ldom, basis, bdry_pts, int_pts, bdry_nodes, int_nodes, ltol, rtol, ppl, 2)

    # compute estimated relative error bound
    relerr_est = tensions*peon_bound

    print("eigs:", eigs)
    print("relerr_est:", relerr_est)

    return eigs, tensions, relerr_est, peon_bound
    

if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="L-shaped domain dirichlet eigenvalues")
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
        outfile = f"l_shape_e{n_eigs}_n{n_basis}"
    else:
        outfile = args.outfile
    
    eigs, tensions, relerr_est, peon_bound = L_shape(n_eigs, n_basis, ltol, rtol, ppl)
    # save results
    np.savez(outfile, eigs=eigs, tensions=tensions, relerr_est=relerr_est, peon_bound=peon_bound)