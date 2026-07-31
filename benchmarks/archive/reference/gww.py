def gww(n_eigs, n_fb_basis, n_fs_basis, ltol, rtol, ppl):
    # GWW domain: Dirichlet eigenvalues and error estimates
    from lappy import geometry, FourierBesselBasis, FundamentalBasis
    from benchmarking import estimate_peon, precise_eigs

    # get domain (Dirichlet and Neumann boundary)
    gww = geometry.GWW1()
    gww_neu = geometry.GWW1(bc='neu')

    # basis info
    # set up Fourier-Bessel Basis
    if n_fb_basis > 0:
        int_angles = gww.int_angles
        reg_corner_idx = ((np.pi/int_angles % 1) < 1e-16)
        singular_angles = int_angles[~reg_corner_idx]
        orders = np.zeros(int_angles.shape, dtype=int)
        orders[~reg_corner_idx] = np.ceil(n_fb_basis*singular_angles/singular_angles.sum()).astype(int)
        while orders.sum() > n_fb_basis:
            orders[orders.argmax()] -= 1
        while orders.sum() < n_fb_basis:
            orders[orders.argmin()] += 1

    # set up fundamental solution basis
    seg_lens = np.array([seg.len for seg in gww.bdry.segments])
    if n_fs_basis > 0:
        n_per_seg = np.ceil(n_fs_basis*seg_lens/seg_lens.sum()).astype(int)
        while n_per_seg.sum() > n_fs_basis:
            n_per_seg[n_per_seg.argmax()] -= 1
        while n_per_seg.sum() < n_fs_basis:
            n_per_seg[n_per_seg.argmin()] += 1

    # bdry and int pts
    n_bdry = 2*(n_fb_basis + n_fs_basis)
    pts_per_seg = np.ceil(n_bdry*seg_lens/seg_lens.sum()).astype(int)
    bdry_pts, bdry_normals = gww.bdry_data(pts_per_seg)[:2]
    int_pts = gww.int_pts(npts_rand=(n_fb_basis + n_fs_basis))
    print(f"len(basis)={n_fb_basis+n_fs_basis}")
    print(f"len(bdry_pts)={len(bdry_pts)}, len(int_pts)={len(int_pts)}")

    # bdry and int nodes
    bdry_nodes = gww.bdry_pts(pts_per_seg, weights=True)
    int_nodes = gww.int_pts('mesh', mesh_size=0.25, weights=True)
    print(f"len(bdry_nodes)={len(bdry_nodes)}, len(int_nodes)={len(int_nodes)}")

    # estimate Poisson extension operator norm
    # C(Omega) <= 1/sqrt(inradius(Omega)*mu_1(Omega))
    # where mu_1(Omega) is the first nonzero Neumann eigenvalue
    print("Estimating Poisson extension operator norm")
    basis_neu = FourierBesselBasis.from_domain(gww_neu, orders)
    if n_fs_basis > 0:
        fs_basis_neu = FundamentalBasis.from_domain(gww_neu, n_per_seg, 0.15)
        basis_neu = basis_neu + fs_basis_neu
    basis_neu = basis_neu.to_normalized((bdry_pts, int_pts))
    peon_bound = estimate_peon(gww_neu, basis_neu, bdry_pts, int_pts, bdry_normals, ltol, 1e-6, ppl, verbose=2)
    print(f"estimated bound on C(Omega):", peon_bound)

    # solve for Dirichlet eigs
    print("Solving for Dirichlet eigenvalues")
    basis = FourierBesselBasis.from_domain(gww, orders)
    if n_fs_basis > 0:
        fs_basis = FundamentalBasis.from_domain(gww, n_per_seg, 0.15)
        basis = basis + fs_basis
    basis = basis.to_normalized((bdry_pts, int_pts))
    eigs, tensions = precise_eigs(n_eigs, gww, basis, bdry_pts, int_pts, bdry_nodes, int_nodes, ltol, rtol, ppl, 2)

    # compute estimated relative error bound
    relerr_est = tensions*peon_bound

    print("eigs:", eigs)
    print("relerr_est:", relerr_est)

    return eigs, tensions, relerr_est, peon_bound
    

if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="GWW domain dirichlet eigenvalues")
    parser.add_argument("n_eigs", type=int, help="Number of eigenvalues to compute")
    parser.add_argument("n_fb_basis", type=int, help="Number of Fourier-Bessel basis functions to use")
    parser.add_argument("n_fs_basis", type=int, help="Number of FS sources to use")
    parser.add_argument("--ltol", type=float, help="Eigenvalue relative tolerance", default=5e-16)
    parser.add_argument("--rtol", type=float, help="Regularization tolerance", default=1e-12)
    parser.add_argument("--ppl", type=int, help="points per level", default=10)
    parser.add_argument("--outfile", type=str, help="results outfile", default="")

    args = parser.parse_args()
    n_eigs = args.n_eigs
    n_fb_basis = args.n_fb_basis
    n_fs_basis = args.n_fs_basis
    ltol = args.ltol
    rtol = args.rtol
    ppl = args.ppl
    if args.outfile == "":
        outfile = f"gww_e{n_eigs}_nfb{n_fb_basis}_fns{n_fs_basis}"
    else:
        outfile = args.outfile
    
    eigs, tensions, relerr_est, peon_bound = gww(n_eigs, n_fb_basis, n_fs_basis, ltol, rtol, ppl)
    # save results
    np.savez(outfile, eigs=eigs, tensions=tensions, relerr_est=relerr_est, peon_bound=peon_bound)
