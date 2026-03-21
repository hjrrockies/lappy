import numpy as np

def iso_tri(h, n_eigs, n_fb_basis, n_fs_basis, ltol, rtol, ppl):
    # isosceles triangle domain: Dirichlet eigenvalues and error estimates
    from lappy import geometry, FourierBesselBasis, FundamentalBasis, asymp
    from benchmarking import estimate_peon, precise_eigs

    # get domain (Dirichlet and Neumann boundary)
    tri = geometry.iso_tri(h)
    tri_neu = geometry.iso_tri(h, bc='neu')

    # basis info
    int_angles = tri.int_angles
    orders = np.ceil(n_fb_basis*int_angles/int_angles.sum()).astype(int)
    print(orders)

    lens = np.array([seg.len for seg in tri.bdry.segments])
    sources_per_seg = np.ceil(n_fs_basis*lens/lens.sum()).astype(int)
    print(sources_per_seg)

    # bdry and int pts
    n_basis = n_fb_basis + n_fs_basis
    n_bdry = 2*n_basis
    n_per_seg = np.ceil(n_bdry*lens/lens.sum()).astype(int)
    bdry_pts, bdry_normals = tri.bdry_data(n_per_seg)[:2]
    int_pts = tri.int_pts(npts_rand=n_basis)
    print(f"len(basis) = {n_basis}")
    print(f"len(bdry_pts)={len(bdry_pts)}, len(int_pts)={len(int_pts)}")

    # bdry and int nodes
    bdry_nodes = tri.bdry_pts(n_per_seg, weights=True)
    int_nodes = tri.int_pts('mesh', mesh_size=0.25)
    print(f"len(bdry_nodes)={len(bdry_nodes)}, len(int_nodes)={len(int_nodes)}")

    # estimate Poisson extension operator norm
    # C(Omega) <= 1/sqrt(inradius(Omega)*mu_1(Omega))
    # where mu_1(Omega) is the first nonzero Neumann eigenvalue
    print("Estimating Poisson extension operator norm")
    fb_basis_neu = FourierBesselBasis.from_domain(tri_neu, orders)
    d_neu = 2*np.pi/asymp.weyl_est(1, tri_neu)
    fs_basis_neu = FundamentalBasis.from_domain(tri_neu, sources_per_seg, d=d_neu)
    basis_neu = (fb_basis_neu + fs_basis_neu).to_normalized(bdry_pts + int_pts)
    peon_bound = estimate_peon(tri_neu, basis_neu, bdry_pts, int_pts, bdry_normals, ltol, rtol, ppl, verbose=2)
    print(f"estimated bound on C(Omega):", peon_bound)

    # solve for Dirichlet eigs
    print("Solving for Dirichlet eigenvalues")
    fb_basis = FourierBesselBasis.from_domain(tri, orders).to_normalized(bdry_pts + int_pts)
    d = 2*np.pi/asymp.weyl_est(1, tri)
    fs_basis = FundamentalBasis.from_domain(tri, sources_per_seg, d=d)
    basis = (fb_basis + fs_basis).to_normalized(bdry_pts + int_pts)
    eigs, tensions = precise_eigs(n_eigs, tri, basis, bdry_pts, int_pts, bdry_nodes, int_nodes, ltol, rtol, ppl, 2)

    # compute estimated relative error bound
    relerr_est = tensions*peon_bound

    print(eigs)
    print(relerr_est)

    return eigs, tensions, relerr_est, peon_bound
    

if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="triron domain dirichlet eigenvalues")
    parser.add_argument("--h", type=float, help="h param", default=1.0)
    parser.add_argument("n_eigs", type=int, help="Number of eigenvalues to compute")
    parser.add_argument("n_fb_basis", type=int, help="Number of FB basis functions to use")
    parser.add_argument("n_fs_basis", type=int, help="Number of FS basis functions to use")
    parser.add_argument("--ltol", type=float, help="Eigenvalue relative tolerance", default=5e-16)
    parser.add_argument("--rtol", type=float, help="Regularization tolerance", default=1e-14)
    parser.add_argument("--ppl", type=int, help="points per level", default=10)
    parser.add_argument("--outfile", type=str, help="results outfile", default="")

    args = parser.parse_args()
    h = args.h
    n_eigs = args.n_eigs
    n_fb_basis = args.n_fb_basis
    n_fs_basis = args.n_fs_basis
    ltol = args.ltol
    rtol = args.rtol
    ppl = args.ppl
    if args.outfile == "":
        outfile = f"iso_tri_{h}_e{n_eigs}_fb{n_fb_basis}_fs{n_fs_basis}"
    else:
        outfile = args.outfile
    
    eigs, tensions, relerr_est, peon_bound = iso_tri(h, n_eigs, n_fb_basis, n_fs_basis, ltol, rtol, ppl)
    # save results
    np.savez(outfile, eigs=eigs, tensions=tensions, relerr_est=relerr_est, peon_bound=peon_bound)