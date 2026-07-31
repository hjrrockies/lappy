from lappy import *
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile

gww = geometry.GWW1()
n_fb_basis = 600
n_fs_basis = 0
d = 0
ltol = 5e-16
rtol = 1e-14

# build collocation points
n_bdry = 2*n_fb_basis
seg_lens = np.array([seg.len for seg in gww.bdry.segments])
n_per_seg = np.ceil(n_bdry*seg_lens/seg_lens.sum()).astype(int)
bdry_nodes = gww.bdry_pts(n_per_seg, weights=True)
int_nodes = gww.int_pts(method='mesh', mesh_size=.5, weights=True)

@profile
def gww_eig1_tension():
    """Solve for the first eigenvalue of the GWW-1 domain using n_fb_basis Fourier-Bessel
    basis functions (count proportional to interior angle of singular corners) and n_fs_basis 
    fundamental solution sources (proportional to edge length) spaced distance d from the boundary"""

    # set up Fourier-Bessel Basis
    if n_fb_basis > 0:
        int_angles = gww.int_angles
        reg_corner_idx = ((np.pi/int_angles % 1) < 1e-16)
        singular_angles = int_angles[~reg_corner_idx]
        orders = np.zeros(int_angles.shape, dtype=int)
        orders[~reg_corner_idx] = np.ceil(n_fb_basis*singular_angles/singular_angles.sum()).astype(int)
        fb_basis = FourierBesselBasis.from_domain(gww, orders)
        while orders.sum() > n_fb_basis:
            orders[orders.argmax()] -= 1
        while orders.sum() < n_fb_basis:
            orders[orders.argmin()] += 1

    # set up fundamental solution basis
    seg_lens = np.array([seg.len for seg in gww.bdry.segments])
    if n_fs_basis > 0:
        n_per_seg = np.ceil(n_fs_basis*seg_lens/seg_lens.sum()).astype(int)
        fs_basis = FundamentalBasis.from_domain(gww, n_per_seg, d)
        while n_per_seg.sum() > n_fs_basis:
            n_per_seg[n_per_seg.argmax()] -= 1
        while n_per_seg.sum() < n_fs_basis:
            n_per_seg[n_per_seg.argmin()] += 1

    # combine
    if n_fb_basis > 0 and n_fs_basis > 0:
        basis = fb_basis + fs_basis
    elif n_fb_basis > 0:
        basis = fb_basis
    elif n_fs_basis > 0:
        basis = fs_basis
    else:
        raise ValueError("need nonzero number of basis functions")

    # normalize basis
    basis = basis.to_normalized(bdry_nodes + int_nodes)
    
    # setup solver
    solver = MPSEigensolver(basis, bdry_nodes, int_nodes, rtol=rtol, ltol=ltol)

    # use golden search for first eigenvalue
    eig1 = opt.golden_search(solver.sigma, 2.53794, 2.53795, tol=2.54*ltol)[0]

    return eig1, solver.sigma(eig1), solver

if __name__ == "__main__":
    gww_eig1_tension()