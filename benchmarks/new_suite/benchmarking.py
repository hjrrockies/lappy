from lappy import *
import numpy as np

def build_fb_basis(dom, n, strat='singular_weighted'):
    # get corners, interior angles   
    corners = dom.corners
    int_angles = dom.int_angles
    ratio = np.pi / int_angles
    reg_corner_idx = np.abs(ratio - np.round(ratio))/ratio < 1e-15
    sing_corner_idx = ~reg_corner_idx
    sing_angles = int_angles[sing_corner_idx]
    
    if sing_corner_idx.sum() == 0:
        if strat == 'singular_weighted': strat = 'weighted'
        elif strat == 'singular': strat = 'uniform'
    if strat == 'uniform':
        orders = np.ceil(n/len(corners)).astype(int)*np.ones(len(corners))
    elif strat == 'weighted':
        orders = np.ceil(n*int_angles/int_angles.sum()).astype(int)
    elif strat == 'singular_uniform':
        orders = np.zeros(len(corners), dtype=int)
        orders[sing_corner_idx] = np.ceil(n/sing_corner_idx.sum()).astype(int)
    elif strat == 'singular_weighted':
        orders = np.zeros(len(corners), dtype=int)
        orders[sing_corner_idx] = np.ceil(n*sing_angles/sing_angles.sum()).astype(int)
    return FourierBesselBasis.from_domain(dom, orders)

def build_fs_basis(dom, n, d=1.0):
    sources_per_seg = np.ceil(n*dom.seg_lens/dom.seg_lens.sum()).astype(int)
    return FundamentalBasis.from_domain(dom, sources_per_seg, d)

def build_eigprob(dom, n_fb_basis, fb_strat='singular_weighted', n_fs_basis=0, fs_dist=1.0,
                  rtol=None, ltol=5e-16, n_bdry_mult=2, n_int=50, normalize=True):
    """computes the first n_eigs of a domain"""
    if n_fb_basis == 0 and n_fs_basis == 0:
        raise ValueError("must have a nonzero number of basis functions")
    
    # build fb_basis
    if n_fb_basis > 0:
        fb_basis = build_fb_basis(dom, n_fb_basis, fb_strat)

    # build fs_basis
    if n_fs_basis > 0:
        fs_basis = build_fs_basis(dom, n_fs_basis, fs_dist)

    # combine bases, if needed
    if n_fb_basis > 0 and n_fs_basis > 0:
        basis = fb_basis + fs_basis
    elif n_fs_basis == 0:
        basis = fb_basis
    elif n_fb_basis == 0:
        basis = fs_basis

    # build bdry_pts and int_pts, normalize basis
    n_bdry = n_bdry_mult*len(basis)
    pts_per_seg = np.ceil(n_bdry*dom.seg_lens/dom.seg_lens.sum()).astype(int)
    bdry_pts = dom.bdry_pts(pts_per_seg)
    int_pts = dom.int_pts(npts_rand=n_int)
    if normalize:
        basis = NormalizedBasis(basis, (bdry_pts, int_pts))

    # build solver
    solver = MPSEigensolver(basis, bdry_pts, int_pts, ltol=ltol)
    
    # handle rtol
    if rtol is None:
        lam_fk = bounds.faber_krahn(dom)
        solver.rtol = solver.adapt_rtol(lam_fk, 1.5*lam_fk)
    else:
        solver.rtol = rtol

    # build eigenproblem
    eigprob = Eigenproblem(dom, solver)
    return eigprob