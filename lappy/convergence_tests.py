from . import bases
from .bases import FourierBesselBasis, FundamentalBasis, NormalizedBasis
from . import geometry
from .mps import MPSEigensolver
from .opt import golden_search
import numpy as np

def make_fb_corner_basis(domain, n):
    """Makes a Fourier-Bessel basis for the domain on the corners"""
    if n == 0:
        return None
    orders = bases.fb_corner_orders(domain, n)
    return FourierBesselBasis.at_corners(domain, orders)

def make_fs_bdry_basis(domain, n, d, order=1, spacing='even'):
    """Makes a fundamental solution basis for the domain along the boundary"""
    if n == 0:
        return None
    sources_per_seg = bases.fs_bdry_sps(domain, n, order)
    return FundamentalBasis.by_boundary(domain, sources_per_seg, d, order, spacing)

def make_fs_corner_basis(domain, n, C, sigma, order=1, mults=(0,1,1)):
    """Makes a fundamental solution basis for the domain using exponentially spaced 'lightning' points at the corners"""
    if n == 0:
        return None
    fracs = bases.fs_corner_fraction(domain, mults[0], mults[1], mults[2])
    sources_per_corner = bases.fs_corner_orders(domain, n, fracs, order)
    return FundamentalBasis.by_corners(domain, sources_per_corner, C, sigma, order)

def make_bdry_pts(domain, fb_basis=None, fs_basis=None, mult=2, min_per_seg=3):
    """makes boundary collocation points"""
    if fb_basis is None and fs_basis is None:
        raise Exception("at least one basis must be provided")
    n_segments = len(domain.bdry.segments)
    pts_per_seg = np.zeros(n_segments, dtype=int)
    if fb_basis is not None:
        pps_fb = geometry.pts_per_seg(domain, fb_basis, mult, min_per_seg)
        pts_per_seg += pps_fb
    if fs_basis:
        seg_lens = domain.seg_lens
        pps_fs = np.round(len(fs_basis)*mult*seg_lens/seg_lens.sum()).astype(int)
        pts_per_seg += pps_fs
    pts_per_seg[pts_per_seg < min_per_seg] = min_per_seg
    return domain.bdry_pts(pts_per_seg)

def make_solver(domain, n_fb, n_fs_bdry, n_fs_corner, fs_bdry_kwargs, fs_corner_kwargs,
                rtol=1e-14, bdry_pts=None, int_pts=None, bdry_mult=2, bdry_mps=3):
    fb_basis = make_fb_corner_basis(domain, n_fb)
    fs_bdry_basis = make_fs_bdry_basis(domain, n_fs_bdry, **fs_bdry_kwargs)
    fs_corner_basis = make_fs_corner_basis(domain, n_fs_corner, **fs_corner_kwargs)

    if fs_bdry_basis is not None and fs_corner_basis is not None:
        fs_basis = fs_bdry_basis + fs_corner_basis
    elif fs_bdry_basis is not None:
        fs_basis = fs_bdry_basis
    elif fs_corner_basis is not None:
        fs_basis = fs_corner_basis
    else:
        fs_basis = None

    if fb_basis is not None and fs_basis is not None:
        basis = fb_basis + fs_basis
    elif fb_basis is not None:
        basis = fb_basis
    elif fs_basis is not None:
        basis = fs_basis
    if bdry_pts is None:
        bdry_pts = make_bdry_pts(domain, fb_basis, fs_basis, bdry_mult, bdry_mps)
    if int_pts is None:
        np.random.seed(0)
        int_pts = domain.int_pts(npts_rand=len(basis))

    basis = NormalizedBasis(basis, (bdry_pts, int_pts))
    return MPSEigensolver(basis, bdry_pts, int_pts, rtol=rtol)

def n_bases(n, fs_bdry_frac, fs_corner_frac):
    if fs_bdry_frac + fs_corner_frac > 1:
        raise ValueError("fs_bdry_frac and fs_corner_frac must add to at most 1")
    fb_frac = 1 - fs_bdry_frac - fs_corner_frac
    n_fb = np.round(n*fb_frac).astype(int)
    n_fs_bdry = np.round(n*fs_bdry_frac).astype(int)
    n_fs_corner = np.round(n*fs_corner_frac).astype(int)
    return n_fb, n_fs_bdry, n_fs_corner

def test_convergence(domain, eig_lb, eig_ub, N, fs_bdry_frac, fs_corner_frac, 
                     fs_bdry_kwargs, fs_corner_kwargs, rtol, bdry_pts, int_pts):
    """tests MPS convergence for an eigenvalue of a domain"""

    tensions = np.empty(N.shape)
    N_basis = np.empty_like(N)
    N_normed = np.empty_like(N)
    N_reg = np.empty_like(N)
    gsvd_rank = np.empty_like(N)
    for i,n in enumerate(N[::-1]):
        n_fb, n_fs_bdry, n_fs_corner = n_bases(n, fs_bdry_frac, fs_corner_frac)
        solver = make_solver(domain, n_fb, n_fs_bdry, n_fs_corner, fs_bdry_kwargs, fs_corner_kwargs, 
                            rtol, bdry_pts, int_pts)
        if i == 0 and eig_ub-eig_lb > 0:
            eig = golden_search(solver.sigma, eig_lb, eig_ub, tol=eig_ub*1e-15)[0]
        elif eig_ub == eig_lb:
            eig = eig_lb
        diagnostics = solver._tension_diagnostics(eig)
        tensions[-(i+1)] = diagnostics['sigma']
        N_basis[-(i+1)] = len(solver.basis)
        N_normed[-(i+1)] = diagnostics['n']
        N_reg[-(i+1)] = diagnostics['n_reg']
        gsvd_rank[-(i+1)] = diagnostics['gsvd_rank']
    return eig, tensions, N_basis, N_normed, N_reg, gsvd_rank