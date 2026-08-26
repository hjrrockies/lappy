import numpy as np
from lappy.bases import FourierBesselBasis, FundamentalBasis, MultiBasis, fb_corner_orders, fs_corner_orders, fs_bdry_sps
from dataclasses import dataclass

@dataclass
class BasisSpec:
    fs_corner_frac: float = 0.0
    fs_bdry_frac: float = 0.0
    fs_corner_sigma: float = 1.0
    fs_corner_C: float = 1.0
    fs_corner_order: int = 1
    fs_bdry_d: float = 1.0
    fs_bdry_order: int = 1

def make_basis(domain, n_target, basis_spec):
    bs = basis_spec
    fb_frac = 1.0 - bs.fs_corner_frac - bs.fs_bdry_frac
    n_fb = np.ceil(n_target*fb_frac).astype(int)
    n_fs_corner = np.ceil(n_target*bs.fs_corner_frac).astype(int)
    n_fs_bdry = np.ceil(n_target*bs.fs_bdry_frac).astype(int)

    fb_basis = None
    if n_fb > 0:
        fb_orders = fb_corner_orders(domain, n_fb)
        fb_basis = FourierBesselBasis.at_corners(domain, fb_orders)

    fs_corner_basis = None
    if n_fs_corner > 0:
        fs_orders = fs_corner_orders(domain, n_fs_corner, order=bs.fs_corner_order)
        fs_corner_basis = FundamentalBasis.by_corners(domain, fs_orders, bs.fs_corner_C, 
                                                      bs.fs_corner_sigma, bs.fs_corner_order)

    fs_bdry_basis = None
    if n_fs_bdry > 0:
        sps = fs_bdry_sps(domain, n_fs_bdry, bs.fs_bdry_order)
        fs_bdry_basis = FundamentalBasis.by_boundary(domain, sps, bs.fs_bdry_d, bs.fs_bdry_order)

    bases = [b for b in [fb_basis, fs_corner_basis, fs_bdry_basis] if b is not None]
    if len(bases) == 1: return bases[0]
    else: return MultiBasis(bases)
            
