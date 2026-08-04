"""Basis-composition laboratory.

Measures the one thing that actually matters for MPS accuracy: how small a
tension a given basis can achieve at a *known* eigenvalue. Everything else
(digits, convergence rate, escalation schedules) is downstream of that.

Used to answer "is this domain resolution-limited, and if so by which part
of the basis?" without running a full eigenvalue search each time.
"""
import numpy as np

from lappy import geometry as G, bases, mps, MPSEigensolver
from lappy.bases import (FourierBesselBasis, FundamentalBasis,
                         fb_corner_orders, fs_corner_orders)
from lappy.symmetry import (SymmetrizedBasis, prune_columns, fundamental_bdry_pts,
                            fundamental_int_pts)


def probe(domain, basis, lam, bdry_mult=3, int_npts=None, rtol=1e-14, seed=0,
          group=None, sector=None):
    """Tension at ``lam`` for ``basis``, optionally projected onto a symmetry
    sector. Returns ``(sigma, n_cols, n_reg)``."""
    np.random.seed(seed)
    order = group.order if group is not None else 1
    n_per_seg = mps.pts_per_seg(domain, basis, mult=bdry_mult * order)
    if group is None:
        bdry_pts = domain.bdry_pts(n_per_seg)
        int_pts = domain.int_pts(method='random',
                                 npts_rand=int_npts or 3 * len(basis))
        b = basis
    else:
        bdry_pts = fundamental_bdry_pts(domain, group, n_per_seg)
        int_pts = fundamental_int_pts(domain, group,
                                      int_npts or 3 * len(basis) // order)
        b = prune_columns(SymmetrizedBasis(basis, group, sector), lam,
                          np.concatenate([bdry_pts.pts, int_pts.pts]))
    n_cols = len(b)
    if n_cols == 0:
        return np.inf, 0, 0
    solver = MPSEigensolver(b.to_normalized((bdry_pts, int_pts)), bdry_pts,
                            int_pts, rtol=rtol, ttol=1e-3)
    d = solver._tension_diagnostics(lam)
    return solver.sigma(lam), n_cols, d['n_reg']


def fb_only(domain, orders):
    return FourierBesselBasis.from_domain(domain, list(orders))


def fb_plus_fs_corners(domain, fb_orders, n_fs, fs_order=2, C=10.0, sigma=1.0):
    fb = FourierBesselBasis.from_domain(domain, list(fb_orders))
    spc = fs_corner_orders(domain, n_fs, order=fs_order)
    fs = FundamentalBasis.by_corners(domain, spc, C, sigma, fs_order)
    return fb + fs


def fb_plus_fs_bdry(domain, fb_orders, n_fs, d=1.0, fs_order=1):
    fb = FourierBesselBasis.from_domain(domain, list(fb_orders))
    n_per = bases.fs_bdry_sps(domain, n_fs, order=fs_order)
    fs = FundamentalBasis.by_boundary(domain, n_per, d=d, order=fs_order)
    return fb + fs


def sweep(label, rows):
    print(f'\n=== {label} ===')
    print(f"{'configuration':<44} {'cols':>5} {'n_reg':>6} {'sigma':>11}")
    for name, sig, nc, nr in rows:
        print(f'{name:<44} {nc:5d} {nr:6d} {sig:11.3e}')
