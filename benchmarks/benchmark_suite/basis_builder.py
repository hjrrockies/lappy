"""Utilities to construct and normalize particular solution bases for benchmarking."""
import numpy as np
from lappy import FourierBesselBasis, FundamentalBasis, geometry
from lappy.asymp import weyl_est


def build_fb_basis(domain, n_fb, strategy='singular_angle_weighted'):
    """
    Build a Fourier-Bessel basis for *domain* using *n_fb* functions total.

    Parameters
    ----------
    domain : Domain
    n_fb   : int — total number of FB basis functions
    strategy : str
        'uniform'                 — equal orders per corner
        'angle_weighted'          — orders ∝ interior corner angle
        'singular_only'           — all orders at non-regular corners; zero at regular
        'singular_angle_weighted' — orders ∝ singular corner angle, zero for regular corners

    Returns
    -------
    FourierBesselBasis
    """
    int_angles = domain.int_angles
    n_corners = len(int_angles)
    orders = np.zeros(n_corners, dtype=int)

    if n_fb == 0:
        return FourierBesselBasis.from_domain(domain, orders)

    if strategy == 'uniform':
        # equal orders per corner
        base = n_fb // n_corners
        orders[:] = base
        remainder = n_fb - base * n_corners
        orders[:remainder] += 1

    elif strategy == 'angle_weighted':
        # proportional to interior angle
        weights = int_angles / int_angles.sum()
        orders = np.floor(weights * n_fb).astype(int)
        _adjust_sum(orders, n_fb)

    elif strategy == 'singular_only':
        # all orders at non-regular corners (alpha not a rational multiple of pi)
        reg_mask = (np.pi / int_angles % 1) < 1e-10
        singular_mask = ~reg_mask
        n_singular = singular_mask.sum()
        if n_singular == 0:
            # fall back to uniform
            base = n_fb // n_corners
            orders[:] = base
            remainder = n_fb - base * n_corners
            orders[:remainder] += 1
        else:
            base = n_fb // n_singular
            orders[singular_mask] = base
            remainder = n_fb - base * n_singular
            idx = np.where(singular_mask)[0]
            orders[idx[:remainder]] += 1

    elif strategy == 'singular_angle_weighted':
        # proportional to singular angle, zero for regular corners
        reg_mask = (np.pi / int_angles % 1) < 1e-10
        singular_mask = ~reg_mask
        singular_angles = int_angles[singular_mask]
        if len(singular_angles) == 0:
            # fall back to uniform
            base = n_fb // n_corners
            orders[:] = base
            remainder = n_fb - base * n_corners
            orders[:remainder] += 1
        else:
            weights = singular_angles / singular_angles.sum()
            singular_orders = np.floor(weights * n_fb).astype(int)
            _adjust_sum(singular_orders, n_fb)
            orders[singular_mask] = singular_orders

    else:
        raise ValueError(f"Unknown FB strategy: {strategy!r}")

    return FourierBesselBasis.from_domain(domain, orders)


def build_fs_basis(domain, n_fs, d_strategy='fixed', d=1.0, d_scale=1.0,
                   seg_strategy='length_weighted'):
    """
    Build a FundamentalSolution basis for *domain* using *n_fs* sources total.

    Parameters
    ----------
    domain       : Domain
    n_fs         : int — total source count
    d_strategy   : str
        'fixed'             — use d directly
        'inradius_fraction' — d = d_scale * domain.inradius
        'spectral'          — d = 2π / sqrt(weyl_est(1, domain))
    d            : float — distance from boundary (for 'fixed' strategy)
    d_scale      : float — scale factor (for 'inradius_fraction' and 'spectral' strategies)
    seg_strategy : str
        'length_weighted' — sources ∝ segment length
        'uniform'         — equal sources per segment

    Returns
    -------
    FundamentalBasis
    """
    if d_strategy == 'fixed':
        dist = d
    elif d_strategy == 'inradius_fraction':
        dist = d_scale * domain.inradius
    elif d_strategy == 'spectral':
        lam1 = weyl_est(1, domain)
        dist = d_scale * 2 * np.pi / np.sqrt(max(lam1, 1e-6))
    else:
        raise ValueError(f"Unknown FS distance strategy: {d_strategy!r}")

    seg_lens = np.array([seg.len for seg in domain.bdry.segments])
    n_segs = len(seg_lens)

    if seg_strategy == 'length_weighted':
        weights = seg_lens / seg_lens.sum()
        n_per_seg = np.floor(weights * n_fs).astype(int)
        _adjust_sum(n_per_seg, n_fs)
    elif seg_strategy == 'uniform':
        base = n_fs // n_segs
        n_per_seg = np.full(n_segs, base, dtype=int)
        remainder = n_fs - base * n_segs
        n_per_seg[:remainder] += 1
    else:
        raise ValueError(f"Unknown FS segment strategy: {seg_strategy!r}")

    # ensure at least 1 source per segment where total > 0
    n_per_seg = np.maximum(n_per_seg, 0)

    return FundamentalBasis.from_domain(domain, n_per_seg, d=dist)


def build_basis(domain, n_fb, n_fs, fb_strategy='singular_angle_weighted',
                fs_d_strategy='fixed', fs_d=1.0, fs_d_scale=1.0,
                fs_seg_strategy='length_weighted',
                normalize=True, norm_pts=None):
    """
    Build a combined FB+FS particular solution basis.

    Parameters
    ----------
    domain         : Domain
    n_fb           : int — FB basis count (0 = no FB basis)
    n_fs           : int — FS basis count (0 = no FS basis)
    fb_strategy    : str — FB order allocation strategy
    fs_d_strategy  : str — FS source distance strategy
    fs_d           : float — FS source distance (for 'fixed')
    fs_d_scale     : float — FS source distance scale factor
    fs_seg_strategy: str — FS source density per segment strategy
    normalize      : bool — normalize basis via to_normalized()
    norm_pts       : PointSet or (bdry_pts, int_pts) or None
                     Points for L² norm; if None, will use bdry+interior pts with default counts.

    Returns
    -------
    ParticularBasis (FourierBesselBasis, FundamentalBasis, MultiBasis, or NormalizedBasis)
    """
    if n_fb > 0 and n_fs > 0:
        fb = build_fb_basis(domain, n_fb, strategy=fb_strategy)
        fs = build_fs_basis(domain, n_fs, d_strategy=fs_d_strategy, d=fs_d,
                            d_scale=fs_d_scale, seg_strategy=fs_seg_strategy)
        basis = fb + fs
    elif n_fb > 0:
        basis = build_fb_basis(domain, n_fb, strategy=fb_strategy)
    elif n_fs > 0:
        basis = build_fs_basis(domain, n_fs, d_strategy=fs_d_strategy, d=fs_d,
                               d_scale=fs_d_scale, seg_strategy=fs_seg_strategy)
    else:
        raise ValueError("At least one of n_fb or n_fs must be > 0")

    if normalize:
        if norm_pts is None:
            n_total = n_fb + n_fs
            # default: 2x boundary oversampling, 1x interior
            n_bdry = max(2 * n_total, 20)
            n_int = max(n_total, 10)
            seg_lens = np.array([seg.len for seg in domain.bdry.segments])
            n_per_seg = np.maximum(
                np.round(n_bdry * seg_lens / seg_lens.sum()).astype(int), 1
            )
            bdry_pts = domain.bdry_pts(n_per_seg)
            int_pts = domain.int_pts(npts_rand=n_int)
            norm_pts = (bdry_pts, int_pts)
        basis = basis.to_normalized(norm_pts)

    return basis


# --- helper ---

def _adjust_sum(arr, target):
    """Adjust integer array (in-place) so arr.sum() == target by incrementing/decrementing argmax/argmin."""
    while arr.sum() < target:
        arr[arr.argmin()] += 1
    while arr.sum() > target:
        arr[arr.argmax()] -= 1
