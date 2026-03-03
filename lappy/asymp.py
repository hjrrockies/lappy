"""lappy.asymp — Weyl asymptotic expansions for Laplacian eigenvalues."""

import numpy as np
import scipy.optimize


def _parse_domain_or_scalars(domain, area, perim, bc_type, need_perim=True):
    """Extract (area, perim, bc_type) from domain or keyword scalars."""
    if domain is not None:
        area = domain.area
        perim = domain.perimeter
        bc_type = domain.bc_type
    else:
        if area is None:
            raise ValueError("provide either a Domain or area=")
        if need_perim and perim is None:
            raise ValueError("provide either a Domain or both area= and perim=")
    return area, perim, bc_type


def _check_bc(bc_type):
    if bc_type == 'dir':
        sign = -1
    elif bc_type == 'neu':
        sign = +1
    elif bc_type in ('mixed', 'rob'):
        raise NotImplementedError(f"weyl asymptotics not implemented for bc_type={bc_type!r}")
    else:
        raise ValueError(f"unknown bc_type {bc_type!r}")
    return sign


def weyl_count(lam, domain=None, *, area=None, perim=None, bc_type='dir'):
    """Two-term Weyl counting function N(λ) ≈ (A·λ ∓ P·√λ) / (4π).

    Parameters
    ----------
    lam : float or ndarray
        Eigenvalue argument (vectorized).
    domain : Domain, optional
        If provided, area, perim, and bc_type are extracted automatically.
    area, perim : float, keyword-only
        Geometric scalars used when domain is None.
    bc_type : {'dir', 'neu'}, keyword-only
        'dir' uses minus sign (default); 'neu' uses plus sign.
        'mixed' and 'rob' raise NotImplementedError.
    """
    area, perim, bc_type = _parse_domain_or_scalars(domain, area, perim, bc_type)
    sign = _check_bc(bc_type)
    lam = np.asarray(lam, dtype=float)
    return (area * lam + sign * perim * np.sqrt(lam)) / (4 * np.pi)


def weyl_est(k, domain=None, *, area=None, perim=None, bc_type='dir'):
    """Asymptotic estimate for the k-th eigenvalue (closed-form inverse of weyl_count).

    Parameters
    ----------
    k : float or ndarray
        Eigenvalue index (vectorized). May be non-integer for interpolation.
    domain : Domain, optional
    area, perim : float, keyword-only
    bc_type : {'dir', 'neu'}, keyword-only
    """
    area, perim, bc_type = _parse_domain_or_scalars(domain, area, perim, bc_type)
    sign = _check_bc(bc_type)
    k = np.asarray(k, dtype=float)
    # Dirichlet: (+P + sqrt(P²+16π A k)) / (2A), all squared
    # Neumann:   (-P + sqrt(P²+16π A k)) / (2A), all squared
    # sign=-1 for dir (P term is positive in numerator)
    # sign=+1 for neu (P term is negative in numerator)
    num_sign = -sign  # flip: dir -> +P, neu -> -P
    return ((num_sign * perim + np.sqrt(perim**2 + 16 * np.pi * area * k)) / (2 * area)) ** 2


def _corner_correction(angles):
    """Polygon corner correction: Σ (π² - α²) / (24π α)."""
    angles = np.asarray(angles, dtype=float)
    return np.sum((np.pi**2 - angles**2) / (24 * np.pi * angles))


def weyl_count_poly(lam, domain=None, *, area=None, perim=None, angles=None, bc_type='dir'):
    """Three-term Weyl counting function for polygonal domains.

    N(λ) ≈ (A·λ ∓ P·√λ) / (4π)  +  Σ_k (π² − α_k²) / (24π·α_k)

    Parameters
    ----------
    lam : float or ndarray
    domain : Domain or Polygon, optional
        If a Polygon instance, int_angles is extracted automatically.
        If a generic Domain, `angles` must be supplied explicitly.
    area, perim : float, keyword-only
    angles : array-like, keyword-only
        Interior angles in radians. Required when domain is not a Polygon.
    bc_type : {'dir', 'neu'}, keyword-only
    """
    from .geometry import Polygon as LappyPolygon

    if domain is not None:
        area = domain.area
        perim = domain.perimeter
        bc_type = domain.bc_type
        if isinstance(domain, LappyPolygon):
            angles = domain.int_angles
        elif angles is None:
            raise TypeError(
                "domain is not a Polygon; supply angles= explicitly"
            )
    else:
        if area is None or perim is None:
            raise ValueError("provide either a Domain or both area= and perim=")
        if angles is None:
            raise ValueError("provide angles= when no domain is given")

    sign = _check_bc(bc_type)
    lam = np.asarray(lam, dtype=float)
    two_term = (area * lam + sign * perim * np.sqrt(lam)) / (4 * np.pi)
    return two_term + _corner_correction(angles)


def weyl_est_poly(k, domain=None, *, area=None, perim=None, angles=None, bc_type='dir'):
    """Numerically inverted three-term Weyl estimate for the k-th eigenvalue.

    Uses scipy.optimize.brentq, seeded by weyl_est for the bracket.

    Parameters
    ----------
    k : int or float
        Eigenvalue index.
    domain : Domain or Polygon, optional
    area, perim : float, keyword-only
    angles : array-like, keyword-only
    bc_type : {'dir', 'neu'}, keyword-only
    """
    from .geometry import Polygon as LappyPolygon

    # resolve scalars once
    if domain is not None:
        _area = domain.area
        _perim = domain.perimeter
        _bc_type = domain.bc_type
        if isinstance(domain, LappyPolygon):
            _angles = domain.int_angles
        elif angles is None:
            raise TypeError("domain is not a Polygon; supply angles= explicitly")
        else:
            _angles = angles
    else:
        if area is None or perim is None:
            raise ValueError("provide either a Domain or both area= and perim=")
        if angles is None:
            raise ValueError("provide angles= when no domain is given")
        _area, _perim, _bc_type, _angles = area, perim, bc_type, angles

    def _f(lam, k_target):
        return weyl_count_poly(lam, area=_area, perim=_perim, angles=_angles, bc_type=_bc_type) - k_target

    def _solve_one(k_scalar):
        # seed bracket from two-term estimate
        lam0 = weyl_est(k_scalar, area=_area, perim=_perim, bc_type=_bc_type)
        # bracket: search around lam0 by factor of 4
        a = lam0 / 4.0
        b = lam0 * 4.0
        # make sure bracket is valid
        fa, fb = _f(a, k_scalar), _f(b, k_scalar)
        # widen if needed
        for _ in range(20):
            if fa * fb < 0:
                break
            a /= 2.0
            b *= 2.0
            fa, fb = _f(a, k_scalar), _f(b, k_scalar)
        return scipy.optimize.brentq(_f, a, b, args=(k_scalar,))

    k = np.asarray(k, dtype=float)
    scalar = k.ndim == 0
    k_flat = np.atleast_1d(k).ravel()
    result = np.array([_solve_one(ki) for ki in k_flat])
    return float(result[0]) if scalar else result.reshape(k.shape)
