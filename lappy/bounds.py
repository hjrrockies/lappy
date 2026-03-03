"""lappy.bounds — rigorous Laplacian eigenvalue inequalities."""

import numpy as np
from scipy.special import jn_zeros

# Bessel zeros computed once at module level
_j01 = jn_zeros(0, 1)[0]   # first zero of J_0  ≈ 2.4048
_j11 = jn_zeros(1, 1)[0]   # first zero of J_1  ≈ 3.8317


def _get_area(domain, area):
    """Extract area from domain or keyword, with validation."""
    if domain is not None:
        return domain.area
    if area is None:
        raise ValueError("provide either a Domain or area=")
    return area


def _validate_bc(domain, expected):
    """Raise ValueError if domain.bc_type != expected."""
    if domain is not None and domain.bc_type != expected:
        raise ValueError(
            f"This bound requires bc_type='{expected}', "
            f"but domain has bc_type='{domain.bc_type}'"
        )


def faber_krahn(domain=None, *, area=None):
    """Dirichlet lower bound on λ₁: λ₁ ≥ π·j₀₁²/|Ω|.

    Equality iff Ω is a disk (Faber–Krahn inequality).

    Parameters
    ----------
    domain : Domain, optional
        Must have bc_type='dir'. Area is extracted automatically.
    area : float, keyword-only
        Used when domain is None.
    """
    _validate_bc(domain, 'dir')
    area = _get_area(domain, area)
    return np.pi * _j01**2 / area


def szego_weinberger(domain=None, *, area=None):
    """Neumann upper bound on μ₁: μ₁ ≤ π·j₁₁²/|Ω|.

    Equality iff Ω is a disk (Szegő–Weinberger inequality).

    Parameters
    ----------
    domain : Domain, optional
        Must have bc_type='neu'. Area is extracted automatically.
    area : float, keyword-only
        Used when domain is None.
    """
    _validate_bc(domain, 'neu')
    area = _get_area(domain, area)
    return np.pi * _j11**2 / area


def payne_polya_weinberger(domain=None):
    """Dirichlet ratio bound: λ₂/λ₁ ≤ (j₁₁/j₀₁)² ≈ 2.539.

    This is a pure constant (Payne–Pólya–Weinberger conjecture, proved by Ashbaugh–Benguria).
    `domain` is optional and used only for BC validation.

    Parameters
    ----------
    domain : Domain, optional
        If provided, must have bc_type='dir'.
    """
    _validate_bc(domain, 'dir')
    return (_j11 / _j01) ** 2


def inradius_upper(domain=None, *, inradius=None):
    """Dirichlet upper bound on λ₁ from inscribed disk: λ₁ ≤ j₀₁²/R².

    Because Ω ⊃ disk(R), domain monotonicity gives this upper bound.
    Equality iff Ω is a disk.

    Parameters
    ----------
    domain : Domain, optional
        Must have bc_type='dir'. inradius is extracted automatically.
    inradius : float, keyword-only
        Radius of the largest inscribed disk. Used when domain is None.
    """
    _validate_bc(domain, 'dir')
    if domain is not None:
        inradius = domain.inradius
    if inradius is None:
        raise ValueError("provide either a Domain or inradius=")
    return _j01**2 / inradius**2


def polya(k, domain=None, *, area=None):
    """Pólya lower bound on λ_k: λ_k ≥ 4π·k/|Ω| (Dirichlet only).

    Proven for tiling domains; conjectured in general.

    Parameters
    ----------
    k : int or ndarray
        Eigenvalue index (vectorized).
    domain : Domain, optional
        Must have bc_type='dir'. Area extracted automatically.
    area : float, keyword-only
        Used when domain is None.
    """
    _validate_bc(domain, 'dir')
    area = _get_area(domain, area)
    return 4 * np.pi * np.asarray(k) / area
