"""lappy.reference — closed-form Laplacian eigenvalue formulas for special geometries."""

import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros
from scipy.optimize import brentq

# Rectangles
def rect_eig(m, n, L, H):
    """Exact eigenvalue λ_{m,n} = π²m²/L² + π²n²/H² for an L×H rectangle.

    Pure formula; no bc_type. Caller is responsible for valid indices:
    m, n ≥ 1 for Dirichlet; m, n ≥ 0 for Neumann.
    """
    return m**2 * np.pi**2 / L**2 + n**2 * np.pi**2 / H**2


def rect_eigs(k, L, H, bc_type='dir', ret_mn=False):
    """First k eigenvalues of an L×H rectangle, sorted ascending.

    Parameters
    ----------
    k : int
        Number of eigenvalues to return.
    L, H : float or ndarray
        Rectangle dimensions. Vectorized: output shape is (*L.shape, k).
    bc_type : {'dir', 'neu'}
        Boundary condition type. 'dir' starts indices at 1 (no zero eigenvalue);
        'neu' starts at 0 (includes zero eigenvalue for m=n=0).
    ret_mn : bool
        If True, also return (m_arr, n_arr) index arrays.
    """
    L, H = np.asarray(L), np.asarray(H)
    if bc_type == 'dir':
        start = 1
    elif bc_type == 'neu':
        start = 0
    else:
        raise ValueError(f"bc_type must be 'dir' or 'neu', got {bc_type!r}")

    mn = np.arange(start, k + start)
    M, N = np.meshgrid(mn, mn, indexing='ij')
    eigs = rect_eig(
        M.flatten()[np.newaxis],
        N.flatten()[np.newaxis],
        L.flatten()[:, np.newaxis],
        H.flatten()[:, np.newaxis],
    )

    idx = np.argsort(eigs, axis=-1)
    eigs = np.take_along_axis(eigs, idx, axis=-1)[:, :k]
    eigs = eigs.reshape((*L.shape, k))
    if ret_mn:
        m = np.take_along_axis(M.flatten()[np.newaxis], idx, axis=-1)[:, :k]
        n = np.take_along_axis(N.flatten()[np.newaxis], idx, axis=-1)[:, :k]
        return eigs, m.reshape((*L.shape, k)), n.reshape((*L.shape, k))
    else:
        return eigs


def rect_eig_grad(m, n, L, H):
    """Derivatives of λ_{m,n} with respect to L and H.

    Returns (dλ/dL, dλ/dH). BC-independent (formula is the same for any bc_type).
    """
    m, n = np.asarray(m), np.asarray(n)
    L, H = np.asarray(L), np.asarray(H)
    return (-2 * (np.pi * m.T) ** 2 / L**3).T, (-2 * (np.pi * n.T) ** 2 / H**3).T


def rect_eig_bound_idx(bound, L, H, bc_type='dir'):
    """Indices (m, n) of all eigenvalues ≤ bound for an L×H rectangle.

    Parameters
    ----------
    bound : float
        Upper bound on eigenvalue.
    L, H : float
        Rectangle dimensions.
    bc_type : {'dir', 'neu'}
        Sets the starting index (1 for Dirichlet, 0 for Neumann).
    """
    if bc_type == 'dir':
        start = 1
    elif bc_type == 'neu':
        start = 0
    else:
        raise ValueError(f"bc_type must be 'dir' or 'neu', got {bc_type!r}")

    m_max = start
    while True:
        eig = rect_eig(m_max, start, L, H)
        if eig > bound:
            break
        m_max += 1

    n_max = start
    while True:
        eig = rect_eig(start, n_max, L, H)
        if eig > bound:
            break
        n_max += 1

    M = np.arange(start, m_max + 1)[:, np.newaxis]
    N = np.arange(start, n_max + 1)[np.newaxis]
    Lambda = rect_eig(M, N, L, H)
    return np.argwhere(Lambda <= bound) + start


def rect_eig_mult(lambda_, L, H, bc_type='dir', maxind=1000):
    """Find all index pairs (m, n) whose eigenvalue matches lambda_.

    For multiplicity analysis. Returns (m_arr, n_arr).

    Parameters
    ----------
    lambda_ : float
        Target eigenvalue.
    L, H : float
        Rectangle dimensions.
    bc_type : {'dir', 'neu'}
        Sets the starting index (1 for Dirichlet, 0 for Neumann).
    maxind : int
        Maximum index to search up to.
    """
    if bc_type == 'dir':
        start = 1
    elif bc_type == 'neu':
        start = 0
    else:
        raise ValueError(f"bc_type must be 'dir' or 'neu', got {bc_type!r}")

    idx = np.arange(start, maxind + start)
    Lam = rect_eig(idx[np.newaxis], idx[:, np.newaxis], L, H)
    diff = np.abs(lambda_ - Lam)
    tot = (diff < 1e-12).sum()
    ind = np.unravel_index(np.argsort(diff, axis=None), diff.shape)
    return (ind[0] + start)[:tot], (ind[1] + start)[:tot]


def rect_eig_mult_mn(m, n, L, H, bc_type='dir'):
    """Find all index pairs duplicating the (m, n) eigenvalue.

    Convenience wrapper around rect_eig_mult.
    """
    return rect_eig_mult(
        rect_eig(m, n, L, H), L, H, bc_type=bc_type, maxind=10 * max(m, n)
    )


# ── Shared infrastructure ─────────────────────────────────────────────────────

def _take_k_from_grid(eig_fn, m_vals, n_vals, k):
    """Sort eigenvalues from an (m, n) index grid, filter Nones, return first k."""
    raw = [eig_fn(int(m), int(n)) for m in m_vals for n in n_vals]
    vals = np.array([v for v in raw if v is not None], dtype=float)
    if len(vals) < k:
        raise ValueError(
            f"Grid produced only {len(vals)} eigenvalues; need {k}. "
            "Increase index range."
        )
    return np.sort(vals)[:k]


def _bessel_zero(nu, n):
    """n-th positive zero of J_nu, located by scanning for sign changes.

    Scans J_nu on a coarse grid (step π/4) to find the n-th sign change,
    then refines with brentq. Correct for all nu ≥ 0, n ≥ 1.
    """
    x_max = (n + nu / 2 + 3) * np.pi
    step = np.pi / 4
    xs = np.arange(0.5, x_max + step, step)
    ys = jv(nu, xs)
    crossings = np.where(ys[:-1] * ys[1:] < 0)[0]
    if len(crossings) < n:
        raise ValueError(f"Could not find {n}-th zero of J_{nu}")
    idx = crossings[n - 1]
    return brentq(lambda x: jv(nu, x), xs[idx], xs[idx + 1])


# ── Isosceles right triangle (legs a, Dirichlet) ──────────────────────────────

def iso_right_tri_eig(m, n, l):
    """Exact eigenvalue for an isosceles right triangle with legs l.

    Valid only for m > n ≥ 1 (strict inequality avoids the trivially zero
    antisymmetric combination). Returns None for invalid indices.

    Formula: λ_{m,n} = π²(m² + n²) / l²
    """
    if n < 1 or m <= n:
        return None
    return np.pi**2 * (m**2 + n**2) / l**2


def iso_right_tri_eigs(k, l):
    """First k Dirichlet eigenvalues of an isosceles right triangle with legs l.

    Parameters
    ----------
    k : int
        Number of eigenvalues to return.
    l : float
        Leg length.
    """
    max_idx = 10*k
    m_vals = range(2, max_idx + 1)   # m ≥ 2 so that m > n ≥ 1 is possible
    n_vals = range(1, max_idx)
    return _take_k_from_grid(lambda m, n: iso_right_tri_eig(m, n, l), m_vals, n_vals, k)


# ── Equilateral triangle (side length a, Dirichlet) ──────────────────────────────

def eq_tri_eig(m, n, l=1):
    """Eigenvalues of the equilateral triangle with side length l"""
    if (m < 0) or (n < 1):
        return None
    elif m >= n:
        return None
    elif (m-n)%2 != 0:
        return None
    return (4/3)*np.pi**2*((m**2)/3 + n**2)/l**2

def eq_tri_eigs(k, l=1):
    max_idx = 10*k
    # for m,n > 0, each eigenvalue has multiplicity 2
    m_vals = range(1,max_idx+1)
    n_vals = range(1,max_idx+1)
    eigs1 = _take_k_from_grid(lambda m,n: eq_tri_eig(m, n, l), m_vals, n_vals, k)
    # for m=0, each eigenvalue has multiplicity 1
    eigs2 = _take_k_from_grid(lambda m,n: eq_tri_eig(m, n, l), [0], n_vals, k)
    eigs = np.concatenate((eigs1, eigs1, eigs2))
    return np.sort(eigs)[:k]
    
# ── Circular sector (radius R, opening angle alpha, Dirichlet) ────────────────

def sector_eig(m, n, R, alpha):
    """Exact eigenvalue for a circular sector of radius R and opening angle alpha.

    Indices: m ≥ 1 (angular mode), n ≥ 1 (radial mode). All modes have
    multiplicity 1 (angular factor is sin(mπθ/alpha)).

    Formula: λ_{m,n} = (j_{mπ/alpha, n} / R)²
    where j_{nu, n} is the n-th positive zero of J_nu.
    """
    nu = m * np.pi / alpha
    return _bessel_zero(nu, n) ** 2 / R**2


def sector_eigs(k, R, alpha):
    """First k Dirichlet eigenvalues of a circular sector.

    Parameters
    ----------
    k : int
        Number of eigenvalues to return.
    R : float
        Radius.
    alpha : float
        Opening angle in radians (0 < alpha ≤ 2π).
    """
    # Area = alpha*R²/2 → Weyl: λ_k ≈ 4πk/(alpha*R²)
    # j_{nu,n} ≈ (n + nu/2)*π → rough bound: n + m*π/(2*alpha) ≲ sqrt(λ_k)*R/π
    max_idx = int(np.ceil(2 * np.sqrt(k) + 4))
    m_vals = range(1, max_idx + 1)
    n_vals = range(1, max_idx + 1)
    return _take_k_from_grid(lambda m, n: sector_eig(m, n, R, alpha), m_vals, n_vals, k)

# ── Disk (radius r) ───────────────────────────────────────────────────────────

def disk_eig(m, n, r, bc_type='dir'):
    """Exact eigenvalue λ_{m,n} for a disk of radius r.

    Parameters
    ----------
    m : int
        Angular order (≥ 0). Multiplicity is 1 for m=0, 2 for m≥1.
    n : int
        Radial index (≥ 1).
    r : float
        Disk radius.
    bc_type : {'dir', 'neu'}
        Dirichlet: λ = (j_{m,n} / r)², where j_{m,n} is the n-th positive zero of J_m.
        Neumann:   λ = (j'_{m,n} / r)², where j'_{m,n} is the n-th positive zero of J_m'.
                   The zero eigenvalue (constant mode) is not returned by this function.
    """
    if bc_type == 'dir':
        return jn_zeros(m, n)[-1] ** 2 / r**2
    elif bc_type == 'neu':
        return jnp_zeros(m, n)[-1] ** 2 / r**2
    else:
        raise ValueError(f"bc_type must be 'dir' or 'neu', got {bc_type!r}")


def disk_eigs(k, r, bc_type='dir'):
    """First k eigenvalues of a disk of radius r, sorted ascending (with multiplicity).

    Parameters
    ----------
    k : int
        Number of eigenvalues to return.
    r : float
        Disk radius.
    bc_type : {'dir', 'neu'}
        Dirichlet: λ_{m,n} = (j_{m,n}/r)², m≥0, n≥1. Multiplicity 2 for m≥1.
        Neumann:   λ_{m,n} = (j'_{m,n}/r)², m≥0, n≥1. Multiplicity 2 for m≥1.
                   The zero eigenvalue (m=n=0 constant mode) is prepended automatically.
    """
    if bc_type not in ('dir', 'neu'):
        raise ValueError(f"bc_type must be 'dir' or 'neu', got {bc_type!r}")

    # Weyl estimate: λ_k ≈ 4k/r²; j_{m,n} ≈ (n + m/2)*π.
    # Over-estimate the required index range to be safe.
    max_m = int(np.ceil(2 * np.sqrt(k) + 6))
    max_n = int(np.ceil(np.sqrt(k) + 6))

    eigs = [] if bc_type == 'dir' else [0.0]  # Neumann includes λ=0
    zero_fn = jn_zeros if bc_type == 'dir' else jnp_zeros
    for m in range(0, max_m + 1):
        zeros = zero_fn(m, max_n)
        mult = 1 if m == 0 else 2
        eigs.extend((z / r) ** 2 for z in zeros for _ in range(mult))

    eigs = np.sort(eigs)
    if len(eigs) < k:
        raise ValueError(
            f"Grid produced only {len(eigs)} eigenvalues; need {k}. "
            "Increase index range."
        )
    return eigs[:k]


# ── Domains without closed-form eigenvalues ─────────────────────────────────────────────────────
def gww_eigs(k):
    """The first 25 Dirichlet eigenvalues of the GWW isospectral domains, accurate to 12 digits, in sorted order"""
    if k > 25:
        raise ValueError("Only the first 25 eigenvalues are available")
    # just letting numpy sort because Driscoll's table wasn't easy to copy and paste!
    eigs = np.sort([2.53794399980, 9.20929499840, 14.3138624643, 20.8823950433, 24.6740110027, 
                    3.65550971352, 10.5969856913, 15.871302620, 21.2480051774, 26.0802400997, 
                    5.17555935622, 11.5413953956, 16.9417516880, 22.2328517930, 27.3040189211, 
                    6.53755744376, 12.3370055014, 17.6651184368, 23.7112974848, 28.1751285815, 
                    7.24807786256, 13.0536540557, 18.9810673877, 24.4792340693, 29.5697729132])
    return eigs[:k]

def L_shape_eigs(k):
    """The first 25 eigenvalues of the L-shaped domain, accurate to at least 14 digits"""
    if k > 25:
        raise ValueError("Only the first 25 eigenvalues are available")
    return np.array([  9.639723844021946,  15.197251926454308,  19.739208802178716,
                      29.521481114144848,  31.912635957137752,  41.47450989021491,
                      44.94848778135119,   49.34802200544678,   49.34802200544678,
                      56.70960988738507,   65.37653570984583,   71.05775564851349,
                      71.57267968033655,   78.95683520871486,   89.30166835196012,
                      92.3069067630492,    97.38072264602184,   98.69604401089357,
                      98.69604401089357,  101.60529408377867,  112.36860922562566,
                     115.5201730946677,   128.30485721416164,  128.30485721416164,
                     130.11902885096785])[:k]

def chevron_eigs(k, h1=1.0, h2=2.0):
    """The first 10 eigenvalues of the chevron with heights h1,h2, accurate to 12 digits"""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    
    if h1 == 1 and h2 == 2:
        return np.array([ 39.66587536762846,  77.66316267381548,  81.88608149069968,
                        111.42970385691103, 120.59489370950362, 152.06601346806502,
                        161.16007983417921, 179.80395817996902, 204.7047973867004 ,
                        205.98199724200455])[:k]
    else:
        raise ValueError("eigenvalues not available for this h1,h2 pair")

def iso_tri_eigs(k, h=1.0):
    """The first 10 eigenvalues of the isosceles triangle, accurate to 12 digits"""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    if h==20.0:
        return np.array([ 3.538204270133983,  4.552162970620473,  5.539932740921296,
                        6.544007125902493,  7.578949068369423,  8.651562362007823,
                        9.765571848342804, 10.923228519465912, 12.125988718163306,
                        12.4252704009599  ])[:k]
    else:
        raise ValueError("eigenvalues not available for this value of h")

