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
    """n-th positive zero of J_nu, to full double precision.

    Computed in extended precision with ``mpmath.besseljzero`` and rounded
    once. These values define the *reference* spectra for sector and disk
    domains, so they must be more accurate than anything checked against
    them -- otherwise a reference error is misread as a solver error.

    That is not hypothetical. The previous implementation scanned for a sign
    change and refined with ``scipy.optimize.brentq`` at its default
    tolerance, which is fine at integer order but loses 2-3 digits at
    fractional order, where ``scipy.special.jv`` is itself less accurate:

        nu = 4/3   |J_nu(z)| = 1.1e-13 at the returned zero (should be ~1e-16)
        nu = 1.512 |J_nu(z)| = 6.5e-14
        nu = 2, 4, 6                    ~1e-16, i.e. correct

    Fractional orders are exactly what the sector domains need
    (``nu = m*pi/alpha``), and the resulting eigenvalues were off by up to
    1.4e-13 relative. That made `disk_sector` solves look like they had
    *violated* their own Moler--Payne bound -- certified 13.7 digits against a
    "true" error of 12.9 -- when the solver was right and the reference was
    wrong. mpmath brings the residuals back to ~1e-16 at every order.
    """
    import mpmath as mp
    with mp.workdps(40):
        return float(mp.besseljzero(mp.mpf(float(nu)), int(n)))


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
    """The first 25 Dirichlet eigenvalues of the GWW isospectral domains, accurate to 12 digits, in sorted order.
    Cross-checked (see benchmarks/reference/gww.py, TUNING_LOG.md) via
    independent MPS solves of GWW1 and GWW2 separately at n_basis=320:
    GWW1 mostly reaches 9.5-9.9 digits (2 outliers at 6.1, 3.9 digits),
    GWW2 mostly reaches 7.2-8.7 digits (1 mode at 13.2 digits) -- both
    agree with this table to within their respective precision."""
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
    """The first 25 eigenvalues of the L-shaped domain, accurate to at least
    14 digits. Cross-checked (see benchmarks/reference/L_shape.py,
    TUNING_LOG.md) via an independent MPS solve at n_basis=240 reaching
    12.9-13.3 digits for the first 10 -- agrees with this table to ~1e-13,
    within that precision."""
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

def ellipse_eigs(k, a=2.0, b=1.0):
    """The first 10 Dirichlet eigenvalues of the ellipse with semi-axes a,b
    (computed via benchmarks/reference/ellipse.py, MPS with a boundary
    fundamental-solution basis -- no closed form). a=2,b=1: 13.3-14.4
    digits (n_basis=240, re-verified with the corrected pipeline -- see
    TUNING_LOG.md). a=3,4 (b=1) were NOT re-verified with the corrected
    pipeline (a re-check run was killed after 5+ minutes without
    finishing, unusually slow compared to a=2's ~1-2 minutes) -- their
    values are from the earlier, less careful pass: accurate to at least
    7 digits, worst case ~6.7 digits at a=3."""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    if a == 2.0 and b == 1.0:
        return np.array([ 3.566726599853406,  6.275430620157517, 10.028401620452271,
                          11.736665434133736, 14.877304135375340, 15.923963976853598,
                          20.846866237831982, 21.023029232908417, 24.885731654879255,
                          27.080637209915011])[:k]
    elif a == 3.0 and b == 1.0:
        return np.array([ 3.108128700342726,  4.595055082905808,  6.509341614142352,
                           8.881928158123285, 11.030611158910741, 11.734242285459748,
                          13.539805518163716, 15.080890472747482, 16.430451193912770,
                          18.931804562846718])[:k]
    elif a == 4.0 and b == 1.0:
        return np.array([ 2.920251003362351,  3.934862295633269,  5.175723543179423,
                           6.659014663915840,  8.397619666159422, 10.401610755766617,
                          10.714369529583474, 12.505571121120370, 12.678776996497007,
                          14.503899387838459])[:k]
    else:
        raise ValueError("eigenvalues not available for this a,b pair")

def reg_ngon_eigs(k, N):
    """The first 10 Dirichlet eigenvalues (counting multiplicity) of a
    regular N-gon with unit circumradius, N=5..8 (computed via
    benchmarks/reference/reg_ngon.py, MPS with a mixed Fourier-Bessel +
    fundamental-solution basis -- every corner is singular for these N).
    Accurate to at least 10.2 digits for N=5,6,7; for N=8, 8 of 9 unique
    eigenvalues are accurate to at least 8.8 digits, but one
    (lambda=29.536810903327243, a near-neighbor of the doubled eigenvalue
    at 29.540564727244991) is only accurate to ~2.8 digits -- see
    reg_ngon.py and TUNING_LOG.md for the diagnostics ruling out
    regularization/collocation as the cause (it's a genuinely under-
    resolved mode at n_basis=120)."""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    if N == 5:
        return np.array([  7.957089389349433,  20.106281644908393,  20.106281644908393,
                           35.654694217826744,  35.654694217826744,  41.313935812799116,
                           55.700798288206641,  55.700798288206641,  64.521882950899112,
                           64.521882950899112])[:k]
    elif N == 6:
        return np.array([  7.155339133926017,  18.131677865530712,  18.131677865530712,
                           32.451857514400388,  32.451857514400388,  37.491352876797706,
                           47.629365773857295,  52.637890139143394,  60.105112094166572,
                           60.105112094166572])[:k]
    elif N == 7:
        return np.array([  6.735099196687083,  17.083807204748648,  17.083807204748648,
                           30.643654921220552,  30.643654921220552,  35.397032810525708,
                           47.100728432751367,  47.100728432751367,  57.017683247909424,
                           57.017683247909424])[:k]
    elif N == 8:
        return np.array([  6.484933493724240,  16.456119030336588,  16.456119030336588,
                           29.536810903327243,  29.540564727244991,  29.540564727244991,
                           34.124475285748161,  45.529751487885697,  45.529751487885697,
                           55.049743550878873])[:k]
    else:
        raise ValueError("eigenvalues not available for this N")

def cut_square_eigs(k, r=0.25):
    """The first 10 Dirichlet eigenvalues of a unit square with one corner
    cut by a circular arc of radius r (computed via
    benchmarks/reference/cut_square.py, MPS with a pure Fourier-Bessel basis
    -- all remaining corners are regular right angles). Accurate to at
    least 6.4 digits for r=0.25 (n_basis=320); accurate to at least 9.1
    digits for r=0.5 (n_basis=320, close to the 10+ digit target)."""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    if r == 0.25:
        return np.array([ 20.585592337197024,  49.422034168078973,  53.607796386908106,
                           84.371488667211239,  99.437928066752903, 104.562833789125236,
                          129.187535527364219, 146.994370017546117, 170.060990914169764,
                          170.553669971897790])[:k]
    elif r == 0.5:
        return np.array([ 28.081015645274604,  55.146340053352930,  72.407039289994984,
                          100.899496342046007, 115.233911594759078, 139.429979543240051,
                          153.042769322697268, 182.635738771611955, 191.634273757009765,
                          202.708585647789022])[:k]
    else:
        raise ValueError("eigenvalues not available for this r")

def mushroom_eigs(k, a=1.0, b=1.0, r=1.5):
    """The first 10 Dirichlet eigenvalues of mushroom(a,b,r) -- a half-disk
    cap on a rectangular stem, with two 270-degree reentrant corners at the
    junction (computed via benchmarks/reference/mushroom.py, MPS with a
    mixed Fourier-Bessel + fundamental-solution basis). Accurate to at
    least 11.3 digits."""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    if a == 1.0 and b == 1.0 and r == 1.5:
        return np.array([  5.497868889097452,  11.507908981960103,  13.363962538819910,
                           18.067786790669235,  20.805793683510373,  25.550152545706617,
                           29.124676104888923,  32.589926048677256,  34.194889649910934,
                           41.911982648353757])[:k]
    else:
        raise ValueError("eigenvalues not available for this a,b,r triple")

def H_shape_eigs(k):
    """The first 10 Dirichlet eigenvalues of the fixed 12-vertex H-shaped
    domain (4 reentrant 270-degree corners; computed via
    benchmarks/reference/H_shape.py, MPS with a mixed Fourier-Bessel +
    fundamental-solution basis). Accurate to at least 7.8 digits
    (n_basis=320), except index 6 (0-indexed), lambda=19.739208802178766
    (suspiciously close to 2*pi^2 -- likely a mode that vanishes on the
    connecting web and is effectively an exact rectangle eigenvalue), which
    reaches 13.3 digits."""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    return np.array([  7.733088853276283,   8.551726848574045,  13.927633223859791,
                       13.931597881917137,  14.305229961336140,  17.706735223095627,
                       19.739208802178766,  24.788709817219047,  26.370982345700579,
                       26.423167326708011])[:k]

def chevron_eigs(k, h1=1.0, h2=2.0):
    """The first 10 eigenvalues of the chevron with heights h1,h2.

    h1=1,h2=2 is accurate to 12 digits (an earlier, independently-derived
    table). The other 3 pairs (computed via benchmarks/reference/chevron.py
    at n_basis=160) are much less precise -- only ~5-7 digits for
    h1=1,h2=1.5/2.0, and ~3-4 digits for h1=2,h2=3/4 -- because their
    sharpest corner (as small as ~11 degrees) genuinely needs much more
    basis than is practical to solve quickly at this Fourier-Bessel order
    (see that script's module docstring and TUNING_LOG.md for the several
    approaches tried and ruled out: corner-order reweighting, denser
    collocation, and moderate basis increases all confirmed this is a slow-
    convergence resolution limit, not a quick fix; h1=1,h2=1.25 was
    excluded outright as too hard, similar to why geometry.spiral() is
    excluded from the benchmark set)."""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")

    if h1 == 1 and h2 == 2:
        return np.array([ 39.66587536762846,  77.66316267381548,  81.88608149069968,
                        111.42970385691103, 120.59489370950362, 152.06601346806502,
                        161.16007983417921, 179.80395817996902, 204.7047973867004 ,
                        205.98199724200455])[:k]
    elif h1 == 1 and h2 == 1.5:
        return np.array([113.734766669427927, 189.448519906295729, 214.171972707017147,
                          272.053387695407821, 283.244291179057882, 348.908624511603080,
                          355.061155405508714, 428.693725570045387, 438.828078645126823,
                          518.577130919274964])[:k]
    elif h1 == 2 and h2 == 3:
        return np.array([ 64.708985605874176, 121.539374925198445, 130.253934421114678,
                          155.535111621981400, 173.830602900292064, 194.130498772824097,
                          214.340324596339258, 226.620355923715977, 253.734091427685115,
                          263.032759596165249])[:k]
    elif h1 == 2 and h2 == 4:
        return np.array([ 24.427905596255293,  43.891947613356443,  56.871718718195481,
                           63.509810642063812,  77.629441744211917,  79.004363501607074,
                           96.362085037989488,  98.349053725219605, 112.019515962249514,
                          139.714706538389208])[:k]
    else:
        raise ValueError("eigenvalues not available for this h1,h2 pair")

def iso_tri_eigs(k, h=1.0):
    """The first 10 eigenvalues of the isosceles triangle (base 2, height h),
    computed via MPS (benchmarks/reference/iso_tri.py). Accuracy varies by
    height (worst tension-implied digit count in each set): h=0.5 ~10.8
    digits, h=1.0 ~13.0 digits, h=2.0 ~12.0 digits, h=4.0 ~11.8 digits,
    h=8.0 ~11.3 digits (all comfortably at or above the 10-digit target,
    n_basis=120). h=16.0 and h=20.0 were not re-verified with the improved
    pipeline (h=16 got unusually slow, see iso_tri.py) -- their values are
    from an earlier, less careful solve: h=16 ~7.8 digits, h=20 ~12
    digits."""
    if k > 10:
        raise ValueError("Only the first 10 eigenvalues are available")
    if h==0.5:
        return np.array([ 67.349455544425652, 111.036404225384885, 151.473895577570715,
                          199.539129893408813, 221.691797716339380, 253.997396523095404,
                          298.689466549538338, 325.281558433280395, 357.434635325396926,
                          394.821748879388792])[:k]
    elif h==1.0:
        return np.array([ 24.674011002723478,  49.348022005446687,  64.152428607080637,
                           83.891637409259246,  98.696044010893758, 123.370055013616835,
                          128.304857214161217, 143.109263815796027, 167.783274818519516,
                          182.587681420152677])[:k]
    elif h==2.0:
        return np.array([ 11.456820359432427,  25.694945637505505,  27.759101056346598,
                           44.337951419766441,  49.884782473352665,  50.842728850066379,
                           68.247856785464919,  74.672366637382709,  81.112315003885087,
                           81.320389608320156])[:k]
    elif h==4.0:
        return np.array([  6.726526574153285,  12.333521955030589,  19.071767666600952,
                           19.074704193994812,  27.005508461745229,  29.098013503262695,
                           36.056833854046154,  37.246116430239411,  39.954321850759705,
                           46.498064800315710])[:k]
    elif h==8.0:
        return np.array([  4.719098710033834,   7.227058278292141,   9.950620918236773,
                           12.946772679734305,  15.010378933747058,  16.233055784524762,
                           19.816898996168046,  20.000479851124531,  23.701929385100062,
                           24.972774558019974])[:k]
    elif h==16.0:
        return np.array([  3.741313107211302,   4.988436354293714,   6.231679923092005,
                            7.517545607747248,   8.861653817098958,  10.271173428254006,
                           11.749965646537081,  12.884145973648904,  13.300328814508363,
                           14.923728877388257])[:k]
    elif h==20.0:
        return np.array([ 3.538204270133983,  4.552162970620473,  5.539932740921296,
                        6.544007125902493,  7.578949068369423,  8.651562362007823,
                        9.765571848342804, 10.923228519465912, 12.125988718163306,
                        12.4252704009599  ])[:k]
    else:
        raise ValueError("eigenvalues not available for this value of h")


# ── Closed-form eigenfunctions and exact L² norms ────────────────────────────────
# Each *_eigfun returns (u, norm2), where u(z) is a vectorized callable taking
# complex coordinates z = x + iy and norm2 = ∫_Ω |u|² dA is the exact squared L²
# norm. These are used to verify that a cubature rule reproduces eigenfunction L²
# norms (diagonal Gram entries) and orthogonality (off-diagonal entries) to the
# requested precision. Conventions match the corresponding *_eig functions and the
# geometry factory placement (e.g. rect() has a corner at the origin).

def rect_eigfun(m, n, L, H, bc='dir'):
    """Eigenfunction and exact squared L² norm for the L×H rectangle [0,L]×[0,H].

    Matches ``lappy.geometry.rect(L, H)`` (corner at the origin).

    Dirichlet (m, n ≥ 1): u = sin(mπx/L) sin(nπy/H).
    Neumann   (m, n ≥ 0): u = cos(mπx/L) cos(nπy/H).

    Returns
    -------
    (u, norm2) : callable, float
    """
    if bc == 'dir':
        if m < 1 or n < 1:
            raise ValueError("Dirichlet rectangle modes require m, n ≥ 1")
        def u(z):
            return np.sin(m*np.pi*np.real(z)/L) * np.sin(n*np.pi*np.imag(z)/H)
        norm2 = (L/2) * (H/2)
    elif bc == 'neu':
        if m < 0 or n < 0:
            raise ValueError("Neumann rectangle modes require m, n ≥ 0")
        def u(z):
            return np.cos(m*np.pi*np.real(z)/L) * np.cos(n*np.pi*np.imag(z)/H)
        # ∫cos²(mπx/L)dx = L/2 for m≥1, L for m=0
        norm2 = (L if m == 0 else L/2) * (H if n == 0 else H/2)
    else:
        raise ValueError(f"bc must be 'dir' or 'neu', got {bc!r}")
    return u, norm2


def iso_right_tri_eigfun(m, n, l):
    """Dirichlet eigenfunction and exact squared L² norm for the isosceles
    right triangle with legs ``l`` (vertices 0, l, i*l).

    Matches ``lappy.geometry.iso_right_tri(l)``.

    The triangle is the ``l``x``l`` square folded on its diagonal, so its modes
    are the square's antisymmetrized under ``(x,y) -> (y,x)``::

        u = sin(m pi x/l) sin(n pi y/l) - sin(n pi x/l) sin(m pi y/l),  m > n >= 1

    Norm: the two terms are orthogonal on the square and each integrates to
    ``l^2/4``, giving ``l^2/2`` there; ``u`` is antisymmetric about the
    diagonal, so the triangle carries exactly half, ``l^2/4``.

    Returns
    -------
    (u, norm2) : callable, float
    """
    if not (m > n >= 1):
        raise ValueError("isosceles right triangle modes require m > n >= 1")

    def u(z):
        x, y = np.real(z), np.imag(z)
        return (np.sin(m*np.pi*x/l) * np.sin(n*np.pi*y/l)
                - np.sin(n*np.pi*x/l) * np.sin(m*np.pi*y/l))

    return u, l**2 / 4


def normalized_eigfun(eigfun_result):
    """Turn a ``(u, norm2)`` pair into an L²(Omega)-orthonormal callable.

    The reference eigenfunctions all return their exact squared norm, so
    normalization is exact rather than quadrature-based. That matters for
    validating quadrature: an independently-normalized eigenfunction is a
    ground truth that ``interior_l2`` and the Rellich identity can be checked
    *against*, rather than something they define.
    """
    u, norm2 = eigfun_result
    scale = 1.0 / np.sqrt(norm2)
    return lambda z: scale * u(z)


def disk_eigfun(m, n, R, parity='cos'):
    """Dirichlet eigenfunction and exact squared L² norm for a disk of radius R.

    Matches ``lappy.geometry.disk(R)`` (centered at the origin).

    u = J_m(k r) · trig(m θ), with k = j_{m,n}/R, r = |z|, θ = arg(z).
    ``parity`` selects cos(mθ) or sin(mθ); for m = 0 only 'cos' is valid.

    Radial norm: ∫₀ᴿ J_m(k r)² r dr = (R²/2) J_{m+1}(j_{m,n})².
    Angular norm: ∫₀²π cos²(mθ)dθ = π (m≥1) or 2π (m=0); sin² = π (m≥1).

    Returns
    -------
    (u, norm2) : callable, float
    """
    if m < 0 or n < 1:
        raise ValueError("disk modes require m ≥ 0, n ≥ 1")
    if m == 0 and parity == 'sin':
        raise ValueError("parity='sin' is trivially zero for m=0")
    j_mn = jn_zeros(m, n)[-1]
    k = j_mn / R
    trig = np.cos if parity == 'cos' else np.sin
    def u(z):
        r = np.abs(z)
        theta = np.angle(z)
        return jv(m, k*r) * trig(m*theta)
    radial = (R**2 / 2) * jv(m+1, j_mn)**2
    angular = (2*np.pi if m == 0 else np.pi)
    return u, angular * radial


def sector_eigfun(m, n, R, alpha):
    """Dirichlet eigenfunction and exact squared L² norm for a circular sector.

    Matches ``lappy.geometry.disk_sector(R, alpha)`` (apex at the origin, angular
    extent [0, alpha]).

    u = J_ν(k r) sin(ν θ), with ν = mπ/alpha, k = j_{ν,n}/R, θ = arg(z) mod 2π.

    Radial norm: (R²/2) J_{ν+1}(j_{ν,n})²; angular norm: ∫₀^alpha sin²(νθ)dθ = alpha/2.

    Returns
    -------
    (u, norm2) : callable, float
    """
    if m < 1 or n < 1:
        raise ValueError("sector modes require m ≥ 1, n ≥ 1")
    nu = m * np.pi / alpha
    j_nu_n = _bessel_zero(nu, n)
    k = j_nu_n / R
    def u(z):
        r = np.abs(z)
        theta = np.mod(np.angle(z), 2*np.pi)
        return jv(nu, k*r) * np.sin(nu*theta)
    radial = (R**2 / 2) * jv(nu+1, j_nu_n)**2
    angular = alpha / 2
    return u, angular * radial

