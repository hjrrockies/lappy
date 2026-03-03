"""lappy.exact — closed-form Laplacian eigenvalue formulas for special geometries."""

import numpy as np


def rect_eig(m, n, L, H):
    """Exact eigenvalue λ_{m,n} = π²m²/L² + π²n²/H² for an L×H rectangle.

    Pure formula; no bc_type. Caller is responsible for valid indices:
    m, n ≥ 1 for Dirichlet; m, n ≥ 0 for Neumann.
    """
    return m**2 * np.pi**2 / L**2 + n**2 * np.pi**2 / H**2


def rect_eigs_k(k, L, H, bc_type='dir', ret_mn=False):
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
