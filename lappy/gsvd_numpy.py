"""
gsvd_numpy.py — GSVD via QR + SVD (pure NumPy / SciPy, no rank truncation).

Algorithm
---------
1. Thin QR of [A; B]:  M = Q_qr @ R_qr   (Q_qr: (m+n)×p, R_qr: p×p upper-tri)
2. SVD of Q1 = Q_qr[:m, :]:  Q1 = U @ diag(c) @ Vh   (c values in [0, 1])
3. V from Q2_W = Q2 @ Vh.conj().T:  columns are orthogonal with norms s = sqrt(1-c²)
   (This follows from Q_qr^H Q_qr = I_p, not from a CSD algorithm.)
4. X = R_qr.conj().T @ Vh.conj().T   (the shared right factor)

Note on the "CS decomposition" label
-------------------------------------
The CS decomposition proper is a joint algorithm applied to a square unitary
matrix; see scipy.linalg.cossin / LAPACK DORCSD.  This module does NOT call
that algorithm.  Instead it exploits the fact that for a matrix [Q1; Q2] with
orthonormal columns, Q1 and Q2 share the same right singular vectors (Vh),
which can be recovered from the SVD of Q1 alone.  The result is mathematically
equivalent to the CSD of a completed square unitary Q_full, but the derivation
is "SVD of one block + normalization of the other."

Unlike gsvd4py, no rank truncation is applied: k=0 and l=p always (all p pairs
are treated as finite generalized singular values).  This is intentional — the
solver is designed for diagnostic use when you want to see the full spectrum
without thresholding.

Matches the gsvd() / gsvdvals() signatures of gsvd4py, with these differences:
  - No tola/tolb/lwork/overwrite_* parameters.
  - k is always 0, l is always p.
  - Requires m >= p and n >= p (the typical MPS setting).

References
----------
Paige & Saunders (1981), "Towards a Generalized Singular Value Decomposition".
"""

import numpy as np
from scipy.linalg import qr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gsvd(a, b, mode='full', compute_u=True, compute_v=True, compute_right=True,
         check_finite=True):
    """Generalized SVD via CS decomposition (pure Python + NumPy/SciPy).

    Parameters
    ----------
    a : (m, p) array_like
        First matrix.  Must satisfy m >= p.
    b : (n, p) array_like
        Second matrix.  Must satisfy n >= p.
    mode : {'full', 'econ', 'separate'}, default 'full'
        'full'     — U (m×m), V (n×n), C (m×p), S (n×p), X (p×p).
        'econ'     — U (m×p), V (n×p), C (p×p), S (p×p), X (p×p).
        'separate' — U, V, D1 (m×p), D2 (n×p), R (p×p), Q (p×p), k=0, l=p.
    compute_u : bool
        Compute left singular vectors of a (U).
    compute_v : bool
        Compute left singular vectors of b (V).
    compute_right : bool
        Compute the shared right factor X (or Q in separate mode).
    check_finite : bool
        Validate that a and b contain only finite values.

    Returns
    -------
    mode='full' / 'econ' (compute_u=compute_v=compute_right=True):
        U, V, C, S, X
    mode='separate':
        U, V, D1, D2, R, Q, k, l
    Omitted factors (compute_* = False) are dropped from the front/end of the
    tuple in the same order as gsvd4py.

    Notes
    -----
    C diagonal is in non-increasing order; S diagonal in non-decreasing order,
    matching the scipy / gsvd4py convention.
    k = 0 and l = p always (no rank truncation).
    """
    if mode not in ('full', 'econ', 'separate'):
        raise ValueError(f"mode must be 'full', 'econ', or 'separate', got {mode!r}")

    a = np.asarray(a)
    b = np.asarray(b)

    # dtype promotion (mirror gsvd4py: integers → float64, keep float32/complex)
    dtype = np.result_type(a, b, np.float64)
    if not (np.issubdtype(dtype, np.floating) or
            np.issubdtype(dtype, np.complexfloating)):
        dtype = np.float64
    a = np.array(a, dtype=dtype)
    b = np.array(b, dtype=dtype)

    if a.ndim != 2:
        raise ValueError(f"a must be 2-D, got shape {a.shape}")
    if b.ndim != 2:
        raise ValueError(f"b must be 2-D, got shape {b.shape}")

    m, p = a.shape
    n = b.shape[0]
    if b.shape[1] != p:
        raise ValueError(
            f"a and b must have the same number of columns: {p} != {b.shape[1]}"
        )
    if m < p:
        raise ValueError(
            f"gsvd_cs requires m >= p (rows of a >= cols); got m={m}, p={p}. "
            "Increase the number of boundary points or decrease basis size."
        )
    if n < p:
        raise ValueError(
            f"gsvd_cs requires n >= p (rows of b >= cols); got n={n}, p={p}. "
            "Increase the number of interior points or decrease basis size."
        )

    if check_finite:
        if not np.all(np.isfinite(a)):
            raise ValueError("a contains non-finite values.")
        if not np.all(np.isfinite(b)):
            raise ValueError("b contains non-finite values.")

    # ------------------------------------------------------------------ #
    # Step 1: Thin QR of [A; B]
    #   M = Q_qr @ R_qr,  Q_qr: (m+n)×p,  R_qr: p×p upper-triangular
    # ------------------------------------------------------------------ #
    M = np.vstack([a, b])
    Q_qr, R_qr = qr(M, mode='economic')  # Q_qr: (m+n)×p, R_qr: p×p

    # Canonicalise sign so R_qr has non-negative diagonal (unique thin QR).
    diag_signs = np.sign(np.real(np.diag(R_qr)))
    diag_signs[diag_signs == 0] = 1.0
    Q_qr = Q_qr * diag_signs[np.newaxis, :]
    R_qr = diag_signs[:, np.newaxis] * R_qr

    Q1 = Q_qr[:m, :]   # m × p
    Q2 = Q_qr[m:, :]   # n × p

    # ------------------------------------------------------------------ #
    # Step 2: SVD of Q1  →  cosines, U, and rotation Vh
    #   Q1 = U_thin @ diag(c) @ Vh   (c in [0,1], descending)
    # ------------------------------------------------------------------ #
    full_u = compute_u and (mode == 'full')
    U_svd, c_vals, Vh = np.linalg.svd(Q1, full_matrices=full_u)
    # full_u=False: U_svd m×p, Vh p×p
    # full_u=True : U_svd m×m, Vh p×p  (numpy always returns full Vh for square Q1? No.)
    # numpy docs: full_matrices=True → u: (M,M), vh: (N,N); False → u: (M,K), vh: (K,N)
    # Q1 is m×p with m>=p → K=p, so:
    #   full_matrices=False: U_svd m×p, Vh p×p
    #   full_matrices=True : U_svd m×m, Vh p×p  (Vh is always p×p here since K=p)

    c_vals = np.clip(c_vals, 0.0, 1.0)
    s_vals = np.sqrt(np.maximum(0.0, 1.0 - c_vals**2))

    # ------------------------------------------------------------------ #
    # Step 3: Build V from Q2
    #   Q2_W = Q2 @ Vh.conj().T  has orthogonal columns with norms s_vals
    #   (follows from Q_qr^H Q_qr = I_p  →  Q1^H Q1 + Q2^H Q2 = I_p
    #    →  Vh diag(c²) Vh^H + Q2^H Q2 = I  →  (Q2 Vh^H)^H (Q2 Vh^H) = diag(s²))
    # ------------------------------------------------------------------ #
    Vh_p = Vh   # p×p (always square for m>=p)

    Q2_W = Q2 @ Vh_p.conj().T  # n×p, columns orthogonal with norms s_vals

    # Build V (n×p) by column-normalising Q2_W.
    # Q2_W has orthogonal columns with norms s_vals[i] in ASCENDING order
    # (because c_vals is descending → s ascending).  A naive SVD would sort
    # them in descending order and break the pairing with c_vals, so we
    # normalise directly instead.
    eps_s = np.finfo(np.float64).eps ** 0.5 * max(np.linalg.norm(a), np.linalg.norm(b), 1.0)
    V_thin = np.zeros((n, p), dtype=dtype)
    nonzero = s_vals > eps_s
    if np.any(nonzero):
        V_thin[:, nonzero] = Q2_W[:, nonzero] / s_vals[nonzero]
    if np.any(~nonzero):
        # Columns of Q2_W with s≈0 are near-zero; fill with orthonormal
        # vectors spanning the complement of the non-zero V columns.
        V_thin = _fill_zero_v_columns(V_thin, nonzero, n)
    full_v = compute_v and (mode == 'full')

    # ------------------------------------------------------------------ #
    # Step 4: Build output
    #   k=0, l=p  (no rank truncation)
    # ------------------------------------------------------------------ #
    k = 0
    l = p   # q = k + l = p

    # -- Diagonal matrices --
    # C: m×p with C[i,i] = c_vals[i]  (m>=p so all p diagonal entries fit)
    # S: n×p with S[i,i] = s_vals[i]  (n>=p so all p diagonal entries fit)
    C_full = np.zeros((m, p), dtype=float)
    S_full = np.zeros((n, p), dtype=float)
    idx = np.arange(p)
    C_full[idx, idx] = c_vals
    S_full[idx, idx] = s_vals

    # -- Right factor X (for full/econ) or Q factor (for separate) --
    # X = Q_right @ R.conj().T  where Q_right = Vh_p^H  (matching gsvd4py convention)
    # Equivalently: X^H = Vh_p @ R_qr  →  A = U C X^H, B = V S X^H.
    if compute_right:
        # Q_right = Vh_p^H = Vh_p.conj().T  (p×p unitary)
        Q_right = Vh_p.conj().T
        # A = U C Vh R_qr = U C X^H  →  X^H = Vh R_qr = Q_right^H R_qr
        # so X = R_qr^H Q_right = R_qr.conj().T @ Q_right
        X = R_qr.conj().T @ Q_right    # p×p

    if mode == 'separate':
        result = []
        if compute_u:
            if full_u:
                result.append(U_svd)
            else:
                result.append(_extend_to_unitary(U_svd, m))
        if compute_v:
            if full_v:
                result.append(_extend_to_unitary(V_thin, n))
            else:
                result.append(_extend_to_unitary(V_thin, n))
        result += [C_full, S_full, R_qr]
        if compute_right:
            result.append(Q_right)   # p×p right orthogonal factor
        result += [k, l]
        return tuple(result)

    # -- full / econ modes --
    if mode == 'full':
        U_out = U_svd if compute_u else None
        V_out = _extend_to_unitary(V_thin, n) if (compute_v and full_v) else None
        if compute_v and not full_v:
            V_out = _extend_to_unitary(V_thin, n)
        C_out = C_full   # m×p
        S_out = S_full   # n×p
    else:  # 'econ'
        U_out = (U_svd if U_svd.shape[1] == p else U_svd[:, :p]) if compute_u else None
        V_out = V_thin if compute_v else None   # already n×p
        C_out = C_full[:p, :]   # p×p
        S_out = S_full[:p, :]   # p×p

    result = []
    if compute_u:
        result.append(U_out)
    if compute_v:
        result.append(V_out)
    result += [C_out, S_out]
    if compute_right:
        result.append(X)
    return tuple(result)


def gsvdvals(a, b, check_finite=True):
    """Generalized singular value pairs of (a, b) via CS decomposition.

    Parameters
    ----------
    a : (m, p) array_like
    b : (n, p) array_like
    check_finite : bool

    Returns
    -------
    c : ndarray, shape (p,)
        Generalized cosines in non-increasing order.
    s : ndarray, shape (p,)
        Generalized sines in non-decreasing order.
    """
    C, S = gsvd(a, b, mode='econ', compute_u=False, compute_v=False,
                compute_right=False, check_finite=check_finite)
    c = np.diag(C)   # p values from diagonal of p×p C block
    s = np.diag(S)
    return c, s


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_zero_v_columns(V_thin, nonzero, n):
    """Fill zero columns of V_thin (n×p) for indices where nonzero is False.

    Replaces the zero columns with orthonormal vectors spanning the complement
    of the already-filled columns.
    """
    p = V_thin.shape[1]
    zero_cols = np.where(~nonzero)[0]
    if len(zero_cols) == 0:
        return V_thin
    # Build basis for the orthogonal complement of the non-zero columns
    good = V_thin[:, nonzero]   # n × (number of non-zero)
    rng = np.random.default_rng(seed=1)
    rand = rng.standard_normal((n, len(zero_cols)))
    if np.issubdtype(V_thin.dtype, np.complexfloating):
        rand = rand + 1j * rng.standard_normal((n, len(zero_cols)))
    rand = rand - good @ (good.conj().T @ rand)
    extra, _ = qr(rand, mode='economic')
    V_out = V_thin.copy()
    V_out[:, zero_cols] = extra[:, :len(zero_cols)]
    return V_out


def _extend_to_unitary(Q_thin, n):
    """Extend a thin unitary matrix Q_thin (n×k, k<=n) to a full n×n unitary.

    The first k columns are preserved exactly; the remaining n-k columns span
    the orthogonal complement, computed via QR of a random matrix projected
    onto that complement.
    """
    n_rows, k = Q_thin.shape
    assert n_rows == n
    if k == n:
        return Q_thin
    # Build null-space complement via QR of (I - Q Q^H) applied to random vecs
    rng = np.random.default_rng(seed=0)
    rand = rng.standard_normal((n, n - k))
    if np.issubdtype(Q_thin.dtype, np.complexfloating):
        rand = rand + 1j * rng.standard_normal((n, n - k))
    # Project out the span of Q_thin
    rand -= Q_thin @ (Q_thin.conj().T @ rand)
    # QR of the projected matrix
    extra, _ = qr(rand, mode='economic')
    return np.hstack([Q_thin, extra])
