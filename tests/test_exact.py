"""Tests for lappy.exact — closed-form rectangle eigenvalue formulas."""

import numpy as np
import pytest
from scipy.special import jn_zeros

import lappy.exact as exact
from lappy.utils import rect_loss_std, rect_loss_reciprocal, rect_loss_outerlog


# ── rect_eig ──────────────────────────────────────────────────────────────────

def test_rect_eig_unit_square():
    """λ_{1,1} for unit square equals 2π²."""
    assert np.isclose(exact.rect_eig(1, 1, 1, 1), 2 * np.pi**2)


def test_rect_eig_formula():
    """λ_{m,n} = π²m²/L² + π²n²/H²."""
    m, n, L, H = 2, 3, 4.0, 5.0
    expected = np.pi**2 * 4 / 16 + np.pi**2 * 9 / 25
    assert np.isclose(exact.rect_eig(m, n, L, H), expected)


def test_rect_eig_vectorized():
    """rect_eig broadcasts over m, n, L, H."""
    m = np.array([1, 2])
    n = np.array([1, 1])
    L, H = 1.0, 1.0
    result = exact.rect_eig(m, n, L, H)
    expected = np.array([2 * np.pi**2, 5 * np.pi**2])
    np.testing.assert_allclose(result, expected)


# ── rect_eigs ───────────────────────────────────────────────────────────────

def test_rect_eigs_dir_starts_at_one():
    """Dirichlet: first eigenvalue is rect_eig(1,1,1,1) = 2π²."""
    eigs = exact.rect_eigs(3, 1, 1, bc_type='dir')
    assert np.isclose(eigs[0], 2 * np.pi**2)


def test_rect_eigs_neu_starts_at_zero():
    """Neumann: first eigenvalue is 0 (m=n=0 mode)."""
    eigs = exact.rect_eigs(3, 1, 1, bc_type='neu')
    assert np.isclose(eigs[0], 0.0)


def test_rect_eigs_dir_sorted():
    """Dirichlet eigenvalues are returned in ascending order."""
    eigs = exact.rect_eigs(6, 1, 1, bc_type='dir')
    assert np.all(np.diff(eigs) >= 0)


def test_rect_eigs_ret_mn_dir():
    """ret_mn=True returns index arrays consistent with rect_eig."""
    eigs, m, n = exact.rect_eigs(4, 1.0, 2.0, bc_type='dir', ret_mn=True)
    recomputed = exact.rect_eig(m, n, 1.0, 2.0)
    np.testing.assert_allclose(eigs, recomputed)


def test_rect_eigs_vectorized_L_H():
    """rect_eigs handles batched L, H arrays."""
    L = np.array([1.0, 2.0])
    H = np.array([1.0, 1.0])
    eigs = exact.rect_eigs(3, L, H, bc_type='dir')
    assert eigs.shape == (2, 3)
    # first eigenvalue of 1x1 square
    assert np.isclose(eigs[0, 0], 2 * np.pi**2)
    # first eigenvalue of 2x1 rectangle
    assert np.isclose(eigs[1, 0], exact.rect_eig(1, 1, 2.0, 1.0))


def test_rect_eigs_invalid_bc():
    with pytest.raises(ValueError, match="bc_type"):
        exact.rect_eigs(3, 1, 1, bc_type='mixed')


# ── rect_eig_grad ─────────────────────────────────────────────────────────────

def test_rect_eig_grad_finite_difference():
    """rect_eig_grad matches centered finite differences."""
    m, n, L, H = 2, 3, 3.0, 4.0
    dL, dH = exact.rect_eig_grad(m, n, L, H)
    eps = 1e-6
    fd_dL = (exact.rect_eig(m, n, L + eps, H) - exact.rect_eig(m, n, L - eps, H)) / (2 * eps)
    fd_dH = (exact.rect_eig(m, n, L, H + eps) - exact.rect_eig(m, n, L, H - eps)) / (2 * eps)
    assert np.isclose(dL, fd_dL, rtol=1e-5)
    assert np.isclose(dH, fd_dH, rtol=1e-5)


def test_rect_eig_grad_sign():
    """Derivatives are negative (eigenvalue decreases as domain expands)."""
    dL, dH = exact.rect_eig_grad(1, 1, 1.0, 1.0)
    assert dL < 0
    assert dH < 0


# ── rect_eig_bound_idx ────────────────────────────────────────────────────────

def test_rect_eig_bound_idx_dir():
    """Dir: all returned indices have eigenvalues ≤ bound."""
    bound = 20.0
    idx = exact.rect_eig_bound_idx(bound, 1.0, 1.0, bc_type='dir')
    eigs = exact.rect_eig(idx[:, 0], idx[:, 1], 1.0, 1.0)
    assert np.all(eigs <= bound + 1e-12)
    assert np.all(idx >= 1)


def test_rect_eig_bound_idx_neu():
    """Neu: indices start at 0; (0,0) eigenvalue is 0 which is ≤ any positive bound."""
    bound = 10.0
    idx = exact.rect_eig_bound_idx(bound, 1.0, 1.0, bc_type='neu')
    eigs = exact.rect_eig(idx[:, 0], idx[:, 1], 1.0, 1.0)
    assert np.all(eigs <= bound + 1e-12)
    assert np.any(np.all(idx == 0, axis=1))  # (0,0) is included


# ── rect_eig_mult ─────────────────────────────────────────────────────────────

def test_rect_eig_mult_square():
    """Unit square: λ_{1,2} = λ_{2,1} → multiplicity 2."""
    L, H = 1.0, 1.0
    m_arr, n_arr = exact.rect_eig_mult(exact.rect_eig(1, 2, L, H), L, H, bc_type='dir')
    assert len(m_arr) == 2
    pairs = set(zip(m_arr, n_arr))
    assert (1, 2) in pairs
    assert (2, 1) in pairs


def test_rect_eig_mult_mn_consistent():
    """rect_eig_mult_mn gives same result as rect_eig_mult."""
    L, H = 1.0, 1.0
    m_arr1, n_arr1 = exact.rect_eig_mult_mn(1, 2, L, H, bc_type='dir')
    m_arr2, n_arr2 = exact.rect_eig_mult(exact.rect_eig(1, 2, L, H), L, H, bc_type='dir')
    np.testing.assert_array_equal(sorted(zip(m_arr1, n_arr1)), sorted(zip(m_arr2, n_arr2)))


# ── rect_loss_* still work after migration ───────────────────────────────────

def test_rect_loss_std_runs():
    eigs_true = exact.rect_eigs(3, 1.0, 1.0)
    loss = rect_loss_std(1.0, 1.0, eigs_true)
    assert np.isclose(loss, 0.0)


def test_rect_loss_reciprocal_runs():
    eigs_true = exact.rect_eigs(3, 1.0, 1.0)
    loss = rect_loss_reciprocal(1.0, 1.0, eigs_true)
    assert np.isclose(loss, 0.0)


def test_rect_loss_outerlog_runs():
    eigs_true = exact.rect_eigs(3, 1.0, 1.0)
    loss = rect_loss_outerlog(1.0, 1.0, eigs_true)
    # at exact solution loss should be very small (log of ~0)
    assert np.isfinite(loss)


# ── tri_isosc_eig ──────────────────────────────────────────────────────────────

def test_tri_isosc_eig_first():
    """First eigenvalue of isosceles right triangle: λ_{2,1} = 5π²/a²."""
    a = 1.0
    assert np.isclose(exact.tri_isosc_eig(2, 1, a), 5 * np.pi**2)


def test_tri_isosc_eig_invalid_returns_none():
    """m ≤ n or n < 1 returns None."""
    assert exact.tri_isosc_eig(1, 1, 1.0) is None   # m == n
    assert exact.tri_isosc_eig(1, 2, 1.0) is None   # m < n
    assert exact.tri_isosc_eig(2, 0, 1.0) is None   # n < 1


def test_tri_isosc_eig_scales_with_a():
    """λ scales as 1/a²."""
    val1 = exact.tri_isosc_eig(2, 1, 1.0)
    val2 = exact.tri_isosc_eig(2, 1, 2.0)
    assert np.isclose(val1 / val2, 4.0)


# ── tri_isosc_eigs ─────────────────────────────────────────────────────────────

def test_tri_isosc_eigs_count():
    """Returns exactly k values."""
    eigs = exact.tri_isosc_eigs(10, 1.0)
    assert len(eigs) == 10


def test_tri_isosc_eigs_sorted():
    """Returned eigenvalues are sorted ascending."""
    eigs = exact.tri_isosc_eigs(10, 1.0)
    assert np.all(np.diff(eigs) >= 0)


def test_tri_isosc_eigs_first_matches_eig():
    """First eigenvalue matches tri_isosc_eig(2, 1, a)."""
    a = 3.0
    eigs = exact.tri_isosc_eigs(5, a)
    assert np.isclose(eigs[0], exact.tri_isosc_eig(2, 1, a))


def test_tri_isosc_eigs_scales_with_a():
    """All eigenvalues scale as 1/a²."""
    eigs1 = exact.tri_isosc_eigs(6, 1.0)
    eigs2 = exact.tri_isosc_eigs(6, 2.0)
    np.testing.assert_allclose(eigs1 / eigs2, 4.0)


# ── _bessel_zero ───────────────────────────────────────────────────────────────

def test_bessel_zero_integer_order():
    """_bessel_zero matches scipy jn_zeros for integer nu."""
    from scipy.special import jn_zeros
    for m in [0, 1, 2, 3]:
        for n in [1, 2, 3]:
            expected = jn_zeros(m, n)[-1]
            got = exact._bessel_zero(m, n)
            assert np.isclose(got, expected, rtol=1e-8), f"m={m}, n={n}"


# ── sector_eig ────────────────────────────────────────────────────────────────

def test_sector_eig_half_disk():
    """Sector with alpha=π is a half-disk; angular modes are integer-order Bessel."""
    from scipy.special import jn_zeros
    R = 1.0
    alpha = np.pi
    # m=1 → nu=1, n=1 → j_{1,1}
    expected = jn_zeros(1, 1)[0] ** 2 / R**2
    assert np.isclose(exact.sector_eig(1, 1, R, alpha), expected)


def test_sector_eig_scales_with_R():
    """λ scales as 1/R²."""
    val1 = exact.sector_eig(1, 1, 1.0, np.pi / 2)
    val2 = exact.sector_eig(1, 1, 2.0, np.pi / 2)
    assert np.isclose(val1 / val2, 4.0)


def test_sector_eig_quarter_disk():
    """Quarter disk (alpha=π/2): nu=2 for m=1; check against _bessel_zero."""
    R, alpha = 1.5, np.pi / 2
    nu = 1 * np.pi / alpha  # = 2
    expected = exact._bessel_zero(nu, 1) ** 2 / R**2
    assert np.isclose(exact.sector_eig(1, 1, R, alpha), expected)


# ── sector_eigs ───────────────────────────────────────────────────────────────

def test_sector_eigs_count():
    """Returns exactly k values."""
    eigs = exact.sector_eigs(8, 1.0, np.pi / 2)
    assert len(eigs) == 8


def test_sector_eigs_sorted():
    """Returned eigenvalues are sorted ascending."""
    eigs = exact.sector_eigs(8, 1.0, np.pi / 2)
    assert np.all(np.diff(eigs) >= 0)


def test_sector_eigs_scales_with_R():
    """All eigenvalues scale as 1/R²."""
    eigs1 = exact.sector_eigs(6, 1.0, np.pi / 3)
    eigs2 = exact.sector_eigs(6, 2.0, np.pi / 3)
    np.testing.assert_allclose(eigs1 / eigs2, 4.0, rtol=1e-8)


def test_sector_eigs_half_disk_first():
    """Half-disk first eigenvalue = j_{1,1}²/R²."""
    from scipy.special import jn_zeros
    R = 1.0
    eigs = exact.sector_eigs(5, R, np.pi)
    expected_first = jn_zeros(1, 1)[0] ** 2 / R**2
    assert np.isclose(eigs[0], expected_first)
