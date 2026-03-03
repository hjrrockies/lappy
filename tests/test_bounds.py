"""Tests for lappy.bounds — rigorous Laplacian eigenvalue inequalities."""

import numpy as np
import pytest
from scipy.special import jn_zeros

import lappy.bounds as bounds
from lappy import Polygon
from lappy.geometry import MultiSegment


# Bessel zeros (reference values)
_j01 = jn_zeros(0, 1)[0]
_j11 = jn_zeros(1, 1)[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def dir_square():
    return Polygon(np.array([0, 1, 1 + 1j, 1j]))


def neu_square():
    vertices = np.array([0, 1, 1 + 1j, 1j])
    bdry = MultiSegment.from_vertices(vertices, bc='neu')
    return Polygon(bdry=bdry)


# ── faber_krahn ───────────────────────────────────────────────────────────────

def test_faber_krahn_scalar():
    """Scalar path: π·j₀₁²/area."""
    assert np.isclose(bounds.faber_krahn(area=1.0), np.pi * _j01**2)


def test_faber_krahn_domain_path():
    """Domain path matches scalar path for unit square."""
    dom = dir_square()
    result_domain = bounds.faber_krahn(dom)
    result_scalar = bounds.faber_krahn(area=dom.area)
    assert np.isclose(result_domain, result_scalar)


def test_faber_krahn_raises_on_neumann():
    """Raises ValueError when domain has Neumann bc."""
    with pytest.raises(ValueError, match="bc_type"):
        bounds.faber_krahn(neu_square())


def test_faber_krahn_missing_area():
    with pytest.raises(ValueError):
        bounds.faber_krahn()


# ── szego_weinberger ──────────────────────────────────────────────────────────

def test_szego_weinberger_scalar():
    """Scalar path: π·j₁₁²/area."""
    assert np.isclose(bounds.szego_weinberger(area=1.0), np.pi * _j11**2)


def test_szego_weinberger_domain_path():
    """Domain path matches scalar path for unit Neumann square."""
    dom = neu_square()
    result_domain = bounds.szego_weinberger(dom)
    result_scalar = bounds.szego_weinberger(area=dom.area)
    assert np.isclose(result_domain, result_scalar)


def test_szego_weinberger_raises_on_dirichlet():
    """Raises ValueError when domain has Dirichlet bc."""
    with pytest.raises(ValueError, match="bc_type"):
        bounds.szego_weinberger(dir_square())


# ── payne_polya_weinberger ────────────────────────────────────────────────────

def test_payne_polya_weinberger_value():
    """Returns (j₁₁/j₀₁)²."""
    assert np.isclose(bounds.payne_polya_weinberger(), (_j11 / _j01) ** 2)


def test_payne_polya_weinberger_no_domain_required():
    """Can be called without any arguments."""
    result = bounds.payne_polya_weinberger()
    assert np.isfinite(result)
    assert result > 2.5  # known value ≈ 2.539


def test_payne_polya_weinberger_domain_ok():
    """Accepts a Dirichlet domain for BC validation."""
    result = bounds.payne_polya_weinberger(dir_square())
    assert np.isclose(result, (_j11 / _j01) ** 2)


def test_payne_polya_weinberger_raises_on_neumann():
    """Raises if domain is Neumann."""
    with pytest.raises(ValueError, match="bc_type"):
        bounds.payne_polya_weinberger(neu_square())


# ── inradius_upper ────────────────────────────────────────────────────────────

def test_inradius_upper_scalar():
    """inradius=0.5 → j₀₁²/0.25 = 4·j₀₁²."""
    result = bounds.inradius_upper(inradius=0.5)
    assert np.isclose(result, _j01**2 / 0.25)


def test_inradius_upper_domain_path():
    """Domain path uses domain.inradius."""
    dom = dir_square()
    result_domain = bounds.inradius_upper(dom)
    result_scalar = bounds.inradius_upper(inradius=dom.inradius)
    assert np.isclose(result_domain, result_scalar)


def test_inradius_upper_unit_square_value():
    """Unit square inradius=0.5 → upper bound = 4·j₀₁²."""
    dom = dir_square()
    assert np.isclose(dom.inradius, 0.5, atol=1e-3)
    result = bounds.inradius_upper(dom)
    assert np.isclose(result, 4 * _j01**2, rtol=1e-3)


def test_inradius_upper_raises_on_neumann():
    with pytest.raises(ValueError, match="bc_type"):
        bounds.inradius_upper(neu_square())


def test_inradius_upper_missing_inradius():
    with pytest.raises(ValueError):
        bounds.inradius_upper()


# ── polya ─────────────────────────────────────────────────────────────────────

def test_polya_scalar():
    """Scalar k: λ_k ≥ 4π·k/area."""
    result = bounds.polya(5, area=1.0)
    assert np.isclose(result, 4 * np.pi * 5)


def test_polya_ndarray():
    """Vectorized k."""
    k = np.array([1, 2, 3, 4, 5])
    result = bounds.polya(k, area=1.0)
    assert result.shape == (5,)
    np.testing.assert_allclose(result, 4 * np.pi * k)


def test_polya_domain_path():
    """Domain path extracts area automatically."""
    dom = dir_square()
    result_domain = bounds.polya(5, dom)
    result_scalar = bounds.polya(5, area=dom.area)
    assert np.isclose(result_domain, result_scalar)


def test_polya_raises_on_neumann():
    with pytest.raises(ValueError, match="bc_type"):
        bounds.polya(5, neu_square())


def test_polya_values_increase():
    """Pólya bound is strictly increasing in k."""
    k = np.arange(1, 10)
    result = bounds.polya(k, area=1.0)
    assert np.all(np.diff(result) > 0)
