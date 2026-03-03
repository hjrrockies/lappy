"""Tests for lappy.asymp — Weyl asymptotic expansions."""

import numpy as np
import pytest

import lappy.asymp as asymp
from lappy import Polygon
from lappy.geometry import MultiSegment


# ── Helpers ───────────────────────────────────────────────────────────────────

def unit_square():
    return Polygon(np.array([0, 1, 1 + 1j, 1j]))


def neu_square():
    """Unit square with Neumann boundary."""
    vertices = np.array([0, 1, 1 + 1j, 1j])
    bdry = MultiSegment.from_vertices(vertices, bc='neu')
    return Polygon(bdry=bdry)


# ── weyl_count ────────────────────────────────────────────────────────────────

def test_weyl_count_domain_vs_scalar():
    """Domain path and scalar path give identical results."""
    dom = unit_square()
    lam = 100.0
    result_domain = asymp.weyl_count(lam, dom)
    result_scalar = asymp.weyl_count(lam, area=dom.area, perim=dom.perimeter, bc_type='dir')
    assert np.isclose(result_domain, result_scalar)


def test_weyl_count_dir_formula():
    """Dirichlet: N ≈ (A·λ − P·√λ) / (4π)."""
    area, perim, lam = 1.0, 4.0, 100.0
    expected = (area * lam - perim * np.sqrt(lam)) / (4 * np.pi)
    assert np.isclose(asymp.weyl_count(lam, area=area, perim=perim, bc_type='dir'), expected)


def test_weyl_count_neu_greater_than_dir():
    """Neumann N > Dirichlet N for same λ > 0 (perimeter term adds vs subtracts)."""
    lam = 50.0
    area, perim = 1.0, 4.0
    n_dir = asymp.weyl_count(lam, area=area, perim=perim, bc_type='dir')
    n_neu = asymp.weyl_count(lam, area=area, perim=perim, bc_type='neu')
    assert n_neu > n_dir


def test_weyl_count_ndarray():
    """weyl_count accepts ndarray lam."""
    lam = np.array([10.0, 50.0, 100.0])
    result = asymp.weyl_count(lam, area=1.0, perim=4.0, bc_type='dir')
    assert result.shape == (3,)
    assert np.all(np.diff(result) > 0)


def test_weyl_count_missing_args():
    with pytest.raises(ValueError):
        asymp.weyl_count(100.0, area=1.0)  # missing perim


def test_weyl_count_mixed_raises():
    with pytest.raises(NotImplementedError):
        asymp.weyl_count(100.0, area=1.0, perim=4.0, bc_type='mixed')


# ── weyl_est ──────────────────────────────────────────────────────────────────

def test_weyl_est_roundtrip_dir():
    """weyl_count(weyl_est(k)) ≈ k for Dirichlet."""
    area, perim = 1.0, 4.0
    for k in [1, 5, 20]:
        lam_est = asymp.weyl_est(k, area=area, perim=perim, bc_type='dir')
        k_back = asymp.weyl_count(lam_est, area=area, perim=perim, bc_type='dir')
        assert np.isclose(k_back, k, rtol=1e-10)


def test_weyl_est_roundtrip_neu():
    """weyl_count(weyl_est(k)) ≈ k for Neumann."""
    area, perim = 1.0, 4.0
    for k in [1, 5, 20]:
        lam_est = asymp.weyl_est(k, area=area, perim=perim, bc_type='neu')
        k_back = asymp.weyl_count(lam_est, area=area, perim=perim, bc_type='neu')
        assert np.isclose(k_back, k, rtol=1e-10)


def test_weyl_est_domain_path():
    """Domain path gives same result as scalar path."""
    dom = unit_square()
    k = 10
    est_domain = asymp.weyl_est(k, dom)
    est_scalar = asymp.weyl_est(k, area=dom.area, perim=dom.perimeter, bc_type='dir')
    assert np.isclose(est_domain, est_scalar)


def test_weyl_est_ndarray():
    """weyl_est accepts ndarray k."""
    k = np.array([1.0, 5.0, 10.0])
    result = asymp.weyl_est(k, area=1.0, perim=4.0, bc_type='dir')
    assert result.shape == (3,)
    assert np.all(np.diff(result) > 0)


# ── weyl_count_poly ───────────────────────────────────────────────────────────

def test_weyl_count_poly_correction_is_constant():
    """Corner correction is independent of λ: difference between two λ values
    equals the two-term Weyl difference (correction cancels out)."""
    dom = unit_square()
    lam1, lam2 = 50.0, 100.0
    diff_poly = asymp.weyl_count_poly(lam2, dom) - asymp.weyl_count_poly(lam1, dom)
    diff_two = asymp.weyl_count(lam2, dom) - asymp.weyl_count(lam1, dom)
    assert np.isclose(diff_poly, diff_two)


def test_weyl_count_poly_unit_square_correction():
    """Unit square: 4 right-angle corners give correction = 1/4."""
    dom = unit_square()
    # correction = 4 * (π² - (π/2)²) / (24π · (π/2))
    #            = 4 * (3π²/4) / (12π²) = 4/16 = 1/4
    correction = asymp.weyl_count_poly(0.0, dom)  # at lam=0 two-term is 0
    assert np.isclose(correction, 0.25, rtol=1e-10)


def test_weyl_count_poly_polygon_domain():
    """Polygon domain: angles extracted automatically."""
    dom = unit_square()
    result = asymp.weyl_count_poly(100.0, dom)
    result_scalar = asymp.weyl_count_poly(
        100.0, area=dom.area, perim=dom.perimeter,
        angles=dom.int_angles, bc_type='dir'
    )
    assert np.isclose(result, result_scalar)


def test_weyl_count_poly_raises_on_non_polygon_no_angles():
    """Generic Domain without angles= raises TypeError."""
    from lappy.geometry import MultiSegment, Domain
    # Build a Domain (not Polygon) from a polygon boundary
    vertices = np.array([0, 1, 1 + 1j, 1j])
    bdry = MultiSegment.from_vertices(vertices)
    generic_domain = Domain(bdry)  # Domain, not Polygon
    with pytest.raises(TypeError, match="angles"):
        asymp.weyl_count_poly(100.0, generic_domain)


# ── weyl_est_poly ─────────────────────────────────────────────────────────────

def test_weyl_est_poly_roundtrip():
    """weyl_count_poly(weyl_est_poly(k)) ≈ k."""
    dom = unit_square()
    for k in [1, 5, 15]:
        lam_est = asymp.weyl_est_poly(k, dom)
        k_back = asymp.weyl_count_poly(lam_est, dom)
        assert np.isclose(k_back, k, rtol=1e-6)
