import pytest
from lappy import Domain, Polygon, ParametricSegment
from lappy import geometry as geo
import numpy as np

@pytest.fixture
def unit_square_domain():
    vertices = np.array([0,1,1+1j,1j])
    return Polygon(vertices)

@pytest.fixture
def rect_domain():
    """2×1 rectangle."""
    return Polygon(np.array([0, 2, 2+1j, 1j]))

@pytest.fixture
def right_triangle():
    """3-4-5 right triangle."""
    return Polygon(np.array([0, 3, 3+4j]))

@pytest.fixture
def unit_disk_seg():
    """Unit disk as a ParametricSegment."""
    return ParametricSegment(
        lambda t: np.exp(1j*t),
        lambda t: 1j*np.exp(1j*t),
        0, 2*np.pi, val_closed=True, val_simple=False
    )

# ── Domain fixtures for the cubature verification suite ──────────────────────────

@pytest.fixture
def disk_domain():
    """Unit disk (curved boundary)."""
    return geo.disk(1.0)

@pytest.fixture
def sector_domain():
    """Quarter-disk sector, radius 1, opening π/2 (curved boundary, convex)."""
    return geo.disk_sector(1.0, np.pi/2)

@pytest.fixture
def sector_reflex_domain():
    """Reflex disk sector, radius 1, opening 3π/2 (curved boundary + reentrant apex)."""
    return geo.disk_sector(1.0, 3*np.pi/2)

@pytest.fixture
def Lshape_domain():
    """L-shaped polygon with a 270° reentrant corner at the origin."""
    return geo.L_shape()

@pytest.fixture
def eq_tri_domain():
    """Equilateral triangle, side length 1."""
    return geo.eq_tri(1.0)

@pytest.fixture
def iso_right_tri_domain():
    """Isosceles right triangle, legs of length 1."""
    return geo.iso_right_tri(1.0)