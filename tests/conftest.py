import pytest
from lappy import Domain, Polygon, ParametricSegment
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