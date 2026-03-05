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
def unit_circle_seg():
    """Unit circle as a ParametricSegment."""
    return ParametricSegment(
        lambda t: np.exp(1j*t),
        lambda t: 1j*np.exp(1j*t),
        0, 2*np.pi, val_closed=True, val_simple=False
    )

@pytest.fixture
def gww_dir_eigs():
    """The first 25 Dirichlet eigenvalues of the GWW isospectral domains, accurate to 12 digits, in sorted order"""
    # just letting numpy sort because Driscoll's table wasn't easy to copy and paste!
    eigs = np.sort([2.53794399980, 9.20929499840, 14.3138624643, 20.8823950433, 24.6740110027, 
                    3.65550971352, 10.5969856913, 15.871302620, 21.2480051774, 26.0802400997, 
                    5.17555935622, 11.5413953956, 16.9417516880, 22.2328517930, 27.3040189211, 
                    6.53755744376, 12.3370055014, 17.6651184368, 23.7112974848, 28.1751285815, 
                    7.24807786256, 13.0536540557, 18.9810673877, 24.4792340693, 29.5697729132])
    return eigs