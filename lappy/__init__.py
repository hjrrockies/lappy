# Do not delete this file. It tells python that lappy is a module you can import from.
# public facing functions should be imported here so they can be used directly
name = "lappy"

# modules available on package load
from . import mps
from . import bases
from . import quad
from . import utils
from . import geometry

# classes and functions available directly from lappy
from .mps import MPSProblem, solve_eigs_interval
from .evp import Eigenproblem
from .bases import PointSet, ParticularBasis, MultiBasis, FourierBesselBasis, NormalizedBasis
from .quad import triangular_mesh,  tri_quad, boundary_nodes_polygon
from .geometry import PointSet, Domain, Polygon, Segment, LineSegment, MultiSegment, make_boundary_pts, make_interior_pts
