# Do not delete this file. It tells python that lappy is a module you can import from.
# public facing functions should be imported here so they can be used directly
name = "lappy"

# modules available on package load
from . import mps
from . import bases
from . import quad
from . import utils
from . import geometry
from . import reference
from . import asymp
from . import bounds
from . import opt

# classes and functions available directly from lappy
from .mps import MPSEigensolver
from .core import EigensolverFailure
from .evp import Eigenproblem
from .bases import ParticularBasis, MultiBasis, FourierBesselBasis, NormalizedBasis, FundamentalBasis
from .geometry import PointSet, Domain, Polygon, ParametricSegment, LineSegment, MultiSegment
