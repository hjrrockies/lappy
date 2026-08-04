# `lappy` design principles
The initial object is a `Domain` object, which inherits from `BaseDomain`. A domain object represents a simply-connected
planar region with a specified boundary, which itself is a `MultiSegment` or `Segment` object. The boundary needs to be
a simple, closed curve.

The next step up on the hierarchy is a `BaseEigensolver` object. All eigensolvers should have a `from_domain` constructor
method which accepts a `BaseDomain` object and optional parameters. The optional parameters should all have sensible
default fallbacks, which can depend on the domain in question. `BaseEigensolver` objects, depending on their type,
may create or depend on sub-objects (such as particular solution bases for MPS eigensolvers). The `__init__` method
of an eigensolver object should accept the minimal number of parameters to uniquely characterize the problem instance,
and thus shouldn't depend directly on the domain object.

The top of the hierarchy is the `Eigenproblem` object, which *should* accept a `Domain` object in its `__init__`
constructor, and optionally eigensolver objects.

# Segment (and MultiSegment) design and defaults
- Segments are always taken in the standard CCW/positive orientation.
- Segment objects should use lazy evaluation whenever possible.
- Segment objects should have a reliable method for conversion to SplineSegments.

Segments have two kinds of point sampling:
- Polyline points: points which help check for intersections, and help for converting to splines, etc.
- Collocation/quadrature point sampling (including normal and tangent vectors). These are used by the MPS solver,
    and must include weights (when desired).

Unless otherwise specified:
- Segments should assume Dirichlet conditions.
- Segments should assume Gauss-Legendre points for collocation/quadrature points.
- Curved segments should use adaptive integration for arclength reparameterization 
    and polyline point samples.

# Domain design and defaults
- Domains are (for now) always simply-connected.
- Domains have a MultiSegment boundary (in CCW orientation).
- Domains should provide methods for extracting boundary collocation/quadrature data (points, normals, weights).
- Domains should provide methods for extracting interior collocation/cubature data (points and weights).
    - In particular, Domains should provide random interior points and mesh-derived cubature points.
- Domains should use lazy evaluation whenever possible.
- Domains should provide easy access to data about boundary corners.
- All methods should have cheaper fallbacks for the Polygon case.

Unless otherwise specified:
- Domains should check for boundary closure, simplicity, and MultiSegment contiguity on construction.
- Collocation point sampling should assume the weight-free case, with Gauss-Legendre points along the boundary and
    random points in the interior.

# MPSEigensolver design and defaults
- MPSEigensolver instances should directly depend only on the ParticularBasis and on the boundary/interior point data
    (as well as other settings)
    - This lets the user set up a problem with bases and points that were manually built.
- However, MPSEigensolver should have a highly robust `from_domain` method which builds an MPS eigensolver for a given
    domain using good heuristics.

Unless otherwise specified:
- MPSEigensolver should use sensible defaults for parameters/settings from `mps.py`. That is, it should be sufficient to
    provide a basis and boundary/interior points.
- MPSEigensolver should assume the Dirichlet boundary condition.

## MPSEigensolver.from_domain design and defaults
`MPSEigensolver.from_domain` should depend on the domain, a precision parameter (for desired precision in eigenvalue
computation), and a an approximate maximum value of the spectral parameter $\lambda$. Only the domain should be a 
required argument, with the precision set to a default of 1e-8 and the max spectral parameter set to a 2-term Weyl
estimate of the 6th eigenvalue (remembering to include sensible behavior for the Neumann, Robin, and Zaremba boundary
cases too).

The first step is determining an appropriate basis of particular solutions for the domain. This should dispatch to
`bases.make_default_basis`.

### make_default_basis design and defaults
(Probably saving this for later consideration.)

The next step is determining appropriate sets of collocation points (and potentially weights). There need to be two new
functions (placed appropriately) `make_default_bdry_pts` and `make_default_int_pts`.

### make_default_bdry_pts design and defaults
- This function should use Gauss-Jacobi points along each boundary segment, with the left- and right-hand singular
    exponents chosen to match the known eigenfunction asymptotics at each corner. If a segment endpoints have corner
    angles $\phi_1 = \frac{\pi}{\alpha}$ and $\phi_2 = \frac{\pi}{\beta}$, then $\alpha$ and $\beta$ are the singular
    exponents for the Gauss-Jacobi points.
- By default, it should return points only (no weights). With `weights=True`, weights are returned.
- This function should allow for other types of points (Gauss-Legendre, Chebyshev, etc)
- This function should also produce Gauss-Jacobi points and weights which are adapted to outward-normal derivatives of
    eigenfunctions, which have singular exponents of $\alpha-1$ and $\beta-1$.

### make_default_int_pts design and defaults
- This function should default to to random interior points.
- By default, it should return points only (no weights). With `weights=True`, weights are returned.
- This function should also provide cubature points (and optionally weights) which are adaptively determined to give
    a desired level of precision for $L^2$ inner product and norm estimation. The default level of precision should
    match the precision of the MPSEigensolver.

## Eigenproblem design and defaults
(Probably saving this for later consideration.)


