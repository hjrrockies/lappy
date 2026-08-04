# Overview
`lappy` needs to have better approaches for generating high-quality cubature rules for planar domains. These cubature
rules are responsible for ensuring $L^2$ orthnormality of eigenfunctions.

# Proposal
Build a new system for generating cubature nodes and weights for `Domain` and `Polygon` objects. The new system should
take in:

1. a `Domain` or `Polygon` object,
2. an (approximate) maximum of the spectral parameter $\lambda$, and finally
3. a user-specified `precision` for the desired level of relative accuracy in computing $L^2$ norms on the domain.

It should first generate a mesh for the domain which respects the curved parts of the boundary. It should then place
appropriate cubature rules on each element (potentially different rules for each element, if useful). The mesh and
cubature rules should have the following properties:

1. Corner singularities should be appropriately handled
2. All cubature weights should be positive.

The meshing algorithm should use `pygmsh` if possible. The `Polygon` case should dispatch to a method which bypasses
any algorithms which handle curved boundaries.

# Verification
A verification scheme needs to be designed as well. The following domains may serve as useful test cases:

1. Rectangles
2. Equilateral triangles
3. L-shaped domain (purpose: reentrant corner singularity)
4. Disks (purpose: curved boundary)
5. Disk sectors (purpose: curved boundary and potentially reentrant singularity)

Other domains should be included in the list, if useful.