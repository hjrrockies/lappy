# lappy design goal: automated basis selection and MPS settings
A key aspect of `lappy` design is to have reliable heuristics for default particular solution bases and MPS solver settings.
The ultimate goal is to make it so the standard workflow for solving eigenproblems is:

```python
dom = Domain(...) # domain object with boundary conditions
eigprob = Eigenproblem(dom) # eigenproblem object
eigs = eigprob.solve(n_eigs=10, ltol=1e-10) # solve for first n_eigs=10 eigenvalues to relative precision ltol=1e-10
```

This requires that a good choice of particular solution basis, boundary points, interior points, and MPS hyperparameters
needs to be filled in automatically. Currently, a comprehensive portrait of how to pick these settings is not available.

# benchmark tests for data collection: overview
To resolve this issue, we need to run a suite of benchmarking tests that search over the various particular basis choices
and MPS settings on a variety of domains. The output of these tests will be a dataset that gives insights for how to pick
good particular solution bases and MPS settings on a given domain in order to obtain a particular degree of precision
in the eigenvalue computation.

# Model problem
The model problem is that a user will specify a domain `dom`, a number of eigenvalues `n_eigs` to compute, and a relative
precision `ltol`. Benchmarking this problem involves considering a variety of domains, number of desired eigenvalues, and
the desired level of precision.

# Particular solution basis choices
We need to benchmark the following aspects of particular solution bases:
1. Type of particular solutions (Fourier-Bessel, fundamental solutions, or combined FB-FS bases)
2. Number of basis functions of each type
3. Distribution of Fourier-Bessel functions: how many to put at each corner?
    a. How does this vary with corner angle?
    b. How does this vary with singular-vs-regular corners?
    c. How does this change for reentrant corners?
4. Location of fundamental solution terms
    a. How should they be spaced along the boundary?
    b. How far from the boundary should they be placed?

# MPS settings
1. How much regularization is needed? How much regularization can be afforded, given the decrease in achievable accuracy
based on the amount of regularization?

# Benchmark domains
The following domains can serve as benchmarks:
1. Rectangles with varying aspect ratio
2. Right trapezoids (trapezoids with two adjacent right angles) with varying side lengths
3. Parallelograms with varying side lengths and skewness
3. Equilateral triangle
4. Isosceles right triangle
5. General isosceles triangles, with varying height
6. Regular N-gons of varying N
7. L-shaped domain
8. GWW domains
9. Chevron domains
10. Cut square
11. Disk
12. Disk sectors of varying angle (small wedges up to angles above pi (reentrant corner))
13. Mushroom domain
14. H-shape domain
15. Ellipses
16. Stadium domain

# Benchmark test: domains needing reference values
The following domains do not have closed-form eigenvalues and do not yet have high-precision reference values:
1. Chevron domains
2. Cut square
3. Mushroom
4. H-shape
5. Ellipses
6. Stadium
7. General isosceles triangles
8. Regular N-gons, N > 4
9. Right trapezoids
10. Parallelograms

I will manage making reference tables for these domains (to put in `lappy.reference`)

# Benchmark test design principles
- Tests should isolate one or two variables at a time
- Tests should compare the results of `eigprob.solve(n_eigs, ltol=1e-15)` to reference values
- Tests should save the settings, the computed eigenvaues for each setting, the MPS tensions at each setting, and the
relative accuracy of the computed eigenvalues
