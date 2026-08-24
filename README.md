# lappy

Laplacian eigenvalues, eigenfunctions and boundary integrals for planar domains, by the Method
of Particular Solutions (GSVD form).

**Version 0.9 — beta.** The version number is a claim about the *polygon* path specifically:
what it promises, what has been measured, and where the envelope ends are all written down in
**[`docs/polygon_contract.md`](docs/polygon_contract.md)**. Read that before building on this.

## Install

```sh
git clone https://github.com/hjrrockies/lappy && cd lappy && pip install -e .
```

## Use

Posing and solving an eigenproblem is three lines:

```python
from lappy import Eigenproblem, geometry as geo

dom  = geo.L_shape()
evp  = Eigenproblem(dom, precision=1e-10)
eigs = evp.solve(4)            # 159 columns, ~11.5 true digits, nothing configured
```

`precision` is one dial at both stages: it sizes the basis *and* sets the eigenvalue search's
tolerance. It is a hint with roughly a digit of margin, not a guarantee —
`evp.check_precision(eigs)` measures what was actually achieved.

The detailed workflow is still exposed when you want it:

```python
import numpy as np
from lappy import Polygon, basis_plan, mps

rect   = Polygon(np.array([0, 2, 2+1j, 1j]))
basis  = basis_plan.polygon_default_basis(rect, lam_max=200, target=1e-10)
solver = mps.MPSEigensolver.from_domain(rect, basis=basis)
eigs   = Eigenproblem(rect, eval_solver=solver).solve(4)
```

### In an optimization loop

Plan once, realize per iterate, and follow the mode by value rather than by index:

```python
from lappy.asymp import weyl_est

plan = basis_plan.plan_basis(dom0, weyl_est(6, dom0), target=1e-10)
lam  = Eigenproblem(dom0, precision=1e-10).solve(1)[0]      # cold start, once
for dom in family:
    solver = mps.MPSEigensolver.from_domain(dom, basis=basis_plan.realize(plan, dom))
    lam = Eigenproblem(dom, eval_solver=solver, precision=1e-10).track(lam)
```

The frozen plan gives an identical basis size on every member, so a change in `lambda` came from
the shape and not from the basis. `track` is ~2.7x faster than a fresh `solve` and raises rather
than returning a plausible wrong number if it loses the mode.

For shape derivatives, use `solver.hadamard_quad` — **not** `solver.bdry_quad`, which is the
normalization node set and loses three digits when a velocity moves a reentrant corner. See §2
of the contract.

## What it is good at

Certified accuracy on corner domains. The eigenvalue search minimizes a GSVD tension, and the
basis is derived from the geometry rather than chosen by hand: `basis_plan` sets corner
Fourier–Bessel budgets and boundary-offset source arcs from the domain, `lam_max` and a target,
then measures the result instead of predicting it.

Measured on ten suite polygons through the three-line path: 8.6–13.4 true digits at 60–364
columns. Sharp-cornered families (chevrons, thin isoceles triangles) fall short of a requested
precision and say so.

## Tests

```sh
pytest                      # ~1100 tests, several minutes
pytest -m "not slow"        # the fast subset
```

## Documentation

- [`docs/polygon_contract.md`](docs/polygon_contract.md) — the v0.9 promise, envelope and known defects
- [`docs/scope_and_downstream.md`](docs/scope_and_downstream.md) — where lappy stops and a shape package begins
- [`benchmarks/basis_lab/PLAN_LAB.md`](benchmarks/basis_lab/PLAN_LAB.md) — how the basis planner was designed and validated
- [`docs/rellich.md`](docs/rellich.md), [`docs/eigfun_integrals.md`](docs/eigfun_integrals.md) — the boundary-integral machinery
