"""Tests for lappy.fem — FEMEigensolver."""

import numpy as np
import pytest

from lappy import Polygon, Eigenproblem
from lappy.fem import FEMEigensolver
from lappy.reference import rect_eigs


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def unit_square():
    return Polygon(np.array([0, 1, 1+1j, 1j]))


@pytest.fixture(scope='module')
def fem_solver(unit_square):
    return FEMEigensolver(unit_square, mesh_size=0.05)


# ── Construction ──────────────────────────────────────────────────────────────

def test_constructor_stores_params():
    sq = Polygon(np.array([0, 1, 1+1j, 1j]))
    s = FEMEigensolver(sq, mesh_size=0.1, k_max=50)
    assert s.mesh_size == 0.1
    assert s.k_max == 50
    assert not s._assembled


def test_from_domain(unit_square):
    s = FEMEigensolver.from_domain(unit_square)
    assert isinstance(s, FEMEigensolver)


def test_rejects_neumann():
    from lappy.geometry import MultiSegment
    bdry = MultiSegment.from_vertices(np.array([0, 1, 1+1j, 1j]), bc='neu')
    poly = Polygon(bdry=bdry, val_simple=False)
    with pytest.raises(NotImplementedError):
        FEMEigensolver(poly)


def test_evp_fem_string():
    sq = Polygon(np.array([0, 1, 1+1j, 1j]))
    evp = Eigenproblem(sq, eval_solver='fem')
    assert isinstance(evp.eval_solver, FEMEigensolver)


# ── Lazy assembly ─────────────────────────────────────────────────────────────

def test_lazy_assembly():
    sq = Polygon(np.array([0, 1, 1+1j, 1j]))
    s = FEMEigensolver(sq, mesh_size=0.1)
    assert not s._assembled
    s._build_and_assemble()
    assert s._assembled
    assert s._K.shape[0] == s._N_dof
    assert s._M.shape == s._K.shape


def test_K_symmetric():
    sq = Polygon(np.array([0, 1, 1+1j, 1j]))
    s = FEMEigensolver(sq, mesh_size=0.1)
    s._build_and_assemble()
    diff = s._K - s._K.T
    assert abs(diff).max() < 1e-12


def test_M_symmetric():
    sq = Polygon(np.array([0, 1, 1+1j, 1j]))
    s = FEMEigensolver(sq, mesh_size=0.1)
    s._build_and_assemble()
    diff = s._M - s._M.T
    assert abs(diff).max() < 1e-12


# ── Eigenvalue accuracy (unit square, Dirichlet) ──────────────────────────────
#
# Exact eigenvalues: π²(m² + n²), m,n ≥ 1.
# First 10: 2π², 5π², 5π², 8π², 10π², 10π², 13π², 13π², 17π², 18π²
# P1 FEM overestimates — expect values slightly above reference.
# With mesh_size=0.05, errors should be well under 1%.

EXACT_10 = rect_eigs(10, 1.0, 1.0)


def test_first_eigenvalue(fem_solver):
    eigs, mults, _ = fem_solver.solve_interval(0, 30)
    assert len(eigs) >= 1
    lam1 = eigs[0]
    assert lam1 > 2 * np.pi**2          # FEM overestimates
    assert abs(lam1 - 2 * np.pi**2) / (2 * np.pi**2) < 0.01


def test_first_10_eigenvalues(fem_solver):
    a = 0
    b = EXACT_10[-1] * 1.05
    eigs, mults, _ = fem_solver.solve_interval(a, b)
    eigs_flat = np.repeat(eigs, mults)
    assert len(eigs_flat) >= 10, f"only found {len(eigs_flat)} eigenvalues"

    found = np.sort(eigs_flat)[:10]
    rel_err = np.abs(found - EXACT_10) / EXACT_10
    # P1 FEM with mesh_size=0.05 gives ~3% error on higher eigenvalues
    assert np.all(rel_err < 0.03), f"max rel error {rel_err.max():.4f}"


def test_fem_overestimates(fem_solver):
    """P1 FEM gives variational upper bounds — all values must be >= exact."""
    a, b = 0, EXACT_10[-1] * 1.05
    eigs, mults, _ = fem_solver.solve_interval(a, b)
    eigs_flat = np.sort(np.repeat(eigs, mults))[:10]
    assert np.all(eigs_flat >= EXACT_10 - 1e-6)


def test_solve_interval_return_types(fem_solver):
    eigs, mults, fevals = fem_solver.solve_interval(0, 30)
    assert isinstance(eigs, np.ndarray)
    assert isinstance(mults, np.ndarray)
    assert mults.dtype == int
    assert fevals == 1
    assert len(eigs) == len(mults)
    assert np.all(mults >= 1)


def test_eigenproblem_solve_10():
    sq = Polygon(np.array([0, 1, 1+1j, 1j]))
    evp = Eigenproblem(sq, eval_solver='fem')
    evp.eval_solver.mesh_size = 0.05
    eigs = evp.solve(10)
    assert len(eigs) == 10
    rel_err = np.abs(eigs - EXACT_10) / EXACT_10
    assert np.all(rel_err < 0.03), f"max rel error {rel_err.max():.4f}"
