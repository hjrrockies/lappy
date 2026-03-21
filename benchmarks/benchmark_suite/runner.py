"""BenchmarkConfig, BenchmarkResult, and run_benchmark()."""
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from lappy import MPSEigensolver, Eigenproblem

from .domains import DomainSpec
from .basis_builder import build_basis


@dataclass
class SweepDomainEntry:
    domain_name: str
    domain_params: dict = field(default_factory=dict)
    sweep_overrides: dict = field(default_factory=dict)  # {var_name: [values]}
    fixed_overrides: dict = field(default_factory=dict)  # {param_name: value}


@dataclass
class BenchmarkConfig:
    domain_name: str
    domain_params: dict

    n_eigs: int = 10

    # basis
    n_fb: int = 100
    n_fs: int = 0
    fb_strategy: str = 'singular_angle_weighted'
    fs_d_strategy: str = 'fixed'
    fs_d: float = 0.15
    fs_d_scale: float = 0.15
    fs_seg_strategy: str = 'length_weighted'

    # collocation grid
    bdry_pts_factor: float = 2.0   # bdry_pts = ceil(factor * (n_fb+n_fs)) total, prop to seg length
    int_pts_factor: float = 1.0    # int_pts  = ceil(factor * (n_fb+n_fs)) total random

    # solver tolerances
    rtol: float = 1e-14
    ltol: float = 1e-15
    ppl: int = 10


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    eigs: np.ndarray
    tensions: np.ndarray
    rel_errors: Optional[np.ndarray]   # None if no reference eigenvalues available
    wall_time: float
    n_basis_actual: int
    n_bdry_pts: int
    n_int_pts: int


def run_benchmark(spec: DomainSpec, config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run a single benchmark.

    Steps
    -----
    1. Build domain from spec.factory(**config.domain_params)
    2. Compute bdry_pts (Gauss-Legendre, no weights) and int_pts (random, no weights)
    3. Build basis via build_basis(...)
    4. Create MPSEigensolver + Eigenproblem
    5. Solve for config.n_eigs eigenvalues
    6. Compute tensions and relative errors (if reference available)
    """
    # --- 1. domain ---
    domain = spec.factory(**config.domain_params)

    n_total = config.n_fb + config.n_fs
    if n_total == 0:
        raise ValueError("n_fb + n_fs must be > 0")

    # --- 2. collocation points ---
    import numpy as np

    seg_lens = np.array([seg.len for seg in domain.bdry.segments])
    n_bdry_total = math.ceil(config.bdry_pts_factor * n_total)
    n_per_seg = np.maximum(
        np.round(n_bdry_total * seg_lens / seg_lens.sum()).astype(int), 1
    )
    bdry_pts = domain.bdry_pts(n_per_seg, kind='legendre')  # no weights

    n_int = math.ceil(config.int_pts_factor * n_total)
    int_pts = domain.int_pts(npts_rand=n_int)               # no weights

    # --- 3. basis (normalised on the same collocation pts) ---
    basis = build_basis(
        domain,
        config.n_fb,
        config.n_fs,
        fb_strategy=config.fb_strategy,
        fs_d_strategy=config.fs_d_strategy,
        fs_d=config.fs_d,
        fs_d_scale=config.fs_d_scale,
        fs_seg_strategy=config.fs_seg_strategy,
        normalize=True,
        norm_pts=(bdry_pts, int_pts),
    )

    # --- 4. solver ---
    solver = MPSEigensolver(
        basis, bdry_pts, int_pts,
        rtol=config.rtol,
        ltol=config.ltol,
    )
    evp = Eigenproblem(domain, eval_solver=solver)

    # --- 5. solve ---
    t0 = time.perf_counter()
    eigs = evp.solve(config.n_eigs, ppl=config.ppl)
    wall_time = time.perf_counter() - t0

    # --- 6. tensions ---
    tensions = np.array([solver.sigma(float(eig)) for eig in eigs])

    # --- 7. relative errors ---
    rel_errors = None
    if spec.ref_eigs is not None:
        try:
            ref = spec.ref_eigs(config.n_eigs, config.domain_params)
            if ref is not None:
                ref = np.asarray(ref, dtype=float)
                # match lengths in case reference returns fewer values
                k = min(len(eigs), len(ref))
                rel_errors = np.abs(eigs[:k] - ref[:k]) / np.abs(ref[:k])
        except Exception:
            pass

    return BenchmarkResult(
        config=config,
        eigs=eigs,
        tensions=tensions,
        rel_errors=rel_errors,
        wall_time=wall_time,
        n_basis_actual=len(basis),
        n_bdry_pts=len(bdry_pts),
        n_int_pts=len(int_pts),
    )
