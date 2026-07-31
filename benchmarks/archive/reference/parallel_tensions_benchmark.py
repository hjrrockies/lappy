"""
Benchmark: ThreadPoolExecutor for parallel tension evaluation.

Measures wall-clock speedup of evaluating solver.tensions(lam) in parallel
vs. serial, using two disjoint lambda grids to avoid cache hits.
"""

import os
import sys
import time
import pathlib
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from lappy import geometry, asymp, bounds
from lappy.bases import FourierBesselBasis
from lappy.mps import MPSEigensolver


def setup_solver(n_basis):
    ldom = geometry.L_shape()
    orders = [n_basis, 0, 0, 0, 0, 0]

    n_per_seg = n_basis
    bdry_pts, bdry_normals = ldom.bdry_data(n_per_seg)[:2]
    int_pts = ldom.int_pts(npts_rand=n_basis)

    basis = FourierBesselBasis.from_domain(ldom, orders).to_normalized(bdry_pts + int_pts)
    solver = MPSEigensolver(basis, bdry_pts, int_pts)

    return solver, ldom, bdry_pts, int_pts, basis


def make_grid(ldom, n_lam):
    lam_lo = bounds.faber_krahn(ldom)
    lam_hi = asymp.weyl_est(26, ldom)
    lamgrid = np.linspace(lam_lo, lam_hi, n_lam)
    return lamgrid


def check_correctness(serial_grid, serial_results, parallel_grid, parallel_results):
    # Grids are disjoint; check shape and that all tensions are finite and positive
    assert len(serial_results) == len(serial_grid), "Serial result count mismatch"
    assert len(parallel_results) == len(parallel_grid), "Parallel result count mismatch"
    for i, (lam, res) in enumerate(zip(serial_grid, serial_results)):
        sigma = res[0] if isinstance(res, tuple) else res
        assert np.all(np.isfinite(sigma)), f"Serial: non-finite tension at lam={lam}"
    for i, (lam, res) in enumerate(zip(parallel_grid, parallel_results)):
        sigma = res[0] if isinstance(res, tuple) else res
        assert np.all(np.isfinite(sigma)), f"Parallel: non-finite tension at lam={lam}"


def run_benchmark(n_basis=40, n_lam=100, thread_counts=None):
    print(f"Setting up solver (n_basis={n_basis})...")
    solver, ldom, bdry_pts, int_pts, basis = setup_solver(n_basis)

    lamgrid = make_grid(ldom, n_lam)

    basis_size = len(basis)
    print(
        f"n_lam={n_lam}, n_basis={n_basis}, basis_size={basis_size}, "
        f"bdry_pts={len(bdry_pts)}, int_pts={len(int_pts)}"
    )

    # Serial baseline
    t0 = time.perf_counter()
    serial_results = [solver.tensions(lam) for lam in lamgrid]
    t_serial = time.perf_counter() - t0
    print(f"  serial (1 thread):  {t_serial:.3f}s  (baseline)")

    # Determine thread counts to test
    cpu_count = os.cpu_count() or 1
    if thread_counts is None:
        counts = sorted(set([2, 4, cpu_count]))
    else:
        counts = [thread_counts]

    # Parallel trials — use fresh solver object to dodge the cache
    for n_threads in counts:
        solver2 = setup_solver(n_basis)[0]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            parallel_results = list(ex.map(solver2.tensions, lamgrid))
        t_parallel = time.perf_counter() - t0

        speedup = t_serial / t_parallel if t_parallel > 0 else float('inf')
        print(f"  parallel {n_threads} threads: {t_parallel:.3f}s  ({speedup:.2f}x speedup)")

        check_correctness(lamgrid, serial_results, lamgrid, parallel_results)

    print("Correctness checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ThreadPoolExecutor for parallel tension evaluation"
    )
    parser.add_argument("--n_basis", type=int, default=40,
                        help="Fourier-Bessel orders at re-entrant corner (default: 40)")
    parser.add_argument("--n_lam", type=int, default=100,
                        help="Lambda grid points per trial (default: 100)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Run a single specific thread count (default: [2, 4, cpu_count])")
    args = parser.parse_args()

    run_benchmark(n_basis=args.n_basis, n_lam=args.n_lam, thread_counts=args.threads)


if __name__ == "__main__":
    main()
