"""
Benchmark: n_workers speedup in MPSEigensolver.solve_interval and tensions_batch.

Tests the built-in parallel paths added to mps.py:
  - solver.tensions_batch(lams, n_workers=N)
  - solver.solve_interval(a, b, n_pts, n_workers=N)
"""

import os
import sys
import time
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from lappy import geometry, asymp, bounds
from lappy.bases import FourierBesselBasis
from lappy.mps import MPSEigensolver


def make_solver(n_basis=40):
    ldom = geometry.L_shape()
    bdry_pts, bdry_normals = ldom.bdry_data(n_basis)[:2]
    int_pts = ldom.int_pts(npts_rand=n_basis)
    basis = FourierBesselBasis.from_domain(ldom, [n_basis, 0, 0, 0, 0, 0]).to_normalized(
        bdry_pts + int_pts
    )
    return MPSEigensolver(basis, bdry_pts, int_pts), ldom


def bench_tensions_batch(n_basis=40, n_lam=200, thread_counts=None):
    print(f"\n=== tensions_batch  (n_basis={n_basis}, n_lam={n_lam}) ===")
    cpu = os.cpu_count() or 1
    counts = thread_counts or sorted({2, 4, cpu})

    solver, ldom = make_solver(n_basis)
    lam_lo = bounds.faber_krahn(ldom)
    lam_hi = asymp.weyl_est(30, ldom)
    lams = list(np.linspace(lam_lo, lam_hi, n_lam))

    # warm up the matrix-build cache (A_B / A_I calls)
    _ = solver.tensions(lams[0])

    # serial baseline — fresh solver to avoid cache
    s0, _ = make_solver(n_basis)
    t0 = time.perf_counter()
    r_serial = s0.tensions_batch(lams, n_workers=1)
    t_serial = time.perf_counter() - t0
    print(f"  serial (n_workers=1):  {t_serial:.3f}s  (baseline)")

    for n in counts:
        sp, _ = make_solver(n_basis)
        t0 = time.perf_counter()
        r_par = sp.tensions_batch(lams, n_workers=n)
        t_par = time.perf_counter() - t0
        speedup = t_serial / t_par if t_par > 0 else float("inf")
        # correctness: results should be finite (cross-instance tension values aren't
        # comparable because to_normalized uses a randomized normalization)
        assert all(np.all(np.isfinite(r)) for r in r_par), "tensions_batch returned non-finite"
        print(f"  parallel n_workers={n}:  {t_par:.3f}s  ({speedup:.2f}x)")


def bench_solve_interval(n_basis=40, n_pts=150, thread_counts=None):
    print(f"\n=== solve_interval  (n_basis={n_basis}, n_pts={n_pts}) ===")
    cpu = os.cpu_count() or 1
    counts = thread_counts or sorted({2, 4, cpu})

    solver_s, ldom = make_solver(n_basis)
    lam_lo = bounds.faber_krahn(ldom)
    lam_hi = asymp.weyl_est(20, ldom)

    t0 = time.perf_counter()
    eigs_s, mults_s, _ = solver_s.solve_interval(lam_lo, lam_hi, n_pts)
    t_serial = time.perf_counter() - t0
    print(f"  serial (n_workers=None):  {t_serial:.3f}s  (baseline, {len(eigs_s)} eigs)")

    for n in counts:
        solver_p, _ = make_solver(n_basis)
        t0 = time.perf_counter()
        eigs_p, mults_p, _ = solver_p.solve_interval(lam_lo, lam_hi, n_pts, n_workers=n)
        t_par = time.perf_counter() - t0
        speedup = t_serial / t_par if t_par > 0 else float("inf")
        assert np.allclose(eigs_s, eigs_p, rtol=1e-6), (
            f"solve_interval eig mismatch at n_workers={n}:\n  serial: {eigs_s}\n  parallel: {eigs_p}"
        )
        print(f"  parallel n_workers={n}:  {t_par:.3f}s  ({speedup:.2f}x)  {len(eigs_p)} eigs")

    print("  correctness OK")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n_basis", type=int, default=40)
    p.add_argument("--n_lam", type=int, default=200)
    p.add_argument("--n_pts", type=int, default=150)
    p.add_argument("--threads", type=int, nargs="+", default=None)
    args = p.parse_args()

    bench_tensions_batch(args.n_basis, args.n_lam, args.threads)
    bench_solve_interval(args.n_basis, args.n_pts, args.threads)
