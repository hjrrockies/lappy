"""
Performance profile of the boundary-integral (Rellich/Cauchy) normalization code
(lappy/cauchy.py, lappy/rellich.py), under the shape-optimization-style "cold start"
workload: a fresh Domain/basis/MPSEigensolver each trial, normalizing a handful of
eigenvalues per trial.

Scope: lappy.cauchy / lappy.rellich only -- NOT the GSVD eigensolve. All timings below
call solver._cauchy_gram(lam) directly (the Rellich Gram-matrix assembly itself) rather
than solving for real eigenvalues first, so results aren't diluted by GSVD/root-finding
cost that's out of scope here. `lam` values are synthetic (a linspace of plausible
spectral-parameter magnitudes) -- the Gram assembly is valid at any lam, not just true
eigenvalues.

As of the Kress-style graded-mesh rewrite (docs/rellich_hadamard_mps.pdf Sec. 6.1,
replacing the old SS/SR/RS/RR singularity-subtraction quadrature in lappy.cauchy),
rellich_gram_basis makes exactly one basis_cauchy_data call per Gram build regardless of
corner count or per-corner mode count -- the redundant-evaluation failure modes Sec. 3
used to quantify (per-exponent-group full-basis re-evaluation, duplicate node sets across
kernels, missing jacgauss caching) no longer apply by construction. Section 3 below is
now just a single sanity check confirming that invariant continues to hold.

Sections (each independently toggleable via CLI flags, default: run all):
  1. Macro wall-clock timing: cold-start (fresh solver + one Gram build) and
     warm-many-eigenvalues (one solver, many distinct lam), across corner count,
     basis size, and MultiBasis (FB+FS).
  2. cProfile hotspot breakdown of one _cauchy_gram(lam) call, small vs. many-corner.
  3. Sanity check: exactly one basis_cauchy_data call per Gram build.
"""
import sys
import time
import pstats
import cProfile
import pathlib
import argparse
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from lappy.geometry import Polygon
from lappy.bases import FourierBesselBasis, FundamentalBasis
from lappy.mps import MPSEigensolver
from lappy.rellich import rellich_gram_basis
from lappy.asymp import weyl_est


# ── domain/basis construction ────────────────────────────────────────────────

def make_domain(n_corners, bc='dir', radius=1.0):
    """A regular n_corners-gon. For n_corners > 4 this has genuinely singular
    (non-integer alpha) corners -- exactly the case the SS/SR/RS/RR machinery exists
    for -- so it's a representative, controllable-corner-count stress case."""
    theta = np.linspace(0, 2*np.pi, n_corners + 1)[:-1]
    verts = radius*np.exp(1j*theta)
    return Polygon(verts, bc=bc)


def make_basis(domain, order_per_corner, use_fs=False, n_fs=4):
    n_corners = len(domain.corners)
    basis = FourierBesselBasis.from_domain(domain, orders=[order_per_corner]*n_corners)
    if use_fs:
        r_fs = 3.0*np.max(np.abs(domain.corners))
        theta = np.linspace(0, 2*np.pi, n_fs + 1)[:-1]
        sources = r_fs*np.exp(1j*theta)
        basis = basis + FundamentalBasis(sources, orders=2)
    return basis


def make_solver(n_corners, order_per_corner, bc='dir', use_fs=False,
                bulk_mult=2, bulk_min_per_seg=4, margin=2.0):
    domain = make_domain(n_corners, bc=bc)
    basis = make_basis(domain, order_per_corner, use_fs=use_fs)
    lam_max = weyl_est(6, domain)
    solver = MPSEigensolver.from_domain(
        domain, lam_max=lam_max, basis=basis,
        rellich_mult=bulk_mult, rellich_min_per_seg=bulk_min_per_seg,
        rellich_margin=margin,
    )
    return solver, domain, basis


def synthetic_lams(domain, n_lam):
    """Plausible spectral-parameter magnitudes, not real eigenvalues (unneeded here --
    Gram assembly is valid at any lam)."""
    return np.linspace(weyl_est(2, domain), weyl_est(30, domain), n_lam)


# ── Section 1: macro wall-clock timing ───────────────────────────────────────

def bench_cold_start(corner_counts, order_per_corner=6, n_lam=3, use_fs=False, **knobs):
    print(f"\n=== cold start (fresh solver each trial, {n_lam} Gram builds/trial, "
          f"order_per_corner={order_per_corner}, use_fs={use_fs}) ===")
    for nc in corner_counts:
        t0 = time.perf_counter()
        solver, domain, basis = make_solver(nc, order_per_corner, use_fs=use_fs, **knobs)
        t_setup = time.perf_counter() - t0

        lams = synthetic_lams(domain, n_lam)
        t0 = time.perf_counter()
        for lam in lams:
            G = solver._cauchy_gram(float(lam))
            assert np.all(np.isfinite(G)), f"non-finite Gram at n_corners={nc}, lam={lam}"
        t_grams = time.perf_counter() - t0

        print(f"  n_corners={nc:3d}  N_basis={len(basis):4d}  "
              f"setup={t_setup:.3f}s  {n_lam} Gram builds={t_grams:.3f}s  "
              f"({t_grams/n_lam*1000:.1f}ms/build)  total={t_setup+t_grams:.3f}s")


def bench_basis_size(orders, n_corners=8, n_lam=3, **knobs):
    print(f"\n=== basis size sweep (n_corners={n_corners}, {n_lam} Gram builds/trial) ===")
    for order in orders:
        t0 = time.perf_counter()
        solver, domain, basis = make_solver(n_corners, order, **knobs)
        t_setup = time.perf_counter() - t0

        lams = synthetic_lams(domain, n_lam)
        t0 = time.perf_counter()
        for lam in lams:
            solver._cauchy_gram(float(lam))
        t_grams = time.perf_counter() - t0

        print(f"  order_per_corner={order:3d}  N_basis={len(basis):4d}  "
              f"setup={t_setup:.3f}s  {n_lam} Gram builds={t_grams:.3f}s  "
              f"({t_grams/n_lam*1000:.1f}ms/build)")


def bench_multibasis(n_corners=8, order_per_corner=6, n_lam=3, **knobs):
    print(f"\n=== FB-only vs. FB+FS MultiBasis (n_corners={n_corners}, "
          f"order_per_corner={order_per_corner}) ===")
    for use_fs, label in [(False, "FB only"), (True, "FB + FS")]:
        solver, domain, basis = make_solver(n_corners, order_per_corner, use_fs=use_fs, **knobs)
        lams = synthetic_lams(domain, n_lam)
        t0 = time.perf_counter()
        for lam in lams:
            solver._cauchy_gram(float(lam))
        t_grams = time.perf_counter() - t0
        print(f"  {label:8s}  N_basis={len(basis):4d}  "
              f"{n_lam} Gram builds={t_grams:.3f}s  ({t_grams/n_lam*1000:.1f}ms/build)")


def bench_quadrature_knobs(knob_configs, n_corners=8, order_per_corner=6, n_lam=3):
    print(f"\n=== quadrature knob sweep (n_corners={n_corners}, order_per_corner={order_per_corner}) ===")
    for knobs in knob_configs:
        solver, domain, basis = make_solver(n_corners, order_per_corner, **knobs)
        lams = synthetic_lams(domain, n_lam)
        t0 = time.perf_counter()
        for lam in lams:
            solver._cauchy_gram(float(lam))
        t_grams = time.perf_counter() - t0
        print(f"  {knobs}  {n_lam} Gram builds={t_grams:.3f}s  ({t_grams/n_lam*1000:.1f}ms/build)")


def bench_warm_many_eigs(n_lams=(5, 20, 50), n_corners=8, order_per_corner=6, **knobs):
    print(f"\n=== warm, many eigenvalues (one solver, n_corners={n_corners}, "
          f"order_per_corner={order_per_corner}) ===")
    solver, domain, basis = make_solver(n_corners, order_per_corner, **knobs)
    max_n = max(n_lams)
    all_lams = synthetic_lams(domain, max_n)
    for n in n_lams:
        # fresh solver so _cauchy_gram's instance_lru_cache doesn't roll over between trials
        solver, domain, basis = make_solver(n_corners, order_per_corner, **knobs)
        lams = synthetic_lams(domain, n)
        t0 = time.perf_counter()
        for lam in lams:
            solver._cauchy_gram(float(lam))
        t_total = time.perf_counter() - t0
        print(f"  n_lam={n:3d}  total={t_total:.3f}s  ({t_total/n*1000:.2f}ms/lam)")


# ── Section 2: cProfile hotspot breakdown ────────────────────────────────────

def profile_hotspots(n_corners, order_per_corner=6, label="", save_prof=None):
    print(f"\n=== cProfile hotspots: {label} (n_corners={n_corners}, "
          f"order_per_corner={order_per_corner}) ===")
    solver, domain, basis = make_solver(n_corners, order_per_corner)
    lam = float(synthetic_lams(domain, 1)[0])

    profiler = cProfile.Profile()
    profiler.enable()
    solver._cauchy_gram(lam)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)
    if save_prof:
        stats.dump_stats(save_prof)
        print(f"  (saved: {save_prof})")


# ── Section 3: sanity check ───────────────────────────────────────────────────

@contextmanager
def _patch(module, name, make_wrapper):
    original = getattr(module, name)
    setattr(module, name, make_wrapper(original))
    try:
        yield
    finally:
        setattr(module, name, original)


def check_single_basis_evaluation(n_corners=8, order_per_corner=6):
    """Confirms the Kress-graded-mesh rewrite's central invariant: rellich_gram_basis makes
    exactly one basis_cauchy_data call per Gram build, regardless of corner count or
    per-corner mode count (the old SS/SR/RS/RR quadrature made one call per distinct
    exponent group -- see git history for this file's previous measure_full_basis_waste/
    measure_redundant_rr/measure_jacgauss_reuse, which quantified that cost)."""
    print(f"\n=== single-basis-evaluation sanity check (n_corners={n_corners}, "
          f"order_per_corner={order_per_corner}) ===")
    solver, domain, basis = make_solver(n_corners, order_per_corner)
    lam = float(synthetic_lams(domain, 1)[0])

    calls = []

    def make_wrapper(original):
        def wrapped(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        return wrapped

    import lappy.rellich as rellich_mod
    with _patch(rellich_mod, 'basis_cauchy_data', make_wrapper):
        rellich_gram_basis(solver.basis, lam, solver._cauchy_data)

    print(f"  {len(calls)} basis_cauchy_data call(s) for this Gram build (expected: 1)")
    assert len(calls) == 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sections', nargs='+',
                   choices=['macro', 'profile', 'counts'], default=['macro', 'profile', 'counts'])
    p.add_argument('--corner_counts', type=int, nargs='+', default=[4, 8, 16, 32])
    p.add_argument('--orders', type=int, nargs='+', default=[3, 6, 12, 24])
    p.add_argument('--n_lam', type=int, default=3)
    p.add_argument('--save_prof', type=str, default=None,
                   help='if set, saves cProfile .prof files with this prefix')
    args = p.parse_args()

    if 'macro' in args.sections:
        bench_cold_start(args.corner_counts, n_lam=args.n_lam)
        bench_basis_size(args.orders, n_lam=args.n_lam)
        bench_multibasis(n_lam=args.n_lam)
        bench_quadrature_knobs([
            dict(bulk_mult=2, bulk_min_per_seg=4, margin=2.0),
            dict(bulk_mult=4, bulk_min_per_seg=8, margin=4.0),
            dict(bulk_mult=2, bulk_min_per_seg=4, margin=4.0),
        ], n_lam=args.n_lam)
        bench_warm_many_eigs()

    if 'profile' in args.sections:
        small_prof = f"{args.save_prof}_small.prof" if args.save_prof else None
        big_prof = f"{args.save_prof}_big.prof" if args.save_prof else None
        profile_hotspots(4, order_per_corner=6, label="small/simple", save_prof=small_prof)
        profile_hotspots(32, order_per_corner=12, label="many-corner/large-basis", save_prof=big_prof)

    if 'counts' in args.sections:
        check_single_basis_evaluation()
        check_single_basis_evaluation(n_corners=32, order_per_corner=12)


if __name__ == "__main__":
    main()
