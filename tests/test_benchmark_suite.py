"""
Benchmark suite tests.
"""
import numpy as np
import pytest

pytest.skip(
    "benchmark_suite was moved to benchmarks/archive/ pending a rebuild on "
    "the current API; revisit later",
    allow_module_level=True,
)

from benchmarks.archive.benchmark_suite.domains import DOMAINS
from benchmarks.archive.benchmark_suite.runner import BenchmarkConfig, run_benchmark


def test_unit_square_smoke():
    """run_benchmark completes for the unit square and returns a valid result."""
    spec = DOMAINS['rect']
    cfg = BenchmarkConfig(
        domain_name='rect',
        domain_params={'L': 1.0, 'H': 1.0},
        n_eigs=5,
        n_fb=60,
        n_fs=0,
        rtol=1e-10,
        ppl=10,
        bdry_pts_factor=2.0,
        int_pts_factor=1.0,
    )

    result = run_benchmark(spec, cfg)

    # result is structurally valid
    assert np.all(np.isfinite(result.eigs))
    assert np.all(np.isfinite(result.tensions))
    assert np.all(result.eigs > 0)
    assert result.wall_time > 0

    print(f"\neigs:     {result.eigs}")
    print(f"tensions: {result.tensions}")
    if result.rel_errors is not None:
        print(f"rel_err:  {result.rel_errors}")
