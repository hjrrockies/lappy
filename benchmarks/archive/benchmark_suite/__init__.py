"""Benchmark suite for lappy eigenvalue solver configuration studies."""
from .domains import DOMAINS, DomainSpec
from .runner import BenchmarkConfig, BenchmarkResult, run_benchmark
from .results import save_result, load_result, results_to_dataframe
