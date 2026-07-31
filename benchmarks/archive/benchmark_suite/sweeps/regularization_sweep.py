"""
Sweep 4: rtol effect on eigenvalue accuracy
============================================
Question: How much regularization is needed / can be afforded?

Variables
---------
rtol ∈ {1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16}

Fixed
-----
n_fb=100, n_fs=100, n_eigs=10, ppl=10

Domains
-------
rect(2,1), L_shape, GWW1
"""
import numpy as np
from ..domains import DOMAINS
from ..runner import BenchmarkConfig, run_benchmark, SweepDomainEntry

SWEEP_NAME = 'regularization'

DOMAIN_ENTRIES = [
    SweepDomainEntry('rect',    {'L': 2.0, 'H': 1.0}),
    SweepDomainEntry('L_shape', {}),
    SweepDomainEntry('GWW1',    {}),
    # To add a domain with a tighter rtol range:
    # SweepDomainEntry('disk', {'r': 1.0}, sweep_overrides={'rtol': [1e-10, 1e-12, 1e-14]}),
]

RTOL_VALUES = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]

FIXED = dict(
    n_fb=100,
    n_fs=100,
    n_eigs=10,
    ppl=10,
    bdry_pts_factor=2.0,
    int_pts_factor=1.0,
)


def make_configs(domain_filter=None):
    configs = []
    for entry in DOMAIN_ENTRIES:
        if domain_filter and entry.domain_name not in domain_filter:
            continue
        fixed = {**FIXED, **entry.fixed_overrides}
        rtol_values = entry.sweep_overrides.get('rtol', RTOL_VALUES)
        for rtol in rtol_values:
            cfg = BenchmarkConfig(
                domain_name=entry.domain_name,
                domain_params=entry.domain_params,
                rtol=rtol,
                **fixed,
            )
            configs.append(cfg)
    return configs


def run(outdir, domain_filter=None, n_workers=1, dry_run=False):
    from ..results import save_result
    from concurrent.futures import ProcessPoolExecutor

    configs = make_configs(domain_filter)

    if dry_run:
        for cfg in configs:
            print(f"[DRY RUN] {SWEEP_NAME}: {cfg.domain_name} rtol={cfg.rtol:.0e}")
        return []

    def _run_one(args):
        spec, cfg = args
        return run_benchmark(spec, cfg)

    tasks = [(DOMAINS[cfg.domain_name], cfg) for cfg in configs]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            result_list = list(pool.map(_run_one, tasks))
    else:
        result_list = [_run_one(t) for t in tasks]

    results = []
    for cfg, result in zip(configs, result_list):
        param_str = '_'.join(f"{k}{v}" for k, v in sorted(cfg.domain_params.items()))
        tag = f"rtol{cfg.rtol:.0e}"
        if param_str:
            tag = f"{param_str}_{tag}"
        path = save_result(result, outdir, SWEEP_NAME, extra_tag=tag)
        print(f"Saved: {path}")
        results.append(result)

    return results
