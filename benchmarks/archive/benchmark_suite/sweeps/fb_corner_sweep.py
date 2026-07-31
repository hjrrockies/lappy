"""
Sweep 1: FB corner order allocation
====================================
Question: How should FB orders be distributed across corners?

Variables
---------
fb_strategy ∈ {uniform, angle_weighted, singular_only, singular_angle_weighted}

Fixed
-----
n_fb=100, n_fs=0, bdry_factor=2, int_factor=1, rtol=1e-12, ppl=10, n_eigs=10

Domains
-------
rect(2,1), L_shape, iso_right_tri, reg_ngon(6), disk_sector(pi/3), disk_sector(4pi/3)
"""
import numpy as np
from ..domains import DOMAINS
from ..runner import BenchmarkConfig, run_benchmark, SweepDomainEntry

SWEEP_NAME = 'fb_corner'

DOMAIN_ENTRIES = [
    SweepDomainEntry('rect',         {'L': 2.0, 'H': 1.0}),
    SweepDomainEntry('L_shape',      {}),
    SweepDomainEntry('iso_right_tri',{'l': 1.0}),
    SweepDomainEntry('reg_ngon',     {'N': 6}),
    SweepDomainEntry('disk_sector',  {'theta': np.pi / 3}),
    SweepDomainEntry('disk_sector',  {'theta': 4 * np.pi / 3}),
]

FB_STRATEGIES = ['uniform', 'angle_weighted', 'singular_only', 'singular_angle_weighted']

FIXED = dict(
    n_fb=100,
    n_fs=0,
    bdry_pts_factor=2.0,
    int_pts_factor=1.0,
    rtol=1e-12,
    ppl=10,
    n_eigs=10,
)


def make_configs(domain_filter=None):
    configs = []
    for entry in DOMAIN_ENTRIES:
        if domain_filter and entry.domain_name not in domain_filter:
            continue
        fixed = {**FIXED, **entry.fixed_overrides}
        strategies = entry.sweep_overrides.get('fb_strategy', FB_STRATEGIES)
        for strategy in strategies:
            cfg = BenchmarkConfig(
                domain_name=entry.domain_name,
                domain_params=entry.domain_params,
                fb_strategy=strategy,
                **fixed,
            )
            configs.append(cfg)
    return configs


def run(outdir, domain_filter=None, n_workers=1, dry_run=False):
    from ..results import save_result
    from concurrent.futures import ProcessPoolExecutor
    import os

    configs = make_configs(domain_filter)

    results = []
    if dry_run:
        for cfg in configs:
            print(f"[DRY RUN] {SWEEP_NAME}: {cfg.domain_name} params={cfg.domain_params} fb_strategy={cfg.fb_strategy}")
        return results

    def _run_one(args):
        spec, cfg = args
        return run_benchmark(spec, cfg)

    tasks = [(DOMAINS[cfg.domain_name], cfg) for cfg in configs]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            result_list = list(pool.map(_run_one, tasks))
    else:
        result_list = [_run_one(t) for t in tasks]

    for cfg, result in zip(configs, result_list):
        tag = f"{cfg.fb_strategy}"
        # encode domain params in tag
        param_str = '_'.join(f"{k}{v}" for k, v in sorted(cfg.domain_params.items()))
        if param_str:
            tag = f"{param_str}_{tag}"
        path = save_result(result, outdir, SWEEP_NAME, extra_tag=tag)
        print(f"Saved: {path}")
        results.append(result)

    return results
