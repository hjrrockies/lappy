"""
Sweep 2: FS source distance and density
========================================
Question: How far from the boundary and how dense?

Variables
---------
A) fs_d_strategy='inradius_fraction', d_scale ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5}
B) fs_seg_strategy ∈ {length_weighted, uniform}

Fixed
-----
n_fs=100, n_fb=0, rtol=1e-12, n_eigs=10

Domains
-------
disk, rect(2,1), GWW1, chevron(1,2)
"""
import numpy as np
from ..domains import DOMAINS
from ..runner import BenchmarkConfig, run_benchmark, SweepDomainEntry

SWEEP_NAME = 'fs_placement'

DOMAIN_ENTRIES = [
    SweepDomainEntry('disk',    {'r': 1.0}),
    SweepDomainEntry('rect',    {'L': 2.0, 'H': 1.0}),
    SweepDomainEntry('GWW1',    {}),
    SweepDomainEntry('chevron', {'h1': 1.0, 'h2': 2.0}),
]

D_SCALES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
SEG_STRATEGIES = ['length_weighted', 'uniform']

FIXED = dict(
    n_fb=0,
    n_fs=100,
    fs_d_strategy='inradius_fraction',
    rtol=1e-12,
    ppl=10,
    n_eigs=10,
    bdry_pts_factor=2.0,
    int_pts_factor=1.0,
)


def make_configs(domain_filter=None):
    configs = []
    for entry in DOMAIN_ENTRIES:
        if domain_filter and entry.domain_name not in domain_filter:
            continue
        fixed = {**FIXED, **entry.fixed_overrides}
        d_scales = entry.sweep_overrides.get('d_scale', D_SCALES)
        seg_strategies = entry.sweep_overrides.get('seg_strategy', SEG_STRATEGIES)
        for d_scale in d_scales:
            for seg_strategy in seg_strategies:
                cfg = BenchmarkConfig(
                    domain_name=entry.domain_name,
                    domain_params=entry.domain_params,
                    fs_d_scale=d_scale,
                    fs_seg_strategy=seg_strategy,
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
            print(f"[DRY RUN] {SWEEP_NAME}: {cfg.domain_name} d_scale={cfg.fs_d_scale} seg={cfg.fs_seg_strategy}")
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
        tag = f"d{cfg.fs_d_scale}_{cfg.fs_seg_strategy}"
        if param_str:
            tag = f"{param_str}_{tag}"
        path = save_result(result, outdir, SWEEP_NAME, extra_tag=tag)
        print(f"Saved: {path}")
        results.append(result)

    return results
