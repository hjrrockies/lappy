"""
Sweep 5: Scaling with n_eigs
==============================
Question: How large a basis is needed to compute k eigenvalues accurately?

Variables
---------
A) n_eigs ∈ {5, 10, 15, 20, 25}
B) n_basis ∈ {50, 100, 200, 400}
   fb_fraction = 0.5 for polygonal domains, 0.0 for smooth domains (disk, ellipse, stadium)
   Use fixed_overrides={'fb_fraction': 0.0} on a SweepDomainEntry for smooth domains.

Fixed
-----
fb_fraction=0.5, rtol=1e-12, ppl=10

Domains
-------
rect(2,1), L_shape, GWW1
"""
import numpy as np
from ..domains import DOMAINS
from ..runner import BenchmarkConfig, run_benchmark, SweepDomainEntry

SWEEP_NAME = 'n_eigs'

DOMAIN_ENTRIES = [
    SweepDomainEntry('rect',    {'L': 2.0, 'H': 1.0}),
    SweepDomainEntry('L_shape', {}),
    SweepDomainEntry('GWW1',    {}),
    # Smooth domain example: SweepDomainEntry('disk', {'r': 1.0}, fixed_overrides={'fb_fraction': 0.0}),
]

N_EIGS_VALUES  = [5, 10, 15, 20, 25]
N_BASIS_VALUES = [50, 100, 200, 400]

FIXED = dict(
    fb_fraction=0.5,
    rtol=1e-12,
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
        fb_fraction = fixed.pop('fb_fraction')
        n_eigs_values  = entry.sweep_overrides.get('n_eigs',  N_EIGS_VALUES)
        n_basis_values = entry.sweep_overrides.get('n_basis', N_BASIS_VALUES)
        for n_eigs in n_eigs_values:
            for n_basis in n_basis_values:
                n_fb = round(fb_fraction * n_basis)
                n_fs = n_basis - n_fb
                cfg = BenchmarkConfig(
                    domain_name=entry.domain_name,
                    domain_params=entry.domain_params,
                    n_eigs=n_eigs,
                    n_fb=n_fb,
                    n_fs=n_fs,
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
            print(f"[DRY RUN] {SWEEP_NAME}: {cfg.domain_name} n_eigs={cfg.n_eigs} n_basis={cfg.n_fb+cfg.n_fs}")
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
        tag = f"k{cfg.n_eigs}_nb{cfg.n_fb + cfg.n_fs}"
        if param_str:
            tag = f"{param_str}_{tag}"
        path = save_result(result, outdir, SWEEP_NAME, extra_tag=tag)
        print(f"Saved: {path}")
        results.append(result)

    return results
