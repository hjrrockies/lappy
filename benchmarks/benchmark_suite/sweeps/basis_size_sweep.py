"""
Sweep 3: Total basis size and FB/FS ratio
==========================================
Question: How large must the basis be? What FB/FS mix is best?

Variables
---------
A) n_basis ∈ {20, 40, 80, 160, 320, 640}
B) fb_fraction ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
   → n_fb = round(fb_fraction * n_basis), n_fs = n_basis - n_fb

Fixed
-----
default allocation strategies, rtol=1e-12, n_eigs=10, ppl=10

Domains
-------
All domains with reference eigenvalues:
rect, L_shape, iso_right_tri, eq_tri, disk_sector, GWW1
"""
import numpy as np
from ..domains import DOMAINS
from ..runner import BenchmarkConfig, run_benchmark, SweepDomainEntry

SWEEP_NAME = 'basis_size'

# Only domains with reference eigenvalues
rect = SweepDomainEntry('rect', {'L': 2.0, 'H': 1.0}, 
                        sweep_overrides={'n_basis':np.arange(10,100,10)})
L_shape = SweepDomainEntry('L_shape', {}, 
                           sweep_overrides={'n_basis':np.arange(20,200,20)})
iso_right_tri = SweepDomainEntry('iso_right_tri', {'l': 1.0}, 
                                 sweep_overrides={'n_basis':np.arange(10,120,10)})
equilateral_tri = SweepDomainEntry('eq_tri', {},
                                   sweep_overrides={'n_basis':np.arange(10,120,10)})
disk_sector = SweepDomainEntry('disk_sector', {'theta': np.pi*np.sqrt(2)/2})
DOMAIN_ENTRIES = [
    rect,
    L_shape,
    iso_right_tri,
    equilateral_tri,
    disk_sector,
    SweepDomainEntry('GWW1',            {}, sweep_overrides={'rtol':1e-12}),
]

N_BASIS_VALUES = [20, 40, 80, 160, 320, 640]
FB_FRACTIONS   = [0.0, 0.25, 0.5, 0.75, 1.0]

FIXED = dict(
    n_eigs=10,
    rtol=1e-14,
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
        n_basis_values = entry.sweep_overrides.get('n_basis', N_BASIS_VALUES)
        fb_fractions = entry.sweep_overrides.get('fb_fraction', FB_FRACTIONS)
        for n_basis in n_basis_values:
            for fb_fraction in fb_fractions:
                n_fb = round(fb_fraction * n_basis)
                n_fs = n_basis - n_fb
                if n_fb == 0 and n_fs == 0:
                    continue
                cfg = BenchmarkConfig(
                    domain_name=entry.domain_name,
                    domain_params=entry.domain_params,
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
            print(f"[DRY RUN] {SWEEP_NAME}: {cfg.domain_name} n_fb={cfg.n_fb} n_fs={cfg.n_fs}")
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
        tag = f"nb{cfg.n_fb + cfg.n_fs}_fb{cfg.n_fb}"
        if param_str:
            tag = f"{param_str}_{tag}"
        path = save_result(result, outdir, SWEEP_NAME, extra_tag=tag)
        print(f"Saved: {path}")
        results.append(result)

    return results
