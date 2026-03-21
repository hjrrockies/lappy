"""
Run a single benchmark config by index (called by each Slurm array task).

Usage:
    .venv/bin/python -m benchmarks.benchmark_suite.slurm.run_job \\
        --config_file /path/to/basis_size_configs.pkl [--index 42]

If --index is omitted, $SLURM_ARRAY_TASK_ID is used.
"""
import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--index', type=int, default=None,
                        help='Config index; defaults to $SLURM_ARRAY_TASK_ID')
    args = parser.parse_args()

    idx = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])

    with open(args.config_file, 'rb') as f:
        data = pickle.load(f)

    cfg    = data['configs'][idx]
    sweep  = data['sweep']
    outdir = data['outdir']

    from benchmarks.benchmark_suite.domains import DOMAINS
    from benchmarks.benchmark_suite.runner import run_benchmark
    from benchmarks.benchmark_suite.results import save_result

    spec   = DOMAINS[cfg.domain_name]
    result = run_benchmark(spec, cfg)

    param_str = '_'.join(f"{k}{v}" for k, v in sorted(cfg.domain_params.items()))
    tag = f"j{idx:04d}"
    if param_str:
        tag = f"{param_str}_{tag}"

    path = save_result(result, outdir, sweep, extra_tag=tag)
    print(f"[{idx}] {cfg.domain_name}: saved {path}")


if __name__ == '__main__':
    main()
