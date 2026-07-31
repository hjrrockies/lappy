"""
Serialize benchmark configs to a pickle file for a Slurm array job.

Usage (from repo root):
    .venv/bin/python -m benchmarks.benchmark_suite.slurm.prepare_jobs \\
        --sweep basis_size [--domains rect,L_shape] [--outdir benchmarks/results]

Prints ARRAY_SIZE=N and CONFIG_FILE=path for the submission script to parse.
"""
import argparse
import pathlib
import pickle

SWEEP_NAMES = ['fb_corner', 'fs_placement', 'basis_size', 'regularization', 'n_eigs']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', required=True, choices=SWEEP_NAMES)
    parser.add_argument('--domains', default=None,
                        help='Comma-separated domain filter (default: all)')
    parser.add_argument('--outdir', default='benchmarks/results',
                        help='Directory where results will be written by run_job.py')
    parser.add_argument('--jobs_dir', default='benchmarks/benchmark_suite/slurm/jobs',
                        help='Directory to write the config pickle')
    args = parser.parse_args()

    from benchmarks.benchmark_suite.sweeps import (
        fb_corner_sweep, fs_placement_sweep, basis_size_sweep,
        regularization_sweep, n_eigs_sweep,
    )
    SWEEPS = {
        'fb_corner':      fb_corner_sweep,
        'fs_placement':   fs_placement_sweep,
        'basis_size':     basis_size_sweep,
        'regularization': regularization_sweep,
        'n_eigs':         n_eigs_sweep,
    }

    domain_filter = set(args.domains.split(',')) if args.domains else None
    configs = SWEEPS[args.sweep].make_configs(domain_filter)

    jobs_dir = pathlib.Path(args.jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    config_file = jobs_dir / f'{args.sweep}_configs.pkl'

    with open(config_file, 'wb') as f:
        pickle.dump({'configs': configs, 'sweep': args.sweep, 'outdir': args.outdir}, f)

    print(f"Wrote {len(configs)} configs to {config_file}")
    print(f"ARRAY_SIZE={len(configs)}")
    print(f"CONFIG_FILE={config_file.resolve()}")


if __name__ == '__main__':
    main()
