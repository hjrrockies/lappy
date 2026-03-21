#!/usr/bin/env python3
"""
Master CLI script for the lappy benchmark suite.

Usage examples
--------------
# Run all sweeps, all domains
python run_all.py --outdir results/

# Only basis-size sweep, only rect and L_shape
python run_all.py --sweeps basis_size --domains rect,L_shape --outdir /tmp/bench_test

# Dry run: print all configs without executing
python run_all.py --dry_run

# Parallel execution
python run_all.py --n_workers 4 --outdir results/
"""
import argparse
import os
import sys

# Allow running from this directory or from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.benchmark_suite.sweeps import (
    fb_corner_sweep,
    fs_placement_sweep,
    basis_size_sweep,
    regularization_sweep,
    n_eigs_sweep,
)

SWEEPS = {
    'fb_corner':      fb_corner_sweep,
    'fs_placement':   fs_placement_sweep,
    'basis_size':     basis_size_sweep,
    'regularization': regularization_sweep,
    'n_eigs':         n_eigs_sweep,
}


def main():
    parser = argparse.ArgumentParser(
        description='Run lappy benchmark sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--sweeps',
        default=','.join(SWEEPS.keys()),
        help=f'Comma-separated sweep names (default: all). Available: {", ".join(SWEEPS)}',
    )
    parser.add_argument(
        '--domains',
        default=None,
        help='Comma-separated domain names to restrict to (default: all)',
    )
    parser.add_argument(
        '--outdir',
        default='benchmarks/results',
        help='Output directory for results (default: benchmarks/results/)',
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=1,
        help='Number of parallel workers via ProcessPoolExecutor (default: 1)',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print all configs that would be run, without executing',
    )

    args = parser.parse_args()

    sweep_names = [s.strip() for s in args.sweeps.split(',')]
    domain_filter = (
        {d.strip() for d in args.domains.split(',')}
        if args.domains
        else None
    )

    for name in sweep_names:
        if name not in SWEEPS:
            print(f"Unknown sweep: {name!r}. Available: {', '.join(SWEEPS)}", file=sys.stderr)
            sys.exit(1)

    for name in sweep_names:
        module = SWEEPS[name]
        print(f"\n{'='*60}")
        print(f"Running sweep: {name}")
        print(f"{'='*60}")
        module.run(
            outdir=args.outdir,
            domain_filter=domain_filter,
            n_workers=args.n_workers,
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
