#!/bin/bash
#SBATCH --job-name=fb_basis_corners
#SBATCH --account=siallocation
#SBATCH --partition=normal_q
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

VENV="$HOME/venvs/lappy"
PYTHON_MODULE="Python/3.12.3-GCCcore-13.3.0"

module load "$PYTHON_MODULE"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

"$VENV/bin/python" -m benchmarks.new_suite.fb_basis_corners
