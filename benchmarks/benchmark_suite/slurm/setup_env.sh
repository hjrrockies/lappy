#!/bin/bash
# One-time setup: create the Python virtual environment on the cluster.
# Run this once from the repo root in an interactive job.
#
# Edit PYTHON_MODULE to match whatever `module avail python` shows on your cluster.

set -euo pipefail

PYTHON_MODULE="Python/3.12.3-GCCcore-13.3.0"
VENV_DIR="$HOME/venvs/.lappy"

module load "$PYTHON_MODULE"
mkdir -p "$(dirname "$VENV_DIR")"
python -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -e .

echo ""
echo "Environment ready. Test with:"
echo "  $VENV_DIR/bin/python -c 'import lappy; print(\"lappy OK\")'"
