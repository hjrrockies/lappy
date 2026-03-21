#!/bin/bash
# One-time setup: create the Python virtual environment on the cluster.
# Run this once from the repo root after cloning/transferring the project.
#
# Edit PYTHON_MODULE to match whatever `module avail python` shows on your cluster.

set -euo pipefail

PYTHON_MODULE="python/3.11"   # <-- adjust to match your cluster

module load "$PYTHON_MODULE"
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

echo ""
echo "Environment ready. Test with:"
echo "  .venv/bin/python -c 'import lappy; print(\"lappy OK\")'"
