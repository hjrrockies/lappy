#!/bin/bash
# Submit a Slurm array job for one lappy benchmark sweep.
#
# Usage:
#   bash benchmarks/benchmark_suite/slurm/submit_array.sh \
#       --sweep basis_size [--domains rect,L_shape] [--outdir benchmarks/results]
#
# Before first use:
#   1. Fill in ACCOUNT, PARTITION, and PYTHON_MODULE below.
#   2. Run setup_env.sh once in an interactive job to create the venv.
#   3. Then run this script to prepare and submit.

set -euo pipefail

# ── Cluster settings (edit these) ─────────────────────────────────────────────
ACCOUNT="siallocation"
PARTITION="normal_q"
PYTHON_MODULE="Python/3.12.3-GCCcore-13.3.0"          # match what you used in setup_env.sh
TIME_LIMIT="24:00:00"                # wall time per task; increase for large sweeps
MEM_PER_TASK="64G"
MAX_CONCURRENT=32                    # max simultaneous tasks (%N in --array)
VENV="$HOME/venvs/.lappy"
# ──────────────────────────────────────────────────────────────────────────────

# Parse arguments
SWEEP=""
DOMAINS_ARG=""
OUTDIR="benchmarks/results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep)   SWEEP="$2";                         shift 2 ;;
        --domains) DOMAINS_ARG="--domains $2";         shift 2 ;;
        --outdir)  OUTDIR="$2";                        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$SWEEP" ]] && {
    echo "Usage: $0 --sweep <name> [--domains a,b] [--outdir dir]"
    echo "Sweeps: fb_corner fs_placement basis_size regularization n_eigs"
    exit 1
}

# Change to repo root (script lives 3 levels deep)
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# Generate config pickle and capture ARRAY_SIZE / CONFIG_FILE
PREPARE_OUT=$($VENV/bin/python -m benchmarks.benchmark_suite.slurm.prepare_jobs \
    --sweep "$SWEEP" $DOMAINS_ARG --outdir "$OUTDIR")
echo "$PREPARE_OUT"

ARRAY_SIZE=$(echo "$PREPARE_OUT" | grep '^ARRAY_SIZE=' | cut -d= -f2)
CONFIG_FILE=$(echo "$PREPARE_OUT" | grep '^CONFIG_FILE=' | cut -d= -f2)

[[ -z "$ARRAY_SIZE" || "$ARRAY_SIZE" -eq 0 ]] && { echo "No configs generated — nothing to submit."; exit 1; }

LAST=$(( ARRAY_SIZE - 1 ))
LOG_DIR="$REPO_ROOT/benchmarks/benchmark_suite/slurm/logs/${SWEEP}"
mkdir -p "$LOG_DIR"

# Submit
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=lappy_${SWEEP}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --array=0-${LAST}%${MAX_CONCURRENT}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --cpus-per-task=1
#SBATCH --output=${LOG_DIR}/task_%a.out
#SBATCH --error=${LOG_DIR}/task_%a.err

module load ${PYTHON_MODULE}
cd ${REPO_ROOT}
${VENV}/bin/python -m benchmarks.benchmark_suite.slurm.run_job \
    --config_file ${CONFIG_FILE}
EOF
)

echo "Submitted job ${JOB_ID}: ${ARRAY_SIZE} tasks for sweep '${SWEEP}'."
echo "Logs: ${LOG_DIR}/"
echo "Monitor: squeue --job ${JOB_ID}"
