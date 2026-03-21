# lappy Benchmark Suite

Systematic benchmarks for the Method of Particular Solutions eigensolver.
Each sweep isolates one design variable (basis strategy, basis size, regularization, etc.)
and measures accuracy against reference eigenvalues across a set of test domains.

## Directory structure

```
benchmark_suite/
├── domains.py          # Domain registry (geometry factories + reference eigenvalues)
├── basis_builder.py    # FB and FS basis construction helpers
├── runner.py           # BenchmarkConfig, BenchmarkResult, run_benchmark()
├── results.py          # save/load results (.npz + .json pairs); DataFrame export
├── run_all.py          # CLI entry point (local runs)
├── sweeps/
│   ├── fb_corner_sweep.py      # Sweep 1: FB order allocation strategy
│   ├── fs_placement_sweep.py   # Sweep 2: FS source distance and density
│   ├── basis_size_sweep.py     # Sweep 3: Total basis size and FB/FS ratio
│   ├── regularization_sweep.py # Sweep 4: rtol effect on accuracy
│   └── n_eigs_sweep.py         # Sweep 5: Scaling with n_eigs
└── slurm/
    ├── setup_env.sh        # One-time venv creation on the cluster
    ├── prepare_jobs.py     # Serialize configs to a pickle for array jobs
    ├── run_job.py          # Single-task runner (called by each array task)
    └── submit_array.sh     # End-to-end: prepare + sbatch submit
```

---

## Running locally

```bash
# All sweeps, all domains
python -m benchmarks.benchmark_suite.run_all --outdir benchmarks/results

# One sweep, filtered to specific domains
python -m benchmarks.benchmark_suite.run_all \
    --sweeps basis_size --domains rect,L_shape --outdir benchmarks/results

# Dry run: print all configs without executing
python -m benchmarks.benchmark_suite.run_all --dry_run

# Parallel execution (ProcessPoolExecutor)
python -m benchmarks.benchmark_suite.run_all --n_workers 4 --outdir benchmarks/results
```

---

## Running on a Slurm cluster

Each sweep is submitted as an independent array job where every task runs one config.

### First-time setup

1. Copy or clone the repo to the cluster.
2. Edit the three cluster-specific variables at the top of `slurm/submit_array.sh`:

   ```bash
   ACCOUNT="your_allocation_account"
   PARTITION="your_partition"
   PYTHON_MODULE="python/3.11"   # match `module avail python`
   ```

3. Create the virtual environment (run once from the repo root):

   ```bash
   bash benchmarks/benchmark_suite/slurm/setup_env.sh
   ```

### Submitting a sweep

```bash
bash benchmarks/benchmark_suite/slurm/submit_array.sh --sweep basis_size

# With a domain filter
bash benchmarks/benchmark_suite/slurm/submit_array.sh \
    --sweep basis_size --domains rect,L_shape

# Custom output directory
bash benchmarks/benchmark_suite/slurm/submit_array.sh \
    --sweep basis_size --outdir /scratch/yourname/lappy_results
```

### Submitting all sweeps

```bash
for sweep in fb_corner fs_placement basis_size regularization n_eigs; do
    bash benchmarks/benchmark_suite/slurm/submit_array.sh --sweep $sweep
done
```

### Monitoring

```bash
squeue --job <JOB_ID>           # progress
cat benchmarks/benchmark_suite/slurm/logs/basis_size/task_0.out   # per-task log
```

### How it works

`submit_array.sh` calls `prepare_jobs.py`, which serializes the full config list for the
sweep to `slurm/jobs/<sweep>_configs.pkl`. Slurm then launches one task per config; each
task calls `run_job.py --config_file ... ` with `$SLURM_ARRAY_TASK_ID` as the index.
Results are written independently by each task to the output directory, so there are no
inter-task dependencies and partial runs are safe to resume by resubmitting failed tasks.

---

## Adding / modifying domains

Each sweep has a `DOMAIN_ENTRIES` list of `SweepDomainEntry` objects:

```python
from ..runner import SweepDomainEntry

DOMAIN_ENTRIES = [
    SweepDomainEntry('rect', {'L': 2.0, 'H': 1.0}),
    # Per-domain sweep range override:
    SweepDomainEntry('disk', {'r': 1.0}, sweep_overrides={'rtol': [1e-10, 1e-12, 1e-14]}),
    # Per-domain fixed param override:
    SweepDomainEntry('disk', {'r': 1.0}, fixed_overrides={'fb_fraction': 0.0}),
]
```

- **Add a domain**: append a new `SweepDomainEntry(...)`.
- **Remove a domain**: delete the line.
- **Custom sweep range**: pass `sweep_overrides={'var': [values]}`.
- **Custom fixed params**: pass `fixed_overrides={'param': value}`.

To register a new domain geometry, add a `DomainSpec` entry to `domains.py`.

---

## Result files

Each run produces a `.npz` + `.json` pair in the output directory:

```
{domain}_{sweep}_{tag}.npz    # eigs, tensions, rel_errors, wall_time arrays
{domain}_{sweep}_{tag}.json   # BenchmarkConfig fields + n_basis_actual, n_bdry_pts, etc.
```

Load and analyse results:

```python
from benchmarks.benchmark_suite.results import load_result, results_to_dataframe
import glob

results = [load_result(p) for p in glob.glob('benchmarks/results/basis_size*.npz')]
df = results_to_dataframe(results)
print(df[['domain_name', 'n_fb', 'n_fs', 'median_rel_error', 'wall_time']])
```
