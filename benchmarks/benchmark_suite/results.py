"""Save/load benchmark results and convert to pandas DataFrames."""
import json
import os
import numpy as np
from dataclasses import asdict


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

from .runner import BenchmarkConfig, BenchmarkResult


def save_result(result: BenchmarkResult, outdir: str, sweep_name: str,
                extra_tag: str = '') -> str:
    """
    Save a BenchmarkResult as a .npz/.json pair.

    File name: {domain}_{sweep_name}[_{extra_tag}].npz/.json
    Returns the base path (without extension).
    """
    os.makedirs(outdir, exist_ok=True)
    cfg = result.config
    base = f"{cfg.domain_name}_{sweep_name}"
    if extra_tag:
        base += f"_{extra_tag}"
    base_path = os.path.join(outdir, base)

    # arrays
    arrays = dict(
        eigs=result.eigs,
        tensions=result.tensions,
        wall_time=np.array(result.wall_time),
    )
    if result.rel_errors is not None:
        arrays['rel_errors'] = result.rel_errors

    np.savez(base_path + '.npz', **arrays)

    # metadata
    meta = {
        'config': asdict(cfg),
        'n_basis_actual': result.n_basis_actual,
        'n_bdry_pts': result.n_bdry_pts,
        'n_int_pts': result.n_int_pts,
        'has_rel_errors': result.rel_errors is not None,
    }
    with open(base_path + '.json', 'w') as f:
        json.dump(meta, f, indent=2, cls=_NumpyEncoder)

    return base_path


def load_result(base_path: str) -> BenchmarkResult:
    """
    Load a BenchmarkResult from a .npz/.json pair.
    *base_path* may include or omit the .npz extension.
    """
    base_path = base_path.removesuffix('.npz').removesuffix('.json')

    data = np.load(base_path + '.npz')
    with open(base_path + '.json') as f:
        meta = json.load(f)

    cfg_dict = meta['config']
    config = BenchmarkConfig(**cfg_dict)

    rel_errors = data['rel_errors'] if meta['has_rel_errors'] else None

    return BenchmarkResult(
        config=config,
        eigs=data['eigs'],
        tensions=data['tensions'],
        rel_errors=rel_errors,
        wall_time=float(data['wall_time']),
        n_basis_actual=meta['n_basis_actual'],
        n_bdry_pts=meta['n_bdry_pts'],
        n_int_pts=meta['n_int_pts'],
    )


def results_to_dataframe(results: list) -> 'pd.DataFrame':
    """
    Flatten a list of BenchmarkResults to a pandas DataFrame.
    One row per result; all BenchmarkConfig fields are columns.
    Scalar summaries of arrays (median tension, median rel_error, max rel_error) are included.

    Requires pandas (``pip install pandas``).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("results_to_dataframe() requires pandas: pip install pandas") from None

    rows = []
    for r in results:
        row = asdict(r.config)
        row['wall_time'] = r.wall_time
        row['n_basis_actual'] = r.n_basis_actual
        row['n_bdry_pts'] = r.n_bdry_pts
        row['n_int_pts'] = r.n_int_pts
        row['n_eigs_returned'] = len(r.eigs)
        row['median_tension'] = float(np.median(r.tensions)) if len(r.tensions) > 0 else None
        row['max_tension'] = float(np.max(r.tensions)) if len(r.tensions) > 0 else None
        if r.rel_errors is not None and len(r.rel_errors) > 0:
            row['median_rel_error'] = float(np.median(r.rel_errors))
            row['max_rel_error'] = float(np.max(r.rel_errors))
        else:
            row['median_rel_error'] = None
            row['max_rel_error'] = None
        rows.append(row)

    return pd.DataFrame(rows)
