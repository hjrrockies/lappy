"""Precompute and persist singular-corner-integrand calibration curves for every rule
in lappy/data/cubature_rules/*.json.

lappy.cubature._choose_corner_rule evaluates _singular_error(kind, deg, beta) for every
rule in the positive-weight ladder each time a reentrant corner is handled; without
persisted data this triggers an mpmath-based analytic-reference computation (expensive
relative to a JSON read, though still only ~milliseconds per call) the first time each
rule is used at a given corner exponent beta. Persisting a fixed beta-grid calibration
here turns that into a cheap JSON read + interpolation, matching the plane-wave capacity
precompute (scripts/precompute_capacity.py).

Usage:
    python scripts/precompute_singular_calibration.py

Idempotent/incremental: rules whose stored singular_grid_version already matches
lappy.cubature.SINGULAR_GRID_VERSION (and whose grid matches _SINGULAR_BETA_GRID) are
skipped, so it's safe to re-run after adding new rule families (only the new rules get
calibrated) or after bumping SINGULAR_GRID_VERSION (everything gets recalibrated).
"""
import json
import os
import time

import numpy as np

from lappy.cubature import (_compute_singular_curve, SINGULAR_GRID_VERSION,
                            _SINGULAR_BETA_GRID)
from lappy.cubature_registry import load_registry

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'lappy', 'data', 'cubature_rules')


def main():
    registry = load_registry()
    n_grid = len(_SINGULAR_BETA_GRID)
    beta_grid_list = _SINGULAR_BETA_GRID.tolist()

    for kind in sorted(registry):
        family = registry[kind]
        changed = False
        for deg_str in sorted(family['rules'], key=int):
            entry = family['rules'][deg_str]
            stored_grid = entry.get('singular_beta_grid')
            if (entry.get('singular_grid_version') == SINGULAR_GRID_VERSION
                    and stored_grid is not None
                    and len(entry.get('singular_E', [])) == n_grid
                    and np.allclose(stored_grid, _SINGULAR_BETA_GRID)):
                continue
            deg = int(deg_str)
            t0 = time.perf_counter()
            E = _compute_singular_curve(kind, deg)
            dt = time.perf_counter() - t0
            entry['singular_grid_version'] = SINGULAR_GRID_VERSION
            entry['singular_beta_grid'] = beta_grid_list
            entry['singular_E'] = E.tolist()
            changed = True
            print(f"{kind} deg={deg}: calibrated ({dt*1000:.1f} ms)")

        if changed:
            out_path = os.path.join(DATA_DIR, f"{kind}.json")
            with open(out_path, 'w') as f:
                json.dump(family, f, indent=2)


if __name__ == '__main__':
    main()
