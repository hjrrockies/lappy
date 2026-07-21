"""Precompute and persist plane-wave capacity calibration curves for every rule
in lappy/data/cubature_rules/*.json.

lappy.cubature.choose_rule evaluates capacity(kind, deg, eps) for every distinct
degree in the positive-weight rule ladder; without persisted data this triggers
an expensive plane-wave calibration (lappy.cubature._compute_capacity_curve) the
first time each rule is used in a process. Persisting the result here turns that
into a cheap JSON read.

Usage:
    python scripts/precompute_capacity.py

Idempotent/incremental: rules whose stored capacity_grid_version already matches
lappy.cubature.CAPACITY_GRID_VERSION are skipped, so it's safe to re-run after
adding new rule families (only the new rules get calibrated) or after bumping
CAPACITY_GRID_VERSION (everything gets recalibrated).
"""
import json
import os
import time

from lappy.cubature import _compute_capacity_curve, CAPACITY_GRID_VERSION, _CAPACITY_S_GRID
from lappy.cubature_registry import load_registry

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'lappy', 'data', 'cubature_rules')


def main():
    registry = load_registry()
    n_grid = len(_CAPACITY_S_GRID)

    for kind in sorted(registry):
        family = registry[kind]
        changed = False
        for deg_str in sorted(family['rules'], key=int):
            entry = family['rules'][deg_str]
            if (entry.get('capacity_grid_version') == CAPACITY_GRID_VERSION
                    and len(entry.get('capacity_E', [])) == n_grid):
                continue
            deg = int(deg_str)
            t0 = time.perf_counter()
            E = _compute_capacity_curve(kind, deg)
            dt = time.perf_counter() - t0
            entry['capacity_grid_version'] = CAPACITY_GRID_VERSION
            entry['capacity_E'] = E.tolist()
            changed = True
            print(f"{kind} deg={deg}: calibrated ({dt*1000:.1f} ms)")

        if changed:
            out_path = os.path.join(DATA_DIR, f"{kind}.json")
            with open(out_path, 'w') as f:
                json.dump(family, f, indent=2)


if __name__ == '__main__':
    main()
