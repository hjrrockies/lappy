"""Loader for the JSON library of triangle cubature rules in lappy/data/cubature_rules/.

Each rule (kind, degree) is stored in barycentric form (bary_coords, bary_weights),
with bary_weights normalized to sum to 1, alongside precomputed diagnostics
(positivity, weight conditioning, inside-triangle check, verified degree of
exactness) used to select rules for a given task (see lappy.cubature).

To add a new rule family, see scripts/build_cubature_registry.py.
"""
import json
from importlib import resources

import numpy as np

_registry = None
_arrays = {}


def load_registry():
    """Reads every *.json in lappy/data/cubature_rules/ once; returns {kind: family_dict}."""
    global _registry
    if _registry is None:
        _registry = {}
        data_dir = resources.files('lappy').joinpath('data', 'cubature_rules')
        for entry in data_dir.iterdir():
            if entry.name.endswith('.json'):
                family = json.loads(entry.read_text())
                _registry[family['kind']] = family
    return _registry


def available_kinds():
    return list(load_registry().keys())


def available_degrees(kind):
    registry = load_registry()
    try:
        family = registry[kind]
    except KeyError:
        raise ValueError(f"rule kind '{kind}' is not defined")
    return sorted(int(deg) for deg in family['rules'])


def get_cubature_rule(kind, deg):
    """Returns a cubature rule of a specified kind and degree in barycentric form."""
    key = (kind, deg)
    if key not in _arrays:
        registry = load_registry()
        try:
            entry = registry[kind]['rules'][str(deg)]
        except KeyError:
            raise ValueError(f"rule of kind '{kind}' and degree {deg} is not defined")
        bary_coords = np.asarray(entry['nodes'], dtype=np.float64)
        bary_weights = np.asarray(entry['weights'], dtype=np.float64)
        _arrays[key] = (bary_coords, bary_weights)
    return _arrays[key]


def iter_rules(positive_only=False, min_degree=None, max_degree=None):
    """Yields diagnostic dicts for every rule in the registry, without decoding
    nodes/weights: {kind, deg, npts, positive, min_weight, max_weight, weight_ratio,
    inside_triangle, verified_degree, verified_tol}."""
    registry = load_registry()
    for kind, family in registry.items():
        for deg_str, entry in family['rules'].items():
            deg = int(deg_str)
            if positive_only and not entry['positive']:
                continue
            if min_degree is not None and deg < min_degree:
                continue
            if max_degree is not None and deg > max_degree:
                continue
            yield {
                'kind': kind,
                'deg': deg,
                'npts': entry['npts'],
                'positive': entry['positive'],
                'min_weight': entry['min_weight'],
                'max_weight': entry['max_weight'],
                'weight_ratio': entry['weight_ratio'],
                'inside_triangle': entry['inside_triangle'],
                'verified_degree': entry['verified_degree'],
                'verified_tol': entry['verified_tol'],
            }
