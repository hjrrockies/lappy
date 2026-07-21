"""Shared diagnostics for building the JSON cubature-rule registry.

Used by both build_cubature_registry.py (legacy one-time migration) and
scrape_cubature_rules.py (ongoing ingestion from the unipd rule library).
"""
from math import factorial

import numpy as np

_REF_AREA = 0.5


def exact_moment(p, q):
    """Exact average of x^p y^q over reference triangle (0,0),(1,0),(0,1), i.e.
    (1/area) * integral, matching the convention that rule weights sum to 1."""
    return (factorial(p) * factorial(q) / factorial(p + q + 2)) / _REF_AREA


def verify_degree(bary_coords, weights, nominal_degree, max_check=None):
    """Returns (verified_degree, verified_tol): highest degree d for which every
    monomial moment x^p y^q (p+q<=d) is exact to near machine precision, walking
    up until the error jumps by orders of magnitude."""
    x = bary_coords[:, 1]
    y = bary_coords[:, 2]
    if max_check is None:
        max_check = nominal_degree + 2

    verified_degree = 0
    verified_tol = 0.0
    for d in range(0, max_check + 1):
        max_err = 0.0
        for p in range(d + 1):
            q = d - p
            approx = np.sum(weights * x**p * y**q)
            exact = exact_moment(p, q)
            max_err = max(max_err, abs(approx - exact))
        if max_err < 1e-9:
            verified_degree = d
            verified_tol = max_err
        else:
            break
    return verified_degree, verified_tol


def compute_diagnostics(arr, nominal_degree):
    bary_coords = arr[:, :3]
    weights = arr[:, 3]
    npts = len(weights)
    min_w = float(weights.min())
    max_w = float(weights.max())
    positive = bool(min_w > 0)
    inside_triangle = bool(np.all(bary_coords >= -1e-12) and np.all(bary_coords <= 1 + 1e-12))
    weight_ratio = (max_w / min_w) if min_w > 0 else None
    verified_degree, verified_tol = verify_degree(bary_coords, weights, nominal_degree)
    if verified_degree < nominal_degree:
        print(f"  WARNING: degree {nominal_degree} rule only verified to degree "
              f"{verified_degree} (tol {verified_tol:.3e})")
    return {
        'npts': npts,
        'nodes': bary_coords.tolist(),
        'weights': weights.tolist(),
        'positive': positive,
        'inside_triangle': inside_triangle,
        'min_weight': min_w,
        'max_weight': max_w,
        'weight_ratio': weight_ratio,
        'verified_degree': verified_degree,
        'verified_tol': verified_tol,
    }
