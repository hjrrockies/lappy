"""Domain registry for benchmark suite."""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from lappy import geometry
from lappy import reference


@dataclass
class DomainSpec:
    name: str
    factory: Callable
    params: list
    ref_eigs: Callable | None  # (k, params_dict) -> ndarray
    bc_type: str
    notes: str = ""


def _rect_ref(k, p):
    return reference.rect_eigs(k, p['L'], p['H'])

def _eq_tri_ref(k, p):
    return reference.eq_tri_eigs(k, p.get('l', 1))

def _iso_tri_ref(k, p):
    return reference.iso_right_tri_eigs(k, p.get('l', 1))

def _L_shape_ref(k, p):
    return reference.L_shape_eigs(k)

def _GWW1_ref(k, p):
    return reference.gww_eigs(k)

def _GWW2_ref(k, p):
    return reference.gww_eigs(k)

def _sector_ref(k, p):
    return reference.sector_eigs(k, p.get('r', 1), p['theta'])

def _disk_ref(k, p):
    return reference.disk_eigs(k, p.get('r', 1))


DOMAINS: dict[str, DomainSpec] = {}

def _register(spec: DomainSpec):
    DOMAINS[spec.name] = spec


# --- Rectangles ---
_register(DomainSpec(
    name='rect',
    factory=lambda **p: geometry.rect(p['L'], p['H']),
    params=[
        {'L': 1.0, 'H': 1.0},
        {'L': 1.5, 'H': 1.0},
        {'L': 2.0, 'H': 1.0},
        {'L': 3.0, 'H': 1.0},
        {'L': 5.0, 'H': 1.0},
        {'L': 10.0, 'H': 1.0},
    ],
    ref_eigs=_rect_ref,
    bc_type='dir',
    notes='Rectangle [0,L]x[0,H], aspect ratios L/H in {1, 1.5, 2, 3, 5, 10}',
))

# --- Right trapezoid ---
_register(DomainSpec(
    name='right_trapezoid',
    factory=lambda **p: geometry.right_trapezoid(p['h1'], p['h2']),
    params=[
        {'h1': 1.0, 'h2': 0.5},
        {'h1': 1.0, 'h2': 1.5},
        {'h1': 1.0, 'h2': 2.0},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Right trapezoid, base 1, left height h1, right height h2',
))

# --- Parallelogram ---
_register(DomainSpec(
    name='parallelogram',
    factory=lambda **p: geometry.parallelogram(p.get('b', 1), p.get('h', 1), p.get('alpha', np.pi/3)),
    params=[
        {'b': 1.0, 'h': 1.0, 'alpha': np.pi / 3},
        {'b': 2.0, 'h': 1.0, 'alpha': np.pi / 3},
        {'b': 1.0, 'h': 1.0, 'alpha': np.pi / 4},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Parallelogram with base b, height h, shear angle alpha',
))

# --- Equilateral triangle ---
_register(DomainSpec(
    name='eq_tri',
    factory=lambda **p: geometry.eq_tri(p.get('l', 1)),
    params=[{'l':1.0}],
    ref_eigs=_eq_tri_ref,
    bc_type='dir',
    notes='Equilateral triangle (unit circumradius)',
))

# --- Isosceles right triangle ---
_register(DomainSpec(
    name='iso_right_tri',
    factory=lambda **p: geometry.iso_right_tri(p.get('l', 1)),
    params=[{'l': 1.0}],
    ref_eigs=_iso_tri_ref,
    bc_type='dir',
    notes='Isosceles right triangle, legs length l',
))

# --- Isosceles triangle (variable height) ---
_register(DomainSpec(
    name='iso_tri',
    factory=lambda **p: geometry.iso_tri(p['h']),
    params=[
        {'h': 0.5},
        {'h': 1.0},
        {'h': 2.0},
        {'h': 4.0},
        {'h': 8.0},
        {'h': 16.0},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Isosceles triangle with base 2 and height h',
))

# --- Regular n-gons ---
_register(DomainSpec(
    name='reg_ngon',
    factory=lambda **p: geometry.reg_ngon(int(p['N'])),
    params=[
        {'N': 5},
        {'N': 6},
        {'N': 7},
        {'N': 8},
    ],
    ref_eigs=lambda k, p: _rect_ref(k, {'L': 1.0, 'H': 1.0}) if int(p['N']) == 4 else None,
    bc_type='dir',
    notes='Regular n-gon, unit circumradius. N=4 has rect reference.',
))

# --- L-shape ---
_register(DomainSpec(
    name='L_shape',
    factory=lambda **p: geometry.L_shape(),
    params=[{}],
    ref_eigs=_L_shape_ref,
    bc_type='dir',
    notes='L-shaped domain (fixed geometry)',
))

# --- GWW isospectral pairs ---
_register(DomainSpec(
    name='GWW1',
    factory=lambda **p: geometry.GWW1(),
    params=[{}],
    ref_eigs=_GWW1_ref,
    bc_type='dir',
    notes='First GWW isospectral domain',
))

_register(DomainSpec(
    name='GWW2',
    factory=lambda **p: geometry.GWW2(),
    params=[{}],
    ref_eigs=_GWW2_ref,
    bc_type='dir',
    notes='Second GWW isospectral domain',
))

# --- Chevron ---
_register(DomainSpec(
    name='chevron',
    factory=lambda **p: geometry.chevron(p['h1'], p['h2']),
    params=[
        {'h1': 1.0, 'h2': 1.25},
        {'h1': 1.0, 'h2': 1.5},
        {'h1': 1.0, 'h2': 2.0},
        {'h1': 2.0, 'h2': 3.0},
        {'h1': 2.0, 'h2': 4.0},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Chevron domain with heights h1, h2',
))

# --- Cut square ---
_register(DomainSpec(
    name='cut_square',
    factory=lambda **p: geometry.cut_square(p['r']),
    params=[
        {'r': 0.25},
        {'r': 0.5},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Unit square with circular cut-out of radius r',
))

# --- Disk ---
_register(DomainSpec(
    name='disk',
    factory=lambda **p: geometry.disk(p.get('r', 1)),
    params=[{'r': 1.0}],
    ref_eigs=_disk_ref,
    bc_type='dir',
    notes='Disk of radius r',
))

# --- Disk sector ---
_register(DomainSpec(
    name='disk_sector',
    factory=lambda **p: geometry.disk_sector(p.get('r', 1), p['theta']),
    params=[
        {'theta': np.pi / 6},
        {'theta': np.pi / 4},
        {'theta': np.pi / 3},
        {'theta': np.pi / 2},
        {'theta': 2 * np.pi / 3},
        {'theta': np.pi},
        {'theta': 4 * np.pi / 3},
        {'theta': 5 * np.pi / 3},
        {'theta': 7 * np.pi / 4},
    ],
    ref_eigs=_sector_ref,
    bc_type='dir',
    notes='Circular disk sector of radius 1 and angle theta',
))

# --- Mushroom ---
_register(DomainSpec(
    name='mushroom',
    factory=lambda **p: geometry.mushroom(p.get('a', 1), p.get('b', 1), p.get('r', 1.5)),
    params=[
        {'a': 1.0, 'b': 1.0, 'r': 1.5},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Mushroom domain (cap + stem)',
))

# --- H-shape ---
_register(DomainSpec(
    name='H_shape',
    factory=lambda **p: geometry.H_shape(),
    params=[{}],
    ref_eigs=None,
    bc_type='dir',
    notes='H-shaped domain (fixed geometry)',
))

# --- Ellipse ---
_register(DomainSpec(
    name='ellipse',
    factory=lambda **p: geometry.ellipse(p.get('a', 2), p.get('b', 1)),
    params=[
        {'a': 2.0, 'b': 1.0},
        {'a': 3.0, 'b': 1.0},
        {'a': 4.0, 'b': 1.0},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Ellipse with semi-axes a, b (parametric boundary, no closed-form reference)',
))

# --- Stadium ---
_register(DomainSpec(
    name='stadium',
    factory=lambda **p: geometry.stadium(p.get('L', 1), p.get('H', 1)),
    params=[
        {'L': 1.0, 'H': 1.0},
    ],
    ref_eigs=None,
    bc_type='dir',
    notes='Bunimovich stadium (parametric boundary, no closed-form reference)',
))
