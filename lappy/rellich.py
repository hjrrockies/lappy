"""Boundary-only L^2(Omega) Gram matrices via the Rellich identity.

See docs/rellich.md for the underlying master identity and its Zaremba
(Dirichlet/Neumann) specialization, and docs/rellich_hadamard_mps.pdf for the
general Cauchy-data/kernel-assembler architecture (lappy.cauchy) and the
SS/SR/RS/RR singularity-subtraction quadrature this module builds on: each
basis-function pair is graded to its own exact corner exponent (via
lappy.cauchy.singular_gram), rather than one leading-mode exponent per
segment applied uniformly to every pair. This module builds, for a given
particular-solution basis and spectral parameter lambda, the basis-space Gram
matrix G with G[i,j] = <phi_i, phi_j>_{L^2(Omega)}, computed purely from
boundary Cauchy data (no interior cubature). MPSEigensolver uses this to
orthonormalize eigenfunction coefficient vectors in place of the
interior-cubature-based approach.

Restricted to pure Dirichlet, pure Neumann, and Zaremba (mixed dir/neu)
boundary conditions -- Robin segments are unsupported here (see
docs/rellich.md's explicit scope note); callers should check
domain.bc_type before using this module.
"""
from collections import namedtuple

import numpy as np
import scipy.linalg as la

from .cauchy import singular_gram, default_x0
from .utils import complex_dot

# Domain-aware data needed to (re-)build the Rellich Gram matrix at any eigenvalue. Unlike Phase
# 1's BoundaryNodes bundle, the SS/SR/RS/RR quadrature (lappy.cauchy.singular_gram) needs live
# access to the domain's corner/segment structure at every call (not just once, geometry-only, at
# construction time) -- so this bundle carries `domain` through as an opaque payload for
# MPSEigensolver, which never inspects it itself.
RellichData = namedtuple('RellichData', ['domain', 'x0', 'panel_frac', 'group_pts',
                                         'bulk_mult', 'bulk_min_per_seg'])


def build_rellich_data(domain, basis, x0=None, mult=2, min_per_seg=4, panel_frac=0.4, group_pts=16):
    """Builds the RellichData bundle needed by rellich_gram_basis. `mult`/`min_per_seg` set the
    plain (ungraded) bulk-region point density (see cauchy.singular_gram's bulk_mult/
    bulk_min_per_seg); `panel_frac`/`group_pts` set the corner-panel radius fraction and
    per-exponent-group point count for the SS/SR/RS blocks there."""
    if x0 is None:
        x0 = default_x0(domain)
    return RellichData(domain, x0, panel_frac, group_pts, mult, min_per_seg)


def rellich_gram_basis(basis, lam, rellich_data):
    """Basis-space L^2(Omega) Gram matrix G[i,j] = <phi_i,phi_j> at spectral parameter lam, via
    the Zaremba-specialized Rellich identity: G = (1/2lam)(I_1 - I_2) + (1/2)I_3 (docs/rellich.md
    Sec. 2), each term a lappy.cauchy.singular_gram call restricted to the relevant
    boundary-condition segments."""
    domain, x0, panel_frac, group_pts, bulk_mult, bulk_min_per_seg = rellich_data

    def weight_fn(pts, normals, tangents):
        return complex_dot(pts - x0, normals)

    kwargs = dict(panel_frac=panel_frac, group_pts=group_pts,
                  bulk_mult=bulk_mult, bulk_min_per_seg=bulk_min_per_seg)

    dir_mask = np.array([seg.bc_type == 'dir' for seg in domain.bdry.segments])
    neu_mask = np.array([seg.bc_type == 'neu' for seg in domain.bdry.segments])

    G = np.zeros((len(basis), len(basis)))
    if dir_mask.any():
        G += singular_gram(basis, domain, lam, 'NN', weight_fn, seg_mask=dir_mask, **kwargs)/(2*lam)
    if neu_mask.any():
        G -= singular_gram(basis, domain, lam, 'TT', weight_fn, seg_mask=neu_mask, **kwargs)/(2*lam)
        G += singular_gram(basis, domain, lam, 'uv', weight_fn, seg_mask=neu_mask, **kwargs)/2
    return G


def orthonormalize_coef(coef, G):
    """Rescales/rotates coefficient columns of `coef` (n_basis x mult) so that
    coef.T @ G @ coef == I, i.e. the resulting eigenfunctions are mutually
    L^2(Omega)-orthonormal (Cholesky whitening of the Gram matrix)."""
    mult = coef.shape[1]
    Gram = coef.T@G@coef
    if mult == 1:
        return coef/np.sqrt(Gram[0, 0])
    L = la.cholesky(Gram, lower=True)
    return coef@la.solve_triangular(L, np.eye(mult), lower=True).T
