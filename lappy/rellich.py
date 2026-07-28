"""Boundary-only L^2(Omega) Gram matrices via the Rellich identity.

See docs/rellich.md for the underlying master identity and its Zaremba
(Dirichlet/Neumann) specialization, and docs/rellich_hadamard_mps.pdf for the
general Cauchy-data/kernel-assembler architecture (lappy.cauchy) and the
Kress-style graded-mesh quadrature (lappy.cauchy.build_boundary_quadrature)
this module builds on: a single shared boundary node set per segment serves
every kernel here, rather than a fresh Cauchy-data evaluation per corner mode.
This module builds, for a given particular-solution basis and spectral
parameter lambda, the basis-space Gram matrix G with
G[i,j] = <phi_i, phi_j>_{L^2(Omega)}, computed purely from boundary Cauchy
data (no interior cubature). MPSEigensolver uses this to orthonormalize
eigenfunction coefficient vectors in place of the interior-cubature-based
approach.

Restricted to pure Dirichlet, pure Neumann, and Zaremba (mixed dir/neu)
boundary conditions -- Robin segments are unsupported here (see
docs/rellich.md's explicit scope note); callers should check
domain.bc_type before using this module.
"""
from collections import namedtuple
import warnings

import numpy as np
import scipy.linalg as la

from .cauchy import build_boundary_quadrature, basis_cauchy_data, assemble_kernel, default_x0
from .utils import complex_dot

# Precomputed, geometry-only boundary quadrature (the Kress-graded/plain node set built once by
# build_boundary_quadrature) plus the reference point x0 for the Rellich weight r.N/r.T. Unlike the
# old SS/SR/RS/RR RellichData, this carries no live `domain` reference and needs no per-eigenvalue
# rebuilding: only the basis's Cauchy data (Phi/Phi_N/Phi_T) depends on lam.
RellichData = namedtuple('RellichData', ['pts', 'normals', 'tangents', 'wts',
                                         'dir_mask', 'neu_mask', 'x0'])


def build_rellich_data(domain, basis, lam_max=None, x0=None, mult=2, min_per_seg=4, margin=2.0,
                       c_lam=1.0, beta=0.2):
    """Builds the RellichData bundle needed by rellich_gram_basis. `mult`/`min_per_seg` set the
    per-segment quadrature point density (see cauchy.graded_pts_per_seg); `margin` pads the
    Kress grading order comfortably past each singular corner's worst exponent (see
    cauchy.corner_grading_orders). `lam_max` (defaulting to weyl_est(6, domain) if not given)
    is the worst-case spectral parameter this node set must stay accurate for, since it is built
    once here and reused, unchanged, for every lam tried in a solve_interval search -- only the
    basis's Cauchy data depends on lam after this point. `c_lam`/`beta` tune, respectively, the
    lam-driven and grading-order-driven point-count terms (see cauchy.graded_pts_per_seg)."""
    if x0 is None:
        x0 = default_x0(domain)
    pts, normals, tangents, wts, dir_mask, neu_mask = build_boundary_quadrature(
        domain, basis, lam_max, mult, min_per_seg, margin, c_lam=c_lam, beta=beta)
    return RellichData(pts, normals, tangents, wts, dir_mask, neu_mask, x0)


def rellich_gram_from_cauchy_data(cd, lam, rellich_data):
    """Rellich-identity Gram matrix G[i,j] = <f_i,f_j> at spectral parameter lam
    (docs/rellich.md Sec. 2: G = (1/2lam)(I_1 - I_2) + (1/2)I_3), from an already-evaluated
    `cauchy.CauchyData` `cd` at `rellich_data`'s shared boundary node set. `cd` may carry any
    number of columns -- the full basis (as in `rellich_gram_basis`), or a small
    (mult-column) eigenfunction cluster evaluated directly from its own coefficients/GSVD-eval
    output (docs/rellich_hadamard_mps.pdf Sec. 3.2, "evaluate first, sandwich never") -- since
    `assemble_kernel` is generic over column count."""
    pts, normals, tangents, wts, dir_mask, neu_mask, x0 = rellich_data
    rN = complex_dot(pts - x0, normals)

    n = cd.Phi.shape[1]
    G = np.zeros((n, n))
    if dir_mask.any():
        G += assemble_kernel(cd, 'NN', rN*dir_mask)/(2*lam)
    if neu_mask.any():
        G += assemble_kernel(cd, 'uv', rN*neu_mask)/2
        G -= assemble_kernel(cd, 'TT', rN*neu_mask)/(2*lam)
    return G


def rellich_gram_basis(basis, lam, rellich_data):
    """Basis-space L^2(Omega) Gram matrix G[i,j] = <phi_i,phi_j> at spectral parameter lam, via
    the Zaremba-specialized Rellich identity (see rellich_gram_from_cauchy_data). A single
    basis_cauchy_data call evaluates the whole (undecomposed) basis once, over the whole
    boundary; dir_mask/neu_mask restrict each kernel to its relevant segments by zeroing the
    per-point weight rather than rebuilding a separate node set."""
    pts, normals, tangents, wts = rellich_data[:4]
    cd = basis_cauchy_data(basis, lam, pts, normals, tangents, wts)
    return rellich_gram_from_cauchy_data(cd, lam, rellich_data)


def lowdin_transform(G, ttol=1e-3):
    """Löwdin (symmetric) orthogonalization transform D (mult x mult) for a small Gram matrix G:
    diagonalize G=Q@diag(w)@Q.T and return D=Q@diag(w**-0.5)@Q.T, so that (for a (n, mult) array
    `vals` of already-safely-evaluated function values -- never a basis-level matrix sandwiched
    between coefficient vectors, see docs/rellich_hadamard_mps.pdf Sec. 3.1-3.2) `vals @ D.T` is
    orthonormal in the inner product G represents.

    Returns None (after warning) if G is deficient -- w.min()/w.max() < ttol, mirroring
    MPSEigensolver's existing deficient-multiplicity check -- rather than letting `w**-0.5`
    blow up on a near-zero or (roundoff-)negative eigenvalue and contaminate every column of D
    via the Q mixing; callers should fall back to the raw (un-orthonormalized) values with a
    warning, the same pattern used when no cauchy_data is available at all."""
    w, Q = la.eigh(G)
    if w.min()/w.max() < ttol:
        warnings.warn(f"Rellich Gram matrix is deficient (w.min()/w.max()="
                      f"{w.min()/w.max():.3e}<{ttol:.3e}); cluster may have wrong multiplicity. "
                      "Falling back to un-orthonormalized values.")
        return None
    return (Q*w**-0.5)@Q.T


def orthonormalize_coef(coef, G):
    """Rescales/rotates coefficient columns of `coef` (n_basis x mult) so that
    coef.T @ G @ coef == I, i.e. the resulting eigenfunctions are mutually
    L^2(Omega)-orthonormal (Cholesky whitening of the Gram matrix).

    NUMERICALLY RISKY for ill-conditioned bases when `G` is a basis-level (N x N) Gram matrix
    (e.g. from rellich_gram_basis): `coef.T @ G @ coef` sandwiches G between two copies of the
    raw (potentially huge, heavily-cancelling) GSVD coefficient vector, which multiplies G's
    independently-rounded error through coef on both sides (docs/rellich_hadamard_mps.pdf
    Sec. 3.1). No longer MPSEigensolver's default path -- see
    MPSEigensolver._orthonorm_transform_coef / lowdin_transform for the safe "evaluate first,
    sandwich never" replacement (Sec. 3.2), which builds an already-small (mult x mult) G from
    already-evaluated Cauchy data instead of sandwiching a basis-level one."""
    mult = coef.shape[1]
    Gram = coef.T@G@coef
    if mult == 1:
        return coef/np.sqrt(Gram[0, 0])
    L = la.cholesky(Gram, lower=True)
    return coef@la.solve_triangular(L, np.eye(mult), lower=True).T
