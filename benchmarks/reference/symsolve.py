"""Symmetry-reduced reference solves.

Wraps ``lappy.symmetry`` into the same shape as ``common.solve_domain_v2``:
build one solver per symmetry sector, run ``manual_solve`` + ``polish_eigs``
in each, then merge. Besides being much more accurate (see TUNING_LOG.md),
this labels every eigenvalue with its sector, which makes the multiplicity
structure an *output* of the calculation rather than something inferred from
a near-degenerate tension curve.
"""
import numpy as np

from lappy import geometry, bases, mps, bounds, asymp
from lappy import MPSEigensolver
from lappy.eigfun_integrals import boundary_quadrature
from lappy.geometry import PointSet
from lappy.symmetry import (SymmetrizedBasis, prune_columns, fundamental_bdry_pts,
                            fundamental_int_pts, domain_symmetry)

from common import manual_solve, polish_eigs, lambda_window


def build_sym_solver(domain, group, sector, n_basis, rtol=None, ttol=1e-3,
                     bdry_mult=2, int_npts=None, prune=True, lam_ref=None,
                     prune_kill_tol=1e-8, prune_dup_tol=1e-14, lam_max=None,
                     orthonorm=False, orthonorm_precision=1e-13, **basis_kwargs):
    """One MPS solver for a single symmetry sector.

    The basis is the ordinary full-domain default basis, projected onto the
    sector; collocation happens only on the part of the boundary inside the
    fundamental domain. ``prune`` drops the structurally dependent columns
    (the projection is ``|G|``-to-one on basis columns) to recover the cubic
    GSVD saving.

    ``orthonorm`` attaches a boundary quadrature over the **whole** boundary,
    not the fundamental domain's share of it. That is deliberate and is the
    same reasoning ``certify.certify_sym`` runs on: a sector eigenfunction is a
    genuine Helmholtz solution on all of ``Omega`` satisfying the Dirichlet
    condition on all of ``dOmega``, so its ``L2(Omega)`` norm is the whole-domain
    Rellich integral. Collocation is reduced by symmetry; the norm is not.
    """
    if rtol is None:
        rtol = mps.rtol_default
    basis = bases.make_default_basis(domain, n_basis, **basis_kwargs)

    # `bdry_mult` is scaled up by |G| because roughly 1/|G| of the generated
    # points survive the fundamental-domain filter, and we still want ~2x
    # oversampling of the *retained* column count.
    n_per_seg = mps.pts_per_seg(domain, basis, mult=bdry_mult * group.order)
    bdry_pts = fundamental_bdry_pts(domain, group, n_per_seg)

    if int_npts is None:
        int_npts = max(n_basis // group.order, 40)
    int_pts = fundamental_int_pts(domain, group, int_npts)

    sym = SymmetrizedBasis(basis, group, sector)
    if prune:
        if lam_ref is None:
            lam_ref = float(asymp.weyl_est(3, domain))
        allpts = np.concatenate([bdry_pts.pts, int_pts.pts])
        sym = prune_columns(sym, lam_ref, allpts, kill_tol=prune_kill_tol,
                             dup_tol=prune_dup_tol)

    bdry_quad = None
    if orthonorm:
        if lam_max is None:
            raise ValueError('build_sym_solver(orthonorm=True) needs lam_max '
                             '(use lambda_window(domain, n_eigs)[1])')
        bdry_quad = boundary_quadrature(domain, lam_max,
                                        precision=orthonorm_precision)

    sym = sym.to_normalized((bdry_pts, int_pts))
    solver = MPSEigensolver(sym, bdry_pts, int_pts, rtol=rtol, ttol=ttol,
                            bdry_quad=bdry_quad)
    return solver


def solve_sym(domain, group, n_basis, n_eigs, a=None, b=None, bracket_xtol=1e-5,
              minimize_tol=1e-12, polish_bracket_rel_width=1e-9, ttol=1e-3,
              rtol=1e-14, n_pts_per_eig=11, n_workers=4, verbose=0,
              max_recurse=8, return_solvers=False, **kwargs):
    """Solve every symmetry sector over the same window and merge.

    Returns ``(eigs, sectors, tensions)`` sorted by eigenvalue, where
    ``sectors[i]`` is the character tuple of the sector eigenvalue ``i`` was
    found in.

    Multiplicity comes from two places and *both* are needed. Degeneracies the
    group distinguishes show up as the same eigenvalue in different sectors.
    But the registered group is only the largest elementary abelian 2-subgroup
    with real characters, which can be strictly smaller than the domain's true
    symmetry -- so degeneracies the full group would split can survive *inside*
    a single sector. The unit square is the standard example: under ``rect D2``
    the pair (1,2)/(2,1) splits across sectors, but (1,3)/(3,1) are both
    odd-odd and land in the same sector, because separating them needs the
    diagonal reflection that D4 has and D2 does not. Those are recovered from
    ``manual_solve``'s per-sector multiplicity estimate.
    """
    if a is None or b is None:
        a, b = lambda_window(domain, n_eigs)
    n_pts = max(n_pts_per_eig * n_eigs, 50)
    # The boundary quadrature (when the caller asked for orthonorm) is sized for
    # eigenfunctions up to the top of the search window.
    kwargs.setdefault('lam_max', b)

    all_eigs, all_sectors, all_tens, solvers = [], [], [], {}
    for sector in group.sectors():
        solver = build_sym_solver(domain, group, sector, n_basis, rtol=rtol,
                                  ttol=ttol, **kwargs)
        solvers[sector] = solver
        if verbose:
            print(f'  sector {sector}: {len(solver.basis)} cols, '
                  f'{len(solver.bdry_pts)} bdry pts, {len(solver.int_pts)} int pts')
        eigs, mults, _ = manual_solve(solver, a, b, n_pts, bracket_xtol=bracket_xtol,
                                      minimize_tol=minimize_tol, ttol=ttol,
                                      max_recurse=max_recurse,
                                      n_workers=n_workers, verbose=0)
        if len(eigs):
            eigs, tens = polish_eigs(solver, eigs, ltol=1e-14,
                                     bracket_rel_width=polish_bracket_rel_width)
            # expand within-sector multiplicity (see docstring)
            if mults is not None and any(m > 1 for m in mults):
                rep = np.repeat(np.arange(len(eigs)), mults[:len(eigs)])
                eigs, tens = eigs[rep], np.asarray(tens)[rep]
            all_eigs.append(eigs)
            all_tens.append(tens)
            all_sectors += [sector] * len(eigs)

    if not all_eigs:
        out = np.array([]), [], np.array([])
        return (*out, solvers) if return_solvers else out

    eigs = np.concatenate(all_eigs)
    tens = np.concatenate(all_tens)
    order = np.argsort(eigs)
    eigs, tens = eigs[order], tens[order]
    sectors = [all_sectors[i] for i in order]
    out = (eigs[:n_eigs], sectors[:n_eigs], tens[:n_eigs])
    return (*out, solvers) if return_solvers else out


def report_sym(name, eigs, sectors, tensions, ref=None):
    print(f'\n=== {name} ===')
    hdr = f"{'eig':>24}  {'tension':>10}  {'~dig':>5}  sector"
    if ref is not None:
        hdr += f"{'':>6}{'|diff| vs ref':>16}"
    print(hdr)
    for i, (e, s, t) in enumerate(zip(eigs, sectors, tensions)):
        dig = -np.log10(t) - 1 if t > 0 else np.inf
        line = f'{e:24.15f}  {t:10.2e}  {dig:5.1f}  {s}'
        if ref is not None and i < len(ref):
            line += f'{"":>6}{abs(e - ref[i]):16.3e}'
        print(line)
    print('\nnp.array([' + ', '.join(f'{e:.15f}' for e in eigs) + '])')
