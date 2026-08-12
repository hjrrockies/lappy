"""The asymmetric L: vary proportion with every corner angle held fixed.

The question this exists for. `fs_frac` is the dominant basis knob and nothing yet predicts its
optimum: corner count, sharpest convex corner and bounding-box aspect are all refuted (see
NOTEBOOK). The one clean signal left is that L_shape -- the sole domain with a reentrant corner
and no sharp convex ones -- is the only place pure Fourier-Bessel wins outright, which suggests
FB is strongest when its entire budget sits on a genuine r^(2/3) singularity.

Testing that needs a second reentrant-only domain, and every candidate in the library has a
problem: `cut_square` has no reentrant corner at all (five corners, all pi/2 -- its concavity is
a smooth arc); `plus_shape` is a polyomino, so its closed-form eigenfunctions vanish on the whole
integer grid and are SMOOTH at every reentrant corner, which is the Leg 2 trap; `mushroom` has
curved segments; `H_shape`'s reference is only 7.8 digits; GWW has 45-degree corners.

An asymmetric L has none of those. Legs `a` and `b` about a corner of thickness `t`: one
reentrant corner at 3pi/2 and five at pi/2, EXACTLY as in `L_shape`, for every choice of a and b.
So sweeping a/b moves proportion while holding the entire corner structure fixed -- the
controlled version of the comparison that `iso_tri(h)` could only make with the angles moving
too. Non-integer leg lengths keep it off the polyomino grid, so no accidental product-form
eigenfunction.

lam* comes from an in-house solve (`escalate_and_solve_v2`, ltol=1e-14), since no reference table
covers this family. That is the C2 path the program was designed around, and the tension reported
alongside is the evidence for how far it can be trusted.
"""
import os
import sys
import warnings

import numpy as np

from lappy.geometry import Polygon


def asym_L(a=2.0, b=1.0, t=0.6):
    """L with legs `a` (along x) and `b` (along y), both of thickness `t`.

    Corners, counter-clockwise from the origin: pi/2 at (0,0), (a,0), (a,t), then 3pi/2 at
    (t,t), then pi/2 at (t,b) and (0,b). One reentrant corner, five right angles, for any
    a, b > t > 0.
    """
    if not (a > t > 0 and b > t > 0):
        raise ValueError('need a > t > 0 and b > t > 0')
    v = np.array([0, a, a + 1j*t, t + 1j*t, t + 1j*b, 1j*b], dtype=complex)
    return Polygon(v, val_simple=False)


def _common():
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference')
    if here not in sys.path:
        sys.path.insert(0, here)
    import common
    return common


def inhouse_eigs(domain, n_eigs=5, n_basis=320, basis=None, bdry_mult=3):
    """(eigs, tensions) at ltol=1e-14, with an EXPLICIT basis.

    `escalate_and_solve_v2` cannot be used here, and finding that out is itself a result. It
    builds its solver through `make_default_basis`, which sends a domain with one singular corner
    to pure Fourier-Bessel -- exactly the family under test. On an asymmetric L at leg ratio 2:1
    it escalated to n_basis=320 and still reported tensions of 1.8e-08; at 5:1, 1.0e-05. So the
    reference it produces is only as good as the default basis, and a domain whose default basis
    is weak cannot be certified by it. Every sigma measured against such a lam* is measuring the
    reference, which is the ellipse a=2 failure in a new place.

    Passing the basis explicitly breaks that. `verify_lam_star` then cross-checks two DIFFERENT
    good bases, since using one blend to certify a comparison that includes blends would only
    move the circularity rather than remove it.
    """
    common = _common()
    solver = common.build_solver(domain, n_basis, basis=basis, bdry_mult=bdry_mult, rtol=1e-12)
    a, b = common.lambda_window(domain, n_eigs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eigs, mults, _ = common.manual_solve(solver, a, b, max(11*n_eigs, 50),
                                             bracket_xtol=1e-5, minimize_tol=1e-12,
                                             ttol=1e-3, n_workers=4)
        eigs, tens = common.polish_eigs(solver, eigs, ltol=1e-14, bracket_rel_width=1e-9)
    return np.asarray(eigs, dtype=float), np.asarray(tens, dtype=float)


def verify_lam_star(domain, n_eigs=4, n_basis=320):
    """Two independent good bases must agree, or lam* is not usable.

    Returns (lam_star, tensions, max relative disagreement). A blend and a boundary-FS blend are
    genuinely different constructions, so agreement between them is evidence rather than a
    restatement of one basis's opinion.

    USE THE DISAGREEMENT, NOT THE WORSE TENSION, as the trust figure. The first version took
    `max(tensions, disagreement)`, which is over-conservative and cost a whole ranking: at leg
    ratio 2:1 the two bases had tensions 1.8e-11 and 3.9e-10 while agreeing to 9.8e-12, and
    taking 3.9e-10 marked every good configuration "below lam* trust" and unreadable. A high
    tension says that BASIS is less accurate; it says nothing about lam* once a second,
    independent construction lands in the same place. The tensions remain worth reporting -- if
    both are poor, agreement could be two bases sharing a defect -- but the disagreement is the
    estimator.
    """
    from benchmarks.basis_lab.probe import build_basis, default_spec, wavelength
    common = _common()
    lam_hi = common.lambda_window(domain, n_eigs)[1]
    specs = [default_spec('mixed', n_basis, lam_hi, fs_frac=0.5),
             default_spec('fb_plus_bdry_fs', n_basis, lam_hi, fs_frac=0.5, fs_d_over_h=2.0)]
    runs = []
    for sp in specs:
        b, _ = build_basis(domain, sp, lam_hi)
        runs.append(inhouse_eigs(domain, n_eigs, n_basis, basis=b))
    (e0, t0), (e1, t1) = runs
    k = min(len(e0), len(e1), n_eigs)
    disagree = float(np.max(np.abs(e0[:k] - e1[:k])/np.abs(e0[:k]))) if k else float('inf')
    return e0[:k], t0[:k], disagree


def geometry_report(a, b, t):
    d = asym_L(a, b, t)
    ang = np.asarray(d.int_angles)
    re = int((ang > np.pi + 1e-9).sum())
    conv = ang[ang <= np.pi + 1e-9]
    return dict(n_corners=len(ang), n_reentrant=re,
                sharpest_convex=float(conv.min()/np.pi),
                max_nu=float(np.pi/ang.min()), area=float(abs(d.area)),
                leg_ratio=float(max(a, b)/min(a, b)))
