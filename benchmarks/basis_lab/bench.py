"""A common measuring stick for basis construction.

Every claim about basis choice in this project so far has been scored differently -- certified
digits from one study, `n_reg/n` from another, tension minima from a third -- so the claims
cannot be compared to each other, and several turned out to be measuring cost rather than
quality. This scores any basis-building callable the same way on the same domains.

It deliberately proposes nothing. `docs/basis_heuristics.md` is *a* theory with partial
evidence (the fundamental-solution offset scaling with wavelength is its empirically grounded
part); the current `make_default_basis` embodies another. Both are hypotheses to be measured
here, not specifications to implement.

OBJECTIVE. Certified relative error from Moler--Payne (`benchmarks/reference/certify.py`),
reported as `digits = -log10(eps)`. It needs no reference values, so it works on every domain
rather than only the ones with analytic truth -- which matters, because the domains where basis
choice is hard are exactly the ones with no closed form. Where analytic truth IS available it
is reported alongside, as a check ON the bound rather than as the score.

GUARD. A median tension far below what Moler--Payne allows away from an eigenvalue means the
basis or the collocation is broken, and digits from such a run are meaningless rather than
merely bad (`benchmarks/suite/preflight.background_suspect`). Runs that trip it are flagged.

COST. `n_basis` and wall time travel with every score, so "better" cannot quietly mean "bigger".
That is not hypothetical: two of the study results this harness exists to re-check were size
effects read as placement effects.

REPRODUCIBILITY. `rng` is threaded to the interior draw. Without it two builds of the same
domain differ in every coefficient and no A/B comparison means anything.

THE SEARCH MUST NOT BE THE LIMIT, and it was. The first version of this harness called
`solver.solve_interval`, whose per-bracket minimizer converges to `ltol_default = 1e-8`. That
caps the measurable accuracy around 10 digits no matter how good the basis is, so every curve
it produced was a picture of the eigenvalue search and not of the basis. It manufactured a
"~10 digit plateau" on L_shape and plus_shape, and non-monotone bouncing everywhere, both of
which were reported as basis findings before the check below was run:

    L_shape, pure_fb, TRUE digits against lappy.reference

        n_basis                 64     160     240
        solve_interval        10.7    12.0     9.3
        manual_solve+polish   15.7    14.5    14.6

Fifteen digits at n=64, where the coarse path reads 10.7 and falls to 9.3 by n=240. So this
harness uses the polished path (`benchmarks/reference/common.solve_domain_v2`: `manual_solve`
with `minimize_tol=1e-12`, then `polish_eigs` at `ltol=1e-14`), which is also what produced the
reference tables in `lappy.reference`.
"""
import time
import traceback
import warnings

import numpy as np

from lappy import bases, bounds
from lappy.bases import (FourierBesselBasis, FundamentalBasis, fb_corner_orders,
                         fs_bdry_sps, fs_corner_orders)
from lappy.mps import MPSEigensolver, weyl_est


def _polished_solve(solver, domain, n_eigs, n_pts_per_eig=11, bracket_xtol=1e-5,
                    minimize_tol=1e-12, ttol=1e-3, n_workers=4):
    """`manual_solve` + `polish_eigs`, i.e. `solve_domain_v2`'s search, on a prebuilt solver.

    Not `solve_interval`: its per-bracket minimizer stops at `ltol_default = 1e-8`, which caps
    measurable accuracy near 10 digits and would make this harness a measurement of the search.
    Imported from `benchmarks/reference/common.py`, which uses flat imports, hence the path
    insertion -- the alternative is duplicating the pipeline that produced `lappy.reference`,
    which would be worse.
    """
    import os
    import sys
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'reference')
    if here not in sys.path:
        sys.path.insert(0, here)
    from common import manual_solve, polish_eigs, lambda_window

    a, b = lambda_window(domain, n_eigs)
    n_pts = max(n_pts_per_eig*n_eigs, 50)
    eigs, mults, _ = manual_solve(solver, a, b, n_pts, bracket_xtol=bracket_xtol,
                                  minimize_tol=minimize_tol, ttol=ttol, n_workers=n_workers)
    eigs, tensions = polish_eigs(solver, eigs, ltol=1e-14, bracket_rel_width=1e-9)
    return eigs, mults, tensions


def evaluate(domain, build_basis, n_eigs=4, rng=7, mp_kwargs=None, truth_fn=None,
             n_extra=2):
    """Score one (domain, basis-constructor) pair.

    `build_basis(domain, lam_max)` returns a ParticularBasis -- deliberately the signature the
    new paradigm is aiming at, so a constructor that sizes itself from `lam_max` needs no
    special case here and today's `n_basis` constructors get wrapped instead.

    Returns a dict; `ok=False` with `error`/`traceback` on failure rather than raising, so one
    broken cell does not abandon a sweep.
    """
    from benchmarks.reference.certify import moler_payne

    out = {'ok': False}
    t0 = time.time()
    try:
        lam_max = weyl_est(n_eigs + n_extra, domain)
        basis = build_basis(domain, lam_max)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = MPSEigensolver.from_domain(domain, basis=basis, rng=rng)
            eigs, _mults, tensions = _polished_solve(solver, domain, n_eigs)
        eigs = np.atleast_1d(np.asarray(eigs)).ravel()
        out['tensions'] = list(np.atleast_1d(tensions)[:n_eigs])
        out.update(n_basis=len(basis), n_found=len(eigs), eigs=eigs[:n_eigs], lam_max=lam_max)

        digits = []
        for lam in eigs[:n_eigs]:
            u = solver.eigenfunction(float(lam), mult=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mp = moler_payne(domain, lambda z, u=u: u(z)[:, 0], float(lam),
                                 **(mp_kwargs or {}))
            digits.append(float(mp['digits']))
        out['digits'] = digits
        out['worst_digits'] = min(digits) if digits else float('nan')

        if truth_fn is not None:
            truth = np.asarray(truth_fn(n_eigs), dtype=float)
            k = min(len(truth), len(out['eigs']))
            out['true_digits'] = [float(-np.log10(abs(a - b)/b))
                                  for a, b in zip(out['eigs'][:k], truth[:k])]
        out['ok'] = True
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
        out['traceback'] = traceback.format_exc()
    out['seconds'] = time.time() - t0
    return out


def report(name, r):
    """One line per (domain, construction). Failures print their reason, never a number."""
    if not r['ok']:
        print(f"  {name:22} FAILED  {r['error'][:70]}  ({r['seconds']:.0f}s)")
        return
    d = ' '.join(f'{x:5.1f}' for x in r['digits'])
    extra = ('   true: ' + ' '.join(f'{x:5.1f}' for x in r['true_digits'])
             if 'true_digits' in r else '')
    print(f"  {name:22} n={r['n_basis']:4d} found={r['n_found']:2d} "
          f"worst={r['worst_digits']:5.1f}  MP digits: {d}{extra}  ({r['seconds']:.0f}s)")


# ── Constructions, separated from the branch logic that currently picks between them ─────────
#
# `make_default_basis` selects one of these from corner structure alone: pure boundary FS when
# there are no corners, pure corner Fourier-Bessel with zero or one singular corner, and a
# 50/50 FB + lightning-FS mix with several. Those are good ideas and they are also untested
# ones -- no domain has been scored under a construction its branch would not have chosen, so
# the branch boundaries have never been measured. Building each construction directly is what
# makes that cross-product possible.

def pure_fb(domain, lam_max, n=160):
    """Fourier-Bessel at every corner, angle-weighted."""
    return FourierBesselBasis.from_domain(domain, fb_corner_orders(domain, n))


def pure_fs_bdry(domain, lam_max, n=160, d=None, order=1, c_wave=1.0):
    """Fundamental solutions offset along the boundary.

    `d=None` scales the offset with the wavelength at the top of the window,
    `d = c_wave * 2 pi / sqrt(lam_max)`. That is the one part of `docs/basis_heuristics.md`
    with real empirical support (seven of eight measured optima inside 0.73-1.14 h), and it is
    here as a hypothesis to keep testing, not as a settled rule. A float pins `d` instead.
    """
    if d is None:
        d = c_wave*2*np.pi/np.sqrt(lam_max)
    n_sources = int(round(n/(2*(order - 1) + 1)))
    return FundamentalBasis.by_boundary(domain, fs_bdry_sps(domain, n_sources, order),
                                        d=d, order=order)


def fs_corners(domain, lam_max, n=160, C=10.0, sigma=1.0, order=2):
    """Exponentially clustered ('lightning') sources at the corners."""
    return FundamentalBasis.by_corners(domain, fs_corner_orders(domain, n, order=order),
                                       C, sigma, order)


def mixed(domain, lam_max, n=160, fs_frac=0.5, **kw):
    """The multi-singular-corner branch: corner FB plus lightning FS."""
    n_fs = int(round(fs_frac*n))
    return pure_fb(domain, lam_max, n - n_fs) + fs_corners(domain, lam_max, n_fs, **kw)


def fb_plus_bdry_fs(domain, lam_max, n=160, fs_frac=0.5, **kw):
    """Corner FB plus BOUNDARY-offset FS -- a combination the constructor has no branch for.

    Included because the two FS placements are never compared against each other: `by_corners`
    is used wherever FS meets a domain with corners, `by_boundary` only where there are none.
    That split is an assumption about which placement suits which geometry, and it is one of
    the things this harness exists to check.
    """
    n_fs = int(round(fs_frac*n))
    return pure_fb(domain, lam_max, n - n_fs) + pure_fs_bdry(domain, lam_max, n_fs, **kw)


def fixed_n(n):
    """Today's constructor pinned at a size: the baseline any proposal has to beat."""
    def build(domain, lam_max):
        return bases.make_default_basis(domain, n)
    build.__name__ = f'default(n={n})'
    return build


# ── Convergence RATE, which is the thing a single size cannot see ────────────────────────────
#
# Different geometries should converge differently in kind, not merely in amount: corner-centred
# Fourier-Bessel can converge spectrally on a domain whose only singularity it is centred on,
# while several singular corners appear to prevent that. A score at one fixed `n` cannot tell
# those apart, and worse, it can rank two constructions differently depending on which `n` was
# chosen. So the measurement is a curve.
#
# Three rate models are fitted to digits(n), since all three occur in this literature:
#
#     exponential        err ~ exp(-c n)        digits linear in n
#     root-exponential   err ~ exp(-c sqrt(n))  digits linear in sqrt(n)   (lightning/FS at corners)
#     algebraic          err ~ n^-p             digits linear in log10(n)
#
# A WARNING ABOUT TRUTH SOURCES. Do not use polyominoes for this. Their closed-form
# eigenfunctions sin(m pi x) sin(n pi y) vanish on the whole integer grid and are therefore
# SMOOTH at every reentrant corner -- zero singular amplitude -- so a convergence curve measured
# against `polyomino_eig` is a curve for a problem with no corner singularity in it, no matter
# how many reentrant corners the domain has. That is the same trap `test_eigfun_integrals.py`
# documents as Leg 2. Circular sectors are the honest exact-truth option for a singular corner.

def convergence(domain, build_basis, ns, truth=None, n_eigs=1, rng=7, mp_kwargs=None,
                floor_digits=13.0, **kw):
    """digits(n) for one construction. Returns a list of (n, n_basis, digits, seconds).

    `truth` is a float (the exact eigenvalue to compare the first computed one against) or
    None, in which case Moler--Payne supplies a certified bound instead. Exact truth is much
    cheaper and is preferred wherever a closed form exists.
    """
    rows = []
    for n in ns:
        r = evaluate(domain, lambda d, l, n=n: build_basis(d, l, n=n), n_eigs=n_eigs, rng=rng,
                     mp_kwargs=mp_kwargs, **kw)
        if not r['ok']:
            rows.append((n, None, float('nan'), r['seconds'], r['error']))
            continue
        if len(r.get('eigs', [])) == 0:
            # a construction that finds NO eigenvalue in the window has failed, and saying so
            # is the point -- silently scoring nan here would hide a broken branch
            rows.append((n, r['n_basis'], float('nan'), r['seconds'],
                         'no eigenvalue found in window'))
            continue
        if truth is None:
            dig = r['worst_digits']
        else:
            lam = float(r['eigs'][0])
            dig = float(-np.log10(abs(lam - truth)/abs(truth))) if lam != truth else np.inf
        rows.append((n, r['n_basis'], dig, r['seconds'], None))
    return rows


def classify_rate(rows, floor_digits=13.0):
    """Which rate model fits digits(n) best, ignoring saturated points.

    Points at or above `floor_digits` are dropped: once the error reaches the double-precision
    floor the curve flattens for reasons that have nothing to do with the basis, and including
    them makes every construction look algebraic.
    """
    pts = [(n, d) for n, _, d, _, err in rows
           if err is None and np.isfinite(d) and d < floor_digits]
    if len(pts) < 3:
        return {'model': 'insufficient', 'n_used': len(pts)}
    n = np.array([p[0] for p in pts], dtype=float)
    d = np.array([p[1] for p in pts], dtype=float)

    out = {}
    for name, x in (('exponential', n), ('root-exp', np.sqrt(n)), ('algebraic', np.log10(n))):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((d - pred)**2))
        ss_tot = float(np.sum((d - d.mean())**2))
        out[name] = {'slope': float(coef[0]),
                     'r2': 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')}
    best = max(out, key=lambda k: out[k]['r2'])
    # A curve that is flat, or noisy, has no rate to report. Naming the best of three bad fits
    # would dress up noise as a finding -- the sector's FB curve sits at 12-13 digits for every
    # n in the ladder and the "best" model came back with r2 = 0.12.
    spread = float(d.max() - d.min())
    if out[best]['r2'] < 0.9 or spread < 1.0:
        return {'model': 'flat-or-noisy', 'r2': out[best]['r2'], 'spread': spread,
                'level': float(d.mean()), 'all': out, 'n_used': len(pts),
                'hint': 'already at the accuracy floor for this ladder; use smaller n'}
    return {'model': best, 'r2': out[best]['r2'], 'slope': out[best]['slope'],
            'spread': spread, 'all': out, 'n_used': len(pts)}


def report_convergence(name, rows, floor_digits=13.0):
    cls = classify_rate(rows, floor_digits)
    curve = '  '.join(f'{n}:{d:.1f}' if np.isfinite(d) else f'{n}:--' for n, _, d, _, _ in rows)
    if cls['model'] == 'insufficient':
        tail = f"insufficient ({cls['n_used']} unsaturated pts)"
    elif cls['model'] == 'flat-or-noisy':
        tail = f"flat/noisy   at ~{cls['level']:.1f} digits (spread {cls['spread']:.1f})"
    else:
        tail = f"{cls['model']:12} r2={cls['r2']:.3f} slope={cls['slope']:.3f}"
    print(f"  {name:22} {tail}   {curve}")
    return cls
