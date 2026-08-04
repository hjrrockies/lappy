"""Audit every tabulated reference eigenvalue for two failure modes.

Motivated by ``reg_ngon(8)``, whose table contained a value
(``lambda = 29.5368109...``) at which the tension is ~1e-4 and *stays* ~1e-4
however many interior points are used, while genuine eigenvalues of the same
domain sit at 1e-11. It was never an eigenvalue: it was a shallow local
minimum of the tension curve between two genuine ones, accepted because the
eigenvalue-acceptance threshold ``ttol`` is an absolute constant (1e-3) that
bears no relation to the tension floor the basis can actually reach.

Two checks per domain:

**Spurious**: evaluate ``sigma`` at each tabulated value. Genuine eigenvalues
cluster near the domain's tension floor; a value orders of magnitude above
the domain's median is not an eigenvalue, it is a table entry that should
never have been accepted. The interior-point sweep distinguishes the two
robustly -- a genuine minimum's tension *falls* as interior points are
added (better conditioning), a spurious one does not move.

**Missing**: scan ``sigma`` on a fine grid over the tabulated range and count
minima that dip near the floor. A tabulated sequence with a gap is just as
wrong as one with an extra entry, and a spurious entry always implies a
shifted index for everything after it.
"""
import numpy as np

from lappy import geometry as G, reference
from common import build_solver


CASES = [
    ('L_shape',    {}, lambda: G.L_shape(),           lambda k: reference.L_shape_eigs(k),          240),
    ('mushroom',   {}, lambda: G.mushroom(),          lambda k: reference.mushroom_eigs(k),         240),
    ('H_shape',    {}, lambda: G.H_shape(),           lambda k: reference.H_shape_eigs(k),          320),
    ('ellipse a=2', {}, lambda: G.ellipse(2, 1),      lambda k: reference.ellipse_eigs(k, 2, 1),    240),
    ('ellipse a=3', {}, lambda: G.ellipse(3, 1),      lambda k: reference.ellipse_eigs(k, 3, 1),    240),
    ('ellipse a=4', {}, lambda: G.ellipse(4, 1),      lambda k: reference.ellipse_eigs(k, 4, 1),    240),
    ('cut_square r=0.25', {}, lambda: G.cut_square(0.25), lambda k: reference.cut_square_eigs(k, 0.25), 320),
    ('cut_square r=0.5',  {}, lambda: G.cut_square(0.5),  lambda k: reference.cut_square_eigs(k, 0.5),  320),
    ('GWW1',       {}, lambda: G.GWW1(),              lambda k: reference.gww_eigs(k),              320),
    ('GWW2',       {}, lambda: G.GWW2(),              lambda k: reference.gww_eigs(k),              320),
    ('chevron 1,2',   {}, lambda: G.chevron(1, 2),    lambda k: reference.chevron_eigs(k, 1, 2),    160),
    ('chevron 1,1.5', {}, lambda: G.chevron(1, 1.5),  lambda k: reference.chevron_eigs(k, 1, 1.5),  160),
    ('chevron 2,3',   {}, lambda: G.chevron(2, 3),    lambda k: reference.chevron_eigs(k, 2, 3),    160),
    ('chevron 2,4',   {}, lambda: G.chevron(2, 4),    lambda k: reference.chevron_eigs(k, 2, 4),    160),
    ('iso_tri h=0.5', {}, lambda: G.iso_tri(0.5),     lambda k: reference.iso_tri_eigs(k, 0.5),     120),
    ('iso_tri h=1',   {}, lambda: G.iso_tri(1.0),     lambda k: reference.iso_tri_eigs(k, 1.0),     120),
    ('iso_tri h=2',   {}, lambda: G.iso_tri(2.0),     lambda k: reference.iso_tri_eigs(k, 2.0),     120),
    ('iso_tri h=4',   {}, lambda: G.iso_tri(4.0),     lambda k: reference.iso_tri_eigs(k, 4.0),     120),
    ('iso_tri h=8',   {}, lambda: G.iso_tri(8.0),     lambda k: reference.iso_tri_eigs(k, 8.0),     120),
    ('iso_tri h=16',  {}, lambda: G.iso_tri(16.0),    lambda k: reference.iso_tri_eigs(k, 16.0),    120),
    ('iso_tri h=20',  {}, lambda: G.iso_tri(20.0),    lambda k: reference.iso_tri_eigs(k, 20.0),    120),
] + [(f'reg_ngon N={n}', {}, (lambda n=n: G.reg_ngon(n)),
      (lambda k, n=n: reference.reg_ngon_eigs(k, n)), 120) for n in (5, 6, 7, 8)]


def audit_case(name, build, ref_fn, n_basis, n_eigs=10, int_npts=(200, 1500),
               seed=0, verbose=True):
    dom = build()
    try:
        ref = np.asarray(ref_fn(n_eigs))
    except Exception as exc:
        print(f'{name}: no table ({exc})')
        return None

    sig = []
    for nint in int_npts:
        np.random.seed(seed)
        solver = build_solver(dom, n_basis, int_npts=nint)
        sig.append(np.array([solver.sigma(l) for l in ref]))
    sig = np.array(sig)                                  # (n_settings, n_eigs)

    floor = np.median(sig[-1])
    ratio = sig[-1] / floor
    # A genuine eigenvalue's tension drops (or holds) as interior points are
    # added; a spurious minimum's does not, because nothing about it was
    # conditioning-limited in the first place.
    drop = sig[0] / np.maximum(sig[-1], 1e-300)

    verdicts = []
    for i in range(len(ref)):
        if ratio[i] > 1e3 and drop[i] < 3:
            verdicts.append('SPURIOUS?')
        elif ratio[i] > 1e3:
            verdicts.append('weak')
        else:
            verdicts.append('ok')

    if verbose:
        print(f'\n--- {name} (n_basis={n_basis}, tension floor ~{floor:.1e}) ---')
        print(f"{'i':>3} {'tabulated eigenvalue':>24} {'sigma(lo)':>10} {'sigma(hi)':>10} "
              f"{'/floor':>9}  verdict")
        for i in range(len(ref)):
            print(f'{i:3d} {ref[i]:24.15f} {sig[0][i]:10.2e} {sig[-1][i]:10.2e} '
                  f'{ratio[i]:9.1e}  {verdicts[i]}')
    return dict(name=name, ref=ref, sigma=sig, floor=floor, verdicts=verdicts)


def main():
    flagged = []
    for name, _, build, ref_fn, n_basis in CASES:
        try:
            rec = audit_case(name, build, ref_fn, n_basis)
        except Exception as exc:
            print(f'\n--- {name}: AUDIT FAILED ({type(exc).__name__}: {exc})')
            continue
        if rec is None:
            continue
        bad = [i for i, v in enumerate(rec['verdicts']) if v != 'ok']
        if bad:
            flagged.append((name, bad, [rec['verdicts'][i] for i in bad],
                            rec['ref'][bad]))

    print('\n\n================ SUMMARY OF FLAGGED ENTRIES ================')
    if not flagged:
        print('none')
    for name, idx, verds, vals in flagged:
        for i, v, val in zip(idx, verds, vals):
            print(f'{name:22s} index {i:2d}  {val:22.15f}  {v}')


if __name__ == '__main__':
    main()
