"""C1, the negative control: does contrast actually flag a broken basis?

The program's guard against ill-conditioning is the CONTRAST ratio, median sigma off the
eigenvalues over median sigma at them. The reasoning is that a basis which has lost conditioning
drives tension down everywhere, so a small sigma at an eigenvalue stops meaning anything. That
reasoning is only worth acting on if contrast actually fires on bases we KNOW are broken, and
the threshold used to admit candidates in S1 should be set against measured broken bases rather
than against a round number someone liked.

Two deliberately broken constructions, both of which produce a small-looking sigma:

  interior_sources   boundary FS with check_exterior=False at an offset that puts sources INSIDE
                     a reentrant domain. Sources inside are not particular solutions there, so
                     the tension is meaningless -- `bases.py` documents 24 of 240 interior
                     sources taking the background from 5e-02 to 3e-07 and manufacturing a
                     spurious eigenvalue while hiding a real one.
  overcomplete       a large FS set crammed at a tiny offset, so the columns are near-dependent
                     and the regularized pencil is rank-deficient in every direction.

Against a HEALTHY reference build on the same domain, so the comparison is like-for-like.

WHAT CONTRAST ACTUALLY MEANS, which the first version of this gate got wrong. Contrast ~ 1 says
"this basis has no eigenvalue signal". That is the correct reading for an ill-conditioned basis
AND for a merely weak one -- a basis with sigma = 0.3 everywhere has no signal either, and
flagging it is not a false positive. So cases are classified by BOTH numbers:

    good       small sigma at the eigenvalue, high contrast   -> usable
    weak       large sigma, low contrast                      -> correctly flagged, not a failure
    DANGEROUS  small sigma, low contrast                      -> the case contrast exists to catch

Only the third class tests the metric. A basis whose sigma is small and falling looks like it is
converging; if its contrast is ~1 it is measuring nothing, and raw sigma alone would have
promoted it.

PASS: every constructed-broken case lands at low contrast, including the ones whose sigma looks
good, and genuinely good bases sit orders above. The measured broken band then calibrates S1's
admission rule.
FAIL: a broken basis shows healthy contrast -- redesign the metric before relying on it.

Run: python -m benchmarks.basis_lab.c1_gate
"""
import numpy as np

from benchmarks.basis_lab.probe import default_spec, probe, wavelength
from lappy import geometry as G, reference as ref


def _points(eigs, n_eig=4):
    """(eigenvalues, off-eigenvalue midpoints) from a reference table."""
    e = np.sort(np.asarray(eigs, dtype=float))[:n_eig + 1]
    return e[:n_eig], 0.5*(e[:-1] + e[1:])


# sigma at the eigenvalue below which a basis "looks like it is working"
LOOKS_GOOD = 1e-3


def run(domain_key, domain, eigs, cases_for):
    lam_star, lam_off = _points(eigs)
    lam_max = 2.0*float(np.max(lam_star))
    h = wavelength(lam_max)
    nseg = len(domain.bdry.segments)
    colloc = {'mode': 'pinned', 'n_per_seg': [24]*nseg, 'n_int': 400}

    print(f"\n=== {domain_key}  (wavelength h={h:.4f}) ===")
    print(f"  {'case':30} {'cols':>5} {'drop':>5} {'sig@eig':>10} {'sig@off':>10} "
          f"{'contrast':>9}  class")
    rows = []
    for label, intended, spec in cases_for(lam_max, h):
        r = probe(domain, spec, colloc, lam_star, lam_off, lam_max=lam_max, diagnostics=False)
        if not r['ok']:
            print(f"  {label:30} FAILED {r['error_type']}: {str(r['error_msg'])[:34]}")
            continue
        sig, con = r['sigma_eig_median'], r['contrast']
        looks_good = sig < LOOKS_GOOD
        klass = ('good' if looks_good and con > 1e3 else
                 'DANGEROUS' if looks_good else
                 'weak')
        rows.append((label, intended, sig, con, klass))
        print(f"  {label:30} {r['len_basis_post_norm']:5d} {r['n_sources_dropped']:5d} "
              f"{sig:10.2e} {r['sigma_off_median']:10.2e} {con:9.1e}  {klass}")
    return rows


def _cases(lam_max, h):
    return [
        ('good  pure_fb', 'good', default_spec('pure_fb', 96, lam_max)),
        ('good  pure_fb big', 'good', default_spec('pure_fb', 192, lam_max)),
        ('broken interior d=h n=240', 'broken',
         default_spec('pure_fs_bdry', 240, lam_max, fs_d_over_h=1.0, check_exterior=False)),
        ('broken interior d=2h n=240', 'broken',
         default_spec('pure_fs_bdry', 240, lam_max, fs_d_over_h=2.0, check_exterior=False)),
        ('broken overcomplete d=1e-3h', 'broken',
         default_spec('pure_fs_bdry', 384, lam_max, fs_d=1e-3*h, check_exterior=False)),
    ]


def main():
    rows = []
    # H_shape is reentrant, so an outward offset can genuinely place sources inside it.
    rows += run('H_shape', G.H_shape(), ref.H_shape_eigs(6), _cases)
    rows += run('L_shape', G.L_shape(), ref.L_shape_eigs(6), _cases)

    good = [c for _, intent, _, c, k in rows if intent == 'good' and k == 'good']
    broken = [c for _, intent, _, c, _ in rows if intent == 'broken']
    escaped = [(lbl, c) for lbl, intent, _, c, k in rows if intent == 'broken' and k == 'good']

    print("\n--- C1 verdict ---")
    print(f"  good    contrast: {[f'{c:.1e}' for c in good]}")
    print(f"  broken  contrast: {[f'{c:.1e}' for c in broken]}")
    if not good or not broken:
        print("  inconclusive: a class produced no cases")
        return 1
    if escaped:
        print(f"\n  C1 FAILS: broken bases passed as good -> {escaped}")
        print("  Do not use contrast as an admission rule until it is redesigned.")
        return 1
    band, floor = max(broken), min(good)
    print(f"\n  C1 PASSES: no broken basis reached the good class.")
    print(f"  Broken band tops out at {band:.1e}; worst good basis is {floor:.1e} "
          f"({floor/max(band, 1e-300):.0e}x above).")
    print(f"  Calibration for S1: admit a candidate only at contrast >= {100*max(band, 1.0):.0e}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
