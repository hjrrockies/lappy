"""S0: the per-domain controls that must pass before any knob conclusion is admissible.

C2  reference floor and the frozen domain card
C3  collocation -- choose a pinned setting, and verify insensitivity around it
C4  RNG blocking -- does the interior draw move the answer
C5  rtol -- is pencil regularization a real axis on this domain
C6  off-eigenvalue validity -- does a "midpoint" actually sit away from the spectrum

COLLOCATION IS PINNED RELATIVE TO THE LARGEST BASIS IN THE LADDER, not to a constant. Measured
on L_shape with boundary points fixed at 144, `pure_fb` goes 1.19e-14 at n=64 (ratio 2.25) to
1.30e-16 at n=96 (1.50) to 3.64e-20 with contrast 1.1 at n=128 (1.13) to identically zero beyond,
with `n_reg = n` throughout -- so it is not the regularizer, it is the pencil acquiring null
directions at every lambda once the columns approach the number of boundary constraints. A rule
satisfiable everywhere locates nothing. A size sweep at fixed collocation therefore crosses that
threshold partway up, and the collapse would be misread as the basis running out of
approximation power. Everything here sizes collocation from `n_max`.

Run: python -m benchmarks.basis_lab.s0 <domain_key>
"""
import sys

import numpy as np

from benchmarks.basis_lab import ledger
from benchmarks.basis_lab.probe import KNOB_FIELDS, default_spec, probe, wavelength
from lappy import geometry as G, reference as ref

# Boundary points per basis column at the TOP of the ladder. 1.5 is where the tension stops
# tracking Moler-Payne (docs/basis_heuristics.md) and where contrast had already fallen four
# orders in the L_shape measurement above, so the default sits well clear of it.
BDRY_RATIO = 3.0
INT_RATIO = 2.0


def pinned_colloc(domain, n_max, bdry_ratio=BDRY_RATIO, int_ratio=INT_RATIO, min_per_seg=4):
    """A pinned collocation spec sized from the LARGEST basis the ladder will reach.

    Points are distributed proportionally to segment length -- deliberately NOT via
    `pts_per_seg`, whose rule differs by basis family (Fourier-Bessel skips a corner's own two
    edges) and would therefore reintroduce a family-dependent collocation difference into a
    study whose whole point is to vary the family.
    """
    lens = np.asarray(domain.seg_lens, dtype=float)
    total = int(round(bdry_ratio*n_max))
    raw = total*lens/lens.sum()
    n_per_seg = np.maximum(np.floor(raw).astype(int), min_per_seg)
    deficit = total - int(n_per_seg.sum())
    if deficit > 0:                                  # largest-remainder, so the total is exact
        for i in np.argsort(raw - np.floor(raw))[::-1][:deficit]:
            n_per_seg[i] += 1
    return {'mode': 'pinned', 'n_per_seg': [int(x) for x in n_per_seg],
            'n_int': int(round(int_ratio*n_max))}


def probe_grid_for(lam1, ref_digits, n=7, widen=10.0):
    """Fixed grid around the first eigenvalue, spanning `widen` times the reference's own error.

    Frozen per domain: `sigma_star` is the minimum over THIS grid for every build. A per-build
    adaptive minimum would hand a better basis more search and rebuild the search-tolerance
    confound in new clothing.
    """
    rel = widen*10.0**(-float(ref_digits))
    return list(lam1*(1.0 + np.linspace(-rel, rel, n)))


def _ref_basis(domain, lam_max, n):
    """A known-good reference build for the control stages: pure FB where the domain has
    corners, boundary FS at the wavelength offset where it does not."""
    fam = 'pure_fb' if len(domain.corners) else 'pure_fs_bdry'
    return default_spec(fam, n, lam_max)


def c6_off_points(domain, lam_off, lam_star, spec, colloc, lam_max, rtol=1e-12):
    """Validate each off-eigenvalue: a midpoint sitting on an unlisted mode shows as a dip."""
    keep, dropped = [], []
    for lo in lam_off:
        trio = [lo*(1 - 1e-3), lo, lo*(1 + 1e-3)]
        r = probe(domain, spec, colloc, lam_star, trio, lam_max=lam_max, rtol=rtol,
                  diagnostics=False)
        if not r['ok']:
            dropped.append((float(lo), 'probe failed'))
            continue
        s = r['sigma_off']
        # a genuine non-eigenvalue is flat here; a hidden mode makes the centre a sharp minimum
        if s[1] < min(s[0], s[2])/3.0:
            dropped.append((float(lo), f'dip: {s[0]:.2e} {s[1]:.2e} {s[2]:.2e}'))
        else:
            keep.append(float(lo))
    return keep, dropped


def detect_floor(domain, lam_star, lam_off, lam_max, colloc, n_max, agree=3.0):
    """The level below which sigma is measuring the REFERENCE, not the basis.

    Detected, not assumed, and not taken from one basis. If the reference is the binding limit
    then every good basis bottoms out at the SAME value -- that is exactly how the wrong
    ellipse a=2 eigenvalue announced itself (8.46e-10 for six unrelated configurations,
    identical to three figures). So: build several diverse families at a large size, and if the
    best two agree within `agree`, that common value is the floor. If they disagree, no floor
    has been reached and nothing is censored.

    The first version took the censor from a single `_ref_basis` build, which is pure Fourier-
    Bessel on any domain with corners. On iso_tri(h=8) -- a thin triangle where pure FB is known
    to be the worst family -- that produced a censor of 4.65e-04, marking every genuinely good
    config "below the reference floor" and admitting 0 of 90. A censor derived from one
    arbitrary basis is a measurement of that basis.
    """
    fams = []
    if len(domain.corners):
        fams += [('pure_fb', {}), ('mixed', {'fs_frac': 0.5}),
                 ('fb_plus_bdry_fs', {'fs_frac': 0.5, 'fs_d_over_h': 2.0})]
    fams += [('pure_fs_bdry', {'fs_d_over_h': 2.0})]
    vals = []
    for fam, over in fams:
        try:
            r = probe(domain, default_spec(fam, n_max//2, lam_max, **over), colloc,
                      lam_star, lam_off, lam_max=lam_max, diagnostics=False)
        except Exception:
            continue
        if r['ok'] and np.isfinite(r['sigma_eig_median']) and r['sigma_eig_median'] > 0:
            vals.append((float(r['sigma_eig_median']), fam))
    vals.sort()
    if len(vals) < 2:
        return 0.0, 'fewer than two families built; nothing censored'
    (v0, f0), (v1, f1) = vals[0], vals[1]
    if v1 <= agree*v0:
        return float(np.sqrt(v0*v1)), f'{f0} and {f1} agree ({v0:.2e}, {v1:.2e}) -> common floor'
    return 0.0, (f'best two disagree ({f0} {v0:.2e} vs {f1} {v1:.2e}); '
                 'no common floor, nothing censored')


def build_card(domain_key, domain, eigs, ref_digits, provenance, n_eig=4, n_max=128):
    """C2 + C2b + C6 -> the frozen domain card everything downstream references."""
    e = np.sort(np.asarray(eigs, dtype=float))[:n_eig + 1]
    lam_star, lam_off = [float(x) for x in e[:n_eig]], [float(x) for x in 0.5*(e[:-1] + e[1:])]
    lam_max = 2.0*float(max(lam_star))
    colloc = pinned_colloc(domain, n_max)
    spec = _ref_basis(domain, lam_max, n_max//2)

    keep, dropped = c6_off_points(domain, lam_off, lam_star, spec, colloc, lam_max)
    print(f"  C6 off-points: kept {len(keep)} of {len(lam_off)}"
          + (f"; dropped {dropped}" if dropped else ""))

    censor, why = detect_floor(domain, lam_star, keep, lam_max, colloc, n_max)
    print(f"  C2 censor: {censor:.2e} ({why})")
    card = dict(domain_key=domain_key, lam_star=lam_star, lam_off=keep,
                lam_star_provenance=provenance, ref_floor_digits=float(ref_digits),
                probe_grid=probe_grid_for(lam_star[0], ref_digits), lam_max=lam_max,
                n_max=n_max, bdry_ratio=BDRY_RATIO, int_ratio=INT_RATIO,
                n_per_seg=colloc['n_per_seg'], n_int=colloc['n_int'],
                sigma_censor=censor, off_points_dropped=dropped)
    cid = ledger.put_card(card)
    print(f"  card {cid}: lam1={lam_star[0]:.12f} ({provenance}, ~{ref_digits} digits), "
          f"n_bdry={sum(colloc['n_per_seg'])} n_int={colloc['n_int']}, censor={censor:.2e}")
    return cid, card


def informative_sizes(domain, card, ladder=(8, 12, 16, 24, 32, 48, 64, 96, 128),
                      headroom=1e3, n_pick=2):
    """Sizes at which the controls can actually detect anything.

    The first version of this ran C3/C4/C5 at a hardcoded (48, 96) and reported every knob
    "insensitive" on the square -- because at n>=48 the square sits at the machine floor
    (3-6e-16) and NOTHING can move there. A control run at saturation measures the floor, not
    the knob. So the sizes come from the domain's own curve: keep those whose sigma* is at least
    `headroom` above the censor level, and pick the largest such (closest to interesting) plus
    one well below it.
    """
    spec_of = lambda n: _ref_basis(domain, card['lam_max'], n)
    colloc = pinned_colloc(domain, card['n_max'])
    floor = max(card.get('sigma_censor') or 1e-16, 1e-16)
    usable = []
    for n in ladder:
        r = _run(domain, card, spec_of(n), colloc, stage='S0.sizes')
        if r['ok'] and np.isfinite(r['sigma_star']) and r['sigma_star'] > headroom*floor:
            usable.append((n, r['sigma_star']))
    if not usable:
        return tuple(ladder[:n_pick])
    top = usable[-1][0]
    low = usable[max(0, len(usable)//2 - 1)][0]
    picked = tuple(sorted({low, top}))
    print(f"  informative sizes (sigma* > {headroom:.0e} x censor {floor:.1e}): "
          + ', '.join(f'n={n}:{s:.1e}' for n, s in usable)
          + f'  -> using {picked}')
    return picked


def _run(domain, card, spec, colloc, rtol=1e-12, seed=0, stage='S0'):
    r = probe(domain, spec, colloc, card['lam_star'], card['lam_off'],
              probe_grid=card['probe_grid'], lam_max=card['lam_max'], rtol=rtol, seed=seed)
    r['domain_card_id'] = card.get('card_id')
    r['stage'] = stage
    return r


def c3_collocation(domain_key, domain, card, ns, ratios=(1.5, 2, 3, 4, 6),
                   int_ratios=(1, 2, 3, 5)):
    """Sweep both collocation axes at fixed bases; report where the answer stops moving."""
    print(f"\n  C3 collocation ({domain_key})")
    print(f"    {'n':>4} {'bdry_r':>7} {'int_r':>6} {'n_bdry':>7} {'n_int':>6} "
          f"{'sigma*':>10} {'contrast':>9}")
    rows = []
    for n in ns:
        for br in ratios:
            for ir in int_ratios:
                c = pinned_colloc(domain, card['n_max'], bdry_ratio=br, int_ratio=ir)
                spec = _ref_basis(domain, card['lam_max'], n)
                r = _run(domain, card, spec, c, stage='S0.C3')
                r['bdry_ratio'], r['int_ratio_used'] = br, ir
                ledger.append(domain_key, r, KNOB_FIELDS)
                if r['ok']:
                    rows.append((n, br, ir, r['sigma_star'], r['contrast']))
                    print(f"    {n:>4} {br:>7.1f} {ir:>6.1f} {r['n_bdry_total']:>7} "
                          f"{r['n_int']:>6} {r['sigma_star']:10.2e} {r['contrast']:9.1e}")
    return rows


def c4_rng(domain_key, domain, card, ns, seeds=(0, 1, 2, 3, 4)):
    print(f"\n  C4 rng blocking ({domain_key})")
    out = {}
    for n in ns:
        vals = []
        for sd in seeds:
            spec = _ref_basis(domain, card['lam_max'], n)
            c = pinned_colloc(domain, card['n_max'])
            r = _run(domain, card, spec, c, seed=sd, stage='S0.C4')
            ledger.append(domain_key, r, KNOB_FIELDS)
            if r['ok'] and r['sigma_star'] > 0:
                vals.append(np.log10(r['sigma_star']))
        if vals:
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            out[n] = iqr
            print(f"    n={n:<4} log10(sigma*) spread: iqr={iqr:.2f} "
                  f"range={max(vals)-min(vals):.2f}")
    return out


def c5_rtol(domain_key, domain, card, ns, rtols=(1e-14, 1e-12, 1e-10)):
    print(f"\n  C5 rtol ({domain_key})")
    out = {}
    for n in ns:
        vals = []
        for rt in rtols:
            spec = _ref_basis(domain, card['lam_max'], n)
            c = pinned_colloc(domain, card['n_max'])
            r = _run(domain, card, spec, c, rtol=rt, stage='S0.C5')
            ledger.append(domain_key, r, KNOB_FIELDS)
            if r['ok']:
                vals.append(np.log10(max(r['sigma_star'], 1e-300)))
                print(f"    n={n:<4} rtol={rt:.0e}  sigma*={r['sigma_star']:.2e}  "
                      f"n_reg={r['n_reg']}/{r['len_basis_post_norm']}")
        if vals:
            out[n] = float(max(vals) - min(vals))
    return out


# (builder, eigenvalue table, documented digits, provenance). The digit figure is what gates
# censoring, so it is the DOCUMENTED claim -- and ellipse a=2 lambda_1 is a standing reminder
# that a documented claim can be four orders optimistic, which is why C2 exists.
DOMAINS = {
    'square':      (lambda: G.rect(1.0, 1.0), lambda k: ref.rect_eigs(k, 1.0, 1.0),
                    15.0, 'analytic'),
    'disk':        (lambda: G.disk(1.0), lambda k: ref.disk_eigs(k, 1.0), 15.0, 'analytic'),
    'iso_right_tri': (lambda: G.iso_right_tri(1.0), lambda k: ref.iso_right_tri_eigs(k, 1.0),
                      15.0, 'analytic'),
    'L_shape':     (G.L_shape, ref.L_shape_eigs, 14.0, 'reference_table'),
    'iso_tri_h1':  (lambda: G.iso_tri(1.0), lambda k: ref.iso_tri_eigs(k, 1.0),
                    13.0, 'reference_table'),
    'iso_tri_h05': (lambda: G.iso_tri(0.5), lambda k: ref.iso_tri_eigs(k, 0.5),
                    10.8, 'reference_table'),
    'iso_tri_h2':  (lambda: G.iso_tri(2.0), lambda k: ref.iso_tri_eigs(k, 2.0),
                    12.0, 'reference_table'),
    'iso_tri_h4':  (lambda: G.iso_tri(4.0), lambda k: ref.iso_tri_eigs(k, 4.0),
                    11.8, 'reference_table'),
    'iso_tri_h8':  (lambda: G.iso_tri(8.0), lambda k: ref.iso_tri_eigs(k, 8.0),
                    11.3, 'reference_table'),
    'chevron_1_2': (lambda: G.chevron(1.0, 2.0), lambda k: ref.chevron_eigs(k, 1.0, 2.0),
                    12.0, 'reference_table'),
    'H_shape':     (G.H_shape, ref.H_shape_eigs, 7.8, 'reference_table'),
}


def main(argv):
    key = argv[1] if len(argv) > 1 else 'square'
    build, eigs_fn, digits, prov = DOMAINS[key]
    domain = build()
    print(f"=== S0 controls: {key} ===")
    cid, card = build_card(key, domain, eigs_fn(6), digits, prov)
    card = dict(card, card_id=cid)

    ns = informative_sizes(domain, card)
    c3 = c3_collocation(key, domain, card, ns)
    c4 = c4_rng(key, domain, card, ns)
    c5 = c5_rtol(key, domain, card, ns)

    print(f"\n--- S0 summary: {key} ---")
    if c3:
        by_n = {}
        for n, br, ir, sig, con in c3:
            by_n.setdefault(n, []).append(np.log10(max(sig, 1e-300)))
        spread3 = max(max(v) - min(v) for v in by_n.values())
        print(f"  C3 colloc  : {'SENSITIVE' if spread3 > 0.3 else 'insensitive'}"
              f"  (log10 sigma* spread across the grid: {spread3:.2f})")
    print(f"  sizes used : {ns}")
    print(f"  C4 rng     : {'SENSITIVE' if any(v > 0.3 for v in c4.values()) else 'insensitive'}"
          f"  (iqr {[f'{v:.2f}' for v in c4.values()]})")
    print(f"  C5 rtol    : {'SENSITIVE' if any(v > 0.3 for v in c5.values()) else 'insensitive'}"
          f"  (spread {[f'{v:.2f}' for v in c5.values()]})")
    print(f"  rows written to {ledger.domain_path(key)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
