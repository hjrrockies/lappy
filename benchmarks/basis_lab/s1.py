"""S1: the coarse screen. One factor at a time from each family's centre, plus two small
factorials, at two basis sizes -- then an admission rule that says which configs earn a fine
sweep.

WHY OFAT AND NOT A FACTORIAL. A full grid over d x order x spacing x C x sigma x fs_frac x size
is ~2000 builds per domain and produces a table nobody reads. OFAT from the CENTRE that
`make_default_basis` currently uses means the reference point is the status quo, so every number
is "better or worse than what we ship". The known cost of OFAT is that it cannot see an
interaction between two knobs that are each unremarkable alone, so two 3x3 factorials are added
where interaction is most likely -- (d, order) for boundary sources and (C, sigma) for
lightning ones.

OBJECTIVE is `sigma_hat` (collocation-normalized) and CONTRAST, both from `probe`. Raw sigma is
recorded but not ranked on: it carries a sqrt(n_bdry/n_int) factor that has nothing to do with
the basis.

ADMISSION to S2 requires all of:
  * zero dropped sources, and no warnings at all
  * contrast >= CONTRAST_MIN, calibrated by C1 against deliberately broken bases (the broken
    band topped out at 3.7; good bases sat at 2.5e+04 and above)
  * sigma_hat within one decade of the best uncensored config at that size
  * not censored -- i.e. not below the reference's own resolution, where the number is the
    reference's error rather than the basis's

Run: python -m benchmarks.basis_lab.s1 <domain_key>
"""
import sys

import numpy as np

from benchmarks.basis_lab import ledger, s0
from benchmarks.basis_lab.probe import KNOB_FIELDS, default_spec, probe
from lappy import geometry as G, reference as ref

CONTRAST_MIN = 4e2          # from C1: 100x the measured broken band


def families_for(domain):
    """Which constructions even apply. A corner-free domain has no Fourier-Bessel branch and no
    lightning corners, so offering them would only manufacture failed rows."""
    if len(domain.corners) == 0:
        return ('pure_fs_bdry',)
    return ('pure_fb', 'pure_fs_bdry', 'fs_corners', 'mixed', 'fb_plus_bdry_fs')


def plan_specs(domain, lam_max, n):
    """(label, spec) for every coarse-screen point at one size."""
    fams = families_for(domain)
    out = []

    def add(label, fam, **over):
        if fam in fams:
            out.append((label, default_spec(fam, n, lam_max, **over)))

    # FB: prior findings say re-allocation is nearly inert, so confirm cheaply rather than sweep
    add('fb centre', 'pure_fb')

    # boundary FS: offset, multipole order, spacing
    for dh in (0.05, 0.15, 0.5, 1.0, 2.0, 4.0):
        add(f'fs_bdry d/h={dh}', 'pure_fs_bdry', fs_d_over_h=dh)
    for o in (1, 2, 3):
        add(f'fs_bdry order={o}', 'pure_fs_bdry', fs_order=o)
    for sp in ('even', 'legendre'):
        add(f'fs_bdry spacing={sp}', 'pure_fs_bdry', fs_spacing=sp)

    # lightning corner FS: far distance, clustering rate, multipole order
    for C in (2.0, 10.0, 50.0):
        add(f'fs_corn C={C}', 'fs_corners', fs_C=C)
    for sg in (0.5, 1.0, 2.0):
        add(f'fs_corn sigma={sg}', 'fs_corners', fs_sigma=sg)
    for o in (1, 2, 3):
        add(f'fs_corn order={o}', 'fs_corners', fs_order=o)

    # blends
    for fr in (0.25, 0.5, 0.75):
        add(f'mixed frac={fr}', 'mixed', fs_frac=fr)
        add(f'fb+bdry frac={fr}', 'fb_plus_bdry_fs', fs_frac=fr)

    # the two interaction probes OFAT cannot see
    for dh in (0.15, 1.0, 4.0):
        for o in (1, 2, 3):
            add(f'FACT d/h={dh},order={o}', 'pure_fs_bdry', fs_d_over_h=dh, fs_order=o)
    for C in (2.0, 10.0, 50.0):
        for sg in (0.5, 1.0, 2.0):
            add(f'FACT C={C},sigma={sg}', 'fs_corners', fs_C=C, fs_sigma=sg)
    return out


def screen(domain_key, domain, card, ns):
    colloc = s0.pinned_colloc(domain, card['n_max'])
    censor = card.get('sigma_censor') or 0.0
    all_rows = []
    for n in ns:
        print(f"\n  --- {domain_key}, n={n} ---")
        print(f"    {'config':26} {'cols':>5} {'drop':>4} {'sigma_hat':>10} {'contrast':>9}  flag")
        rows = []
        for label, spec in plan_specs(domain, card['lam_max'], n):
            r = probe(domain, spec, colloc, card['lam_star'], card['lam_off'],
                      probe_grid=card['probe_grid'], lam_max=card['lam_max'], rtol=1e-12)
            r['domain_card_id'] = card['card_id']
            r['stage'] = 'S1'
            r['label'] = label
            ledger.append(domain_key, r, KNOB_FIELDS)
            if not r['ok']:
                print(f"    {label:26} FAILED {r['error_type']}")
                continue
            r['censored'] = bool(r['sigma_star'] <= censor)
            rows.append(r)
        if not rows:
            continue
        live = [r for r in rows if not r['censored']]
        best = min((r['sigma_hat'] for r in live), default=None)
        for r in rows:
            flags = []
            if r['n_sources_dropped']:
                flags.append(f"drop{r['n_sources_dropped']}")
            if r['warnings']:
                flags.append('warn')
            if r['censored']:
                flags.append('censored')
            if (r['contrast'] or 0) < CONTRAST_MIN:
                flags.append('low-contrast')
            if best is not None and not r['censored'] and r['sigma_hat'] > 10*best:
                flags.append('off-pace')
            r['admitted'] = not flags
            print(f"    {r['label']:26} {r['len_basis_post_norm']:5d} "
                  f"{r['n_sources_dropped']:4d} {r['sigma_hat']:10.2e} "
                  f"{(r['contrast'] or 0):9.1e}  {'ADMIT' if r['admitted'] else ','.join(flags)}")
        all_rows += rows
    return all_rows


def main(argv):
    key = argv[1] if len(argv) > 1 else 'disk'
    build, eigs_fn, digits, prov = s0.DOMAINS[key]
    domain = build()
    cards = [c for c in ledger.load_cards() if c['domain_key'] == key]
    if not cards:
        print(f"no domain card for {key}; run s0 first")
        return 1
    card = cards[-1]
    ns = s0.informative_sizes(domain, card)

    print(f"=== S1 coarse screen: {key} (card {card['card_id']}) ===")
    rows = screen(key, domain, card, ns)

    admitted = [r for r in rows if r.get('admitted')]
    print(f"\n--- S1 summary: {key} ---")
    print(f"  {len(admitted)} of {len(rows)} configs admitted")
    if admitted:
        best = sorted(admitted, key=lambda r: r['sigma_hat'])[:8]
        print(f"  best admitted (by sigma_hat):")
        for r in best:
            print(f"    n={r['n_requested']:<4} {r['label']:26} sigma_hat={r['sigma_hat']:.2e} "
                  f"contrast={r['contrast']:.1e}")
    if len(admitted) < 2:
        print("  FEWER THAN 2 CANDIDATES -- escalate size or mark the domain hard")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
