"""The measurement primitive for the basis-knob program: one build, one record.

WHY THIS EXISTS RATHER THAN `bench.py`'s tension path. `MPSEigensolver.from_domain` derives
BOTH collocation axes from the basis -- `mps.py:348` calls `pts_per_seg(domain, basis)` with
`mult=2` hardwired and no pass-through, and `mps.py:353` sets interior points to
`npts_rand=len(basis)`. So a comparison of two families at the same `n`, or of one family across
`n`, silently varies boundary point count, boundary point DISTRIBUTION (the `pts_per_seg` rule
differs for Fourier-Bessel and fundamental solutions), and interior point count at the same
time. Nothing measured through `from_domain` can attribute a difference to the basis.

This builds the point sets explicitly and constructs `MPSEigensolver` directly, so basis knobs
can be varied alone. `benchmarks/reference/basis_lab.probe` is the same shape and was the model;
this adds the full record, warning capture, and the pinned-collocation mode.

WHAT IS RECORDED, and why each field is not optional:

* Every knob, including ones the chosen family ignores, always present and null where unused --
  so a later reader can never confuse "this knob was not set" with "this knob defaulted".
* Realized column counts at three stages. `len(FundamentalBasis)` is `sum(2*order - 1)`, not the
  source count, while Fourier-Bessel is one column per order; and `NormalizedBasis` prunes
  zero-norm columns, so the realized width is not the requested `n` under either family.
* Dropped sources, from the warning text. `FundamentalBasis._exterior_sources_only` warns and
  silently drops sources that land inside the domain -- 25% of them on H_shape at a wavelength
  offset, measured this session. `bench.py` wraps its builds in `simplefilter('ignore')`, which
  is that signal being actively discarded. Here every warning is captured verbatim.
* `rtol`. It is live in every `sigma` evaluation (it truncates the pencil before the GSVD) and
  was recorded by neither prior harness: `bench.py` left it at the 1e-12 default and
  `reference/basis_lab.probe` pins 1e-14, so results from the two are not comparable.

WHAT IS *NOT* RELEVANT HERE. `ltol`/`prec` does nothing in this path -- no minimization runs
when you evaluate `sigma` at a given lambda. It matters only where a solve happens: producing
`lam_star`, and the final validation stage.

Dirichlet only: boundary normals are not passed, matching `reference/basis_lab.probe`, so
`bc_param` is 0 throughout. A Neumann/Robin study would need the normals threaded.
"""
import time
import traceback
import warnings

import numpy as np

from lappy import mps
from lappy.bases import (FourierBesselBasis, FundamentalBasis, fb_corner_orders,
                         fs_bdry_sps, fs_corner_orders)
from lappy.mps import MPSEigensolver

# Every knob any family understands. Present on every record, null where the family ignores it,
# so "what was held fixed here?" is answerable from a single row without re-running anything.
KNOB_FIELDS = ('family', 'n_requested', 'fs_frac', 'fb_strategy', 'fs_placement',
               'fs_d', 'fs_d_over_h', 'fs_order', 'fs_spacing', 'fs_C', 'fs_sigma',
               'check_exterior')

_DROPPED = 'dropped'


def wavelength(lam_max):
    """The wavelength at the top of the window, `h = 2 pi / sqrt(lam_max)`.

    The unit `fs_d` is most usefully expressed in: seven of eight offsets measured in the
    bucket-2 study landed in 0.73-1.14 h across domains whose absolute optima differed tenfold.
    Recorded as `fs_d_over_h` alongside the absolute `fs_d` so both readings survive.
    """
    return 2.0*np.pi/np.sqrt(float(lam_max))


# ── basis construction from a knob dict ──────────────────────────────────────────────────────

def build_basis(domain, spec, lam_max):
    """(basis, meta) for a knob dict. `meta` carries realized source counts for the record."""
    fam = spec['family']
    n = int(spec['n_requested'])
    meta = {'n_sources_requested': None, 'fb_orders': None, 'fs_per_unit': None}

    if fam == 'pure_fb':
        orders = fb_corner_orders(domain, n)
        meta['fb_orders'] = [int(o) for o in orders]
        return FourierBesselBasis.from_domain(domain, orders), meta

    if fam == 'pure_fs_bdry':
        per_seg = fs_bdry_sps(domain, n, order=spec['fs_order'])
        meta['fs_per_unit'] = [int(x) for x in per_seg]
        meta['n_sources_requested'] = int(np.sum(per_seg))
        b = FundamentalBasis.by_boundary(domain, per_seg, d=spec['fs_d'],
                                         order=spec['fs_order'], spacing=spec['fs_spacing'],
                                         check_exterior=spec['check_exterior'])
        return b, meta

    if fam == 'fs_corners':
        per_corner = fs_corner_orders(domain, n, order=spec['fs_order'])
        meta['fs_per_unit'] = [int(x) for x in per_corner]
        meta['n_sources_requested'] = int(np.sum(per_corner))
        b = FundamentalBasis.by_corners(domain, per_corner, spec['fs_C'], spec['fs_sigma'],
                                        spec['fs_order'], check_exterior=spec['check_exterior'])
        return b, meta

    if fam in ('mixed', 'fb_plus_bdry_fs'):
        n_fs = int(round(spec['fs_frac']*n))
        n_fb = n - n_fs
        parts, fb_orders = [], None
        if n_fb > 0:
            fb_orders = fb_corner_orders(domain, n_fb)
            parts.append(FourierBesselBasis.from_domain(domain, fb_orders))
        if n_fs > 0:
            sub = dict(spec, n_requested=n_fs,
                       family='fs_corners' if fam == 'mixed' else 'pure_fs_bdry')
            fs, fs_meta = build_basis(domain, sub, lam_max)
            parts.append(fs)
            meta['fs_per_unit'] = fs_meta['fs_per_unit']
            meta['n_sources_requested'] = fs_meta['n_sources_requested']
        if not parts:
            raise ValueError(f'{fam}: fs_frac={spec["fs_frac"]} and n={n} gave an empty basis')
        meta['fb_orders'] = [int(o) for o in fb_orders] if fb_orders is not None else None
        b = parts[0]
        for p in parts[1:]:
            b = b + p
        return b, meta

    raise ValueError(f'unknown family {fam!r}')


def default_spec(family, n, lam_max, **over):
    """A fully-populated knob dict: every field present, family-irrelevant ones null.

    Centres match what the current `make_default_basis` uses, so a sweep's centre point is the
    status quo and any improvement is measured against it rather than against a strawman.
    """
    h = wavelength(lam_max)
    spec = {k: None for k in KNOB_FIELDS}
    spec.update(family=family, n_requested=int(n), check_exterior=True, fb_strategy='angle')
    if family in ('pure_fs_bdry', 'fb_plus_bdry_fs'):
        spec.update(fs_placement='boundary', fs_d=h, fs_d_over_h=1.0, fs_order=1,
                    fs_spacing='even')
    if family in ('fs_corners', 'mixed'):
        spec.update(fs_placement='corners', fs_C=10.0, fs_sigma=1.0, fs_order=2)
    if family in ('mixed', 'fb_plus_bdry_fs'):
        spec['fs_frac'] = 0.5
    for k, v in over.items():
        if k not in spec:
            raise KeyError(f'{k!r} is not a knob; add it to KNOB_FIELDS deliberately')
        spec[k] = v
    # keep the absolute offset and its wavelength ratio consistent whichever was set
    if spec['fs_placement'] == 'boundary':
        if 'fs_d_over_h' in over and 'fs_d' not in over:
            spec['fs_d'] = spec['fs_d_over_h']*h
        elif 'fs_d' in over:
            spec['fs_d_over_h'] = spec['fs_d']/h
    return spec


# ── collocation, pinned or basis-derived ─────────────────────────────────────────────────────

def collocation(domain, basis, colloc):
    """(n_per_seg array, n_int). `colloc['mode']` is 'pinned' or 'basis_derived'.

    'pinned' is the mode the knob study runs in: the caller supplies `n_per_seg` and `n_int`
    outright, so they do not move when the basis does. 'basis_derived' reproduces
    `from_domain`'s coupling on purpose, for the validation pass that checks a ranking survives
    the way production actually builds a solver.
    """
    if colloc['mode'] == 'pinned':
        n_per_seg = np.asarray(colloc['n_per_seg'], dtype=int)
        return n_per_seg, int(colloc['n_int'])
    if colloc['mode'] == 'basis_derived':
        n_per_seg = mps.pts_per_seg(domain, basis, mult=colloc.get('bdry_mult', 2))
        return np.asarray(n_per_seg, dtype=int), int(colloc.get('int_ratio', 1)*len(basis))
    raise ValueError(f"unknown collocation mode {colloc['mode']!r}")


def _dropped_from_warnings(caught):
    """(count, verbatim text) parsed from `_exterior_sources_only`'s message, or (0, None)."""
    for w in caught:
        msg = str(w.message)
        if _DROPPED in msg:
            try:
                after = msg.split(_DROPPED, 1)[1].split()
                return int(after[0]), msg
            except (IndexError, ValueError):
                return -1, msg          # -1: it warned but the text did not parse
    return 0, None


# ── the primitive ────────────────────────────────────────────────────────────────────────────

def probe(domain, spec, colloc, lam_star, lam_off=(), probe_grid=(), lam_max=None,
          rtol=1e-12, ttol=1e-3, seed=0, diagnostics=True):
    """One basis build, sigma everywhere asked for, one flat record.

    `lam_star` are the reference eigenvalues, `lam_off` the reference NON-eigenvalues, and
    `probe_grid` a fixed grid around the first eigenvalue whose minimum defines `sigma_star`.
    The grid is fixed per domain deliberately: a per-build adaptive minimum would give a better
    basis more search and reintroduce a search-tolerance confound in new clothing.

    Never raises for a bad build -- returns `ok=False` with the traceback, so one broken cell
    does not abandon a sweep.
    """
    rec = {k: spec.get(k) for k in KNOB_FIELDS}
    rec.update(ok=False, error_type=None, error_msg=None, traceback=None, warnings=[],
               seconds_build=None, seconds_sigma=None)
    t0 = time.time()
    try:
        lam_max = float(lam_max if lam_max is not None else 2.0*np.max(lam_star))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            basis, meta = build_basis(domain, spec, lam_max)
            n_dropped, drop_text = _dropped_from_warnings(caught)
            rec['warnings'] = [str(w.message) for w in caught]

        n_per_seg, n_int = collocation(domain, basis, colloc)
        bdry_pts = domain.bdry_pts(n_per_seg)
        int_pts = domain.int_pts(method='random', npts_rand=n_int, rng=seed)
        nb = basis.to_normalized((bdry_pts, int_pts))
        rec['seconds_build'] = time.time() - t0

        rec.update(
            n_sources_requested=meta['n_sources_requested'],
            n_sources_realized=(None if meta['n_sources_requested'] is None
                                else meta['n_sources_requested'] - max(n_dropped, 0)),
            n_sources_dropped=n_dropped, dropped_warning_text=drop_text,
            fb_orders=meta['fb_orders'], fs_per_unit=meta['fs_per_unit'],
            len_basis_pre_norm=len(basis), len_basis_post_norm=len(nb),
            colloc_mode=colloc['mode'], bdry_mult=colloc.get('bdry_mult'),
            n_per_seg=[int(x) for x in n_per_seg], n_bdry_total=int(np.sum(n_per_seg)),
            int_ratio=colloc.get('int_ratio'), n_int=n_int, seed=seed,
            rtol=rtol, ttol=ttol, reg_type='svd', lam_max=lam_max, h_wavelength=wavelength(lam_max),
        )

        solver = MPSEigensolver(nb, bdry_pts, int_pts, rtol=rtol, ttol=ttol)
        t1 = time.time()
        sig_star = [float(solver.sigma(float(l))) for l in lam_star]
        sig_off = [float(solver.sigma(float(l))) for l in lam_off]
        sig_grid = [float(solver.sigma(float(l))) for l in probe_grid]
        rec['seconds_sigma'] = time.time() - t1

        best = min(sig_grid) if sig_grid else (sig_star[0] if sig_star else float('nan'))
        rec.update(
            lam_star=[float(l) for l in lam_star], sigma_at_lamstar=sig_star,
            lam_off=[float(l) for l in lam_off], sigma_off=sig_off,
            probe_grid=[float(l) for l in probe_grid], sigma_grid=sig_grid,
            sigma_star=best,
            sigma_eig_median=float(np.median(sig_star)) if sig_star else None,
            sigma_eig_max=float(np.max(sig_star)) if sig_star else None,
            sigma_off_median=float(np.median(sig_off)) if sig_off else None,
        )
        if sig_star and sig_off:
            rec['contrast'] = float(np.median(sig_off)/max(np.median(sig_star), 1e-300))
        else:
            rec['contrast'] = None

        if diagnostics and lam_star:
            d = solver._tension_diagnostics(float(lam_star[0]))
            rec.update(n_reg=int(d['n_reg']), n_cols=int(d['n']), x_norm=float(d['x_norm']),
                       sigma_cond=float(d['sigma_cond']),
                       AB_reg_min_svdval=float(d['AB_reg_min_svdval']),
                       AI_reg_min_svdval=float(d['AI_reg_min_svdval']))
        else:
            rec.update(n_reg=None, n_cols=None, x_norm=None, sigma_cond=None,
                       AB_reg_min_svdval=None, AI_reg_min_svdval=None)
        rec['ok'] = True
    except Exception as e:                                   # noqa: BLE001 -- recorded, not raised
        rec.update(error_type=type(e).__name__, error_msg=str(e),
                   traceback=traceback.format_exc())
    rec['seconds_total'] = time.time() - t0
    return rec
