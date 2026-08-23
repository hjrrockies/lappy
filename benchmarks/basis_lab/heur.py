"""Does `lappy.heuristics.polygon_default_basis` build good bases, and does its `precision`
argument mean anything?

RETIRED 2026-08-13. The recipe this measured now lives in
`benchmarks/archive/mps_heuristics_poc/` and was replaced by `lappy/basis_plan.py`; see that
directory's README for the matched-size comparison, and `PLAN_LAB.md` for the redesign. The
`report_*` functions below still work -- they read the recorded ledger in `run/heur/` and need no
module -- but the stages that BUILD bases will raise unless the archived file is copied back to
`lappy/heuristics.py`. The measurements are the artifact worth keeping; the code under test is not.

`lappy/heuristics.py` translates `docs/mps_heuristics.pdf`'s closed-form recipe into lappy's
basis primitives. `tests/test_heuristics.py` pins its formulas; nothing measures its output.
Three questions, in the order the evidence has to arrive:

  Q1  Is it better than today's constructions -- AT MATCHED SIZE? `bench.py` exists because
      two earlier studies read size effects as placement effects. The recipe derives `n` from
      geometry, so it is free to win by being bigger, and that is not a win. Every comparison
      here rebuilds each baseline at `n = len(heuristic basis)`.
  Q2  Is `precision` optimistic or pessimistic? It is the promise the API is aiming at
      (`Eigenproblem(domain, precision=p)`), so "asked for p, got q" is the headline.
  Q3  Which of `HeuristicConfig`'s ~12 constants move quality, and which only move cost?

STAGES, cheap first, because the cheap ones steer the expensive one.

  h0  `plan_basis` only, no solves. Sizes and per-corner planning tables for every polygon in
      the suite. Seconds. Decides which domains the later stages can afford at all.
  h1  sigma(lam_ref) and contrast for the heuristic over a precision ladder. ~1-3 s a cell.
  h2  the same measurement for four baselines rebuilt at the heuristic's own size. Q1.
  h4  one-factor-at-a-time over `HeuristicConfig`. Q3.
  h5  the combination h4 recommends, against stock defaults -- because OFAT winners are only
      individually safe.
  h3  the expensive one: full polished eigenvalue solve + Moler--Payne certified digits
      (`bench.evaluate`), heuristic against the best matched-size baseline. 30-450 s a cell.

WHY TENSION CARRIES MOST OF THE WEIGHT. sigma(lam_ref) is what the basis directly controls,
needs no minimization, and so cannot be confounded by the eigenvalue search -- the trap that
voided the first convergence study in this directory (see `bench.py`'s header). Certified
digits are the CHECK on the proxy, not the primary instrument, because they cost 100x more.
Both are reported; where they disagree, say so rather than picking one.

WHY NOT `bench.tension_contrast`. It sets `lam_max = 2*max(eigs)`, which for this recipe is a
different basis: `lam_max` enters as `kappa_max = sqrt(lam_max)` in every count. Every stage
here uses `lam_max = weyl_est(n_eigs + 2, domain)` instead -- the same value `bench.evaluate`
uses (and `MPSEigensolver.from_domain`'s own default, mps.py:339), so h1 and h3 measure the
same object. `off_eigenvalue_points` and `sigma_floor_at` are reused verbatim.

WHY NOT THE s0/s1 LEDGER. `ledger.append` hard-requires every one of `probe.KNOB_FIELDS` and
a `n_requested` knob (ledger.py:96-99). The recipe has no size knob -- that IS the recipe --
so the knob vocabulary does not apply. This module reuses the ledger's *design* (append-only
JSONL, one file per domain, resume by hash of the identifying fields, every field present on
every row) with its own field set and its own directory, `run/heur/`.

USAGE
    python -m benchmarks.basis_lab.heur h0 [domain_key ...]
    python -m benchmarks.basis_lab.heur h1 [domain_key ...]
    python -m benchmarks.basis_lab.heur h2 [domain_key ...]
    python -m benchmarks.basis_lab.heur h4 [domain_key ...]
    python -m benchmarks.basis_lab.heur h3 [domain_key ...]
    python -m benchmarks.basis_lab.heur report        # re-print all tables from the ledger

Re-running a stage skips rows already present (hash of stage/domain/family/precision/cfg/n),
so an interrupted sweep resumes by being restarted, and adding a value to a ladder does not
invalidate what is already measured.
"""
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

from lappy import bases, geometry as geo, reference as ref
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

from . import bench
from ..suite.domains import SUITE

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'heur')

# Fields that identify a measurement. Anything here changing means a different row, and a row
# whose id is already on disk is skipped rather than re-measured.
_ID_FIELDS = ('stage', 'domain_key', 'family', 'precision', 'cfg_key', 'n_requested', 'seed')

# Every row carries these, null where the stage does not produce them, so that a missing field
# is never confused with a defaulted one (ledger.py's first rule).
_ROW_FIELDS = _ID_FIELDS + (
    'ok', 'error', 'seconds', 'n_basis', 'lam_max', 'Lambda',
    'n_fb', 'n_curve', 'n_bridge_lightning', 'n_capped_corners', 'corner_kinds', 'plan_table',
    'sigma_eig', 'sigma_eig_med', 'sigma_off_med', 'contrast', 'sigma_floor', 'floor_displaced',
    'floor_n_probe',
    'ref_source', 'ref_digits', 'ref_resolution_lim', 'past_ref_resolution',
    'digits', 'worst_digits', 'true_digits', 'worst_true_digits', 'n_found', 'eigs',
    'warnings', 'dropped_sources', 'suite_status', 'suite_digit_ceiling',
)

PRECISIONS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
PRECISIONS_H3 = (1e-4, 1e-8, 1e-12)
N_EIGS = 4          # bench.evaluate's default; lam_max = weyl_est(N_EIGS + 2, domain)
SEED = 7
SIZE_CAP_H3 = 520   # the heuristics module's own conditioning warning fires at 600


# ── ledger ───────────────────────────────────────────────────────────────────────────────────

def record_id(rec):
    parts = []
    for k in _ID_FIELDS:
        v = rec.get(k)
        parts.append(f'{k}=' + ('null' if v is None else
                                (f'{v:.12g}' if isinstance(v, float) else str(v))))
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]


def _path(domain_key):
    return os.path.join(RUN_DIR, f'{domain_key}.jsonl')


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return [_to_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, float) and not np.isfinite(v):
        return None          # JSON has no Infinity; None reads back as "not measured"
    return v


def append(rec):
    missing = [k for k in _ROW_FIELDS if k not in rec]
    if missing:
        raise KeyError(f'row is missing fields {missing}; null them explicitly')
    out = {k: _to_jsonable(v) for k, v in rec.items()}
    out['record_id'] = record_id(rec)
    out.setdefault('created_utc', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(_path(rec['domain_key']), 'a') as fh:
        fh.write(json.dumps(out, sort_keys=True, default=float) + '\n')
    return out


def load(domain_key):
    p = _path(domain_key)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_all(stage=None, dedupe=True):
    """All rows, newest-wins per `record_id` when `dedupe`.

    Duplicates are possible and benign: `seen` is read once per domain at the top of a stage, so
    two processes running the same stage at once (which happened while producing `HEURISTICS.md`,
    from a chained run overlapping a manual one) both measure the same cell and both append. The
    two rows agree; keeping one keeps the medians honest.
    """
    rows = []
    if os.path.isdir(RUN_DIR):
        for fn in sorted(os.listdir(RUN_DIR)):
            if fn.endswith('.jsonl'):
                rows += load(fn[:-len('.jsonl')])
    rows = [r for r in rows if stage is None or r.get('stage') == stage]
    if not dedupe:
        return rows
    return list({r.get('record_id'): r for r in rows}.values())


def seen(domain_key):
    return {r.get('record_id') for r in load(domain_key)}


def blank_row(**kw):
    row = {k: None for k in _ROW_FIELDS}
    row['warnings'] = []
    row['ok'] = False
    row.update(kw)
    return row


# ── domains and reference values ─────────────────────────────────────────────────────────────

def polygon_keys():
    """Suite keys whose domain is a straight-edged polygon -- the recipe's declared scope."""
    out = []
    for k, d in SUITE.items():
        try:
            if d.domain().bdry.is_polyline:
                out.append(k)
        except Exception:
            pass
    return out


# Reference eigenvalues for suite keys the generated `REFERENCE` table does not cover. Kept
# explicit rather than inferred: each one is a different provenance with a different depth, and
# a row must be readable against the weaker of the two claims about it (chevron_1_2's table
# docstring says 12 digits, its suite entry says 7.1 -- a live disagreement, not a typo).
_EXTRA_REFS = {
    'rect_thin':     (lambda k: ref.rect_eigs(k, L=1, H=8), 13.0),
    'iso_right_tri': (lambda k: ref.iso_right_tri_eigs(k, l=1), 13.0),
    'iso_tri_h05':   (lambda k: ref.iso_tri_eigs(k, 0.5), 10.8),
    'GWW1':          (lambda k: ref.gww_eigs(k), 9.9),
    'chevron_1_2':   (lambda k: ref.chevron_eigs(k, 1.0, 2.0), 7.1),
    'chevron_1_15':  (lambda k: ref.chevron_eigs(k, 1.0, 1.5), 6.3),
    'chevron_2_3':   (lambda k: ref.chevron_eigs(k, 2.0, 3.0), 4.6),
    'chevron_2_4':   (lambda k: ref.chevron_eigs(k, 2.0, 4.0), 5.0),
    'parallelogram_p127': (None, None),
}


def reference_eigs(key, k=N_EIGS + 1):
    """`(eigs, ref_digits, source)` -- the deepest available truth for `key`, and how deep it is.

    `ref_digits` is the limit on what any measurement against it can say: sigma bottoms out
    around C*|lam_ref - lam_true|, so a curve read past the reference's own resolution is a
    picture of the reference (bench.py's "THE REFERENCE IS ALSO A LIMIT").
    """
    d = SUITE[key]
    if d.truth == 'analytic' and d.truth_fn is not None:
        return np.asarray(d.truth_fn(k), dtype=float), 15.0, 'analytic'
    try:
        from ..suite.run.reference_values import REFERENCE
    except Exception:
        REFERENCE = {}
    if key in REFERENCE:
        e = np.asarray(REFERENCE[key]['eigs'], dtype=float)[:k]
        return e, float(REFERENCE[key]['certified_digits']), 'suite_certified'
    fn, dig = _EXTRA_REFS.get(key, (None, None))
    if fn is None:
        return None, None, 'none'
    return np.asarray(fn(k), dtype=float), dig, 'reference_table'


def lam_max_for(domain):
    return float(weyl_est(N_EIGS + 2, domain))


# ── builders ─────────────────────────────────────────────────────────────────────────────────

def _heuristics():
    """Import the archived recipe, with a message that says where it went."""
    try:
        from lappy import heuristics
    except ImportError as exc:
        raise ImportError(
            'lappy.heuristics was retired; it is archived at '
            'benchmarks/archive/mps_heuristics_poc/heuristics.py. Copy it back to '
            'lappy/heuristics.py to re-run these stages, or use lappy.basis_plan (see '
            'benchmarks/basis_lab/plan_lab.py) instead.') from exc
    return heuristics


def _cfg_from(overrides):
    H = _heuristics()
    return H.HeuristicConfig(**overrides) if overrides else H.HeuristicConfig()


def cfg_key(overrides):
    """Canonical string for a `HeuristicConfig` override dict; '' means stock defaults."""
    return ','.join(f'{k}={v}' for k, v in sorted((overrides or {}).items()))


BASELINES = {
    'pure_fb': lambda d, l, n: bench.pure_fb(d, l, n=n),
    'mixed': lambda d, l, n: bench.mixed(d, l, n=n),
    'fb_plus_bdry_fs': lambda d, l, n: bench.fb_plus_bdry_fs(d, l, n=n),
    'default': lambda d, l, n: bases.make_default_basis(d, n),
}


def build(family, domain, lam_max, precision=None, overrides=None, n=None):
    if family == 'heuristic':
        return _heuristics().polygon_default_basis(domain, lam_max, precision=precision,
                                                   cfg=_cfg_from(overrides))
    return BASELINES[family](domain, lam_max, n)


# ── the cheap instrument: sigma at / off reference eigenvalues ───────────────────────────────

def measure_tension(domain, basis_fn, eigs, ref_digits, floor_n_probe=41):
    """Build a basis and score it by sigma at the first `N_EIGS` reference eigenvalues, sigma at
    the midpoints between them, and the ratio.

    Returns a dict of measurement fields (never raises; `ok=False` and `error` on failure).
    Warnings during the build are captured verbatim, not suppressed: `_drop_interior_sources`
    warns when sources land inside the domain, which invalidates both the tension and any
    certified bound, and a row that hides that is worse than no row.

    `floor_n_probe` is `sigma_floor_at`'s grid, and it dominates the cost: 41 points is 41
    extra GSVDs against the 7 the score itself needs, which at n ~ 500 is 90 s a cell. h1
    (one family) pays the full 41; the stages that multiply out over families or knobs pass 9,
    since the floor is a diagnostic ("is the reference the limit?") rather than the score.
    """
    out = {}
    t0 = time.time()
    try:
        at = np.asarray(eigs, dtype=float)[:N_EIGS]
        off = bench.off_eigenvalue_points(np.asarray(eigs, dtype=float)[:N_EIGS + 1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            basis = basis_fn()
            out['n_basis'] = len(basis)
            msgs = [str(w.message) for w in caught]
        out['warnings'] = sorted({m[:120] for m in msgs})
        out['dropped_sources'] = any('lie inside the domain' in m for m in msgs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = MPSEigensolver.from_domain(domain, basis=basis, rng=SEED, prec=1e-14)
            se = np.array([float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in at])
            so = np.array([float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in off])
            _, floor, _ = bench.sigma_floor_at(solver, float(at[0]), n_probe=floor_n_probe)
        out['floor_n_probe'] = floor_n_probe
        sig_e, sig_o = float(np.median(se)), float(np.median(so))
        out['sigma_eig'] = [float(x) for x in se]
        out['sigma_eig_med'] = sig_e
        out['sigma_off_med'] = sig_o
        out['contrast'] = sig_o/max(sig_e, 1e-300)
        out['sigma_floor'] = float(floor)
        out['floor_displaced'] = bool(floor < sig_e/10)
        if ref_digits is not None:
            lim = float(at[0])*10**(-ref_digits)
            out['ref_resolution_lim'] = lim
            out['past_ref_resolution'] = bool(sig_e < lim)
        out['ok'] = True
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
    out['seconds'] = time.time() - t0
    return out


# ── h0: sizing and feasibility ───────────────────────────────────────────────────────────────

def h0(keys=None):
    keys = keys or polygon_keys()
    for key in keys:
        sd = SUITE[key]
        domain = sd.domain()
        lam_max = lam_max_for(domain)
        done = seen(key)
        for p in PRECISIONS:
            row = blank_row(stage='h0', domain_key=key, family='heuristic', precision=p,
                            cfg_key='', n_requested=None, seed=SEED, lam_max=lam_max,
                            suite_status=sd.status, suite_digit_ceiling=sd.digit_ceiling)
            if record_id(row) in done:
                continue
            t0 = time.time()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    pl = _heuristics().plan_basis(domain, lam_max, precision=p)
                    row['warnings'] = sorted({str(w.message)[:120] for w in caught})
                row.update(ok=True, n_basis=pl.n_total, Lambda=pl.Lambda, n_fb=pl.n_fb,
                           n_curve=pl.n_curve, n_bridge_lightning=pl.n_bridge_lightning,
                           n_capped_corners=sum(1 for c in pl.plans if c.capped),
                           corner_kinds=[c.kind for c in pl.plans],
                           plan_table=[dict(i=c.index, kind=c.kind, alpha=c.alpha,
                                            d_c=c.d_c, R_c=c.R_c, nu_osc=c.nu_osc,
                                            nu_cont=c.nu_cont, M=c.M, capped=bool(c.capped),
                                            binding=c.binding) for c in pl.plans])
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'
            row['seconds'] = time.time() - t0
            append(row)
    report_h0()


def report_h0():
    rows = [r for r in load_all('h0') if r['ok']]
    by = {}
    for r in rows:
        by.setdefault(r['domain_key'], {})[r['precision']] = r
    print('\nh0  basis size the recipe derives, n_total(n_fb), by requested precision')
    print(f"  {'domain':22} {'nc':>3} " + ' '.join(f'{p:>11.0e}' for p in PRECISIONS)
          + '   monotone  lam_max')
    for key in sorted(by, key=lambda k: max(by[k][p]['n_basis'] for p in by[k])):
        cells, ns = [], []
        for p in PRECISIONS:
            r = by[key].get(p)
            if r is None:
                cells.append('--')
                continue
            cells.append(f"{r['n_basis']}({r['n_fb']})" + ('*' if r['n_capped_corners'] else ''))
            ns.append(r['n_basis'])
        mono = 'yes' if all(b >= a for a, b in zip(ns, ns[1:])) else 'NO'
        nc = len(by[key][PRECISIONS[0]]['corner_kinds'])
        lm = by[key][PRECISIONS[0]]['lam_max']
        print(f'  {key:22} {nc:>3} ' + ' '.join(f'{c:>11}' for c in cells)
              + f'   {mono:>8}  {lm:7.1f}')
    print('  * = at least one corner hit the Sec-4 conditioning cap')


# ── h1 / h2: tension calibration and matched-size baselines ─────────────────────────────────

def _tension_stage(stage, keys, families, precisions, floor_n_probe=41):
    for key in keys:
        sd = SUITE[key]
        domain = sd.domain()
        lam_max = lam_max_for(domain)
        eigs, ref_digits, source = reference_eigs(key)
        if eigs is None:
            print(f'  {key:22} skipped: no reference eigenvalues')
            continue
        done = seen(key)
        for p in precisions:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                n_heur = _heuristics().plan_basis(domain, lam_max, precision=p).n_total
            for fam in families:
                n_req = None if fam == 'heuristic' else n_heur
                row = blank_row(stage=stage, domain_key=key, family=fam, precision=p,
                                cfg_key='', n_requested=n_req, seed=SEED, lam_max=lam_max,
                                ref_source=source, ref_digits=ref_digits,
                                suite_status=sd.status, suite_digit_ceiling=sd.digit_ceiling)
                if record_id(row) in done:
                    continue
                row.update(measure_tension(
                    domain, lambda f=fam, n=n_req: build(f, domain, lam_max, precision=p, n=n),
                    eigs, ref_digits, floor_n_probe=floor_n_probe))
                append(row)
                tag = f'{key}/{fam}/{p:.0e}'
                if row['ok']:
                    print(f"    {tag:44} n={row['n_basis']:5d} sig={row['sigma_eig_med']:.2e} "
                          f"contrast={row['contrast']:.1e} ({row['seconds']:.1f}s)")
                else:
                    print(f"    {tag:44} FAILED {row['error'][:60]}")


def h1(keys=None):
    keys = keys or tractable_keys()
    print('\nh1  sigma(lam_ref) for the heuristic over the precision ladder')
    _tension_stage('h1', keys, ['heuristic'], PRECISIONS)
    report_h1()


def h2(keys=None):
    keys = keys or tractable_keys()
    print('\nh2  the same measurement for four baselines rebuilt at the heuristic\'s own size')
    _tension_stage('h2', keys, list(BASELINES), PRECISIONS, floor_n_probe=9)
    report_h2()


def tractable_keys():
    """Polygons the recipe sizes affordably, from h0 if it has run, else from `plan_basis`.

    The cut is `n <= SIZE_CAP_H3` at the middle of the ladder. Domains above it are not
    excluded from the study -- their size blow-up is one of the findings -- they are excluded
    from being SOLVED, which at 1000+ columns is both slow and conditioning-limited.
    """
    sizes = {}
    for r in load_all('h0'):
        if r['ok'] and r['precision'] == 1e-8:
            sizes[r['domain_key']] = r['n_basis']
    if not sizes:
        for key in polygon_keys():
            d = SUITE[key].domain()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sizes[key] = _heuristics().plan_basis(d, lam_max_for(d), precision=1e-8).n_total
    out = [k for k, n in sizes.items() if n <= SIZE_CAP_H3]
    out = [k for k in out if reference_eigs(k)[0] is not None]
    return sorted(out, key=lambda k: sizes[k])


def _digits(x):
    return float('nan') if x is None or x <= 0 else -np.log10(x)


def report_h1():
    rows = [r for r in load_all('h1') if r['ok']]
    by = {}
    for r in rows:
        by.setdefault(r['domain_key'], {})[r['precision']] = r
    print('\nh1  achieved tension digits A = -log10(median sigma at lam_ref) '
          'vs requested R = -log10(precision)')
    print(f"  {'domain':22} {'ref':>6} " + ' '.join(f'{-np.log10(p):>13.0f}' for p in PRECISIONS))
    print(f"  {'':22} {'':>6} " + ' '.join(f"{'A (A-R)':>13}" for p in PRECISIONS))
    for key in sorted(by):
        cells = []
        for p in PRECISIONS:
            r = by[key].get(p)
            if r is None:
                cells.append('--')
                continue
            A = _digits(r['sigma_eig_med'])
            R = -np.log10(p)
            flag = '!' if (r['dropped_sources'] or r['contrast'] < 4e2) else \
                   ('~' if r['past_ref_resolution'] else ' ')
            cells.append(f'{A:5.1f}({A - R:+5.1f}){flag}')
        rd = by[key][list(by[key])[0]]['ref_digits']
        print(f"  {key:22} {rd if rd is None else f'{rd:6.1f}'} "
              + ' '.join(f'{c:>13}' for c in cells))
    print('  ~ = sigma is past the reference\'s own resolution (measuring the reference)')
    print('  ! = contrast below 4e2 or sources dropped: the reading is not trustworthy')


def report_h2():
    h1r = {(r['domain_key'], r['precision']): r for r in load_all('h1') if r['ok']}
    h2r = {}
    for r in load_all('h2'):
        if r['ok']:
            h2r.setdefault((r['domain_key'], r['precision']), {})[r['family']] = r
    fams = list(BASELINES)
    print('\nh2  tension digits at MATCHED size: heuristic vs baselines rebuilt at n_heur')
    print(f"  {'domain':22} {'prec':>7} {'n':>5} {'heur':>7} "
          + ' '.join(f'{f:>16}' for f in fams))
    for key, p in sorted(h1r, key=lambda t: (t[0], -t[1])):
        h = h1r[(key, p)]
        cells = []
        for f in fams:
            r = h2r.get((key, p), {}).get(f)
            if r is None:
                cells.append('--')
                continue
            A = _digits(r['sigma_eig_med'])
            flag = '!' if (r['dropped_sources'] or r['contrast'] < 4e2) else ' '
            cells.append(f'{A:5.1f}{flag} (n={r["n_basis"]})')
        print(f"  {key:22} {p:7.0e} {h['n_basis']:5d} {_digits(h['sigma_eig_med']):7.1f} "
              + ' '.join(f'{c:>16}' for c in cells))
    print('  ! = contrast below 4e2 or sources dropped')


# ── h4: knob OFAT ───────────────────────────────────────────────────────────────────────────

KNOB_GRID = {
    'C_omega': (0.3, 1.0, 3.0, 30.0),
    'gamma': (0.25, 0.6),
    'eta': (0.15, 0.5),
    'nyquist_ppw': (2.0, 5.0),
    'delta_frac_D': (0.1, 0.5),
    'handover_frac': (0.6, 0.95),
    'order_margin': (0.0, 10.0),
    'airy_margin': (1.0, 4.0),
    'n_bridge': (5, 20),
    's_min_frac': (0.01, 0.15),
    'include_regular_fb': (False,),
    'max_reflections': (2,),
}

H4_KEYS = ('square', 'L_shape', 'reg_ngon_6', 'iso_tri_h4', 'right_trapezoid', 'chevron_1_2')
H4_PRECISION = 1e-10


def h4(keys=None):
    keys = keys or [k for k in H4_KEYS if k in SUITE]
    print(f'\nh4  one factor at a time off HeuristicConfig defaults, precision={H4_PRECISION:.0e}')
    for key in keys:
        sd = SUITE[key]
        domain = sd.domain()
        lam_max = lam_max_for(domain)
        eigs, ref_digits, source = reference_eigs(key)
        if eigs is None:
            print(f'  {key:22} skipped: no reference eigenvalues')
            continue
        done = seen(key)
        settings = [{}] + [{k: v} for k, vals in KNOB_GRID.items() for v in vals]
        for over in settings:
            row = blank_row(stage='h4', domain_key=key, family='heuristic',
                            precision=H4_PRECISION, cfg_key=cfg_key(over), n_requested=None,
                            seed=SEED, lam_max=lam_max, ref_source=source,
                            ref_digits=ref_digits, suite_status=sd.status,
                            suite_digit_ceiling=sd.digit_ceiling)
            if record_id(row) in done:
                continue
            row.update(measure_tension(
                domain,
                lambda o=over: build('heuristic', domain, lam_max,
                                     precision=H4_PRECISION, overrides=o),
                eigs, ref_digits, floor_n_probe=9))
            append(row)
            tag = f"{key}/{row['cfg_key'] or 'default'}"
            if row['ok']:
                print(f"    {tag:48} n={row['n_basis']:5d} sig={row['sigma_eig_med']:.2e} "
                      f"contrast={row['contrast']:.1e}")
            else:
                print(f"    {tag:48} FAILED {row['error'][:60]}")
    report_h4()


def report_h4():
    rows = [r for r in load_all('h4') if r['ok']]
    by = {}
    for r in rows:
        by.setdefault(r['domain_key'], {})[r['cfg_key']] = r
    print('\nh4  delta from stock defaults: d(tension digits), d(n), and digits per 100 columns')
    for key in sorted(by):
        base = by[key].get('')
        if base is None:
            continue
        A0, n0 = _digits(base['sigma_eig_med']), base['n_basis']
        print(f"  {key}   default: {A0:.1f} digits at n={n0}, contrast {base['contrast']:.0e}")
        items = sorted((k for k in by[key] if k), key=lambda k: -_digits(by[key][k]['sigma_eig_med']))
        for k in items:
            r = by[key][k]
            A, n = _digits(r['sigma_eig_med']), r['n_basis']
            flag = '!' if (r['dropped_sources'] or r['contrast'] < 4e2) else ' '
            print(f'    {k:28} dA={A - A0:+5.1f}  dn={n - n0:+5d}  '
                  f'A/100col={100*A/max(n, 1):5.2f} (dflt {100*A0/max(n0, 1):.2f}){flag}')


# ── h5: does the recommended combination hold up? ───────────────────────────────────────────
#
# h4 is one factor at a time, so its winners are only individually safe: nothing in it says the
# savings add or that the accuracy stays put when they are applied together. A recommendation
# nobody tested as a whole is exactly the kind of claim this directory exists to stop, so the
# combination gets its own measurement against stock defaults on every h1 domain.

LEAN_CFG = dict(eta=0.5, C_omega=1.0, n_bridge=20, s_min_frac=0.15, handover_frac=0.95,
                include_regular_fb=False)

# `lean` costs 6-9 digits on the all-regular-corner domains, and h6's ablation pins that on the
# one ingredient that removes basis functions rather than moving sources around: with the
# stock ambient spacing, dropping the optional regular-corner Fourier--Bessel terms is nearly
# free (h4 measured -0.1 digits on `square`), but combined with a sparser, farther source curve
# there is nothing left to represent the corner behaviour. `safe` is `lean` with that one
# ingredient put back.
SAFE_CFG = {k: v for k, v in LEAN_CFG.items() if k != 'include_regular_fb'}

CANDIDATES = {'lean': LEAN_CFG, 'safe': SAFE_CFG}
H5_PRECISIONS = (1e-8, 1e-12)


def h5(keys=None):
    keys = keys or tractable_keys()
    print('\nh5  stock defaults vs the candidate combinations '
          + '; '.join(f'{n}={cfg_key(c)}' for n, c in CANDIDATES.items()))
    for key in keys:
        sd = SUITE[key]
        domain = sd.domain()
        lam_max = lam_max_for(domain)
        eigs, ref_digits, source = reference_eigs(key)
        if eigs is None:
            continue
        done = seen(key)
        for p in H5_PRECISIONS:
            for over in [{}] + list(CANDIDATES.values()):
                row = blank_row(stage='h5', domain_key=key, family='heuristic', precision=p,
                                cfg_key=cfg_key(over), n_requested=None, seed=SEED,
                                lam_max=lam_max, ref_source=source, ref_digits=ref_digits,
                                suite_status=sd.status, suite_digit_ceiling=sd.digit_ceiling)
                if record_id(row) in done:
                    continue
                row.update(measure_tension(
                    domain,
                    lambda o=over: build('heuristic', domain, lam_max, precision=p, overrides=o),
                    eigs, ref_digits, floor_n_probe=9))
                append(row)
                name = next((n for n, c in CANDIDATES.items() if c == over), 'stock')
                tag = f'{key}/{p:.0e}/{name}'
                print(f"    {tag:44} " + (f"n={row['n_basis']:5d} sig={row['sigma_eig_med']:.2e} "
                                          f"contrast={row['contrast']:.1e}" if row['ok']
                                          else f"FAILED {row['error'][:50]}"))
    report_h5()


def report_h5():
    rows = [r for r in load_all('h5') if r['ok']]
    by = {}
    for r in rows:
        by.setdefault((r['domain_key'], r['precision']), {})[r['cfg_key']] = r
    for name, cfg in CANDIDATES.items():
        ck = cfg_key(cfg)
        print(f'\nh5  stock vs {name} ({ck})')
        print(f"  {'domain':22} {'prec':>7} {'A stock':>8} {'A ' + name:>8} {'dA':>6} "
              f"{'n stock':>8} {'n ' + name:>7} {'dn%':>6}")
        dAs, dns = [], []
        for (key, p), d in sorted(by.items(), key=lambda t: (t[0][0], -t[0][1])):
            if '' not in d or ck not in d:
                continue
            A0, A1 = _digits(d['']['sigma_eig_med']), _digits(d[ck]['sigma_eig_med'])
            n0, n1 = d['']['n_basis'], d[ck]['n_basis']
            dAs.append(A1 - A0)
            dns.append(100*(n1 - n0)/n0)
            flag = '!' if (d[ck]['dropped_sources'] or d[ck]['contrast'] < 4e2) else ''
            print(f'  {key:22} {p:7.0e} {A0:8.1f} {A1:8.1f} {A1 - A0:+6.1f} '
                  f'{n0:8d} {n1:7d} {100*(n1 - n0)/n0:+6.0f} {flag}')
        if dAs:
            print(f'  median: dA={np.median(dAs):+.1f} digits, dn={np.median(dns):+.0f}% '
                  f'(worst dA={min(dAs):+.1f})')


# ── h6: leave-one-out ablation of the lean combination ──────────────────────────────────────
#
# h5 found that the lean combination costs 6-9 digits on the all-regular-corner domains while
# being free or better everywhere else -- an interaction no single-factor sweep predicted (h4
# measured `include_regular_fb=False` at -0.1 digits on `square`). This isolates the cause by
# putting each ingredient back one at a time.

H6_KEYS = ('square', 'rect_thin', 'eq_tri', 'L_shape', 'right_trapezoid', 'iso_tri_h4')


def h6(keys=None):
    keys = keys or [k for k in H6_KEYS if k in SUITE]
    variants = {'lean': dict(LEAN_CFG)}
    for k in LEAN_CFG:
        variants[f'lean-minus-{k}'] = {kk: vv for kk, vv in LEAN_CFG.items() if kk != k}
    print('\nh6  leave-one-out ablation of the lean combination')
    for key in keys:
        sd = SUITE[key]
        domain = sd.domain()
        lam_max = lam_max_for(domain)
        eigs, ref_digits, source = reference_eigs(key)
        if eigs is None:
            continue
        done = seen(key)
        for p in H5_PRECISIONS:
            for name, over in variants.items():
                row = blank_row(stage='h6', domain_key=key, family='heuristic', precision=p,
                                cfg_key=cfg_key(over), n_requested=None, seed=SEED,
                                lam_max=lam_max, ref_source=source, ref_digits=ref_digits,
                                suite_status=sd.status, suite_digit_ceiling=sd.digit_ceiling)
                if record_id(row) in done:
                    continue
                row.update(measure_tension(
                    domain,
                    lambda o=over: build('heuristic', domain, lam_max, precision=p, overrides=o),
                    eigs, ref_digits, floor_n_probe=9))
                append(row)
                print(f"    {key}/{p:.0e}/{name:32} "
                      + (f"n={row['n_basis']:5d} sig={row['sigma_eig_med']:.2e}" if row['ok']
                         else f"FAILED {row['error'][:50]}"))
    report_h6()


def report_h6():
    h5r, h6r = {}, {}
    for r in load_all('h5'):
        if r['ok'] and r['cfg_key'] == '':
            h5r[(r['domain_key'], r['precision'])] = r
    for r in load_all('h6'):
        if r['ok']:
            h6r.setdefault((r['domain_key'], r['precision']), {})[r['cfg_key']] = r
    names = {cfg_key(dict(LEAN_CFG)): 'lean'}
    for k in LEAN_CFG:
        names[cfg_key({kk: vv for kk, vv in LEAN_CFG.items() if kk != k})] = f'+restore {k}'
    print('\nh6  ablation: tension digits (and n) relative to STOCK defaults')
    for (key, p), d in sorted(h6r.items(), key=lambda t: (t[0][0], -t[0][1])):
        base = h5r.get((key, p))
        if base is None:
            continue
        A0, n0 = _digits(base['sigma_eig_med']), base['n_basis']
        print(f'  {key} @ {p:.0e}   stock {A0:.1f} digits at n={n0}')
        for ck, r in sorted(d.items(), key=lambda t: -_digits(t[1]['sigma_eig_med'])):
            A, n = _digits(r['sigma_eig_med']), r['n_basis']
            print(f'    {names.get(ck, ck):28} dA={A - A0:+5.1f}  n={n:5d} '
                  f'({100*(n - n0)/n0:+.0f}%)')


# ── h3: certified digits ────────────────────────────────────────────────────────────────────

def h3(keys=None, precisions=PRECISIONS_H3, baseline='best'):
    """The expensive stage: `bench.evaluate` (polished solve + Moler--Payne) for the heuristic
    and for one matched-size baseline per cell.

    `baseline='best'` picks, per (domain, precision), whichever baseline h2 measured as
    strongest -- comparing against the weakest would be a straw man, and comparing against all
    four would quadruple a stage that already costs minutes a cell.
    """
    keys = keys or [k for k in tractable_keys() if k in SUITE]
    print('\nh3  certified digits (Moler--Payne) and true digits, heuristic vs matched baseline')
    h2r = {}
    for r in load_all('h2'):
        if r['ok']:
            h2r.setdefault((r['domain_key'], r['precision']), {})[r['family']] = r
    for key in keys:
        sd = SUITE[key]
        domain = sd.domain()
        lam_max = lam_max_for(domain)
        truth_fn = sd.truth_fn if sd.truth == 'analytic' else _truth_fn_from_table(key)
        done = seen(key)
        for p in precisions:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                n_heur = _heuristics().plan_basis(domain, lam_max, precision=p).n_total
            if n_heur > SIZE_CAP_H3:
                print(f'    {key}/{p:.0e} skipped: n_heur={n_heur} over the size cap')
                continue
            cand = h2r.get((key, p), {})
            if baseline == 'best' and cand:
                best = min(cand, key=lambda f: cand[f]['sigma_eig_med'])
            else:
                best = baseline if baseline != 'best' else 'default'
            plan = [('heuristic', None), (best, n_heur)]
            for fam, n_req in plan:
                row = blank_row(stage='h3', domain_key=key, family=fam, precision=p,
                                cfg_key='', n_requested=n_req, seed=SEED, lam_max=lam_max,
                                ref_source='moler_payne', ref_digits=None,
                                suite_status=sd.status, suite_digit_ceiling=sd.digit_ceiling)
                if record_id(row) in done:
                    continue
                t0 = time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    r = bench.evaluate(
                        domain,
                        lambda d, l, f=fam, n=n_req: build(f, d, l, precision=p, n=n),
                        n_eigs=N_EIGS, rng=SEED, truth_fn=truth_fn)
                    row['warnings'] = sorted({str(w.message)[:120] for w in caught})
                row['dropped_sources'] = any('lie inside the domain' in w
                                             for w in row['warnings'])
                if r['ok']:
                    row.update(ok=True, n_basis=r['n_basis'], n_found=r['n_found'],
                               eigs=list(np.asarray(r['eigs'], dtype=float)),
                               digits=r['digits'], worst_digits=r['worst_digits'])
                    if 'true_digits' in r:
                        row['true_digits'] = r['true_digits']
                        row['worst_true_digits'] = float(min(r['true_digits']))
                else:
                    row['error'] = r['error']
                row['seconds'] = time.time() - t0
                append(row)
                tag = f'{key}/{fam}/{p:.0e}'
                if row['ok']:
                    td = ('' if row['worst_true_digits'] is None
                          else f" true={row['worst_true_digits']:.1f}")
                    print(f"    {tag:44} n={row['n_basis']:5d} MP={row['worst_digits']:5.1f}"
                          f"{td} ({row['seconds']:.0f}s)")
                else:
                    print(f"    {tag:44} FAILED {row['error'][:60]} ({row['seconds']:.0f}s)")
    report_h3()


def _truth_fn_from_table(key):
    fn, _ = _EXTRA_REFS.get(key, (None, None))
    if fn is not None:
        return fn
    try:
        from ..suite.run.reference_values import REFERENCE
    except Exception:
        return None
    if key in REFERENCE:
        e = np.asarray(REFERENCE[key]['eigs'], dtype=float)
        return lambda k, e=e: e[:k]
    return None


def matched_true_digits(key, eigs):
    """True relative digits for each computed eigenvalue against its NEAREST reference value.

    `bench.evaluate`'s `true_digits` zips computed against reference positionally, which is
    right only when the solve returns the spectrum with the same multiplicities as the table.
    It does not on the high-multiplicity domains: `square`'s reference list is
    19.74, 49.35, 49.35, 78.96 and a solve that reports each distinct eigenvalue once lands
    every later comparison against the wrong mode, reading 0.2 digits when the eigenvalues are
    in fact correct to 13. Matching to the nearest reference fixes the alignment, and because
    nearest-matching would also hide a MISSED eigenvalue, `report_h3` prints `n_found` beside it
    -- read the two together.
    """
    e, _, _ = reference_eigs(key, k=10)
    if e is None or eigs is None:
        return None
    got = np.asarray(eigs, dtype=float)
    e = np.asarray(e, dtype=float)
    out = []
    for lam in got:
        j = int(np.argmin(np.abs(e - lam)))
        out.append(float('inf') if lam == e[j] else float(-np.log10(abs(lam - e[j])/abs(e[j]))))
    return out


def report_h3():
    rows = [r for r in load_all('h3') if r['ok']]
    by = {}
    for r in rows:
        by.setdefault((r['domain_key'], r['precision']), {})[r['family']] = r
    print('\nh3  certified (MP) and true digits vs requested R = -log10(precision)')
    print('    true = worst over the found eigenvalues, each matched to its NEAREST reference '
          'value;\n    read it together with found/expected -- nearest-matching cannot see a '
          'missed eigenvalue')
    print(f"  {'domain':22} {'prec':>7} {'R':>4} {'family':>16} {'n':>5} "
          f"{'MP':>6} {'MP-R':>6} {'true':>6} {'true-R':>7} {'found':>6} {'s':>5}")
    for (key, p), fams in sorted(by.items(), key=lambda t: (t[0][0], -t[0][1])):
        R = -np.log10(p)
        for fam, r in sorted(fams.items(), key=lambda t: t[0] != 'heuristic'):
            mp = r['worst_digits']
            td = matched_true_digits(key, r['eigs'])
            tr = min(td) if td else None
            print(f"  {key:22} {p:7.0e} {R:4.0f} {fam:>16} {r['n_basis']:5d} "
                  f"{mp:6.1f} {mp - R:+6.1f} "
                  + (f'{tr:6.1f} {tr - R:+7.1f}' if tr is not None else f"{'--':>6} {'--':>7}")
                  + f" {r['n_found']:6d} {r['seconds']:5.0f}")


# ── cli ─────────────────────────────────────────────────────────────────────────────────────

def report_all():
    for fn in (report_h0, report_h1, report_h2, report_h4, report_h5, report_h6, report_h3):
        try:
            fn()
        except Exception as e:
            print(f'  ({fn.__name__} unavailable: {type(e).__name__}: {e})')


STAGES = {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5, 'h6': h6}

if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'h0'
    keys = sys.argv[2:] or None
    if stage == 'report':
        report_all()
    elif stage in STAGES:
        STAGES[stage](keys)
    else:
        print(f'unknown stage {stage!r}; expected one of {sorted(STAGES)} or "report"')
        sys.exit(2)
