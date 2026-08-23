"""Measurement harness for the redesigned basis planner (`lappy/basis_plan.py`).

Successor to `heur.py`, which measured the `docs/mps_heuristics.pdf` recipe and found it wanting
(`HEURISTICS.md`). That study's three methodological gaps are closed here:

* it scored only certified eigenvalue digits, while the stated use case (CLAUDE.md principle 4,
  the shape-optimization inner loop) consumes `dlambda = -integral (du/dn)^2 (V.n) ds`, whose
  accuracy nothing in this project has ever compared against the eigenvalue objective
  (`docs/scope_and_downstream.md` section 7 asks exactly this and records that nobody checked);
* it used a single interior-point seed, where `scope_and_downstream.md:121` is explicit that a
  basis study must fix the seed AND report the spread, because the spread is itself the signal
  that a basis is under-determined (`iso_right_tri`: 4.9, 4.0, 2.5 certified digits on three runs
  of identical code);
* it measured tension only AT reference eigenvalues, and so scored `mixed` on L_shape at 14.3
  digits when a real solve with that basis returns 9-14 "eigenvalues" in a 5-eigenvalue window.
  Spurious minima are counted here.

STAGES

  s0a  Where does per-lambda time go: basis evaluation or the GSVD? Settles by measurement what
       HEURISTICS.md could only infer, and quantifies the double evaluation in
       `NormalizedBasis.__call__` before it is fixed.
  s0b  Do certified-lambda accuracy and dlambda accuracy agree? Same bases, matched size, both
       objectives, against closed-form dlambda truth.

Rows land in `run/plan/*.jsonl`, append-only, resume by `record_id`. The ledger design (every
field present on every row, so a null is never confused with a default; hash of the identifying
fields, so adding a rung to a ladder does not invalidate prior rows) is `ledger.py`'s, restated
here rather than imported because the identifying fields differ and `ledger.append` hard-requires
`probe.KNOB_FIELDS`.
"""
import hashlib
import json
import os
import sys
import time
import warnings

from math import sqrt

import numpy as np

from lappy import bases, geometry as geo, mps, reference as ref
from lappy.asymp import weyl_est
from lappy.basis_plan import _residual_by_arc
from lappy.mps import MPSEigensolver

from . import bench

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'plan')

_ID_FIELDS = ('stage', 'case', 'family', 'n_requested', 'seed', 'variant')

_ROW_FIELDS = _ID_FIELDS + (
    'ok', 'error', 'domain_key', 'n_basis', 'lam_max', 'lam',
    # s0a: the cost split
    'm_bdry', 'm_int', 'n_reg', 't_eval_bdry', 't_eval_int', 't_norms', 't_pair',
    't_tensions', 't_sigma', 'eval_share', 'double_eval_factor', 'n_raw_evals',
    # s0b: the two objectives
    'certified_digits', 'true_digits', 'dlam', 'dlam_exact', 'dlam_rel_err',
    'sigma_at_lam', 'contrast',
    # S3: the objective triple, its guards, and cost
    'target', 'kappa_tier', 'tier', 'n_fb', 'n_fs', 'capped', 'shortfall',
    'sigma_eig_med', 'sigma_off_med', 'worst_arc_residual',
    'n_minima', 'n_true_in_window', 'n_spurious', 'ms_per_sigma', 'l2_spread_rel',
    'warnings', 'dropped_sources', 'seconds',
)

SEED = 7
SEEDS = (7, 11, 13)

# Tune on the dev set, confirm on the holdout, and never adopt a constant on dev evidence alone.
# This is not ceremony: a single-factor screen scored `include_regular_fb=False` at -0.1 digits on
# `square` and the combination cost -8.7 there (HEURISTICS.md h4/h5/h6). Six domains spanning the
# mechanisms; everything else is held out.
DEV = ('square', 'L_shape', 'reg_ngon_6', 'iso_tri_h4', 'right_trapezoid', 'chevron_1_2')
TARGETS = (1e-7, 1e-10, 1e-13)


# ── ledger ───────────────────────────────────────────────────────────────────────────────────

def record_id(rec):
    parts = []
    for k in _ID_FIELDS:
        v = rec.get(k)
        parts.append(f'{k}=' + ('null' if v is None else
                                (f'{v:.12g}' if isinstance(v, float) else str(v))))
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return [_to_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _path(stage):
    return os.path.join(RUN_DIR, f'{stage}.jsonl')


def append(rec):
    missing = [k for k in _ROW_FIELDS if k not in rec]
    if missing:
        raise KeyError(f'row is missing fields {missing}; null them explicitly')
    out = {k: _to_jsonable(v) for k, v in rec.items()}
    out['record_id'] = record_id(rec)
    out.setdefault('created_utc', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(_path(rec['stage']), 'a') as fh:
        fh.write(json.dumps(out, sort_keys=True, default=float) + '\n')
    return out


def load(stage, dedupe=True):
    p = _path(stage)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    if not dedupe:
        return rows
    return list({r.get('record_id'): r for r in rows}.values())


def blank_row(**kw):
    row = {k: None for k in _ROW_FIELDS}
    row['warnings'] = []
    row['ok'] = False
    row.update(kw)
    return row


def seen(stage):
    return {r.get('record_id') for r in load(stage)}


# ── s0a: where does per-lambda time go? ──────────────────────────────────────────────────────
#
# One `sigma(lam)` is: evaluate A_B and A_I (transcendental jv/yv over m x n), then
# regularize_pencil (QR of the stacked (m_B+m_I) x n, SVD of R) and gsvdvals. HEURISTICS.md
# inferred that evaluation dominates at n = 100-500 and that the GSVD takes over near 480, from
# one 39.6 ms interior-block timing plus flop counts. Inference is not measurement.
#
# The second thing this measures is the double evaluation: `norms(lam)` evaluates every component
# point set, and `__call__` then evaluated the same points again in its `wts`-falsy branch (the
# default). `n_raw_evals` COUNTS calls into the wrapped basis for one fresh-lambda A_B + A_I pair:
# 4 before S1 (boundary and interior, each twice), 2 after. A count, not a time ratio, because the
# obvious timing proxy -- t_pair/(t_eval_bdry + t_eval_int) -- stops meaning anything once the
# evaluation is cached: the denominator becomes a cache hit and the ratio reads 50-200 instead of
# the 2.1 it read before the fix. `double_eval_factor` keeps that timing proxy for the rows
# recorded before S1 and should be read only there.
#
# Every timing uses a FRESH lambda, because `_tensions_scalar` (LRU 256), `norms` (LRU 128) and
# `_weighted_eval` (LRU 4) all memoize on lam and would otherwise be timing a dict lookup.

def _fresh_lams(lam_max, k=40):
    """Distinct lambdas spread over the window, none of them repeated across a timing run."""
    return np.linspace(0.31*lam_max, 0.97*lam_max, k)


def _time(fn, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    return time.perf_counter() - t0, out


_WARM = [False]


def _warm_up():
    """First `gsvdvals` call in a process pays LAPACK/gsvd4py initialization -- measured at 10.7 ms
    against 2.5 ms for the same factorization immediately after, which is 4x the whole cell it
    landed in. Burn it before any timing is recorded."""
    if _WARM[0]:
        return
    A = np.linalg.qr(np.random.default_rng(0).standard_normal((60, 20)))[0]
    B = np.linalg.qr(np.random.default_rng(1).standard_normal((60, 20)))[0]
    for _ in range(3):
        mps.tensions(A, B, 'svd', 1e-12)
    _WARM[0] = True


def _count_raw_evals(solver, lam):
    """How many times one fresh-lambda `A_B` + `A_I` pair reaches the wrapped basis."""
    nb = solver.basis
    inner = nb.basis
    calls = [0]
    orig = inner._eval_pointset

    def counted(*a, **kw):
        calls[0] += 1
        return orig(*a, **kw)

    inner._eval_pointset = counted
    try:
        solver.A_B(lam)
        solver.A_I(lam)
    finally:
        del inner._eval_pointset          # restore the bound method
    return calls[0]


def s0a(keys=None, ns=(40, 60, 90, 130, 190, 260, 360, 500), variant='stock'):
    keys = keys or ['square', 'L_shape']
    _warm_up()
    done = seen('s0a')
    print('\ns0a  per-lambda cost split: basis evaluation vs the GSVD stack')
    print(f"  {'domain':10} {'n':>5} {'m_B':>5} {'m_I':>5} {'n_reg':>6} {'eval_B':>9} "
          f"{'eval_I':>9} {'pair':>9} {'tensions':>9} {'sigma':>9} {'eval%':>6} {'x2?':>5}")
    for key in keys:
        domain = _domain(key)
        lam_max = float(weyl_est(6, domain))
        for n in ns:
            row = blank_row(stage='s0a', case=key, family='pure_fb', n_requested=n, seed=SEED,
                            variant=variant, domain_key=key, lam_max=lam_max)
            if record_id(row) in done:
                continue
            t0 = time.time()
            try:
                lams = iter(_fresh_lams(lam_max))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    basis = bench.pure_fb(domain, lam_max, n=n)
                    solver = MPSEigensolver.from_domain(domain, basis=basis, rng=SEED,
                                                        prec=1e-14)
                    nb = solver.basis                      # the NormalizedBasis
                    bdry, interior = nb.component_pts

                    # one boundary / one interior evaluation, with norms already warm
                    lam = next(lams); nb.norms(lam)
                    t_eb, _ = _time(nb._eval_pointset, lam, bdry)
                    t_ei, _ = _time(nb._eval_pointset, lam, interior)

                    # norms() alone at a fresh lambda = one evaluation of each component
                    lam = next(lams)
                    t_nrm, _ = _time(nb.norms, lam)

                    # what a sigma actually pays for evaluation, at a fresh lambda
                    lam = next(lams)
                    t_ab, AB = _time(solver.A_B, lam)
                    t_ai, AI = _time(solver.A_I, lam)
                    t_pair = t_ab + t_ai

                    # the factorization stack, on prebuilt matrices
                    t_ten, _ = _time(mps.tensions, AB, AI, 'svd', solver.rtol)
                    _, _, _, s = mps.regularize_pencil(AB, AI, 'svd', solver.rtol)

                    # end-to-end, fresh lambda
                    lam = next(lams)
                    t_sig, _ = _time(solver.sigma, lam)

                    # exact count of evaluations reaching the wrapped basis, fresh lambda
                    n_raw = _count_raw_evals(solver, next(lams))
                row.update(ok=True, n_basis=len(basis), lam=float(lam),
                           m_bdry=AB.shape[0], m_int=AI.shape[0], n_reg=int(len(s)),
                           t_eval_bdry=t_eb, t_eval_int=t_ei, t_norms=t_nrm, t_pair=t_pair,
                           t_tensions=t_ten, t_sigma=t_sig,
                           eval_share=t_pair/max(t_pair + t_ten, 1e-30),
                           double_eval_factor=t_pair/max(t_eb + t_ei, 1e-30),
                           n_raw_evals=n_raw)
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'
            row['seconds'] = time.time() - t0
            append(row)
            if row['ok']:
                print(f"  {key:10} {row['n_basis']:5d} {row['m_bdry']:5d} {row['m_int']:5d} "
                      f"{row['n_reg']:6d} {1e3*row['t_eval_bdry']:8.2f}m "
                      f"{1e3*row['t_eval_int']:8.2f}m {1e3*row['t_pair']:8.2f}m "
                      f"{1e3*row['t_tensions']:8.2f}m {1e3*row['t_sigma']:8.2f}m "
                      f"{100*row['eval_share']:5.0f}% {row['double_eval_factor']:5.2f}")
            else:
                print(f'  {key:10} {n:5d} FAILED {row["error"][:60]}')
    report_s0a()


def report_s0a():
    rows = [r for r in load('s0a') if r['ok']]
    if not rows:
        return
    print('\ns0a  summary (ms per fresh lambda)')
    print(f"  {'domain':10} {'n':>5} {'eval pair':>10} {'GSVD stack':>11} {'eval share':>11} "
          f"{'sigma':>9} {'double-eval':>12}")
    for r in sorted(rows, key=lambda r: (r['case'], r['n_basis'])):
        print(f"  {r['case']:10} {r['n_basis']:5d} {1e3*r['t_pair']:10.2f} "
              f"{1e3*r['t_tensions']:11.2f} {100*r['eval_share']:10.0f}% "
              f"{1e3*r['t_sigma']:9.2f} {r['double_eval_factor']:12.2f}")
    shares = [r['eval_share'] for r in rows]
    print(f'  evaluation share of a sigma: min {100*min(shares):.0f}%, '
          f'max {100*max(shares):.0f}%, median {100*np.median(shares):.0f}% '
          f'-- neither term is negligible at any n measured')
    for variant in sorted({r['variant'] for r in rows}):
        rs = [r for r in rows if r['variant'] == variant]
        counts = sorted({r.get('n_raw_evals') for r in rs if r.get('n_raw_evals')})
        print(f'  variant {variant!r}: evaluations reaching the wrapped basis per fresh-lambda '
              f'A_B+A_I pair = {counts or "not recorded"} '
              f'(4 = every block evaluated twice; 2 = once)')
        print(f'    (pre-S1 timing proxy, meaningful only before the fix: median '
              f'{np.median([r["double_eval_factor"] for r in rs]):.2f})')

    # n_reg is the rank the SVD truncation keeps. Where it saturates, extra columns still cost a
    # full evaluation and a wider QR but contribute nothing to the pencil -- so it is a hard
    # ceiling on useful basis size at this (lam_max, rtol), and worth knowing before sizing.
    print(f"\n  {'domain':10} {'n':>5} {'n_reg':>6} {'n_reg/n':>8}")
    for r in sorted(rows, key=lambda r: (r['case'], r['n_basis'])):
        print(f"  {r['case']:10} {r['n_basis']:5d} {r['n_reg']:6d} "
              f"{r['n_reg']/r['n_basis']:8.2f}")
    for key in sorted({r['case'] for r in rows}):
        rs = sorted((r for r in rows if r['case'] == key), key=lambda r: r['n_basis'])
        cap = max(r['n_reg'] for r in rs)
        first = next((r['n_basis'] for r in rs if r['n_reg'] >= 0.99*cap), None)
        print(f'  {key}: n_reg saturates at {cap} (first reached near n = {first})')


def _domain(key):
    from ..suite.domains import SUITE
    return SUITE[key].domain()


# ── s0b: does the eigenvalue objective agree with the dlambda objective? ─────────────────────
#
# `docs/scope_and_downstream.md` section 7: "Does the eigenvalue-digit objective agree with the
# dlambda-accuracy objective? A basis tuned for one may not be best for the other." Nothing in
# the project has checked, and every basis study so far -- including HEURISTICS.md's 1154 rows --
# scored only the eigenvalue side. If they disagree, the inner-loop redesign has been aiming at
# the wrong target, because a shape optimizer consumes dlambda.
#
# Both objectives are measured AT a closed-form eigenvalue, so no search is involved and neither
# number can be confounded by where the minimizer stopped:
#
#   eigenvalue objective : Moler-Payne certified digits for u at the exact lam
#   dlambda objective    : rel err of -integral (du/dn)^2 (V.n) ds against closed form
#
# The rectangle is the only polygon with both truths in closed form: lam = pi^2(m^2/L^2+n^2/H^2),
# and translating the x = L edge gives dlam/dL = -2 pi^2 m^2 / L^3 exactly. (Dilation is NOT
# usable as an independent check -- the Rellich identity is what `gram` normalizes with, so
# `V = x - x0` is near-tautological. tests/test_shape_derivative.py says so at its own
# `test_dilation_is_the_rellich_identity`.) Singular-corner domains have no closed-form dlambda
# and need a finite-difference reference; that is s0c, after this settles the regular-corner case.
#
# The solver recipe deliberately mirrors tests/test_shape_derivative.py::_solver, which reaches
# 1e-12 on this case with make_default_basis at n=120 -- so `default` near n=130 reproducing
# ~1e-12 is the harness's own correctness check.

RECT_CASES = (
    ('rect2x1_m1n1', 2.0, 1.0, 1, 1),
    ('rect2x1_m2n1', 2.0, 1.0, 2, 1),
    ('rect2x1_m1n2', 2.0, 1.0, 1, 2),
    ('rect2x1_m3n2', 2.0, 1.0, 3, 2),
    ('rect1x8_m1n1', 1.0, 8.0, 1, 1),
    ('rect1x8_m2n1', 1.0, 8.0, 2, 1),
)

# 'heuristic' (the retired docs/mps_heuristics.pdf recipe, now in
# benchmarks/archive/mps_heuristics_poc/) stays in the s0b/s0c rows already on disk but is no
# longer built: `_build` raises for it, and the reports read it from the ledger.
FAMILIES = ('pure_fb', 'mixed', 'fb_plus_bdry_fs', 'default')


def _build(family, domain, lam_max, n):
    if family == 'heuristic':
        raise ImportError('the mps_heuristics recipe is archived at '
                          'benchmarks/archive/mps_heuristics_poc/; its s0b/s0c rows remain in the '
                          'ledger but it is no longer built')
    if family == 'default':
        return bases.make_default_basis(domain, n)
    return getattr(bench, family)(domain, lam_max, n=n)


def _dlam_solver(domain, basis, lam_max, seed):
    """tests/test_shape_derivative.py::_solver, with the interior draw seeded explicitly instead
    of through the global RNG -- the nondeterminism `scope_and_downstream.md:121` warns about is
    exactly what makes a seed-spread column necessary here."""
    from lappy.eigfun_integrals import boundary_quadrature
    bdry = domain.bdry_pts(mps.pts_per_seg(domain, basis, mult=2))
    interior = domain.int_pts(method='random', npts_rand=max(2*len(basis), 400),
                              rng=np.random.default_rng(seed))
    bq = boundary_quadrature(domain, lam_max, precision=1e-13, warn=False)
    return MPSEigensolver(basis.to_normalized((bdry, interior)), bdry, interior,
                          rtol=1e-14, ttol=1e-3, bdry_quad=bq)


def _segment_of_node(bq):
    return np.array([bq.panels[p].seg_idx for p in bq.panel_id])


def _moving_edge_mask(bq, L):
    """The segment whose nodes all sit at x = L. Chosen by geometry, not by a hard-coded index,
    and by SEGMENT rather than by per-node position so a quadrature node that happens to sit
    within tolerance of the corner cannot leak in from a neighbouring edge."""
    seg = _segment_of_node(bq)
    for i in np.unique(seg):
        if np.allclose(bq.pts[seg == i].real, L, atol=1e-9):
            return (seg == i).astype(float)
    raise RuntimeError('failed to identify the moving edge')


# Saturation floors. Below ~1e-16 a relative error is not a measurement -- one seed on
# rect2x1_m2n1 returned dlam bit-identical to the closed form, which a naive -log10 turned into
# "285 digits" and a 285-digit seed spread. Both objectives are clamped, and the ranking analysis
# below only looks at cells where BOTH are unsaturated: at 14.2 vs 14.3 digits the "winner" is
# the interior draw, not the basis, and calling that agreement or disagreement is noise either way.
_DLAM_FLOOR = 2.2e-16
_SATURATED = 13.0


def _dlam_digits(r):
    return -np.log10(max(r['dlam_rel_err'], _DLAM_FLOOR))


def s0b(args=None, ns=(15, 25, 40, 60, 90, 130, 190)):
    from lappy.eigfun_integrals import eigfun_cauchy_data, weighted_integral

    done = seen('s0b')
    print('\ns0b  certified eigenvalue digits vs dlambda accuracy, same bases, matched size')
    for case, L, H, m, nmode in RECT_CASES:
        domain = geo.rect(L, H)
        lam = float(ref.rect_eig(m, nmode, L, H))
        lam_max = 3*lam
        exact = -2*np.pi**2*m**2/L**3
        for family in FAMILIES:
            for n in (ns if family != 'heuristic' else (None,)):
                for seed in SEEDS:
                    row = blank_row(stage='s0b', case=case, family=family, n_requested=n,
                                    seed=seed, variant='stock', domain_key=f'rect{L}x{H}',
                                    lam=lam, lam_max=lam_max, dlam_exact=exact)
                    if record_id(row) in done:
                        continue
                    t0 = time.time()
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            basis = _build(family, domain, lam_max, n)
                            solver = _dlam_solver(domain, basis, lam_max, seed)
                            row['warnings'] = sorted({str(w.message)[:100] for w in caught})
                        row['dropped_sources'] = any('lie inside the domain' in w
                                                     for w in row['warnings'])
                        row['n_basis'] = len(basis)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            # dlambda objective
                            Vn = _moving_edge_mask(solver.bdry_quad, L)
                            coef = solver.eigenfunction_coef(lam, mult=1)
                            ed = eigfun_cauchy_data(solver.basis, lam, coef, solver.bdry_quad)
                            dlam = float(-weighted_integral(ed, 'NN', Vn)[0, 0])
                            # eigenvalue objective, at the same lam and the same basis
                            mp = certify(solver, domain, lam)
                            sig = float(np.atleast_1d(solver.sigma(lam))[0])
                        row.update(ok=True, dlam=dlam,
                                   dlam_rel_err=abs(dlam - exact)/abs(exact),
                                   certified_digits=float(mp['digits']), sigma_at_lam=sig)
                    except Exception as e:
                        row['error'] = f'{type(e).__name__}: {e}'
                    row['seconds'] = time.time() - t0
                    append(row)
                    tag = f'{case}/{family}/{n}/{seed}'
                    if row['ok']:
                        print(f"    {tag:44} n={row['n_basis']:4d} "
                              f"MP={row['certified_digits']:5.1f} "
                              f"dlam_err={row['dlam_rel_err']:.2e} ({row['seconds']:.1f}s)")
                    else:
                        print(f'    {tag:44} FAILED {row["error"][:60]}')
    report_s0b()


def report_s0b():
    rows = [r for r in load('s0b') if r['ok']]
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault((r['case'], r['family'], r['n_requested']), []).append(r)

    print('\ns0b  median over seeds, with spread; dlam digits = -log10(rel err)')
    print(f"  {'case':16} {'family':16} {'n':>5} {'MP':>6} {'MP spr':>7} "
          f"{'dlam dig':>9} {'dlam spr':>9}")
    for (case, family, n), rs in sorted(by.items(), key=lambda t: (t[0][0], t[0][1], t[0][2] or 0)):
        mp = [r['certified_digits'] for r in rs]
        dl = [_dlam_digits(r) for r in rs]
        print(f"  {case:16} {family:16} {rs[0]['n_basis']:5d} {np.median(mp):6.1f} "
              f"{np.ptp(mp):7.1f} {np.median(dl):9.1f} {np.ptp(dl):9.1f}")

    # The question: do the two objectives RANK the families the same way?
    print(f'\n  ranking agreement, UNSATURATED cells only (both objectives < {_SATURATED} digits)')
    from itertools import combinations
    conc = disc = 0
    n_cells = 0
    for case in sorted({r['case'] for r in rows}):
        for n in sorted({r['n_requested'] for r in rows if r['n_requested'] is not None}):
            cell = {f: by.get((case, f, n)) for f in FAMILIES if by.get((case, f, n))}
            mp = {f: float(np.median([r['certified_digits'] for r in rs]))
                  for f, rs in cell.items()}
            dl = {f: float(np.median([_dlam_digits(r) for r in rs])) for f, rs in cell.items()}
            live = [f for f in cell if mp[f] < _SATURATED and dl[f] < _SATURATED]
            if len(live) < 2:
                continue
            n_cells += 1
            for a, b in combinations(live, 2):
                if abs(mp[a] - mp[b]) < 0.3 or abs(dl[a] - dl[b]) < 0.3:
                    continue                     # too close to call on either axis
                if (mp[a] > mp[b]) == (dl[a] > dl[b]):
                    conc += 1
                else:
                    disc += 1
            best_mp = max(live, key=lambda f: mp[f])
            best_dl = max(live, key=lambda f: dl[f])
            flag = '' if best_mp == best_dl else '   <- DISAGREE on the winner'
            print(f'    {case:16} n={n:4d}  best by MP: {best_mp:16} '
                  f'best by dlam: {best_dl:16}{flag}')
            for f in sorted(live):
                print(f'        {f:18} MP {mp[f]:5.1f}   dlam {dl[f]:5.1f}')
    tot = conc + disc
    if not n_cells:
        print('    none: every construction is saturated on both objectives in every cell, so '
              'this case cannot rank them. Use smaller n, or a harder domain (s0c).')
    elif tot:
        print(f'\n  pairwise over {n_cells} unsaturated cells: {conc}/{tot} concordant, '
              f'{disc}/{tot} discordant ({100*conc/tot:.0f}% agreement; pairs within 0.3 digits '
              f'on either axis excluded as ties)')

    # A basis whose accuracy moves with the interior draw is under-determined; report it.
    worst = sorted(rows, key=lambda r: -np.ptp([x['certified_digits']
                                               for x in by[(r['case'], r['family'],
                                                            r['n_requested'])]]))
    seen_k = set()
    print('\n  largest seed spreads (interior draw sensitivity):')
    for r in worst:
        k = (r['case'], r['family'], r['n_requested'])
        if k in seen_k:
            continue
        seen_k.add(k)
        rs = by[k]
        sp_mp = np.ptp([x['certified_digits'] for x in rs])
        sp_dl = np.ptp([_dlam_digits(x) for x in rs])
        if sp_mp < 1.0 and sp_dl < 1.0:
            continue
        print(f'    {k[0]:16} {k[1]:16} n={k[2]}  MP spread {sp_mp:.1f}, '
              f'dlam spread {sp_dl:.1f}')
        if len(seen_k) > 12:
            break


# ── s0c: the same question at a SINGULAR corner, with a finite-difference reference ──────────
#
# s0b answers the question only for all-regular corners, and that is the easy half. At a singular
# corner `du/dn` blows up like r^(alpha-1), so the dlambda integrand (du/dn)^2 (V.n) is precisely
# where this project has already been bitten once: a weight of order r^1 sits outside the corner
# rule's exactness class and cost six orders until `weight_family='integer'` was added
# (tests/test_shape_derivative.py documents the r^p parity table). A basis that gets lambda right
# while getting dlambda wrong is therefore most likely HERE, and if it happens the redesign must
# be tuned to dlambda rather than to certified digits.
#
# No closed-form dlambda exists on the L-shape, so the reference is a finite difference of
# lambda(t) computed with a large basis. Its own resolution is limited: with lambda good to ~1e-12
# relative and h = 2e-3, the five-point formula resolves dlambda to roughly 1e-8 relative. That is
# ample for detecting a DISAGREEMENT (which would show up as a candidate basis being orders worse
# on dlambda than on lambda) and it is not enough to rank two bases that both reach 1e-10. The
# limit is reported with the answer.
#
# The family translates the x = -1 edge outward, which moves two REGULAR corners and leaves the
# reentrant corner fixed. That is deliberate: moving a reentrant corner weakens the Hadamard
# formula's own regularity assumptions (docs/scope_and_downstream.md section 3), and this stage is
# measuring the basis, not the formula's domain of validity.

def _l_family(t):
    """L_shape() with the x=-1 edge translated outward by `t`. t=0 is exactly `geo.L_shape()`."""
    a = 1.0 + t
    return geo.Polygon([0, 1j, -a + 1j, -a - 1j, 1 - 1j, 1], bc='dir', val_simple=False)


def _lam_near(solver, lam0, rel_window=6e-3, npts=9):
    """Polished tension minimum near `lam0`: a small scan for a 3-point bracket, then parabolic
    refinement at relative xtol 1e-15.

    RAISES if the discrete minimum lands on the edge of the scan, because then the window, not the
    tension, decided the answer. That is not a hypothetical: the first version of this function
    scanned a fixed +-2e-3 relative window around lambda(0) for every member of the family, and at
    t = +-2h the eigenvalue had already moved outside it. The minimizer dutifully returned the
    window edge, and the resulting five-point reference was wrong by 16% -- which showed up as
    every single basis being "wrong" by an identical 1.35e-01, the signature of a broken reference
    rather than a broken basis. The three-point estimate, which only used the in-window points,
    was correct all along.
    """
    from lappy.opt import minimize_on_bracket
    f = lambda l: float(np.atleast_1d(solver.sigma(float(l)))[0])   # noqa: E731
    xs = lam0*(1.0 + np.linspace(-rel_window, rel_window, npts))
    ys = np.array([f(x) for x in xs])
    i = int(np.argmin(ys))
    if i == 0 or i == npts - 1:
        raise RuntimeError(f'tension minimum is at the edge of the scan window around {lam0:.9g} '
                           f'(index {i} of {npts}); widen rel_window or reseed from a closer '
                           f'lambda -- the window, not the basis, would set the answer')
    lam, _ = minimize_on_bracket(f, ((xs[i-1], xs[i], xs[i+1]), (ys[i-1], ys[i], ys[i+1])),
                                 1e-15)
    return float(lam)


def _lam_of_t(t, lam_seed, n_ref=240, seed=SEED):
    dom = _l_family(t)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        basis = bases.make_default_basis(dom, n_ref)
        solver = MPSEigensolver.from_domain(dom, basis=basis, rng=seed, prec=1e-14)
        return _lam_near(solver, lam_seed)


def _fd_dlam_reference(h=2e-3, n_ref=240, seed=SEED, lam0=9.639723844):
    """dlambda/dt at t=0 by the five-point central difference, lambda(t) from a large basis.

    Walks outward from t=0 by CONTINUATION -- each solve is seeded from the previous t's
    eigenvalue, not from lambda(0) -- so the scan window follows the eigenvalue instead of being
    outrun by it (see `_lam_near`).

    Returns `(dlam, resolution)`. `resolution` is |five-point(h) - five-point(h/2)|, i.e. the
    reference compared against itself at half the step. That is the honest number: |five - three|
    reports the THREE-point formula's O(h^2) truncation error, which is orders larger and would
    understate the reference by claiming ~4e-6 where the five-point rule actually agrees with
    itself to 1.5e-10 relative between h=2e-3 and h=1e-3.
    """
    def five_at(step):
        lams = {0: _lam_of_t(0.0, lam0, n_ref, seed)}
        for sgn in (+1, -1):
            seed_lam = lams[0]
            for k in (1, 2):
                lams[sgn*k] = seed_lam = _lam_of_t(sgn*k*step, seed_lam, n_ref, seed)
        return (-lams[2] + 8*lams[1] - 8*lams[-1] + lams[-2])/(12*step)

    coarse, fine = five_at(h), five_at(0.5*h)
    return float(fine), abs(float(fine - coarse))


def s0c(args=None, ns=(30, 50, 80, 130)):
    from lappy.eigfun_integrals import eigfun_cauchy_data, weighted_integral

    done = seen('s0c')
    domain = _l_family(0.0)
    lam = float(ref.L_shape_eigs(1)[0])
    lam_max = 3*lam

    print('\ns0c  L_shape (reentrant corner): certified digits vs dlambda, FD reference')
    exact, spread = _fd_dlam_reference()
    print(f'  FD reference dlambda/dt = {exact:.12g}  (five- vs three-point differ by '
          f'{spread:.1e}, i.e. resolution ~{spread/abs(exact):.1e} relative)')

    for family in FAMILIES:
        for n in (ns if family != 'heuristic' else (None,)):
            for seed in SEEDS:
                row = blank_row(stage='s0c', case='L_shape_edge_x', family=family,
                                n_requested=n, seed=seed, variant='stock', domain_key='L_shape',
                                lam=lam, lam_max=lam_max, dlam_exact=exact)
                if record_id(row) in done:
                    continue
                t0 = time.time()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        basis = _build(family, domain, lam_max, n)
                        solver = _dlam_solver(domain, basis, lam_max, seed)
                        row['warnings'] = sorted({str(w.message)[:100] for w in caught})
                    row['dropped_sources'] = any('lie inside the domain' in w
                                                 for w in row['warnings'])
                    row['n_basis'] = len(basis)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        bq = solver.bdry_quad
                        seg = _segment_of_node(bq)
                        Vn = np.zeros(len(seg))
                        for i in np.unique(seg):
                            if np.allclose(bq.pts[seg == i].real, -1.0, atol=1e-9):
                                Vn[seg == i] = 1.0
                        assert Vn.sum() > 0, 'failed to identify the moving edge'
                        coef = solver.eigenfunction_coef(lam, mult=1)
                        ed = eigfun_cauchy_data(solver.basis, lam, coef, bq)
                        dlam = float(-weighted_integral(ed, 'NN', Vn)[0, 0])
                        mp = certify(solver, domain, lam)
                        sig = float(np.atleast_1d(solver.sigma(lam))[0])
                    row.update(ok=True, dlam=dlam,
                               dlam_rel_err=abs(dlam - exact)/abs(exact),
                               certified_digits=float(mp['digits']), sigma_at_lam=sig)
                except Exception as e:
                    row['error'] = f'{type(e).__name__}: {e}'
                row['seconds'] = time.time() - t0
                append(row)
                tag = f'{family}/{n}/{seed}'
                if row['ok']:
                    print(f"    {tag:30} n={row['n_basis']:4d} MP={row['certified_digits']:5.1f} "
                          f"dlam_err={row['dlam_rel_err']:.2e} ({row['seconds']:.1f}s)")
                else:
                    print(f'    {tag:30} FAILED {row["error"][:60]}')
    report_s0c()


def report_s0c():
    rows = [r for r in load('s0c') if r['ok']]
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault((r['family'], r['n_requested']), []).append(r)
    print('\ns0c  L_shape: median over seeds. dlam digits capped by the FD reference (~8)')
    print(f"  {'family':18} {'n':>5} {'MP':>6} {'MP spr':>7} {'dlam dig':>9} {'dlam spr':>9} "
          f"{'MP-dlam':>8}")
    for (family, n), rs in sorted(by.items(), key=lambda t: (t[0][1] or 0, t[0][0])):
        mp = float(np.median([r['certified_digits'] for r in rs]))
        dl = float(np.median([_dlam_digits(r) for r in rs]))
        print(f"  {family:18} {rs[0]['n_basis']:5d} {mp:6.1f} "
              f"{np.ptp([r['certified_digits'] for r in rs]):7.1f} {dl:9.1f} "
              f"{np.ptp([_dlam_digits(r) for r in rs]):9.1f} {mp - dl:8.1f}")
    live = [(f, n, float(np.median([r['certified_digits'] for r in rs])),
             float(np.median([_dlam_digits(r) for r in rs])))
            for (f, n), rs in by.items()]
    floor = max(dl for _, _, _, dl in live)
    print(f'\n  dlam digits saturate at {floor:.1f} = the FD reference\'s own resolution, so no '
          f'cell at or above it\n  is a measurement of the basis. Cells below it:')
    below = [t for t in live if t[3] < floor - 0.2]
    for f, n, mp, dl in sorted(below, key=lambda t: -t[2]):
        print(f'    {f:18} n={n}  MP {mp:5.1f}   dlam {dl:5.1f}')

    # The failure this stage exists to detect: accurate lambda, INACCURATE derivative. It would
    # appear as a cell with high MP and a dlam well below the reference floor.
    suspects = [t for t in below if t[2] >= 10.0]
    if suspects:
        print('\n  *** cells with MP >= 10 but dlam below the reference floor -- lambda accurate, '
              'derivative not:')
        for f, n, mp, dl in suspects:
            print(f'    {f:18} n={n}  MP {mp:5.1f}   dlam {dl:5.1f}')
    else:
        print(f'\n  no cell has MP >= 10 with dlam below the floor: the failure mode this stage '
              f'exists to\n  detect (accurate lambda, inaccurate derivative) does not occur on '
              f'this domain.')
    gaps = [(mp - dl, f, n) for f, n, mp, dl in below]
    if gaps:
        print(f'  MP minus dlam over unsaturated cells: median {np.median([g[0] for g in gaps]):+.1f}'
              f' digits (negative = the certified bound is the CONSERVATIVE one)')


# ── S3: scoring the planner ───────────────────────────────────────────────────────────────────
#
# The score is a TRIPLE plus guards, never a single number:
#
#   accuracy  certified Moler-Payne digits at a reference eigenvalue
#   fidelity  the spurious-minimum count over the window
#   cost      n_basis and milliseconds per sigma
#
# `n_spurious` is the metric whose absence invalidated the previous study's headline. Tension
# measured AT reference eigenvalues, with contrast against the midpoints between them, scored
# `mixed` on L_shape at 14.3 digits -- and a real solve with that basis returned 9 to 14
# "eigenvalues" in a 5-eigenvalue window, certifying MP -1.8. A basis can be excellent where you
# already know the answer and useless for finding it. A dense sigma scan sees that; nothing else
# here does.

def certify(solver, domain, lam, mult=1):
    """Moler--Payne digits with `||u||_L2` from the RELLICH BOUNDARY identity, not cubature.

    `moler_payne(domain, u, lam)` given only a callable has no access to the solver's `bdry_quad`,
    so it falls back to `interior_l2`'s Dunavant mesh. That is the wrong instrument for this
    project: the boundary identity is what `eigfun_integrals` was built for, it reaches ~1e-13 on
    almost every suite domain, it costs three basis evaluations at ~240 nodes against a mesh build,
    and going through it avoids `interior_l2`'s `id(domain)`-keyed cache entirely -- the cache that
    silently corrupted 10 of 150 cells in the first S3 sweep.

    Returns the `moler_payne` dict plus `l2_spread_rel`: the disagreement between `x0` probes, which
    is the boundary rule's own error estimate. `SPREAD_TOL = 1e-8` is where `certify_solver` stops
    trusting it and cross-checks against cubature -- known to matter on the near-slit corners
    (`chevron_2_3`, `chevron_2_4`) and the spirals -- so it is reported rather than assumed away.
    """
    from benchmarks.reference.certify import boundary_l2, moler_payne
    norms, info = boundary_l2(solver, lam, mult=mult)
    u = solver.eigenfunction(lam, mult=mult)
    mp = moler_payne(domain, lambda z, u=u: u(z)[:, 0], lam,
                     l2=float(norms[0]), l2_method=info['l2_method'], l2_info=info)
    spread = info.get('l2_spread') or [0.0]
    mp['l2_spread_rel'] = float(max(spread))/max(float(norms[0])**2, 1e-300)
    return mp


def _count_minima(solver, lo, hi, n=241, rel_depth=1e-2):
    """Local minima of `sigma` on `[lo, hi]`, and the wall time per evaluation.

    A minimum counts if it is a strict discrete local minimum and lies at least `rel_depth` of the
    scan's dynamic range below its neighbours -- otherwise every wiggle of a flat background is a
    "minimum". This is a fidelity check, not an eigenvalue finder: `solve_interval` does the real
    job with bracketing and refinement.
    """
    lams = np.linspace(lo, hi, n)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sig = np.array([float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in lams])
    ms = 1e3*(time.perf_counter() - t0)/n
    lo_s, hi_s = np.log10(max(sig.min(), 1e-300)), np.log10(sig.max())
    thresh = rel_depth*(hi_s - lo_s)
    ls = np.log10(np.maximum(sig, 1e-300))
    count = 0
    for i in range(1, n - 1):
        if ls[i] < ls[i-1] and ls[i] < ls[i+1] and min(ls[i-1], ls[i+1]) - ls[i] > thresh:
            count += 1
    return count, ms, sig


def _score(domain, plan, eigs, seed, want_dlam=None):
    """One cell of the S3 triple, with guards. Never raises."""
    out = {}
    t0 = time.time()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            basis = realize_checked(plan, domain)
            out['warnings'] = sorted({str(w.message)[:100] for w in caught})
        out['dropped_sources'] = any('inside the domain' in w for w in out['warnings'])
        out.update(n_basis=len(basis), n_fb=plan.n_fb, n_fs=plan.n_fs,
                   capped=bool(plan.capped), shortfall=plan.shortfall or None)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = MPSEigensolver.from_domain(domain, basis=basis, rng=seed, prec=1e-14)
            at = np.asarray(eigs, dtype=float)[:4]
            se = np.array([float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in at])
            offs = 0.5*(at[:-1] + at[1:])
            so = np.array([float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in offs])
            mp = certify(solver, domain, float(at[0]))
            arc, cor = _residual_by_arc(plan, domain, solver, float(at[0]))

            lo, hi = 0.92*at[0], 1.08*at[-1]
            n_min, ms, _ = _count_minima(solver, lo, hi)

        n_true = int(((at >= lo) & (at <= hi)).sum())
        out.update(certified_digits=float(mp['digits']),
                   l2_spread_rel=float(mp['l2_spread_rel']),
                   sigma_eig_med=float(np.median(se)), sigma_off_med=float(np.median(so)),
                   contrast=float(np.median(so)/max(np.median(se), 1e-300)),
                   worst_arc_residual=float(max(arc.max(initial=0.0), cor.max(initial=0.0))),
                   n_minima=n_min, n_true_in_window=n_true, n_spurious=max(0, n_min - n_true),
                   ms_per_sigma=float(ms), ok=True)

        if want_dlam is not None:
            from lappy.eigfun_integrals import eigfun_cauchy_data, weighted_integral
            Vn, exact = want_dlam(solver)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                coef = solver.eigenfunction_coef(float(at[0]), mult=1)
                ed = eigfun_cauchy_data(solver.basis, float(at[0]), coef, solver.bdry_quad)
                dl = float(-weighted_integral(ed, 'NN', Vn)[0, 0])
            out.update(dlam=dl, dlam_exact=float(exact),
                       dlam_rel_err=abs(dl - exact)/abs(exact))
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
    out['seconds'] = time.time() - t0
    return out


def realize_checked(plan, domain):
    from lappy.basis_plan import realize
    return realize(plan, domain)


def _poly_keys():
    from ..suite.domains import SUITE
    out = []
    for k in SUITE:
        try:
            if SUITE[k].domain().bdry.is_polyline and reference_eigs_for(k) is not None:
                out.append(k)
        except Exception:
            pass
    return out


def reference_eigs_for(key, k=5):
    from .heur import reference_eigs
    e, _, _ = reference_eigs(key, k)
    return None if e is None else np.asarray(e, dtype=float)


def s3(which='dev', targets=TARGETS, seeds=(SEED,), tiers=('low',)):
    """Score the planner. `which` in {'dev', 'holdout', 'all'} or a list of domain keys.

    The CLI hands every stage a list, so a single-element list naming a group is unwrapped here.
    """
    if isinstance(which, (list, tuple)) and len(which) == 1 and which[0] in ('dev', 'holdout', 'all'):
        which = which[0]
    elif isinstance(which, (list, tuple)) and not which:
        which = 'dev'
    from lappy.basis_plan import plan_basis
    from ..suite.domains import SUITE

    if which == 'dev':
        keys = [k for k in DEV if k in SUITE]
    elif which == 'holdout':
        keys = [k for k in _poly_keys() if k not in DEV]
    elif which == 'all':
        keys = _poly_keys()
    else:
        keys = list(which)

    _warm_up()
    done = seen('s3')
    print(f'\ns3  scoring the planner: {which} ({len(keys)} domains), targets '
          f'{[f"{t:.0e}" for t in targets]}, seeds {list(seeds)}, tiers {list(tiers)}')
    for key in keys:
        domain = SUITE[key].domain()
        eigs = reference_eigs_for(key)
        for tier in tiers:
            lam_max = float(weyl_est(6 if tier == 'low' else 50, domain))
            for target in targets:
                for seed in seeds:
                    row = blank_row(stage='s3', case=key, family='planner',
                                    n_requested=None, seed=seed,
                                    variant=f'{tier}/{target:.0e}', domain_key=key,
                                    target=target, tier=tier, kappa_tier=sqrt(lam_max),
                                    lam_max=lam_max, lam=float(eigs[0]))
                    if record_id(row) in done:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            plan = plan_basis(domain, lam_max, target=target)
                        row.update(_score(domain, plan, eigs, seed))
                    except Exception as e:
                        row['error'] = f'{type(e).__name__}: {e}'
                    append(row)
                    tag = f'{key}/{tier}/{target:.0e}/s{seed}'
                    if row['ok']:
                        print(f"    {tag:42} n={row['n_basis']:4d} MP={row['certified_digits']:5.1f} "
                              f"spur={row['n_spurious']:2d} contrast={row['contrast']:.0e} "
                              f"{row['ms_per_sigma']:6.1f}ms/sig")
                    else:
                        print(f'    {tag:42} FAILED {row["error"][:60]}')
    report_s3()


def report_s3():
    rows = [r for r in load('s3') if r.get('stage') == 's3']
    ok = [r for r in rows if r['ok']]
    if not ok:
        return
    from ..suite.domains import SUITE

    def grp(r):
        return 'dev' if r['case'] in DEV else 'holdout'

    # A certified count above ~15 is the instrument hitting its floor, not an achievement: eps is
    # sqrt(area)*sup|u| with ||u||=1, so once the boundary residual reaches round-off the bound
    # reads 16-17 digits and stops meaning anything. reg_ngon_7 does this (16.3, 17.4) with a
    # SIMPLE lam_1, so it is not a multiplicity artifact -- it is a residual at machine precision.
    # Marked rather than clamped, because the distinction between "certified 14" and "residual at
    # round-off" is worth seeing.
    print('\ns3  the triple, low kappa tier, seed 7   (# = MP at round-off, bound saturated)')
    print(f"  {'set':8} {'domain':18} {'target':>7} {'n':>5} {'MP':>6} {'MP-R':>6} "
          f"{'spur':>5} {'contrast':>9} {'ms/sig':>7} {'ceil':>5}")
    for r in sorted(ok, key=lambda r: (grp(r), r['case'], -(r['target'] or 0))):
        if r['tier'] != 'low' or r['seed'] != SEED:
            continue
        R = -np.log10(r['target'])
        ceil = SUITE[r['case']].digit_ceiling
        flag = '!' if (r['dropped_sources'] or (r['contrast'] or 0) < 4e2
                       or r['n_spurious']) else ('#' if r['certified_digits'] > 15.0 else ' ')
        print(f"  {grp(r):8} {r['case']:18} {r['target']:7.0e} {r['n_basis']:5d} "
              f"{r['certified_digits']:6.1f} {r['certified_digits'] - R:+6.1f} "
              f"{r['n_spurious']:5d} {r['contrast']:9.1e} {r['ms_per_sigma']:7.1f} "
              f"{(f'{ceil:.1f}' if ceil else '-'):>5}{flag}")

    # The cross-check that caught a corrupted instrument: `certified_digits` and
    # `worst_arc_residual` are computed by different code from different quantities, and they agreed
    # to 0.1 digits everywhere the certification was sound. When they diverge, one of them is broken
    # -- that is how `interior_l2`'s id(domain)-keyed cache was found. Kept as a permanent guard.
    drift = [(r, r['certified_digits'] + np.log10(max(r['worst_arc_residual'], 1e-300)))
             for r in ok if r['worst_arc_residual']]
    bad = [(r, g) for r, g in drift if abs(g) > 1.0]
    if bad:
        print('\n  *** certified digits disagree with the per-arc residual by >1 digit -- suspect '
              'the INSTRUMENT, not the basis:')
        for r, g in sorted(bad, key=lambda t: -abs(t[1])):
            print(f"    {r['case']:18} {r['tier']:5} {r['target']:.0e} "
                  f"MP={r['certified_digits']:6.2f} residual="
                  f"{-np.log10(r['worst_arc_residual']):6.2f} gap={g:+5.2f}")
    else:
        print(f'\n  cross-check: all {len(drift)} cells agree with their own per-arc residual '
              f'to within 1 digit (worst {max(abs(g) for _, g in drift):.2f})')

    weak_l2 = [r for r in ok if (r.get('l2_spread_rel') or 0) > 1e-8]
    if weak_l2:
        print(f'  boundary-quadrature L2 spread above SPREAD_TOL=1e-8 in {len(weak_l2)} cells '
              f'(the certification itself is shaky there):')
        for r in sorted(weak_l2, key=lambda r: -(r['l2_spread_rel'] or 0))[:8]:
            print(f"    {r['case']:18} {r['tier']:5} {r['target']:.0e} "
                  f"spread {r['l2_spread_rel']:.1e}")

    for label in ('dev', 'holdout'):
        sub = [r for r in ok if grp(r) == label and r['tier'] == 'low' and r['seed'] == SEED]
        if not sub:
            continue
        met = [r for r in sub if r['certified_digits'] >= -np.log10(r['target'])]
        weak = [r for r in sub if (r['contrast'] or 0) < 4e2]
        spur = [r for r in sub if r['n_spurious']]
        print(f'\n  {label}: {len(met)}/{len(sub)} cells meet their requested target; '
              f'{len(spur)}/{len(sub)} show spurious minima; '
              f'{len(weak)}/{len(sub)} have contrast below 4e2 (reading untrustworthy); '
              f'median MP-R {np.median([r["certified_digits"] + np.log10(r["target"]) for r in sub]):+.1f}')
        if weak:
            print('    contrast collapsed in: ' + ', '.join(
                f'{r["case"]}@{r["target"]:.0e}({r["contrast"]:.0e})' for r in weak))
        if spur:
            print('    spurious in: ' + ', '.join(f'{r["case"]}@{r["target"]:.0e}'
                                                  f'({r["n_spurious"]})' for r in spur))

    # seed spread: a basis whose accuracy moves with the interior draw is under-determined
    by = {}
    for r in ok:
        if r['tier'] == 'low':
            by.setdefault((r['case'], r['target']), []).append(r)
    multi = {k: v for k, v in by.items() if len({r['seed'] for r in v}) > 1}
    if multi:
        print('\n  seed spread over the interior draw (>= 3 seeds), worst first:')
        for (case, target), rs in sorted(multi.items(),
                                         key=lambda kv: -np.ptp([r['certified_digits']
                                                                 for r in kv[1]]))[:12]:
            d = [r['certified_digits'] for r in rs]
            print(f'    {case:18} {target:.0e}  MP {np.median(d):5.1f} '
                  f'spread {np.ptp(d):4.1f}  n={rs[0]["n_basis"]}')

    hi = [r for r in ok if r['tier'] == 'high']
    if hi:
        print('\n  high kappa tier (lam_max from weyl_est(50)): sized for a larger window, '
              'scored on the same low modes')
        print(f"  {'domain':18} {'kappa':>7} {'n':>5} {'MP':>6} {'spur':>5} {'ms/sig':>7}")
        for r in sorted(hi, key=lambda r: r['case']):
            print(f"  {r['case']:18} {r['kappa_tier']:7.1f} {r['n_basis']:5d} "
                  f"{r['certified_digits']:6.1f} {r['n_spurious']:5d} {r['ms_per_sigma']:7.1f}")


# ── smooth: the frozen-plan claim at a SINGULAR corner ────────────────────────────────────────
#
# `tests/test_basis_plan_smoothness.py` proves the frozen-plan architecture on the rectangle, where
# both lambda and dlambda are closed form -- and where every corner is regular. The case that
# matters for this project is a reentrant corner, and there no closed-form dlambda exists, so the
# reference is the same continuation finite difference s0c built. Slow, hence here and not in the
# test suite.

def smooth(args=None, ts=(-0.03, -0.015, 0.0, 0.015, 0.03), target=1e-10):
    from lappy.basis_plan import plan_basis, realize
    from lappy.eigfun_integrals import eigfun_cauchy_data, weighted_integral

    done = seen('smooth')
    base = _l_family(0.0)
    lam_max = 3*float(ref.L_shape_eigs(1)[0])
    exact, resolution = _fd_dlam_reference()
    print(f'\nsmooth  L_shape family, ONE plan frozen at t=0, realized on each member')
    print(f'  FD reference dlambda/dt = {exact:.10f} (resolution ~{resolution/abs(exact):.1e} rel)')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = plan_basis(base, lam_max, target=target)
    print(f'  plan: n={plan.n_total} (fb {plan.n_fb}, fs {plan.n_fs})')
    # NOT "error vs the reference": dlam genuinely varies along the family (-9.23 to -8.91
    # across t = +-0.03), so the last column shows that variation, and the real accuracy check is
    # FD(t) vs Hadamard(t) in the summary below.
    print(f"  {'t':>8} {'n':>5} {'lam':>16} {'sigma':>10} {'dlam':>14} {'d/d(0)-1':>10}")
    for t in ts:
        row = blank_row(stage='smooth', case='L_shape_edge_x', family='planner',
                        n_requested=None, seed=SEED, variant=f't={t:+.4f}',
                        domain_key='L_shape', target=target, tier='low', lam_max=lam_max,
                        dlam_exact=exact)
        if record_id(row) in done:
            continue
        try:
            dom = _l_family(t)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                basis = realize(plan, dom)
                solver = MPSEigensolver.from_domain(dom, basis=basis, rng=SEED, prec=1e-14)
                # Predict where the eigenvalue moved to before searching for it. lambda shifts by
                # dlam*t, which at t=0.03 is 1.4% of lambda -- more than `_lam_near`'s scan window,
                # so seeding from lambda(0) makes its edge guard fire (correctly: the window, not
                # the tension, would otherwise pick the answer).
                lam = _lam_near(solver, float(ref.L_shape_eigs(1)[0]) + exact*t)
                sig = float(np.atleast_1d(solver.sigma(lam))[0])
                bq = solver.bdry_quad
                seg = np.array([bq.panels[p].seg_idx for p in bq.panel_id])
                Vn = np.zeros(len(seg))
                for i in np.unique(seg):
                    if np.allclose(bq.pts[seg == i].real, -(1.0 + t), atol=1e-9):
                        Vn[seg == i] = 1.0
                coef = solver.eigenfunction_coef(lam, mult=1)
                ed = eigfun_cauchy_data(solver.basis, lam, coef, bq)
                dlam = float(-weighted_integral(ed, 'NN', Vn)[0, 0])
            row.update(ok=True, n_basis=len(basis), lam=lam, sigma_at_lam=sig, dlam=dlam,
                       dlam_rel_err=abs(dlam - exact)/abs(exact))
            print(f'  {t:+8.4f} {len(basis):5d} {lam:16.10f} {sig:10.2e} {dlam:14.8f} '
                  f'{dlam/exact - 1.0:+10.2e}')
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
            print(f'  {t:+8.4f} FAILED {row["error"][:60]}')
        append(row)
    report_smooth()


def report_smooth():
    rows = [r for r in load('smooth') if r['ok']]
    if not rows:
        return
    rows.sort(key=lambda r: float(r['variant'].split('=')[1]))
    ts = np.array([float(r['variant'].split('=')[1]) for r in rows])
    lams = np.array([r['lam'] for r in rows])
    dls = np.array([r['dlam'] for r in rows])
    print('\nsmooth  frozen-plan invariants on a reentrant-corner family')
    print(f'  basis size across the family: {sorted({r["n_basis"] for r in rows})} '
          f'(one value = frozen structure held)')
    print(f'  worst sigma at the solved eigenvalue: {max(r["sigma_at_lam"] for r in rows):.2e}')
    if len(rows) >= 3:
        h = ts[1] - ts[0]
        fd = (lams[2:] - lams[:-2])/(2*h)
        print(f'  central FD of the solved lambda: '
              + ' '.join(f'{x:.6f}' for x in fd))
        print(f'  Hadamard dlambda at the same t : '
              + ' '.join(f'{x:.6f}' for x in dls[1:-1]))
        rel = np.abs(fd - dls[1:-1])/np.abs(dls[1:-1])
        print(f'  agreement: worst {rel.max():.2e} relative '
              f'(a central difference at h={h:.3g} truncates at ~{h**2:.1e})')
    at0 = [r for r in rows if abs(float(r['variant'].split('=')[1])) < 1e-12]
    if at0:
        print(f'  at t=0, dlambda vs the independent FD reference: '
              f'{at0[0]["dlam_rel_err"]:.2e} relative (reference resolution ~1.5e-10)')


# ── gate: the pinned regression table ─────────────────────────────────────────────────────────

GATE_PATH = os.path.join(RUN_DIR, 'gate.json')


def gate(args=None, write=False):
    """Pin (domain, target) -> (n, certified digits, spurious count), or check against the pin.

    A redesign that improves one domain and quietly wrecks another has to fail loudly. `--write`
    records the current state; with no argument it re-reads the s3 rows and reports drift.
    """
    write = write or (args and 'write' in args)
    rows = [r for r in load('s3') if r['ok'] and r['tier'] == 'low' and r['seed'] == SEED]
    cur = {f"{r['case']}@{r['target']:.0e}":
           dict(n=r['n_basis'], mp=round(r['certified_digits'], 1), spur=r['n_spurious'])
           for r in rows}
    if write:
        os.makedirs(RUN_DIR, exist_ok=True)
        with open(GATE_PATH, 'w') as fh:
            json.dump(cur, fh, indent=1, sort_keys=True)
        print(f'gate: pinned {len(cur)} cells to {GATE_PATH}')
        return cur
    if not os.path.exists(GATE_PATH):
        print(f'gate: no pin yet; run `gate write` after an s3 sweep')
        return None
    with open(GATE_PATH) as fh:
        pinned = json.load(fh)
    print(f'\ngate  {len(pinned)} pinned cells')
    # Accuracy and cost are separate verdicts. Conflating them made a +1.1-digit gain for +31%
    # columns read as "REGRESSED", which is not what a gate is for: losing accuracy is a defect,
    # spending more columns for more accuracy is a trade for a human to judge.
    worse, better, pricier, missing = [], [], [], []
    for k, v in sorted(pinned.items()):
        if k not in cur:
            missing.append(k)
            continue
        d_mp = cur[k]['mp'] - v['mp']
        d_n = cur[k]['n'] - v['n']
        if d_mp < -0.3 or cur[k]['spur'] > v['spur']:
            worse.append((k, v, cur[k]))
        elif d_mp > 0.3:
            better.append((k, v, cur[k]))
        elif d_n > 0.1*v['n']:
            pricier.append((k, v, cur[k]))
    for k, v, c in worse:
        print(f"  ACCURACY LOST {k:26} MP {v['mp']:5.1f} -> {c['mp']:5.1f}  "
              f"n {v['n']:4d} -> {c['n']:4d}  spur {v['spur']} -> {c['spur']}")
    for k, v, c in better:
        print(f"  improved      {k:26} MP {v['mp']:5.1f} -> {c['mp']:5.1f}  "
              f"n {v['n']:4d} -> {c['n']:4d}")
    for k, v, c in pricier:
        print(f"  costlier      {k:26} MP {v['mp']:5.1f} -> {c['mp']:5.1f}  "
              f"n {v['n']:4d} -> {c['n']:4d}  (same accuracy, more columns)")
    if missing:
        print(f'  not re-measured: {len(missing)} cells')
    print(f'  {len(worse)} lost accuracy, {len(better)} improved, {len(pricier)} costlier at equal '
          f'accuracy, {len(pinned) - len(worse) - len(better) - len(pricier) - len(missing)} unchanged')
    return worse


def sharp(which='dev', refs=(1e9, 3.4, 2.0), target=1e-10, seed=SEED):
    """Sweep `PlanConfig.sharp_ref` -- the alpha above which a corner stops owning a full arc.

    Separate stage because it is the one constant fitted to a measured optimum rather than derived,
    so it needs the dev/holdout discipline applied visibly: sweep on dev, adopt only if the holdout
    agrees. `refs=1e9` disables the mechanism and reproduces the pre-existing behaviour.
    """
    from lappy.basis_plan import PlanConfig, plan_basis
    from ..suite.domains import SUITE
    if isinstance(which, (list, tuple)) and len(which) == 1 and which[0] in ('dev', 'holdout', 'all'):
        which = which[0]
    keys = ([k for k in DEV if k in SUITE] if which == 'dev'
            else [k for k in _poly_keys() if k not in DEV] if which == 'holdout'
            else _poly_keys() if which == 'all' else list(which))
    _warm_up()
    done = seen('sharp')
    print(f'\nsharp  sharp_ref sweep on {which} ({len(keys)} domains) at target {target:.0e}')
    for key in keys:
        domain = SUITE[key].domain()
        eigs = reference_eigs_for(key)
        for sr in refs:
            row = blank_row(stage='sharp', case=key, family='planner', n_requested=None,
                            seed=seed, variant=f'sharp={sr:g}', domain_key=key, target=target,
                            tier='low', lam_max=float(weyl_est(6, domain)), lam=float(eigs[0]))
            if record_id(row) in done:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    plan = plan_basis(domain, row['lam_max'], target=target,
                                      cfg=PlanConfig(sharp_ref=sr))
                row.update(_score(domain, plan, eigs, seed))
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'
            append(row)
            tag = f'{key}/sharp={sr:g}'
            print(f"    {tag:36} " + (f"n={row['n_basis']:4d} MP={row['certified_digits']:5.2f} "
                                      f"contrast={row['contrast']:.0e}" if row['ok']
                                      else f"FAILED {row['error'][:50]}"))
    report_sharp()


def report_sharp():
    rows = [r for r in load('sharp') if r['ok']]
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r['case'], {})[r['variant']] = r
    refs = sorted({r['variant'] for r in rows},
                  key=lambda v: -float(v.split('=')[1]))
    base = refs[0]
    print(f'\nsharp  certified digits by sharp_ref ({base} = mechanism disabled)')
    print(f"  {'set':8} {'domain':18} " + ' '.join(f'{v:>16}' for v in refs))
    deltas = {v: [] for v in refs[1:]}
    costs = {v: [] for v in refs[1:]}
    for key in sorted(by, key=lambda k: (k in DEV, k)):
        cells = by[key]
        if base not in cells:
            continue
        b = cells[base]
        row = [f"{b['certified_digits']:5.2f} (n={b['n_basis']})"]
        for v in refs[1:]:
            if v in cells:
                c = cells[v]
                row.append(f"{c['certified_digits']:5.2f} ({c['certified_digits']-b['certified_digits']:+.1f})")
                deltas[v].append(c['certified_digits'] - b['certified_digits'])
                costs[v].append(c['n_basis']/max(b['n_basis'], 1) - 1.0)
            else:
                row.append('--')
        print(f"  {'dev' if key in DEV else 'holdout':8} {key:18} " + ' '.join(f'{c:>16}' for c in row))
    print()
    for v in refs[1:]:
        if deltas[v]:
            d = np.array(deltas[v])
            print(f'  {v}: median {np.median(d):+.2f} digits, worst {d.min():+.2f}, '
                  f'{int((d > 0.3).sum())} better / {int((d < -0.3).sum())} worse of {len(d)}, '
                  f'cost {100*np.median(costs[v]):+.0f}% columns')


STAGES = {'s0a': s0a, 's0b': s0b, 's0c': s0c, 's3': s3, 'smooth': smooth, 'gate': gate,
          'sharp': sharp}

if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 's0a'
    args = sys.argv[2:] or None
    if stage == 'report':
        report_s0a()
        report_s0b()
        report_s0c()
        report_s3()
    elif stage in STAGES:
        STAGES[stage](args)
    else:
        print(f'unknown stage {stage!r}; expected one of {sorted(STAGES)} or "report"')
        sys.exit(2)
