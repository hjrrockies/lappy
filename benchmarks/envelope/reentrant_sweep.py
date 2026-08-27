"""Where does the corner-block heuristic stop keeping up with a REENTRANT corner?

    uv run python -m benchmarks.envelope.reentrant_sweep [--omegas 200 240 270 300] [--caps 400]

WHY, AND WHERE THE NUMBER 280 COMES FROM. Reentrant corners are in scope, and `douse` has now
measured what they cost downstream: in the multi-start ledgers, 7 of the 13 runs begun at a
reentrant shape were excluded on `worst_sigma` against 2 of the other 66. An instrumented
reproduction of the worst of them (N=6 `lam_3/lam_1`, start 3) put the failure at a specific
angle: at ~280 degrees `Eigenproblem.track_set` began returning a NON-CONSECUTIVE set --
[56.90, 209.97, 274.11, 286.67] -- at a tension of 4.7e-12. Pristine sigma, wrong answer. One of
those endpoints stayed at sigma = 5.3e-04 even when the plan was rebuilt AT it, so the shape was
out of the planner's reach rather than merely stale.

That is a `lappy` question and this probe asks it directly: sweep the reentrant angle and watch
the achieved digits, the tension, and what the planner did about it.

WHAT THE SIZING RULE PREDICTS. `_corner_blocks` gives a corner `M = ceil((nu_osc + nu_cont)/alpha)`
with `alpha = pi/omega`, capped by `_fb_ceiling(alpha) ~ _indep_digits/(ln(1/fb_inner_frac) alpha)`.
Both scale as `1/alpha = omega/pi`, so a reentrant corner asks for more terms AND is allowed more.
The question is which one gives out first, so every row records `M`, the uncapped demand, and the
ceiling -- "the ceiling binds" and "the sizing rule is not asking for enough" are different
diagnoses with different fixes, and the digits alone cannot tell them apart.

THE FAMILY, AND WHY IT IS NOT A QUADRILATERAL. A deep notch has to be paid for by the other
corners: interior angles sum to `(N-2) 180`, so at N=4 an `omega = 330` notch leaves 30 degrees
for the remaining three, and the dart family below really does reach 2.4-degree spikes there.
That confounds the sweep -- the sharp end of the axis and the reentrant end arrive together, and
`_corner_blocks` treats them differently (`sharp_ref/alpha` shrinks a sharp corner's arc toward
`M = 1`). At N=6 the budget is 720 degrees and the same notch leaves an average of 78, so the
neighbours stay moderate: measured, `omega = 283.6` puts them at 38.2 rather than 2.4.

So the default instrument is a REGULAR HEXAGON WITH ONE VERTEX PULLED IN, `notched_ngon`: vertex
0 moves radially to `r*v_0` and `r` is bisected to hit the requested `omega` exactly. N=6 is also
the size the `douse` failure happened at, and 284 degrees is inside its range. The neighbours
still sharpen with `omega` and every row records `min_angle`, so the confound stays visible
rather than being assumed away.

`dart` is kept as the N=4 family (`--family dart`) precisely because it shows that coupling.
`--control` runs the same displacement pushed OUTWARD instead, which is convex at the same
`|r - 1|`: matched in displacement rather than in angle, and the honest way to say so is to print
both angles.

VALIDATION FIRST, ALWAYS. `check_precision` is a residual bound, not an error, and `prec`
dominates what is achieved -- omitting it once cost this project two written-up conclusions that
were then fully reversed. So the probe opens with two domains whose spectra are known in closed
form at the same operating point, and prints the residual-based digits NEXT TO the true error:

    the unit square            lam = pi^2 (m^2 + n^2),          m, n >= 1
    the 45-45-90 triangle      lam = (pi^2/L^2)(m^2 + n^2),     m > n >= 1  (odd reflection)

If those two disagree, nothing below them means anything and the sweep should not be read.
"""
import argparse
import json
import os
import time
import traceback
import warnings
from dataclasses import replace

import numpy as np

from lappy import Eigenproblem, Polygon, basis_plan as BP
from lappy.asymp import weyl_count_poly, weyl_est
from lappy.mps import MPSEigensolver

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
LEDGER = os.path.join(OUT, 'reentrant_sweep.jsonl')
PERIM = 4.0


def weyl_deficit(dom, lams, rtol=1e-6):
    """Is the solved set CONSECUTIVE? Worst shortfall of the count over mid-gap cut points.

    The recipe is `douse.spectral_guard`'s, reimplemented here rather than imported because
    `lappy` does not depend on `douse` -- and it belongs in this probe, because the failure that
    motivated the sweep is not a resolution failure at all. In the enforced reproduction of the
    N=6 run, iterates 199 and 200 (notch angles 281.4 and 280.5 degrees) were refused with a
    deficit of 3.14 having come from a FULL SOLVE, not from tracking: at those angles
    `Eigenproblem.solve` itself returned a set with a mode missing, at a tension in the 1e-11
    band. Digits and sigma cannot see that, so a probe reporting only those two would have
    called these shapes healthy.
    """
    lams = np.sort(np.asarray(lams, dtype=float))
    if lams.size < 2 or not np.all(np.isfinite(lams)) or lams[0] <= 0:
        return float('nan')
    levels = []
    for x in lams:
        if not levels or x > levels[-1]*(1.0 + rtol):
            levels.append(float(x))
    if len(levels) < 2:
        return float('nan')
    return max(float(weyl_count_poly(0.5*(lo + hi), domain=dom))
               - int(np.count_nonzero(lams <= 0.5*(lo + hi)))
               for lo, hi in zip(levels[:-1], levels[1:]))


def _scaled(v, perim=PERIM):
    v = np.asarray(v, dtype=complex)
    return v*perim/np.abs(v - np.roll(v, 1)).sum()


def _angles(v):
    return np.degrees(np.pi - np.angle((np.roll(v, -1) - v)/(v - np.roll(v, 1))))


def notched_ngon(omega_deg, N=6, perim=PERIM, outward=False):
    """Regular `N`-gon with vertex 0 moved radially until its interior angle is `omega_deg`.

    The angle is monotone decreasing in `r`, so a bisection on `r in (-0.95, 1)` inverts it. At
    N=6 the reachable range is `120` (regular) to about `296` degrees; ask for more and the
    bisection says so rather than returning a shape that is not what was requested.
    """
    def build(r):
        v = np.exp(2j*np.pi*np.arange(N)/N)
        v[0] = r*v[0]
        return v

    if outward:
        return _scaled(build(2.0 - _solve_r(omega_deg, build, N)), perim)
    return _scaled(build(_solve_r(omega_deg, build, N)), perim)


def _solve_r(omega_deg, build, N, lo=-0.95, hi=1.0, tol=1e-12):
    f = lambda r: _angles(build(r))[0] - omega_deg                    # noqa: E731
    if f(hi) > 0 or f(lo) < 0:
        raise ValueError(f'omega={omega_deg} is outside the reachable range for N={N}: '
                         f'[{_angles(build(hi))[0]:.1f}, {_angles(build(lo))[0]:.1f}] degrees')
    for _ in range(200):
        mid = 0.5*(lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5*(lo + hi)


def dart(omega_deg, gap=0.75, perim=PERIM):
    """The notch at `D` has interior angle exactly `omega_deg`, for `180 < omega < 360`."""
    if not 180.0 < omega_deg < 360.0:
        raise ValueError(f'omega must be reentrant, got {omega_deg}')
    t = 1.0/np.tan(np.radians(360.0 - omega_deg)/2.0)
    return _scaled([1 + 0j, 1j*(t + gap), -1 + 0j, 1j*t], perim)


def kite(omega_deg, gap=0.75, perim=PERIM):
    """The control: `D` reflected below `AC`, so the same vertex is convex at `360 - omega`."""
    t = 1.0/np.tan(np.radians(360.0 - omega_deg)/2.0)
    return _scaled([1 + 0j, 1j*(t + gap), -1 + 0j, -1j*t], perim)


def exact_square(n, perim=PERIM):
    """`pi^2 (m^2 + n^2)` on the square of side `perim/4`, sorted, first `n`."""
    L = perim/4.0
    m = np.arange(1, 40)
    lam = np.sort((np.pi/L)**2*(m[:, None]**2 + m[None, :]**2).ravel())
    return lam[:n]


def exact_right_isoceles(n, perim=PERIM):
    """The 45-45-90 triangle by odd reflection across the hypotenuse: `m > n >= 1`.

    Legs `L` are set by the perimeter, `L(2 + sqrt(2)) = perim`.
    """
    L = perim/(2.0 + np.sqrt(2.0))
    a = np.arange(1, 40)
    mm, nn = np.meshgrid(a, a, indexing='ij')
    lam = np.sort(((np.pi/L)**2*(mm**2 + nn**2))[mm > nn])
    return lam[:n]


def measure(name, vertices, k, n_cap, target=1e-14, prec=1e-13, exact=None,
            plan_vertices=None):
    """One row: the plan's corner table, the achieved digits, and the tension `douse` gates on.

    `plan_vertices` PLANS ON ONE SHAPE AND SOLVES ON ANOTHER, which is what `douse` does on every
    iterate: `SpectrumEvaluator` freezes a `BasisPlan` and calls `BP.realize(plan, dom)` at each
    new shape. Solving a shape on its own fresh plan is the easy question and this probe answers
    it comfortably at every angle up to 292 degrees; the failure in the wild came from a plan
    frozen a few steps earlier, so `--stale` is the version of the question that matches it.
    """
    rec = dict(shape=name, k=k, n_cap=n_cap, prec=float(prec), target=float(target))
    t0 = time.perf_counter()
    try:
        dom = Polygon(vertices)
        dom_plan = Polygon(plan_vertices) if plan_vertices is not None else dom
        lam_max = weyl_est(max(6, k + 2), dom)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cfg = replace(BP.PlanConfig(), n_cap=n_cap)
            plan = BP.plan_basis(dom_plan, weyl_est(max(6, k + 2), dom_plan),
                                 target=target, cfg=cfg)
            solver = MPSEigensolver.from_domain(dom, lam_max=lam_max,
                                                basis=BP.realize(plan, dom), prec=prec)
            evp = Eigenproblem(dom, eval_solver=solver, precision=prec)
            eigs = np.asarray(evp.solve(k), dtype=float)
            chk = evp.check_precision(eigs)
            # THE TENSION IS WHAT `douse` ACTUALLY GATES ON (`SpectrumEvaluator.replan_tol`,
            # `worst_sigma`), so a probe about what reentrance costs downstream has to report it
            # alongside the digits rather than instead of them.
            sig = max(float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in eigs)
        angles = np.degrees(np.pi - np.angle(
            (np.roll(dom.vertices, -1) - dom.vertices)/(dom.vertices - np.roll(dom.vertices, 1))))
        corners = []
        for c in plan.corners:
            corners.append(dict(omega=float(np.degrees(np.pi/c.alpha)), alpha=float(c.alpha),
                                M=int(c.M), ceiling=int(BP._fb_ceiling(c.alpha, cfg)),
                                demand=float((c.nu_osc + c.nu_cont)/c.alpha)))
        rec.update(ok=True, deficit=float(weyl_deficit(dom, eigs)),
                   n_total=int(plan.n_total), capped=bool(plan.capped),
                   n_fb=int(sum(c.M for c in plan.corners)),
                   n_src=int(sum(a.n_src for a in plan.arcs)),
                   digits=float(chk['digits']), worst_sigma=sig,
                   max_angle=float(np.max(angles)), min_angle=float(np.min(angles)),
                   lams=[float(x) for x in eigs], corners=corners,
                   shortfall=(str(plan.shortfall) if plan.shortfall else None))
        if exact is not None:
            # THE ONLY PLACE "DIGITS" CAN BE CHECKED AGAINST TRUTH. `check_precision` is a
            # residual bound; this is the error.
            err = np.abs(eigs - exact[:len(eigs)])/np.abs(exact[:len(eigs)])
            rec.update(true_err=float(err.max()), true_digits=float(-np.log10(err.max())))
    except Exception as exc:                                          # noqa: BLE001
        rec.update(ok=False, error=f'{type(exc).__name__}: {exc}',
                   trace=traceback.format_exc()[-400:])
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    return rec


def _done():
    out = set()
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            try:
                r = json.loads(line)
                out.add((r['shape'], r['k'], r['n_cap'], float(r.get('prec', -1.0))))
            except (ValueError, KeyError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--omegas', type=float, nargs='*',
                    default=[190, 210, 230, 250, 270, 285, 300, 315, 330])
    ap.add_argument('--caps', type=int, nargs='*', default=[400])
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--prec', type=float, default=1e-13)
    ap.add_argument('--gap', type=float, default=0.75)
    ap.add_argument('--N', type=int, default=6, help='vertices, for the notched_ngon family')
    ap.add_argument('--family', choices=['notch', 'dart'], default='notch')
    ap.add_argument('--control', action='store_true',
                    help='also run the matched CONVEX displacement at each angle')
    ap.add_argument('--stale', type=float, default=None,
                    help='plan at omega and SOLVE at omega+STALE degrees, the way a frozen plan '
                         'meets a shape the optimizer has since walked to.')
    ap.add_argument('--no-validate', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    done = _done()
    rows = []

    def run(name, verts, exact=None, plan_verts=None):
        for cap in args.caps:
            if (name, args.k, cap, float(args.prec)) in done:
                continue
            rec = measure(name, verts, args.k, cap, prec=args.prec, exact=exact,
                          plan_vertices=plan_verts)
            with open(LEDGER, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
            rows.append(rec)
            if rec['ok']:
                extra = (f"  true {rec['true_digits']:5.2f}" if 'true_digits' in rec else '')
                print(f"  {name:16s} cap={cap:4d}  n={rec['n_total']:4d}  "
                      f"digits {rec['digits']:5.2f}{extra}  sigma {rec['worst_sigma']:8.1e}  "
                      f"deficit {rec['deficit']:+5.2f}  {rec['seconds']:6.1f}s", flush=True)
            else:
                print(f"  {name:16s} cap={cap:4d}  {rec['error'][:88]}", flush=True)

    if not args.no_validate:
        print('validation on exactly known spectra (residual-based digits vs TRUE error):')
        s = PERIM/4.0
        run('square', [0, s, s + 1j*s, 1j*s], exact=exact_square(args.k))
        L = PERIM/(2.0 + np.sqrt(2.0))
        run('tri_45_45_90', [0, L, 1j*L], exact=exact_right_isoceles(args.k))

    print(f'\nreentrant sweep, family={args.family}, N={args.N}, k={args.k}, '
          f'prec={args.prec:g}:')
    for w in args.omegas:
        if args.family == 'dart':
            run(f'dart_{w:g}', dart(w, args.gap))
            if args.control:
                run(f'kite_{w:g}', kite(w, args.gap))
        elif args.stale is not None:
            run(f'stale_{w:g}+{args.stale:g}', notched_ngon(w + args.stale, args.N),
                plan_verts=notched_ngon(w, args.N))
        else:
            run(f'notch_{w:g}', notched_ngon(w, args.N))
            if args.control:
                run(f'bump_{w:g}', notched_ngon(w, args.N, outward=True))

    all_rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    sweep = [r for r in all_rows if r.get('ok')
             and r['shape'].startswith(('notch_', 'bump_', 'stale_', 'dart_', 'kite_'))
             and r['k'] == args.k and float(r.get('prec', -1.0)) == float(args.prec)]
    if not sweep:
        return
    print('\n  shape             n_total  digits   worst_sigma  deficit  min_ang   notch '
          'M / ceiling / demand')
    print('  ' + '-'*97)
    def _key(r):
        head, _, tail = r['shape'].partition('_')
        return head, float(tail.split('+')[0])

    for r in sorted(sweep, key=_key):
        notch = max(r['corners'], key=lambda c: c['omega'])
        pin = ' PINNED' if notch['M'] >= notch['ceiling'] else ''
        d = r.get('deficit', float('nan'))
        flag = ' MODE MISSING' if d > 0.5 else ''
        print(f"  {r['shape']:16s} {r['n_total']:6d}  {r['digits']:6.2f}  "
              f"{r['worst_sigma']:11.1e}  {d:+7.2f}  {r['min_angle']:7.1f}   {notch['M']:3d} / "
              f"{notch['ceiling']:3d} / {notch['demand']:6.1f}{pin}{flag}")
    print('\n  deficit > 0.5 means the SOLVED set is not consecutive -- a mode is missing at a\n'
          '  tension the sigma column calls healthy. That is the failure the sweep was built for.')
    print('  M PINNED at the ceiling means `_fb_ceiling` is what limits the block, not the '
          'sizing\n  rule -- a different diagnosis with a different fix. `demand` is the '
          'uncapped\n  `(nu_osc + nu_cont)/alpha` the rule asked for.')


if __name__ == '__main__':
    main()
