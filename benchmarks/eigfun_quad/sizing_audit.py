"""Per-panel error attribution for `boundary_quadrature`: what the sizing model
predicts against what the panel actually delivers.

    python -m benchmarks.eigfun_quad.sizing_audit --keys right_trapezoid,GWW1,ellipse_a2
    python -m benchmarks.eigfun_quad.sizing_audit --exact          # closed-form controls

WHY. The suite re-run measured, via the x0-spread, that the rule misses its
advertised precision on about half the suite (`right_trapezoid` 4.3e-05 at a
claimed 1e-13). The scalar spread says *that* it misses, not *where*. This
attributes the error to individual panels, which is what a sizing fix needs.

INSTRUMENT. Error is attributed by REFINEMENT, one panel at a time: recompute
the whole integral with panel `i` replaced by a finer tiling of the same
interval, and the change is panel `i`'s error. Nothing else moves, so the
attribution is exact up to the refined tiling's own error.

Refinement respects what each rule is for. A legendre panel is bisected
repeatedly at the same order. A CORNER panel keeps its anchored end -- that end
carries the singularity the corner rule exists for and must never be cut off
(the rule for this is already in `eigfun_integrals._split_at_breaks`) -- and its
outer part becomes legendre panels at high order. Raising a corner panel's order
instead would be wrong twice over: accuracy is not monotone in order past a
nu-dependent threshold, and `cornerjac_order_cap` binds.

REFERENCE. Two sources, in order of preference:

  * exact eigenfunctions (`--exact`): polyomino, rect, sector, disk. Zero
    residual and a closed-form norm, so the measured error is the rule's alone.
    `docs/eigfun_integrals.md` is explicit that `mpmath.quad` is NOT a valid
    reference for these integrands, so every reference here is closed-form or
    self-convergent.
  * the MPS eigenfunction for suite geometries with no closed form. Safe here
    for a reason established by measurement, not assumption: holding the node set
    fixed and sweeping `n_basis` 60->480 moves `sup|u|` on the boundary by six
    orders (9.3e-10 -> 1.0e-15) and leaves the x0-spread at 1.68e-06, three
    significant figures identical. Whatever this measures, it is not the
    approximant's residual.
"""
import argparse
import sys
import warnings

import numpy as np

from lappy import quad
from lappy import eigfun_integrals as ei
from lappy.eigfun_integrals import (CornerPanel, EigfunData, assemble_panels,
                                    boundary_quadrature, corner_specs, gram)
from lappy.utils import complex_dot


# ---------------------------------------------------------------------------
# refinement: the measuring device
# ---------------------------------------------------------------------------

def refine_panel(panel, depth=3, smooth_order=48):
    """Replacement tiling of `panel`'s interval, finer but covering it exactly.

    The same rule `eigfun_integrals.refine_quadrature` applies to a whole node set --
    this is the single-panel version, which is what per-panel attribution needs.
    A legendre panel is split into `2**depth` equal pieces at `smooth_order`. A
    corner panel keeps its anchor on a piece `2**depth` times shorter (same rule,
    same order -- the singular end is refined by SHRINKING, not by raising an
    order that is capped and non-monotone) and the vacated part becomes legendre.
    """
    lo, hi = panel.tau0, panel.tau1              # signed: hi<lo means anchored at hi
    if panel.rule == 'legendre':
        edges = np.linspace(lo, hi, 2**depth + 1)
        return [panel._replace(tau0=a, tau1=b, order=smooth_order)
                for a, b in zip(edges[:-1], edges[1:])]
    # corner panel: geometric refinement toward the anchor at tau0
    fracs = 0.5**np.arange(depth, -1, -1)        # e.g. [1/8, 1/4, 1/2, 1]
    taus = lo + (hi - lo)*fracs
    out = [panel._replace(tau1=taus[0])]         # anchored piece, shrunk
    for a, b in zip(taus[:-1], taus[1:]):
        out.append(CornerPanel(panel.seg_idx, a, b, 'legendre', smooth_order,
                               np.nan, None, np.nan, False, -1))
    return out


def model_prediction(panel, k_seg, precision):
    """What the sizing model claims for this panel at the order it was given."""
    if panel.rule == 'legendre':
        span = abs(panel.tau1 - panel.tau0)
        tau, w = quad.cached_leggauss(panel.order)
        k = k_seg*span
        if k <= 0:
            return 0.0
        exact = (np.exp(1j*k) - 1.0)/(1j*k)
        return float(abs(np.sum(w*np.exp(1j*k*tau)) - exact))
    span = abs(panel.tau1 - panel.tau0)
    try:
        return float(quad.corner_model_error(panel.rule, panel.order, panel.nu,
                                             panel.gamma, panel.sub, panel.curved,
                                             k=0.5*k_seg*span))
    except (ValueError, np.linalg.LinAlgError):
        return float('nan')


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------

def audit(domain, lam, ed_for, lam_max=None, precision=1e-13, x0=None,
          depth=3, smooth_order=48, verbose=True):
    """Attribute the Gram's error to individual panels.

    `ed_for(bq)` returns `EigfunData` for any node set -- the caller supplies it,
    so exact and MPS eigenfunctions go through the same path.
    """
    lam_max = lam if lam_max is None else lam_max
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bq = boundary_quadrature(domain, lam_max, precision=precision)
    panels = list(bq.panels)
    if x0 is None:
        x0 = bq.x0
    k_seg = {i: 2.0*np.sqrt(max(lam_max, 0.0))*seg.len
             for i, seg in enumerate(domain.bdry.segments)}

    G0 = float(gram(ed_for(bq), lam, bq, x0=x0)[0, 0])

    rows = []
    for i, p in enumerate(panels):
        repl = panels[:i] + refine_panel(p, depth, smooth_order) + panels[i+1:]
        bq_i = assemble_panels(domain, repl, precision=precision, x0=bq.x0)
        Gi = float(gram(ed_for(bq_i), lam, bq_i, x0=x0)[0, 0])
        rows.append(dict(idx=i, seg=p.seg_idx, rule=p.rule, order=p.order,
                         corner=p.corner, nu=p.nu,
                         span=abs(p.tau1 - p.tau0)*domain.bdry.segments[p.seg_idx].len,
                         measured=abs(Gi - G0),
                         model=model_prediction(p, k_seg[p.seg_idx], precision)))

    # everything refined at once: the total the model is being judged against
    allref = []
    for p in panels:
        allref += refine_panel(p, depth, smooth_order)
    bq_all = assemble_panels(domain, allref, precision=precision, x0=bq.x0)
    G_ref = float(gram(ed_for(bq_all), lam, bq_all, x0=x0)[0, 0])

    if verbose:
        print(f'  nodes={len(bq.pts):5d} -> refined {len(bq_all.pts):6d}   '
              f'G={G0:.14f}  G_ref={G_ref:.14f}  |G-G_ref|={abs(G0-G_ref):.2e}')
        print(f'  {"panel":>5} {"seg":>3} {"rule":>13} {"ord":>4} {"nu":>7} '
              f'{"span":>8} {"model":>10} {"measured":>10} {"ratio":>9}')
        for r in sorted(rows, key=lambda r: -r['measured']):
            ratio = (r['measured']/r['model']) if r['model'] > 0 else float('inf')
            nu = '' if not np.isfinite(r['nu']) else f"{r['nu']:.3f}"
            print(f"  {r['idx']:5d} {r['seg']:3d} {r['rule']:>13} {r['order']:4d} "
                  f"{nu:>7} {r['span']:8.3f} {r['model']:10.2e} "
                  f"{r['measured']:10.2e} {ratio:9.1e}")
    return dict(G=G0, G_ref=G_ref, total=abs(G0 - G_ref), rows=rows,
                nodes=len(bq.pts), nodes_ref=len(bq_all.pts))


# ---------------------------------------------------------------------------
# eigenfunction sources
# ---------------------------------------------------------------------------

def exact_ed(u, grad):
    """`ed_for` from closed-form `u(z)` and `grad(z)` (complex gradient)."""
    def ed_for(bq):
        G = grad(bq.pts)
        return EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts,
                          np.asarray(u(bq.pts))[:, None],
                          complex_dot(G, bq.normals)[:, None],
                          complex_dot(G, bq.tangents)[:, None])
    return ed_for


def mps_ed(solver, lam):
    """`ed_for` from an MPS solver's coefficients at `lam`."""
    coef = solver.eigenfunction_coef(lam, mult=1, orthonorm=False)
    def ed_for(bq):
        return ei.eigfun_cauchy_data(solver.basis, lam, coef, bq)
    return ed_for


def build_mps(domain, n_basis, int_npts=None, seed=0):
    from lappy import bases, mps, MPSEigensolver
    np.random.seed(seed)
    basis = bases.make_default_basis(domain, n_basis)
    bp = domain.bdry_pts(mps.pts_per_seg(domain, basis, mult=2))
    ip = domain.int_pts(method='random', npts_rand=int_npts or max(2*n_basis, 500))
    basis = basis.to_normalized((bp, ip))
    return MPSEigensolver(basis, bp, ip, rtol=mps.rtol_default, ttol=1e-3)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def exact_cases():
    """Closed-form controls: (name, domain, lam, ed_for)."""
    from lappy import reference as ref
    from lappy.geometry import polyomino, rect, disk_sector, disk
    out = []

    for cells, name in (([(0, 0), (1, 0), (0, 1)], 'L-tromino'),
                        ([(i, 1) for i in range(4)] + [(0, 0)], 'long-arm')):
        for (m, n) in ((1, 2), (2, 3)):
            dom = polyomino(cells)
            lam = ref.polyomino_eig(m, n)
            u, norm2 = ref.polyomino_eigfun(m, n, len(cells))
            g = ref.polyomino_eigfun_grad(m, n)
            s = 1.0/np.sqrt(norm2)
            out.append((f'{name}({m},{n})', dom, lam,
                        exact_ed(lambda z, u=u, s=s: s*u(z),
                                 lambda z, g=g, s=s: s*g(z))))

    # rect: integer nu at every corner, so the smooth rule really is exact there --
    # the control that says the machinery itself is sound.
    for (L, H) in ((1.0, 1.0), (2.0, 1.0)):
        for (m, n) in ((1, 2), (3, 2)):
            dom = rect(L, H)
            lam = ref.rect_eig(m, n, L, H)
            u, norm2 = ref.rect_eigfun(m, n, L, H)
            s = 1.0/np.sqrt(norm2)
            a, b = m*np.pi/L, n*np.pi/H
            def grad(z, a=a, b=b, s=s):
                x, y = np.real(z), np.imag(z)
                return s*(a*np.cos(a*x)*np.sin(b*y)
                          + 1j*b*np.sin(a*x)*np.cos(b*y))
            out.append((f'rect({L}x{H},{m},{n})', dom, lam,
                        exact_ed(lambda z, u=u, s=s: s*u(z), grad)))

    # sector: an admissible singular corner (the cornerjac path) at two angles
    for a_over_pi in (0.5, 1.5):
        alpha = a_over_pi*np.pi
        dom = disk_sector(1.0, alpha)
        lam = ref.sector_eig(1, 1, 1.0, alpha)
        u, norm2 = ref.sector_eigfun(1, 1, 1.0, alpha)
        s = 1.0/np.sqrt(norm2)
        g = ref.sector_eigfun_grad(1, 1, 1.0, alpha)
        out.append((f'sector({a_over_pi}pi)', dom, lam,
                    exact_ed(lambda z, u=u, s=s: s*u(z),
                             lambda z, g=g, s=s: s*g(z))))

    # disk: curved, corner-free, EXACT eigenfunction. This is the control that
    # separates "the smooth model is optimistic" from "curved ParametricSegment
    # geometry is the problem" -- a circular arc is exactly parametrized, so any
    # shortfall here is the model's alone.
    from scipy.special import jv, jvp, jn_zeros
    for (m, n) in ((0, 1), (2, 1), (3, 2)):
        R = 1.0
        dom = disk(R)
        j_mn = jn_zeros(m, n)[-1]
        k = j_mn/R
        lam = k**2
        u, norm2 = ref.disk_eigfun(m, n, R)
        s = 1.0/np.sqrt(norm2)
        def grad(z, m=m, k=k, s=s):
            r, th = np.abs(z), np.angle(z)
            ur = k*jvp(m, k*r)*np.cos(m*th)
            ut = -m*jv(m, k*r)*np.sin(m*th)/np.where(r == 0, 1.0, r)
            return s*((ur*np.cos(th) - ut*np.sin(th))
                      + 1j*(ur*np.sin(th) + ut*np.cos(th)))
        out.append((f'disk({m},{n})', dom, lam,
                    exact_ed(lambda z, u=u, s=s: s*u(z), grad)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default='right_trapezoid,GWW1,ellipse_a2')
    ap.add_argument('--exact', action='store_true',
                    help='run the closed-form controls instead of suite domains')
    ap.add_argument('--mode', type=int, default=1, help='which eigenvalue (0-based)')
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--smooth-order', type=int, default=48)
    ap.add_argument('--precision', type=float, default=1e-13)
    ap.add_argument('--generic-x0', action='store_true',
                    help='use a generic x0 instead of the node set\'s own')
    args = ap.parse_args(argv)

    if args.exact:
        for name, dom, lam, ed_for in exact_cases():
            print(f'\n=== {name}  lam={lam:.8f} (exact eigenfunction) ===')
            x0 = (0.37 + 0.181j) if args.generic_x0 else None
            audit(dom, lam, ed_for, lam_max=3*lam, precision=args.precision,
                  x0=x0, depth=args.depth, smooth_order=args.smooth_order)
        return 0

    import json
    import os
    from benchmarks.suite.domains import SUITE
    HERE = os.path.dirname(os.path.abspath(__file__))
    JSONL = os.path.join(os.path.dirname(HERE), 'suite', 'run', 'buckets.jsonl')
    rec = {}
    for line in open(JSONL):
        r = json.loads(line)
        if r.get('tag') == 'orth' and r.get('eigs'):
            rec[r['key']] = r

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'reference'))
    from common import lambda_window

    for key in [k.strip() for k in args.keys.split(',')]:
        r = rec[key]
        dom = SUITE[key].domain()
        lam = r['eigs'][args.mode]
        _, b = lambda_window(dom, SUITE[key].n_eigs)
        solver = build_mps(dom, r['n_basis'])
        print(f'\n=== {key}  lam={lam:.8f}  n_basis={r["n_basis"]} (MPS) ===')
        x0 = (0.37 + 0.181j) if args.generic_x0 else None
        audit(dom, lam, mps_ed(solver, lam), lam_max=b, precision=args.precision,
              x0=x0, depth=args.depth, smooth_order=args.smooth_order)
    return 0


if __name__ == '__main__':
    sys.exit(main())
