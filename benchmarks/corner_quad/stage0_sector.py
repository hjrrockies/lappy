"""Stage 0 gate: corner-adapted rule vs the current Kress rule on disk_sector.

Exact truth: reference.sector_eigfun gives (u, norm2) in closed form and
sector_eigfun_grad the analytic gradient, so an exactly-normalized eigenfunction
fed to the Rellich identity must return 1. No quadrature anywhere in the
reference, and NO mpmath.quad -- as nu -> 1/2 its endpoint behaviour approaches
r^-1 and it errs by 4e-2 even at 40 dps, which manufactures convincing but
entirely spurious plateaus.

Gate: >=4 orders of improvement over Kress before anything enters lappy/.

Run: python -m benchmarks.corner_quad.stage0_sector
"""
import warnings

import numpy as np

from lappy import reference as ref
from lappy.geometry import disk_sector
from lappy.bases import FourierBesselBasis
from lappy.utils import complex_dot
from .proto import (panel_plan, assemble, kress_reference, rellich_norm2,
                    corner_nus, cornerjac, TAU_FLOOR)

warnings.filterwarnings('ignore')
R = 1.0

ALPHAS = [('pi/2', 0.5), ('2pi/3', 2/3), ('1.1pi', 1.1), ('1.25pi', 1.25),
          ('1.5pi', 1.5), ('1.75pi', 1.75), ('1.9pi', 1.9), ('1.99pi', 1.99)]


def sector_setup(alpha_over_pi, m, n):
    alpha = alpha_over_pi*np.pi
    dom = disk_sector(R, alpha)
    u, norm2 = ref.sector_eigfun(m, n, R, alpha)
    g = ref.sector_eigfun_grad(m, n, R, alpha)
    lam = ref.sector_eig(m, n, R, alpha)
    scale = 1.0/np.sqrt(norm2)
    return dom, lam, (lambda z: scale*g(z))


def eval_norm(dom, lam, gn, pts, normals, wts, x0, panel_id=None, groups=None):
    un = complex_dot(gn(pts), normals)
    return rellich_norm2(pts, normals, wts, un, lam, x0, panel_id, groups)


def basis_for(dom, orders=20):
    ncorn = np.shape(dom.corner_angles)[1]
    return FourierBesselBasis.from_domain(dom, [orders]*ncorn)


def x0_choices(alpha):
    """x0 placements. The 'bad' one is what default_x0 effectively returns; the
    'generic' one is chosen OFF the diagonal so the edge terms do not cancel."""
    return [('apex', 0.0 + 0j),
            ('generic', 0.4 + 0.05j),
            ('bbox-ish', 0.000257 - 0.000257j)]


def sweep_alpha(order=16, m=1, n=1):
    print(f"\n=== Leg-1 core: error vs alpha, corner rule (order={order}/panel) "
          f"vs Kress, mode ({m},{n}) ===")
    print(f"{'alpha':>8} {'nu':>6} {'x0':>9} {'corner-jac':>12} {'nodes':>6} "
          f"{'Kress':>12} {'nodes':>6} {'edge share':>11}")
    for name, a in ALPHAS:
        dom, lam, gn = sector_setup(a, m, n)
        basis = basis_for(dom)
        panels = panel_plan(dom, order_corner=order)
        pts, nrm, tng, wts, pid = assemble(dom, panels)
        kp, kn, kt, kw, _ = kress_reference(dom, basis, lam)
        # group panels: which came from a straight edge vs the arc
        groups = {i: ('edge' if p[0] != 1 else 'arc') for i, p in enumerate(panels)}
        nus = corner_nus(dom)
        nu_apex = nus[0]
        for lbl, x0 in x0_choices(a):
            new, parts = eval_norm(dom, lam, gn, pts, nrm, wts, x0, pid, groups)
            old, _ = eval_norm(dom, lam, gn, kp, kn, kw, x0)
            share = abs(parts['edge'])/(abs(parts['edge']) + abs(parts['arc']))
            print(f"{name:>8} {nu_apex:6.3f} {lbl:>9} {new-1:12.2e} {len(pts):6d} "
                  f"{old-1:12.2e} {len(kp):6d} {share:10.1%}")


def sweep_order(a=1.5, m=1, n=1, x0=0.4 + 0.05j):
    print(f"\n=== Convergence in panel order at alpha={a}pi, x0={x0} ===")
    dom, lam, gn = sector_setup(a, m, n)
    print(f"{'order':>6} {'nodes':>6} {'rel err':>12} {'tau_min':>10}")
    for order in [4, 6, 8, 12, 16, 24, 32, 48]:
        panels = panel_plan(dom, order_corner=order)
        pts, nrm, tng, wts, pid = assemble(dom, panels)
        val, _ = eval_norm(dom, lam, gn, pts, nrm, wts, x0)
        nu = corner_nus(dom)[0]
        tau, _ = cornerjac(order, nu)
        flag = ' <-FLOOR' if tau.min() < TAU_FLOOR else ''
        print(f"{order:6d} {len(pts):6d} {val-1:12.2e} {tau.min():10.2e}{flag}")


def sweep_panel_length(m=1, n=1, x0=0.4 + 0.05j, order=16):
    """The plan's open question: the corner expansion converges only within the
    largest disk about the corner inside Omega, so a full-length panel may
    overshoot it. For disk_sector the radii have length R and the inradius at
    the apex is R, so frac=1 should be safe here -- this measures whether it is,
    and gives the shape of the dependence for domains where it is not."""
    print(f"\n=== Panel length (frac of segment) at order={order}, x0={x0} ===")
    print(f"{'alpha':>8} " + " ".join(f"{f'frac={f}':>12}" for f in
                                      [0.25, 0.5, 0.75, 1.0]))
    for name, a in [('1.25pi', 1.25), ('1.5pi', 1.5), ('1.9pi', 1.9)]:
        dom, lam, gn = sector_setup(a, m, n)
        row = []
        for frac in [0.25, 0.5, 0.75, 1.0]:
            panels = panel_plan(dom, order_corner=order, frac=frac)
            pts, nrm, tng, wts, pid = assemble(dom, panels)
            val, _ = eval_norm(dom, lam, gn, pts, nrm, wts, x0)
            row.append(f"{val-1:12.2e}")
        print(f"{name:>8} " + " ".join(row))


def nu_sensitivity(a=1.5, m=1, n=1, x0=0.4 + 0.05j, order=16):
    """Guard that the geometric nu is wired exactly: a 3e-4 relative error in nu
    must cost ~4 digits (docs/corner_quadrature.tex Sec. 4). If this test ever
    reports no loss, a margin-padded or rounded exponent has crept in."""
    print(f"\n=== nu sensitivity at alpha={a}pi (true nu = {1/a:.6f}) ===")
    dom, lam, gn = sector_setup(a, m, n)
    nu_true = corner_nus(dom)[0]
    print(f"{'nu used':>12} {'rel err in nu':>14} {'rel err':>12}")
    for dn in [0.0, 1e-8, 1e-4, 3e-4, 1e-2]:
        nu = nu_true*(1 + dn)
        panels = [(i, t0, t1, k, o, (nu if k == 'jac' else v))
                  for (i, t0, t1, k, o, v) in panel_plan(dom, order_corner=order)]
        pts, nrm, tng, wts, pid = assemble(dom, panels)
        val, _ = eval_norm(dom, lam, gn, pts, nrm, wts, x0)
        print(f"{nu:12.8f} {dn:14.0e} {val-1:12.2e}")


def modes_and_controls(order=16, x0=0.4 + 0.05j):
    print(f"\n=== Higher modes and nu>1 controls at order={order}, x0={x0} ===")
    print(f"{'alpha':>8} {'m,n':>5} {'nu_mode':>8} {'rel err':>12}")
    for name, a in [('1.5pi', 1.5), ('1.9pi', 1.9)]:
        for (m, n) in [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)]:
            dom, lam, gn = sector_setup(a, m, n)
            panels = panel_plan(dom, order_corner=order)
            pts, nrm, tng, wts, pid = assemble(dom, panels)
            val, _ = eval_norm(dom, lam, gn, pts, nrm, wts, x0)
            print(f"{name:>8} {f'{m},{n}':>5} {m/a:8.3f} {val-1:12.2e}")


if __name__ == '__main__':
    sweep_alpha()
    sweep_order()
    sweep_panel_length()
    nu_sensitivity()
    modes_and_controls()
