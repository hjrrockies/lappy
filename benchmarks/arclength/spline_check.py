"""Splines: does the Newton fix carry over, and do knots need their own panels?

A degree-k B-spline is only C^(k-1) at its knots, so |p'| has breaks there. Two consequences:
  * the arclength integral s(t) must not integrate a Gauss panel ACROSS a knot;
  * a boundary quadrature panel spanning a knot sees a C^(k-1) integrand and loses spectral
    convergence even with a perfect t(s).

Both are testable.
"""
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import make_interp_spline

from lappy.geometry import SplineSegment, ellipse
from benchmarks.arclength.newton_prototype import AnalyticArclength


def wobbly_spline(n_ctrl=9, tol=1e-4):
    """A closed-ish cubic spline with genuinely varying speed."""
    th = np.linspace(0, 2*np.pi, n_ctrl)
    r = 1.0 + 0.3*np.cos(3*th)
    pts = r*np.exp(1j*th)
    t = np.linspace(0, 1, n_ctrl)
    sp = make_interp_spline(t, pts, k=3)
    return SplineSegment(sp, 0, 1, tol=tol)


seg = wobbly_spline()
seg.len
m = AnalyticArclength(seg)
print(f"spline: len={seg.len:.6f}  table nodes={len(m.t_nodes)}  "
      f"spline knots in range={np.sum((seg.spline.t > 0) & (seg.spline.t < 1))}")

tau = np.linspace(0, 1, 501)
print(f"  |dp/dtau|-L :  pchip {np.abs(np.abs(seg.dp(tau))-seg.len).max()/seg.len:.2e}"
      f"   newton {np.abs(np.abs(m.dp(tau))-m.L).max()/m.L:.2e}")

# round-trip: is t(s) actually solving the equation?
s = np.linspace(1e-9, m.L - 1e-9, 401)
print(f"  s(t(s)) - s :  pchip {np.abs(seg._s_of_t(seg._t_of_s(s))-s).max()/m.L:.2e}"
      f"   newton {np.abs(m.s_of_t(m.t_of_s(s))-s).max()/m.L:.2e}")

# integrate a smooth function of position over the spline, PCHIP vs Newton, vs a
# high-order reference built with the Newton map itself at very high order
f = lambda z: np.exp(0.3*z.real)*np.cos(2*z.imag)
xr, wr = leggauss(2000)
ref = float(np.sum(wr/2*m.L*f(m.p((xr + 1)/2))))
print(f"\n  {'order':>6} {'pchip':>12} {'newton':>12}   (reference: Newton @ order 2000)")
for order in (32, 64, 128, 256, 512):
    x, w = leggauss(order)
    t_ = (x + 1)/2
    wts = w/2*seg.len
    ep = abs(np.sum(wts*f(seg.p(t_)))/ref - 1)
    en = abs(np.sum(wts*f(m.p(t_)))/ref - 1)
    print(f"  {order:6d} {ep:12.2e} {en:12.2e}")

print("\n  knot-aligned panels (Newton map), same total node count:")
knots = np.unique(np.clip(seg.spline.t[(seg.spline.t >= 0) & (seg.spline.t <= 1)], 0, 1))
s_knots = m.s_of_t(np.clip(knots, m.t_nodes[0], m.t_nodes[-1]))/m.L
s_knots = np.unique(np.clip(s_knots, 0.0, 1.0))
for per in (8, 16, 32):
    x, w = leggauss(per)
    tot = 0.0
    n = 0
    for lo, hi in zip(s_knots[:-1], s_knots[1:]):
        tt = lo + (hi - lo)*(x + 1)/2
        ww = (hi - lo)*w/2*m.L
        tot += float(np.sum(ww*f(m.p(tt))))
        n += per
    print(f"  {n:6d} {'':12} {abs(tot/ref - 1):12.2e}  ({len(s_knots)-1} knot panels x {per})")

print("\ncost per evaluation (vectorized over 2000 points):")
tau = np.linspace(0, 1, 2000)
t0 = time.time(); [seg.p(tau) for _ in range(5)]; t_p = (time.time()-t0)/5
t0 = time.time(); [m.p(tau) for _ in range(5)]; t_n = (time.time()-t0)/5
print(f"  pchip  {t_p*1e3:8.2f} ms      newton {t_n*1e3:8.2f} ms      ratio {t_n/t_p:.0f}x")
print("  (the prototype's s_of_t loops in Python over points; vectorizing is straightforward)")
