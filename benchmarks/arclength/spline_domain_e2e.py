"""End to end on a spline-boundary domain: does boundary_quadrature now integrate spectrally?"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import mpmath as mp
from scipy.interpolate import make_interp_spline

from lappy.geometry import Domain, MultiSegment, SplineSegment
from lappy import eigfun_integrals as ei

mp.mp.dps = 30


def spline_domain(n_ctrl=13, amp=0.25, tol=1e-4):
    """Closed C^2 cubic spline boundary: r(th) = 1 + amp*cos(3 th)."""
    th = np.linspace(0, 2*np.pi, n_ctrl)[:-1]
    r = 1.0 + amp*np.cos(3*th)
    xy = np.column_stack([r*np.cos(th), r*np.sin(th)])
    xy = np.vstack([xy, xy[:1]])
    sp = make_interp_spline(np.linspace(0, 1, len(xy)), xy, k=3, bc_type='periodic')
    seg = SplineSegment(sp, 0, 1, tol=tol)
    return Domain(MultiSegment([seg], val_simple=False, val_contiguous=False),
                  val_simple=False, val_closed=False)


dom = spline_domain()
seg = dom.bdry.segments[0]
print(f"spline domain: perimeter={dom.perimeter:.10f}  break_taus={len(seg.break_taus)}")

# reference: integrate f ds in the spline's own parameter at high precision
f_np = lambda z: np.exp(0.3*z.real)*np.cos(2.0*z.imag)
from numpy.polynomial.legendre import leggauss
xr, wr = leggauss(400)
knots = np.unique(np.concatenate([[0.0], seg.break_taus, [1.0]]))
ref = 0.0
for a, b in zip(knots[:-1], knots[1:]):
    tt = a + (b - a)*(xr + 1)/2
    ref += float(np.sum((b - a)*wr/2*seg.len*f_np(seg.p(tt))))
print(f"reference (knot-panelled, order 400/panel): {ref:.15f}")

print(f"\n{'precision':>10} {'nodes':>6} {'panels':>7} {'sum(w)-perim':>14} {'integral err':>13}")
for prec in (1e-6, 1e-10, 1e-13):
    bq = ei.boundary_quadrature(dom, 25.0, precision=prec, warn=False)
    got = float(np.sum(bq.wts*f_np(bq.pts)))
    print(f"{prec:10.0e} {len(bq.pts):6d} {len(bq.panels):7d} "
          f"{bq.wts.sum()-dom.perimeter:14.2e} {abs(got/ref-1):13.2e}")

print("\nknot-aligned panels vs one global panel, same node budget (manual):")
for order in (32, 64, 128):
    x, w = leggauss(order)
    tau = (x + 1)/2
    glob = float(np.sum(w/2*seg.len*f_np(seg.p(tau))))
    per = max(2, order//(len(knots) - 1))
    xk, wk = leggauss(per)
    tot, n = 0.0, 0
    for a, b in zip(knots[:-1], knots[1:]):
        tt = a + (b - a)*(xk + 1)/2
        tot += float(np.sum((b - a)*wk/2*seg.len*f_np(seg.p(tt))))
        n += per
    print(f"  global order {order:3d} ({order:3d} nodes): {abs(glob/ref-1):9.2e}   "
          f"knot-panelled ({n:3d} nodes): {abs(tot/ref-1):9.2e}")
