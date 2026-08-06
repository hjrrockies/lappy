"""Prototype: an analytic arclength reparametrization, replacing the PCHIP inverse.

The defect is structural. t(s) is a piecewise cubic, so p(tau) = p(t(s)) is only C^1 in tau,
and Gauss-Legendre on a C^1 integrand converges algebraically -- no order fixes it. That caps
every boundary integral on a varying-speed curve at ~1e-6, independent of `tol`.

The fix keeps the adaptive table as a BRACKET and initial guess, then solves exactly:

    s(t) = s_node[i] + int_{t_node[i]}^{t} |p'| dtau     (fixed high-order Gauss, short span)
    t(s): Newton on s(t) - s = 0, with ds/dt = |p'(t)| known analytically

Newton converges quadratically from a cubic initial guess, so 2-3 iterations reach machine
precision. Then p(tau) is as smooth as the underlying curve, and t'(s) = 1/|p'(t)| is analytic
rather than a differentiated interpolant -- both defects go at once.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
import mpmath as mp

from lappy.geometry import ellipse

a, b = 2.0, 1.0
mp.mp.dps = 30
_GX, _GW = leggauss(24)          # per-panel rule for the exact s(t)


class AnalyticArclength:
    def __init__(self, seg, gauss_order=24):
        self.seg = seg
        seg.len                                    # force the existing table
        self.t_nodes = np.asarray(seg._t_of_s.c.shape and seg._s_of_t.x, dtype=float)
        self.speed = seg._speed
        self.p_of_t = seg._p
        self.dp_of_t = seg._dp
        self.guess = seg._t_of_s
        # rebuild the cumulative arclengths at the table's own nodes, to machine precision
        gx, gw = leggauss(gauss_order)
        s = [0.0]
        for lo, hi in zip(self.t_nodes[:-1], self.t_nodes[1:]):
            mid, half = 0.5*(lo + hi), 0.5*(hi - lo)
            s.append(s[-1] + half*np.sum(gw*self.speed(mid + half*gx)))
        self.s_nodes = np.array(s)
        self.L = self.s_nodes[-1]
        self._gx, self._gw = gx, gw

    def s_of_t(self, t):
        """Exact cumulative arclength: anchor at the bracketing node, integrate the remainder."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i = np.clip(np.searchsorted(self.t_nodes, t, side='right') - 1,
                    0, len(self.t_nodes) - 2)
        lo = self.t_nodes[i]
        mid, half = 0.5*(lo + t), 0.5*(t - lo)
        q = np.array([np.sum(self._gw*self.speed(m + h*self._gx))*h
                      for m, h in zip(mid, half)])
        return self.s_nodes[i] + q

    def t_of_s(self, s, iters=4):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        t = np.asarray(self.guess(s), dtype=float)
        for _ in range(iters):
            f = self.s_of_t(t) - s
            d = self.speed(t)
            t = t - f/np.where(np.abs(d) < 1e-300, 1e-300, d)
            t = np.clip(t, self.t_nodes[0], self.t_nodes[-1])
        return t

    def p(self, tau):
        return self.p_of_t(self.t_of_s(self.L*np.atleast_1d(tau)))

    def dp(self, tau):
        """t'(s) = 1/|p'(t)| analytically, so |dp/dtau| == L by construction."""
        t = self.t_of_s(self.L*np.atleast_1d(tau))
        return self.dp_of_t(t)/self.speed(t)*self.L


def reference(f_mp):
    return float(mp.quad(lambda t: f_mp(t)*mp.sqrt((a*mp.sin(t))**2 + (b*mp.cos(t))**2),
                         [0, mp.pi/2, mp.pi, 3*mp.pi/2, 2*mp.pi]))


CASES = {
    'f=1':   (lambda t: mp.mpf(1), lambda z: np.ones_like(z.real)),
    'f=x^2': (lambda t: (a*mp.cos(t))**2, lambda z: z.real**2),
    'f=exp': (lambda t: mp.e**(a*mp.cos(t)/2)*mp.cos(3*b*mp.sin(t)),
              lambda z: np.exp(z.real/2)*np.cos(3*z.imag)),
}

print("Gauss-Legendre of increasing order over the whole ellipse, PCHIP vs Newton\n")
print(f"{'tol':>6} {'order':>6} " + ' '.join(f"{k+' pchip':>14} {k+' newton':>15}"
                                            for k in CASES))
for tol in (1e-4, 1e-6):
    dom = ellipse(a, b, tol=tol)
    seg = dom.bdry.segments[0]
    exact_map = AnalyticArclength(seg)
    for order in (32, 64, 128, 256):
        x, w = leggauss(order)
        tau = (x + 1)/2
        wts = w/2*seg.len
        z_p, z_n = seg.p(tau), exact_map.p(tau)
        cells = []
        for name, (f_mp, f_np) in CASES.items():
            ex = reference(f_mp)
            cells.append(f"{abs(np.sum(wts*f_np(z_p))/ex - 1):14.2e}")
            cells.append(f"{abs(np.sum(wts*f_np(z_n))/ex - 1):15.2e}")
        print(f"{tol:6.0e} {order:6d} " + ' '.join(cells))

print("\nconstant-speed property |dp/dtau|-L:")
for tol in (1e-4, 1e-6):
    seg = ellipse(a, b, tol=tol).bdry.segments[0]
    m = AnalyticArclength(seg)
    tau = np.linspace(0, 1, 501)
    print(f"  tol={tol:.0e}  pchip {np.abs(np.abs(seg.dp(tau))-seg.len).max()/seg.len:.2e}"
          f"   newton {np.abs(np.abs(m.dp(tau))-m.L).max()/m.L:.2e}")
