"""Vectorized Newton arclength reparametrization, and what it costs.

The prototype in newton_prototype.py looped in Python over points inside `s_of_t`, which made
it 657x slower than the PCHIP inverse and told us nothing useful about the real cost. This
version does one `speed` call per Newton iteration, on an (n_points x gauss_order) block.

Two knobs drive the cost:
  * `gauss_order` -- the rule for the partial panel between the bracketing table node and t.
    That span is at most one adaptive-table panel, so it is short and a low order may do.
  * `iters` -- Newton steps from the PCHIP initial guess. Quadratic convergence, so 2-3
    should reach machine precision; more is waste.

Both are measured here against accuracy, so the cheapest sufficient configuration can be
chosen rather than guessed.

Run: python -m benchmarks.arclength.vectorized
"""
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import make_interp_spline

from lappy.geometry import ellipse, SplineSegment, disk_sector


class NewtonArclength:
    """Analytic arclength map for a ParametricSegment, replacing the PCHIP inverse.

    Uses the segment's existing adaptive table only as a bracket and initial guess; the
    arclengths at its nodes are rebuilt to machine precision, and t(s) is then solved rather
    than interpolated. `t'(s) = 1/|p'(t)|` is analytic, so the constant-speed property holds
    by construction instead of being approximated by a differentiated cubic.
    """

    def __init__(self, seg, gauss_order=6, anchor_order=8, iters=3):
        seg.len                                     # force the lazy reparametrization
        self.seg = seg
        self.speed = seg._speed
        self.p_of_t = seg._p
        self.dp_of_t = seg._dp
        self.guess = seg._t_of_s
        self.iters = iters

        self.t_nodes = np.asarray(seg._s_of_t.x, dtype=float)
        ax, aw = leggauss(anchor_order)
        lo, hi = self.t_nodes[:-1], self.t_nodes[1:]
        mid, half = 0.5*(lo + hi), 0.5*(hi - lo)
        pts = mid[:, None] + half[:, None]*ax[None, :]
        vals = self.speed(pts.ravel()).reshape(pts.shape)
        self.s_nodes = np.concatenate([[0.0], np.cumsum(half*(vals @ aw))])
        self.L = float(self.s_nodes[-1])

        self._gx, self._gw = leggauss(gauss_order)

    def s_of_t(self, t):
        """Cumulative arclength: anchor at the bracketing node, integrate the remainder.
        One vectorized `speed` call over an (n_points x gauss_order) block."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        i = np.clip(np.searchsorted(self.t_nodes, t, side='right') - 1,
                    0, len(self.t_nodes) - 2)
        lo = self.t_nodes[i]
        mid, half = 0.5*(lo + t), 0.5*(t - lo)
        pts = mid[:, None] + half[:, None]*self._gx[None, :]
        vals = self.speed(pts.ravel()).reshape(pts.shape)
        return self.s_nodes[i] + half*(vals @ self._gw)

    def t_of_s(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        t = np.asarray(self.guess(s), dtype=float)
        for _ in range(self.iters):
            d = self.speed(t)
            t = t - (self.s_of_t(t) - s)/np.where(np.abs(d) < 1e-300, 1e-300, d)
            np.clip(t, self.t_nodes[0], self.t_nodes[-1], out=t)
        return t

    def p(self, tau):
        return self.p_of_t(self.t_of_s(self.L*np.atleast_1d(tau)))

    def dp(self, tau):
        t = self.t_of_s(self.L*np.atleast_1d(tau))
        return self.dp_of_t(t)/self.speed(t)*self.L


def wobbly_spline(n_ctrl=9, tol=1e-4):
    th = np.linspace(0, 2*np.pi, n_ctrl)
    pts = (1.0 + 0.3*np.cos(3*th))*np.exp(1j*th)
    return SplineSegment(make_interp_spline(np.linspace(0, 1, n_ctrl), pts, k=3),
                         0, 1, tol=tol)


def accuracy(seg, m):
    """Round-trip error of the map, the quantity the quadrature depends on."""
    s = np.linspace(0.0, m.L, 401)
    return float(np.abs(m.s_of_t(m.t_of_s(s)) - s).max()/m.L)


def timeit(fn, n_rep=20):
    fn()                                            # warm
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    return (time.perf_counter() - t0)/n_rep


SEGS = [('ellipse 2x1', lambda: ellipse(2.0, 1.0, tol=1e-4).bdry.segments[0]),
        ('cubic spline', lambda: wobbly_spline()),
        ('circular arc', lambda: disk_sector(1.0, 1.5*np.pi).bdry.segments[1])]

if __name__ == '__main__':
    print("=== knob calibration: accuracy vs gauss_order and Newton iters ===")
    for name, mk in SEGS:
        seg = mk()
        print(f"\n{name}")
        print(f"  {'iters':>5} " + ' '.join(f"{'G=%d' % g:>11}" for g in (6, 8, 12, 24)))
        for iters in (1, 2, 3, 4):
            row = []
            for g in (6, 8, 12, 24):
                row.append(f"{accuracy(seg, NewtonArclength(seg, g, 24, iters)):11.2e}")
            print(f"  {iters:5d} " + ' '.join(row))

    print("\n\n=== speed: t(s) evaluation, vectorized over n points ===")
    print(f"  {'segment':>13} {'n':>6} {'pchip':>10} {'newton':>10} {'ratio':>7} "
          f"{'ns/pt newton':>13}")
    for name, mk in SEGS:
        seg = mk()
        m = NewtonArclength(seg, gauss_order=8, iters=3)
        for n in (100, 1000, 10000):
            s = np.linspace(0.0, m.L, n)
            t_p = timeit(lambda: seg._t_of_s(s))
            t_n = timeit(lambda: m.t_of_s(s))
            print(f"  {name:>13} {n:6d} {t_p*1e3:9.3f}m {t_n*1e3:9.3f}m "
                  f"{t_n/t_p:7.0f} {t_n/n*1e9:13.0f}")

    print("\n\n=== end to end: cost of building one boundary quadrature ===")
    from lappy import eigfun_integrals as ei
    for name, mk in SEGS[:2]:
        seg = mk()
        m = NewtonArclength(seg, gauss_order=8, iters=3)
        n_typical = 250                             # nodes in a real boundary quadrature
        s = np.linspace(0.0, m.L, n_typical)
        t_extra = timeit(lambda: m.t_of_s(s)) - timeit(lambda: seg._t_of_s(s))
        print(f"  {name:>13}: +{t_extra*1e3:.3f} ms for {n_typical} nodes "
              f"(whole quadrature build is currently ~11 ms)")


def confirm_tuned():
    """Does the CHEAP configuration (G=6, iters=3) still restore spectral convergence, and
    what does constructing the map cost?"""
    import mpmath as mp
    mp.mp.dps = 30
    a, b = 2.0, 1.0
    seg = ellipse(a, b, tol=1e-4).bdry.segments[0]

    exact = float(mp.quad(
        lambda t: mp.e**(a*mp.cos(t)/2)*mp.cos(3*b*mp.sin(t))
        * mp.sqrt((a*mp.sin(t))**2 + (b*mp.cos(t))**2),
        [0, mp.pi/2, mp.pi, 3*mp.pi/2, 2*mp.pi]))
    f = lambda z: np.exp(z.real/2)*np.cos(3*z.imag)

    print("\n\n=== tuned config (G=6, iters=3): does it still converge spectrally? ===")
    m = NewtonArclength(seg, gauss_order=6, iters=3)
    print(f"  {'order':>6} {'pchip':>11} {'newton G=6':>12} {'newton G=24':>12}")
    m24 = NewtonArclength(seg, gauss_order=24, iters=3)
    for order in (32, 64, 128, 256):
        x, w = leggauss(order)
        tau, wts = (x + 1)/2, w/2*seg.len
        e_p = abs(np.sum(wts*f(seg.p(tau)))/exact - 1)
        e_6 = abs(np.sum(wts*f(m.p(tau)))/exact - 1)
        e_24 = abs(np.sum(wts*f(m24.p(tau)))/exact - 1)
        print(f"  {order:6d} {e_p:11.2e} {e_6:12.2e} {e_24:12.2e}")

    print("\n=== one-time construction cost (per segment, per solve) ===")
    for name, mk in SEGS:
        sg = mk()
        sg.len
        for anchor in (8, 12, 24):
            t_c = timeit(lambda: NewtonArclength(sg, 6, anchor, 3), n_rep=50)
            acc = accuracy(sg, NewtonArclength(sg, 6, anchor, 3))
            print(f"  {name:>13} anchor_order={anchor:2d}: {t_c*1e3:6.3f} ms  "
                  f"roundtrip {acc:.2e}  ({len(NewtonArclength(sg, 6, anchor, 3).t_nodes)} "
                  f"table nodes)")


if __name__ == '__main__':
    confirm_tuned()
