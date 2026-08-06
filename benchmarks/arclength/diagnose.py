"""Where does the arclength reparametrization lose its digits?

Three candidates, separated:
  (a) the arclength VALUES s(t)     -- 5-pt Gauss per quarter-interval, should be good
  (b) the inverse map t(s)          -- PchipInterpolator, piecewise cubic
  (c) the DERIVATIVE t'(s)          -- the differentiated interpolant, one order worse,
                                       and what |dp/dtau| actually depends on
"""
import numpy as np
from lappy.geometry import ellipse

a, b = 2.0, 1.0


def speed(t):
    return np.hypot(a*np.sin(t), b*np.cos(t))


print(f"{'tol':>7} {'nodes':>6} {'(a) s(t) err':>13} {'(b) t(s) roundtrip':>19} "
      f"{'(c) t_s vs 1/speed':>19} {'|dp/dtau|-L':>13}")
for tol in (1e-4, 1e-5, 1e-6, 1e-7):
    d = ellipse(a, b, tol=tol)
    seg = d.bdry.segments[0]
    seg.len  # force reparam
    t_of_s, s_of_t = seg._t_of_s, seg._s_of_t
    L = seg.len
    n_nodes = len(t_of_s.x)

    s = np.linspace(1e-9, L - 1e-9, 2001)

    # (a) arclength values: compare the table's s(t) against high-order Gauss on |p'|
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(200)
    tq = np.linspace(seg.t0, seg.tf, 101)
    s_exact = [0.0]
    for lo, hi in zip(tq[:-1], tq[1:]):
        mid, half = 0.5*(lo + hi), 0.5*(hi - lo)
        s_exact.append(s_exact[-1] + half*np.sum(wg*speed(mid + half*xg)))
    s_exact = np.array(s_exact)
    err_a = np.abs(s_of_t(tq) - s_exact).max()/L

    # (b) inverse round-trip: s_of_t(t_of_s(s)) should return s
    err_b = np.abs(s_of_t(t_of_s(s)) - s).max()/L

    # (c) the interpolant's derivative against the analytic 1/speed
    t = t_of_s(s)
    err_c = np.abs(t_of_s(s, nu=1)*speed(t) - 1.0).max()

    tau = np.linspace(0, 1, 2001)
    err_dp = np.abs(np.abs(seg.dp(tau)) - L).max()/L
    print(f"{tol:7.0e} {n_nodes:6d} {err_a:13.2e} {err_b:19.2e} {err_c:19.2e} {err_dp:13.2e}")

print("\nIf (a) and (b) are small while (c) tracks |dp/dtau|-L, the loss is entirely")
print("in DIFFERENTIATING the interpolant -- which is avoidable: t'(s) = 1/|p'(t)| exactly.")
