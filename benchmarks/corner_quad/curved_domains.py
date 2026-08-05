"""Test domains with CURVED sides meeting at a singular corner.

No factory in lappy.geometry produces one: `mushroom`'s 3pi/2 corners are
polyline-only, `cut_square`'s arc corners are pi/2, and `disk_sector`'s apex is
between two straight radii. Since the corner-adapted rule must work for curved
sides at a singular corner, the geometry has to be built here.

An arc leaving the origin with initial tangent direction theta0 and (signed)
curvature kappa, parametrized by arclength t:

    c    = (i/kappa) * exp(i theta0)                       # centre
    z(t) = c + (1/kappa) * exp(i (theta0 - pi/2 + kappa t))

so z(0) = 0 and z'(0) = exp(i theta0) exactly, which is what lets the interior
angle at the corner be prescribed rather than discovered.
"""
import numpy as np

from lappy.geometry import Domain, MultiSegment, LineSegment, ParametricSegment


def _arc(theta0, kappa, L, tol=1e-8, bc='dir', z0=0.0 + 0j):
    """Arc of length L leaving z0 with tangent direction theta0 and curvature kappa."""
    c = z0 + (1j/kappa)*np.exp(1j*theta0)
    def p(t):
        return c + (1.0/kappa)*np.exp(1j*(theta0 - np.pi/2 + kappa*t))
    def dp(t):
        return 1j*np.exp(1j*(theta0 - np.pi/2 + kappa*t))
    return ParametricSegment(p, dp, 0.0, L, bc, tol, val_simple=False), p(L)


def curved_wedge(alpha, kappa=0.6, L=1.0, bc='dir', tol=1e-8):
    """A wedge of interior angle `alpha` at the origin whose two sides are ARCS.

    Both sides leave the origin as arcs of curvature +-kappa and are then joined by a straight
    chord, so the singular corner at the origin is bounded by curved segments on both sides
    while the rest of the boundary stays trivial. CCW: the first side leaves along the
    direction alpha and the second returns along 0.
    """
    # side A: leaves origin at angle alpha, curving away from the interior
    segA, endA = _arc(alpha, kappa, L, tol, bc)
    # side B: leaves origin at angle 0, curving the other way; traversed inbound, so build it
    # outbound then reverse by parametrizing backwards
    cB = (1j/(-kappa))*np.exp(1j*0.0)
    def pB(t):   # t from L down to 0 -> arrives at the origin
        return cB + (1.0/(-kappa))*np.exp(1j*(0.0 - np.pi/2 + (-kappa)*(L - t)))
    def dpB(t):
        return 1j*np.exp(1j*(0.0 - np.pi/2 + (-kappa)*(L - t)))*(-1.0)*(-1.0)
    startB = pB(0.0)
    segB = ParametricSegment(pB, dpB, 0.0, L, bc, tol, val_simple=False)
    chord = LineSegment(endA, startB, bc=bc)
    bdry = MultiSegment([segA, chord, segB], val_simple=False, val_contiguous=False)
    return Domain(bdry, val_simple=False, val_closed=False)


def peanut(R=1.0, rho=0.6, d=1.2, bc='dir', tol=1e-8):
    """The UNION of disk(R) and disk(rho) centred at `d` on the real axis.

    The two circles cross at two points and at each the interior angle of the union is
    REENTRANT and bounded by two arcs -- the most natural curved singular corner there is.
    (The difference of the two disks gives *convex* corners there instead, measured at
    alpha=0.52pi, so it is the union that exercises the singular case.) The angle follows
    from the circles' intersection angle, so it is measured from the built domain
    (`corner_int_angles`) rather than prescribed.
    """
    if not (abs(R - rho) < d < R + rho):
        raise ValueError("circles must properly intersect")
    # intersection points of |z|=R and |z-d|=rho
    x = (d*d + R*R - rho*rho)/(2*d)
    y = np.sqrt(R*R - x*x)
    P, Q = x + 1j*y, x - 1j*y

    # Big circle CCW from P round to Q, i.e. the arc OUTSIDE the small disk (through angle pi).
    aP, aQ = np.angle(P), np.angle(Q)
    outer = ParametricSegment(lambda t: R*np.exp(1j*t), lambda t: 1j*R*np.exp(1j*t),
                              aP, aQ + 2*np.pi, bc, tol, val_simple=False)

    # Small circle CCW from Q round to P, i.e. its arc OUTSIDE the big disk (through angle 0).
    # Getting either arc wrong makes the boundary self-intersect, which sends Domain's
    # CCW/polyline machinery into a spin rather than raising -- hence the assertions.
    bP, bQ = np.angle(P - d), np.angle(Q - d)
    s0 = bQ
    s1 = bP if bP > bQ else bP + 2*np.pi
    inner = ParametricSegment(lambda t: d + rho*np.exp(1j*t), lambda t: 1j*rho*np.exp(1j*t),
                              s0, s1, bc, tol, val_simple=False)

    mid_o = R*np.exp(1j*(aP + (aQ + 2*np.pi - aP)/2))
    mid_i = d + rho*np.exp(1j*(s0 + (s1 - s0)/2))
    assert abs(mid_o - d) > rho, "big-circle arc dips inside the small disk"
    assert abs(mid_i) > R, "small-circle arc dips inside the big disk"
    assert abs(outer.pf - inner.p0) < 1e-12 and abs(inner.pf - outer.p0) < 1e-12, \
        "boundary does not close"

    bdry = MultiSegment([outer, inner], val_simple=False, val_contiguous=False)
    return Domain(bdry, val_simple=False, val_closed=False)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    from lappy import eigfun_integrals as ei

    print("bitten_disk: two arcs meeting at two reentrant corners")
    for rho, d in [(0.6, 1.2), (0.5, 1.25), (0.8, 1.4)]:
        try:
            dom = bitten_disk(rho=rho, d=d)
            cia = np.asarray(dom.corner_int_angles)
            print(f"  rho={rho} d={d}: alphas/pi = {np.round(cia/np.pi, 6)}  "
                  f"nu = {np.round(np.pi/cia, 6)}")
            print(ei.singular_corner_report(dom))
            for s in dom.bdry.segments:
                print(f"    seg {type(s).__name__:18s} len={s.len:.6f} "
                      f"param-quality={ei._parametrization_quality(s):.2e}")
        except Exception as e:
            print(f"  rho={rho} d={d}: FAILED {e!r}")

    print("\ncurved_wedge: prescribed alpha, both sides arcs")
    for a in (1.5, 1.25):
        try:
            dom = curved_wedge(a*np.pi)
            cia = np.asarray(dom.corner_int_angles)
            print(f"  alpha={a}pi: alphas/pi = {np.round(cia/np.pi, 6)}")
            print(ei.singular_corner_report(dom))
        except Exception as e:
            print(f"  alpha={a}pi: FAILED {e!r}")
