"""Panel length vs corner clearance -- the last open question from the plan.

A corner-anchored panel is exact for the corner's expansion, which is valid only within the
largest disk about the corner inside Omega (the "clearance"). Every domain used so far has
clearance >= its panel length -- disk_sector and peanut have clearance equal to the edge
length, and H_shape's notch floor has clearance 1 against a 0.5 panel -- so none of them can
measure what happens when a panel reaches past it.

This builds one that can. A 1 x N horizontal strip of cells plus one cell below the left end is
an L with a LONG arm: the reentrant corner's adjacent edge has length N while the clearance
there is the strip width, 1. At N = 6 a full-length panel overshoots the disk sixfold.

Two mechanisms could make a long panel worse, and they need separating:

  1. Class mismatch. Beyond the clearance the integrand is no longer in the corner's exponent
     family, so the rule loses its exactness guarantee.
  2. Resolution. The corner rule clusters nodes AT the corner, so at fixed order the far end of
     a long panel is sparsely sampled -- and it must still resolve the sqrt(lam) oscillation
     over the whole arclength.

Mechanism 2 is already partly handled: corner_order_for_precision sizes the order from a model
integrand at wavenumber sqrt(lam_max)*panel_arclength, so a longer panel asks for a higher
order. Whether that is *sufficient* is the question.

INSTRUMENT. x0-invariance on a real MPS eigenfunction turned out to be useless here: the spread
came back IDENTICAL to three figures across every panel configuration (2.74e-07 at n=6, whether
the panel spanned the whole 5-long edge or a quarter of it) while growing with n. That is the
eigenfunction's own residual, not quadrature error -- the identity
int c.n (du/dn)^2 ds = 0 holds only for an EXACT eigenfunction, and a longer, thinner domain is
simply harder for the basis. Leg 4's documented floor, measured.

So the instrument here is the polyomino's EXACT eigenfunction (zero residual, norm^2 = cells/4
in closed form). It is smooth at the reentrant corner -- Leg 2's limitation -- which makes it
the wrong tool for the singularity but the RIGHT tool for this question: what a long panel
risks is under-resolving the smooth far field, since the corner rule clusters its nodes at the
corner. That is mechanism 2, and it is what this measures.

Mechanism 1 (class mismatch beyond the clearance) is NOT measured here. Any synthetic model of
"the integrand stops being in the corner family past radius R" has to assume what it is trying
to demonstrate, and the exact eigenfunction available on this geometry carries no singular
amplitude to mismatch. Recorded as still open rather than papered over.

Run: python -m benchmarks.corner_quad.panel_length
"""
import warnings

import numpy as np

from lappy import bases, reference as ref
from lappy.geometry import polyomino
from lappy.mps import MPSEigensolver
from lappy.utils import complex_dot
from lappy import eigfun_integrals as ei

warnings.filterwarnings('ignore')

X0S = [0.31 + 0.17j, -0.4 + 0.23j, 2.8 - 0.6j, 1.7 + 1.3j]


def long_arm(n):
    """1 x n strip plus one cell below the left end: clearance 1, long edge n."""
    return polyomino([(i, 1) for i in range(n)] + [(0, 0)])


def report_geometry(n):
    dom = long_arm(n)
    segs = dom.bdry.segments
    print(f"\n=== long_arm(n={n}) ===")
    for s in ei.corner_specs(dom):
        if not (s.singular and s.admissible):
            continue
        clear = ei.corner_clearance(dom, s.point, s.seg_out, s.seg_in)
        for tag, i in (('out', s.seg_out), ('in', s.seg_in)):
            L = segs[i].len
            print(f"  corner {s.idx} at {s.point:+.4g}  {tag}-edge len={L:.3f}  "
                  f"clearance={clear:.3f}  ratio L/clearance={L/clear:.2f}")
    return dom


def x0_spread(solver, eig, coef):
    bq = solver.bdry_quad
    U = solver.basis(eig, bq.pts)@coef
    U_N = solver.basis.ddiff(eig, bq.pts, bq.normals)@coef
    U_T = solver.basis.ddiff(eig, bq.pts, bq.tangents)@coef
    ed = ei.EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, U, U_N, U_T)
    base = ei.gram(ed, eig, bq)[0, 0]
    spread = max(abs(ei.gram(ed, eig, bq, x0)[0, 0] - base) for x0 in X0S)
    return base, spread


def exact_norm_error(dom, cells, m, n, panel_frac, clearance_frac, precision=1e-13):
    """|computed norm^2 - 1| for the EXACT polyomino eigenfunction: zero residual, closed-form
    truth, so every digit of error is quadrature error."""
    lam = ref.polyomino_eig(m, n)
    u, norm2 = ref.polyomino_eigfun(m, n, len(cells))
    g = ref.polyomino_eigfun_grad(m, n)
    sc = 1.0/np.sqrt(norm2)
    bq = ei.boundary_quadrature(dom, lam, precision=precision, panel_frac=panel_frac,
                                clearance_frac=clearance_frac, warn=False)
    grad = sc*g(bq.pts)
    U = (sc*u(bq.pts))[:, None]
    ed = ei.EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, U,
                       complex_dot(grad, bq.normals)[:, None],
                       complex_dot(grad, bq.tangents)[:, None])
    x0 = 1.7 + 1.3j                     # off every corner and every edge line
    return abs(ei.gram(ed, lam, bq, x0)[0, 0] - 1.0), len(bq.pts)


def sweep(n, modes=((1, 1), (2, 3))):
    dom = report_geometry(n)
    cells = [(i, 1) for i in range(n)] + [(0, 0)]
    print(f"\n  {'mode':>6} {'panel_frac':>11} {'clearance':>10} {'nodes':>6} {'norm err':>11}")
    for (m, k) in modes:
        for pf, cf in [(1.0, None), (0.5, None), (0.25, None), (1.0, 0.9), (1.0, 0.4)]:
            err, npts = exact_norm_error(dom, cells, m, k, pf, cf)
            print(f"  {f'({m},{k})':>6} {pf:11.2f} {str(cf):>10} {npts:6d} {err:11.2e}")


if __name__ == '__main__':
    for n in (2, 4, 8, 16):
        sweep(n)
