"""The corner-moving shape derivative, and the exactness class it falls outside.

    python -m benchmarks.hadamard.sector_angle

A circular sector gives closed-form truth for a perturbation that MOVES A CORNER: with
`nu = m pi / alpha` and `lam = (j_{nu,n}/R)^2`,

    dlam/dalpha = 2 j (dj/dnu) (dnu/dalpha) / R^2,      dnu/dalpha = -nu/alpha

and the matching boundary velocity is `V.n = r` on the rotating radius, `0` elsewhere. Both
sides are computed here in 40-digit precision and agreed to 50 digits against an independent
analytic evaluation of the Hadamard integral itself, so the reference is far more accurate than
what it measures -- the discipline `reference._bessel_zero` insists on.

WHAT THIS FOUND. With the EXACT eigenfunction, so that any error is the quadrature's alone:

    alpha      mode    rel err
    0.50 pi   (1,1)    8.6e-16      nu = 2, integer
    0.75 pi   (1,1)    3.2e-08      nu = 4/3
    1.50 pi   (1,1)    6.2e-06      nu = 2/3, reentrant
    1.50 pi   (2,1)    1.3e-09      milder mode at the same corner

Six orders worse than the same node set delivers for the L2 norm (8.6e-15), and it does NOT
improve with refinement: 6.2e-06 at 90 nodes, 3.9e-06 at 166, still ~1e-06 at precision 1e-14.

WHY. The corner rule for a straight edge with `2/nu` integral uses the substitution `t = r^nu`,
which rationalizes the Rellich exponent family `{j nu + 2q}` into polynomials in `t`. A weight
`r^p` contributes `t^(p/nu)`, an integer power only when `p/nu` is. At `nu = 2/3` that means
`t^(3p/2)`: polynomial for EVEN `p`, not for odd. Measured on the same node set:

    weight r^p     p=0      p=1      p=2      p=3      p=4
    rel err      3.4e-14  6.2e-06  3.5e-14  3.2e-11  3.5e-14

Even exact, odd degraded, exactly as predicted.

WHY IT WAS INVISIBLE UNTIL NOW. Every weight lappy has ever integrated is in the exact class.
`gram` uses `r.N`, which with `x0` placed at the singular corner is identically zero on both
straight edges -- the corner contributes nothing at all -- and with a generic `x0` is `O(1)`,
i.e. `p = 0`. The first genuinely new weight is a corner-moving shape velocity, which is
`O(r^1)`, and it is the first one outside the class.

CONSEQUENCE. A downstream shape-optimization package cannot assume the node set built for the
norm serves an arbitrary Hadamard weight. Either the rule is constructed for the product family
(`cached_cornerjacgauss` already takes `gamma` and `sub`, so a caller who declares the weight's
corner exponent can get an exact rule), or the perturbation is restricted to fields vanishing to
even order at singular corners. Perturbations that move a corner are exactly the ones that do
not.
"""
import warnings

import numpy as np
import mpmath as mp

from lappy import reference as ref
from lappy.geometry import disk_sector
from lappy.eigfun_integrals import (boundary_quadrature, corner_specs, EigfunData,
                                    weighted_integral)
from lappy.utils import complex_dot

mp.mp.dps = 40


def dlam_dalpha(m, n, R, alpha):
    """Truth by differentiating `lam = (j_nu/R)^2` through `nu = m pi / alpha`."""
    nu = mp.mpf(m)*mp.pi/mp.mpf(alpha)
    j = mp.besseljzero(nu, n)
    dj = mp.diff(lambda v: mp.besseljzero(v, n), nu)
    return float(2*j*dj*(-nu/mp.mpf(alpha))/mp.mpf(R)**2)


def dlam_dalpha_direct(m, n, R, alpha):
    """The same number, from the Hadamard integral evaluated analytically in high precision.

    An independent route to the reference: `du/dn = c nu cos(nu alpha) J_nu(kr)/r` on the moving
    radius, so `dlam = -c^2 nu^2 cos^2(nu alpha) int_0^R J_nu(kr)^2 dr/r`. Agrees with
    `dlam_dalpha` to 50 digits, which is what licenses either as truth.
    """
    nu = mp.mpf(m)*mp.pi/mp.mpf(alpha)
    j = mp.besseljzero(nu, n)
    k = j/mp.mpf(R)
    norm2 = (mp.mpf(R)**2/2)*mp.besselj(nu+1, j)**2*(mp.mpf(alpha)/2)
    I = mp.quad(lambda r: mp.besselj(nu, k*r)**2/r, [0, R])
    return float(-nu**2*mp.cos(nu*mp.mpf(alpha))**2*I/norm2)


def exact_data(m, n, R, alpha, bq):
    """Cauchy data of the exact, L2-normalized sector eigenfunction at `bq`'s nodes."""
    u, norm2 = ref.sector_eigfun(m, n, R, alpha)
    s = 1.0/np.sqrt(norm2)
    G = s*ref.sector_eigfun_grad(m, n, R, alpha)(bq.pts)
    return EigfunData(bq.pts, bq.normals, bq.tangents, bq.wts, (s*u(bq.pts))[:, None],
                      complex_dot(G, bq.normals)[:, None], complex_dot(G, bq.tangents)[:, None])


def moving_radius(dom, bq, alpha):
    """Mask for the radius at `theta = alpha`, by segment rather than by angle."""
    seg = np.array([bq.panels[p].seg_idx for p in bq.panel_id])
    for i in np.unique(seg):
        pts = bq.pts[seg == i]
        th = np.angle(pts) % (2*np.pi)
        if np.allclose(th, alpha, atol=1e-6) and np.ptp(np.abs(pts)) > 0.5:
            return seg == i
    raise RuntimeError('could not identify the moving radius')


def main():
    R = 1.0
    print('CORNER-MOVING SHAPE DERIVATIVE, exact eigenfunction (error is the quadrature\'s)')
    print(f"{'alpha/pi':>9} {'mode':>7} {'nu':>7} {'dlam/dalpha':>16} {'quadrature':>16} {'rel err':>9}")
    for ao in (0.5, 0.75, 1.0, 1.5, 1.75):
        alpha = ao*np.pi
        for (m, n) in ((1, 1), (2, 1)):
            lam = ref.sector_eig(m, n, R, alpha)
            dom = disk_sector(R, alpha)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bq = boundary_quadrature(dom, 3*lam, precision=1e-13)
            ed = exact_data(m, n, R, alpha, bq)
            mask = moving_radius(dom, bq, alpha)
            Vn = np.where(mask, np.abs(bq.pts), 0.0)
            got = -weighted_integral(ed, 'NN', Vn)[0, 0]
            ex = dlam_dalpha(m, n, R, alpha)
            print(f'{ao:9.2f} {str((m, n)):>7} {m*np.pi/alpha:7.4f} {ex:16.10f} {got:16.10f} '
                  f'{abs(got - ex)/abs(ex):9.1e}', flush=True)

    print('\nREFERENCE CROSS-CHECK (two independent routes, 40 dps)')
    for ao in (0.75, 1.5):
        alpha = ao*np.pi
        a, b = dlam_dalpha(1, 1, R, alpha), dlam_dalpha_direct(1, 1, R, alpha)
        print(f'   alpha={ao}pi:  {a:.12f}  vs  {b:.12f}   rel diff {abs(a-b)/abs(a):.1e}')

    print('\nEXPONENT PARITY: weight r^p on the moving radius of the 1.5pi sector, mode (1,1)')
    print('  the corner rule substitutes t = r^nu, so r^p -> t^(p/nu); at nu=2/3 that is')
    print('  t^(3p/2), a polynomial only for EVEN p.')
    alpha = 1.5*np.pi
    dom = disk_sector(R, alpha)
    lam = ref.sector_eig(1, 1, R, alpha)
    spec = [s for s in corner_specs(dom) if s.singular]
    print(f'  corner rule in use: {[(s.kind, round(s.nu, 4)) for s in spec]}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bq = boundary_quadrature(dom, 3*lam, precision=1e-13)
    ed = exact_data(1, 1, R, alpha, bq)
    mask = moving_radius(dom, bq, alpha)
    nu = mp.mpf(1)*mp.pi/mp.mpf(alpha)
    j = mp.besseljzero(nu, 1)
    k = j/mp.mpf(R)
    norm2 = (mp.mpf(R)**2/2)*mp.besselj(nu+1, j)**2*(mp.mpf(alpha)/2)
    print(f"  {'p':>3} {'truth':>18} {'quadrature':>18} {'rel err':>10}")
    for p in (0, 1, 2, 3, 4):
        ex = float(nu**2*mp.cos(nu*mp.mpf(alpha))**2
                   * mp.quad(lambda r: mp.besselj(nu, k*r)**2*r**(p-2), [0, R])/norm2)
        got = weighted_integral(ed, 'NN', np.where(mask, np.abs(bq.pts)**p, 0.0))[0, 0]
        print(f'  {p:3d} {ex:18.12f} {got:18.12f} {abs(got-ex)/abs(ex):10.1e}')


if __name__ == '__main__':
    main()
