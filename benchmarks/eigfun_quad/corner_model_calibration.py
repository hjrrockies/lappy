"""Calibrating `quad.corner_model_error` against closed-form truth.

    python -m benchmarks.eigfun_quad.corner_model_calibration

`corner_model_error` is the sizing signal for every singular corner: `corner_order_for_precision`
scans it to choose each corner panel's order, and its value becomes the `precision` a
`BoundaryQuad` advertises. This measures how well it predicts, across the range of `nu` the
suite actually contains.

WHY THE TRUTH IS A SERIES AND NOT `mp.quad`. Building the first version of this table, the
harness's own reference self-check flagged `nu = 0.5714` as unstable at 5.3e-07 -- the entire
"true" column there was the REFERENCE's error, not the rule's. That is the fifth time
`mpmath.quad` on a corner integrand has been the weakest link in this project. So the truth
here is closed form, with no endpoint-singularity quadrature anywhere:

    J_nu(k r) = sum_i c_i r^(nu + 2i),    c_i = (-1)^i (k/2)^(nu+2i) / (i! Gamma(nu+i+1))

    int_0^1 r^(p-2) J_nu(k r)^2 dr = sum_{i,j} c_i c_j / (2 nu + 2i + 2j + p - 1)

Every term is exact and the double sum converges factorially. `exact_corner_integral` verifies
itself by term count before returning, so a silently-wrong reference cannot recur.

The integrand is the real one: on an edge leaving a corner of interior angle `alpha = pi/nu`,
the sector eigenfunction has `du/dn ~ J_nu(k r)/r`, so `(du/dn)^2 r^p` is exactly the above
with that `p`. `p = 0` is the Rellich/normalization weight; `p = 1` is a corner-moving shape
velocity.
"""
import numpy as np
import mpmath as mp

from lappy.quad import cached_cornerjacgauss, cached_cornerinterpgauss, corner_model_error

_DPS = 60


def exact_corner_integral(nu, k, p, n_terms=None, verify=True):
    """`int_0^1 r^(p-2) J_nu(k r)^2 dr` in closed form, as a float.

    Self-verifying: recomputes with half the terms and requires 1e-25 agreement, so a
    truncation that is too short reports itself instead of quietly becoming the error floor.
    """
    with mp.workdps(_DPS):
        nu_, k_, p_ = mp.mpf(nu), mp.mpf(k), mp.mpf(p)
        if n_terms is None:
            n_terms = max(30, int(4*k) + 25)

        def total(n):
            c = [(-1)**i*(k_/2)**(nu_ + 2*i)/(mp.factorial(i)*mp.gamma(nu_ + i + 1))
                 for i in range(n)]
            s = mp.mpf(0)
            for i in range(n):
                for j in range(n):
                    s += c[i]*c[j]/(2*nu_ + 2*i + 2*j + p_ - 1)
            return s

        full = total(n_terms)
        if verify:
            half = total(max(4, n_terms//2))
            if abs(full - half) > mp.mpf('1e-25')*abs(full):
                raise RuntimeError(f'truth not converged at nu={nu}, k={k}, p={p}: '
                                   f'{mp.nstr(abs(full-half)/abs(full), 3)}')
        return float(full)


def true_rule_error(kind, order, nu, sub, curved, k, p):
    """Relative error the corner rule actually makes on that integrand."""
    from scipy.special import jv
    gamma = 2.0*nu - 2.0
    if kind == 'cornerjac':
        tau, w = cached_cornerjacgauss(order, nu, gamma, sub)
    else:
        tau, w = cached_cornerinterpgauss(order, nu, gamma, None, curved)
    exact = exact_corner_integral(nu, k, p)
    approx = float(np.sum(w*jv(nu, k*tau)**2*tau**(p - 2.0)))
    return abs(approx - exact)/abs(exact)


REENTRANT = [1.05, 1.1, 1.25, 4/3, 1.5, 1.521236, 1.6, 1.75, 1.9]
CONVEX = [0.6, 0.75, 0.9]
ORDERS = [8, 12, 16, 24, 32, 48, 64]


def sweep(n_j_candidates=(6, 4, 2, 1), ks=(1.0, 4.0, 16.0), ps=(0, 1),
          subs=('nu', 'half'), curveds=(False, True)):
    """Returns records of (predicted, true) over the grid, one per configuration."""
    recs = []
    for aop in REENTRANT + CONVEX:
        nu = 1.0/aop
        for curved in curveds:
            for sub_name in subs:
                sub = nu if sub_name == 'nu' else 0.5
                for k in ks:
                    for order in ORDERS:
                        try:
                            preds = {nj: corner_model_error('cornerjac', order, nu, None, sub,
                                                            curved, k, nj)
                                     for nj in n_j_candidates}
                        except (ValueError, np.linalg.LinAlgError):
                            continue
                        for p in ps:
                            try:
                                t = true_rule_error('cornerjac', order, nu, sub, curved, k, p)
                            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                                continue
                            recs.append(dict(aop=aop, nu=nu, curved=curved, sub=sub_name,
                                             k=k, order=order, p=p, true=t, pred=preds))
    return recs


def _summarize(recs, n_j_candidates):
    """Worst optimism (pred/true < 1 is BAD) and typical tightness, per n_j."""
    print(f"\n{'n_j':>4} {'worst optimism':>15} {'where':>34} {'median pred/true':>17} "
          f"{'frac optimistic':>16}")
    for nj in n_j_candidates:
        ratios, worst, where = [], np.inf, None
        for r in recs:
            if r['true'] <= 0 or r['pred'][nj] <= 0:
                continue
            rat = r['pred'][nj]/r['true']
            ratios.append(rat)
            if rat < worst:
                worst, where = rat, r
        ratios = np.array(ratios)
        w = (f"a={where['aop']:.3f} o={where['order']} k={where['k']:.0f} "
             f"p={where['p']} {where['sub']}" if where else '')
        print(f'{nj:>4} {worst:15.2e} {w:>34} {np.median(ratios):17.2e} '
              f'{np.mean(ratios < 1.0):16.1%}')


def main():
    n_j = (6, 4, 2, 1)
    print(__doc__.split('WHY')[0].strip())
    print('\nreference self-check: exact_corner_integral verifies its own truncation')
    for nu, k, p in ((0.5714, 3.242, 0), (2/3, 3.376, 1), (1/1.9, 3.2, 0)):
        print(f'   nu={nu:.4f} k={k:.3f} p={p}  ->  {exact_corner_integral(nu, k, p):.16e}')

    recs = sweep(n_j_candidates=n_j)
    print(f'\n{len(recs)} configurations measured '
          f'(nu from {1/max(REENTRANT):.3f} to {1/min(CONVEX):.3f})')

    print('\nPREDICTED / TRUE.  < 1 means the model UNDER-predicts the error it will make,')
    print('i.e. advertises a precision it does not deliver. That is the direction that matters.')
    _summarize(recs, n_j)

    for label, sel in (('reentrant (nu < 1)', lambda r: r['nu'] < 1.0),
                       ('convex (nu > 1)', lambda r: r['nu'] > 1.0),
                       ("sub=nu only", lambda r: r['sub'] == 'nu'),
                       ("sub=1/2 only", lambda r: r['sub'] == 'half'),
                       ('Rellich weight p=0', lambda r: r['p'] == 0),
                       ('shape weight p=1', lambda r: r['p'] == 1)):
        sub = [r for r in recs if sel(r)]
        if sub:
            print(f'\n--- {label}  ({len(sub)} configs)')
            _summarize(sub, n_j)


if __name__ == '__main__':
    main()
