"""Tests for lappy.quad's Kress-style graded-mesh quadrature primitives
(kress_w/kress_dw/cached_kressgauss), which replace the SS/SR/RS/RR
singularity-subtraction machinery previously in lappy.cauchy (see
docs/rellich_hadamard_mps.pdf Sec. 6.1). Per that document's own "Confidence
note," the sigmoid formula is checked numerically here rather than trusted
on sight."""

import numpy as np
import pytest

from lappy.quad import (
    kress_v, kress_w, kress_dw, cached_kressgauss, cached_leggauss, _KRESS_TAU_FLOOR,
    cached_cornerjacgauss, cornerjac_order_cap, _CORNER_NU_MIN,
    cached_cornerinterpgauss, corner_exponents, corner_substitution,
    corner_rule_spec, corner_rule_residual, corner_order_for_precision,
    smooth_order_for_precision,
)

QS = [2, 3, 4, 6, 8, 10]

# Reentrant interior angles as multiples of pi, with the corner exponent nu = 1/(alpha/pi).
# 1.5 (the 270-degree corner) and 2/3-style angles with 2/nu integral are the spectral cases;
# the others are the high-order algebraic ones.
CORNER_ALPHAS = [1.1, 1.25, 4/3, 1.5, 1.6, 1.75, 1.9]


@pytest.mark.parametrize("q", QS)
def test_kress_w_endpoints_and_midpoint(q):
    eps = 1e-9
    assert kress_w(eps, q) == pytest.approx(0.0, abs=1e-6)
    assert kress_w(1 - eps, q) == pytest.approx(1.0, abs=1e-6)
    assert kress_w(0.5, q) == pytest.approx(0.5)


@pytest.mark.parametrize("q", QS)
def test_kress_w_monotonic(q):
    # non-decreasing rather than strictly increasing: at high q, nodes extremely close to
    # 0/1 legitimately collapse to the same float64 value (harmless -- those points carry
    # ~zero weight anyway), so strict monotonicity isn't the right invariant to check.
    t = np.linspace(0.001, 0.999, 500)
    w = kress_w(t, q)
    assert np.all(np.diff(w) >= 0)


@pytest.mark.parametrize("q", QS)
def test_kress_w_symmetry(q):
    """w(t) = 1 - w(1-t), the symmetric construction the doc relies on to
    flatten derivatives at both endpoints with a single grading order."""
    t = np.linspace(0.01, 0.99, 50)
    assert np.allclose(kress_w(t, q), 1 - kress_w(1 - t, q))


@pytest.mark.parametrize("q", QS)
def test_kress_w_vanishing_derivatives_at_origin(q):
    """w(t) ~ C*t^q near t=0 (derivatives of order < q vanish there): confirmed
    by checking w(eps)/eps**q converges to a nonzero constant, while
    w(eps)/eps**(q-1) -> 0."""
    eps = np.array([1e-2, 1e-3, 1e-4])
    w = kress_w(eps, q)
    ratio_q = w/eps**q
    ratio_low = w/eps**(q - 1)
    # ratio_q should stabilize (not blow up or vanish) as eps shrinks
    assert ratio_q[-1] == pytest.approx(ratio_q[-2], rel=0.1)
    assert ratio_low[-1] < ratio_low[0]/5


@pytest.mark.parametrize("q", QS)
def test_kress_dw_matches_finite_difference(q):
    t = np.linspace(0.05, 0.95, 20)
    h = 1e-6
    fd = (kress_w(t + h, q) - kress_w(t - h, q))/(2*h)
    assert np.allclose(kress_dw(t, q), fd, atol=1e-6)


@pytest.mark.parametrize("q", QS)
def test_cached_kressgauss_weights_sum_to_one(q):
    """Total measure of [0,1] must be preserved by the reparametrization,
    regardless of grading order, once the base rule is fine enough."""
    tau, wts = cached_kressgauss(60, q)
    assert wts.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(np.diff(tau) >= 0)
    assert tau.min() >= 0 and tau.max() <= 1


@pytest.mark.parametrize("order,q", [(20, 4), (48, 9), (96, 12), (200, 12), (400, 12)])
def test_cached_kressgauss_never_collapses_onto_endpoint(order, q):
    """Regression test: a segment's linear parametrization p(tau) = (1-tau)*p0 + tau*pf
    collapses to exactly p0 once tau underflows below float64 epsilon relative to p0 --
    silently placing a quadrature node exactly on the corner (r=0), which produces NaN in
    any basis whose corner-relative derivatives involve 1/r (see cached_kressgauss's
    docstring). At large order/q, kress_w alone can map nodes many orders of magnitude
    below that threshold (e.g. 1e-23), so cached_kressgauss must clamp -- confirmed here
    for combinations well beyond anything corner_grading_orders' q_min/q_max would produce
    in practice, plus one (400, 12) stress case."""
    tau, wts = cached_kressgauss(order, q)
    assert tau.min() >= _KRESS_TAU_FLOOR
    assert tau.max() <= 1 - _KRESS_TAU_FLOOR
    assert np.all(np.isfinite(wts))


def test_cached_kressgauss_clamp_does_not_affect_weight_normalization():
    """The clamp only nudges already-negligible-weight endpoint nodes; confirm it doesn't
    perceptibly change the total measure even at the stress case above."""
    tau, wts = cached_kressgauss(400, 12)
    assert wts.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("q", QS)
def test_cached_kressgauss_integrates_smooth_function(q):
    tau, wts = cached_kressgauss(60, q)
    f = lambda x: np.exp(x)
    assert np.sum(wts*f(tau)) == pytest.approx(np.e - 1, abs=1e-6)


def test_cached_kressgauss_is_cached():
    assert cached_kressgauss(20, 4) is cached_kressgauss(20, 4)


@pytest.mark.parametrize("q", QS)
def test_cached_kressgauss_resolves_algebraic_singularity(q):
    """Grading order q should let the rule integrate rho**p for p just above
    -1 (a genuine algebraic singularity at t=0) far better than a plain
    Gauss-Legendre rule of the same size, as long as q comfortably exceeds
    the singularity's order."""
    p = -1 + 1.0/q  # mild singularity, within this q's grading power
    tau, wts = cached_kressgauss(40, q)
    u, w = cached_leggauss(40)
    exact = 1.0/(p + 1)
    kress_err = abs(np.sum(wts*tau**p) - exact)
    plain_err = abs(np.sum(w*u**p) - exact)
    assert kress_err < plain_err


### Corner-adapted Gauss-Jacobi rule (docs/corner_quadrature.tex)

@pytest.mark.parametrize("alpha_over_pi", CORNER_ALPHAS)
@pytest.mark.parametrize("order", [8, 16])
def test_cornerjac_exact_on_the_corner_family(alpha_over_pi, order):
    """The rule's defining property: exact for tau**gamma * (polynomial in tau**nu).
    Each monomial tau**(gamma + p*nu) integrates to 1/(gamma + p*nu + 1)."""
    nu = 1.0/alpha_over_pi
    gamma = 2*nu - 2
    tau, w = cached_cornerjacgauss(order, nu)
    for p in range(0, 2*order):
        exact = 1.0/(gamma + p*nu + 1.0)
        got = np.sum(w*tau**(gamma + p*nu))
        assert got == pytest.approx(exact, rel=1e-13), f"p={p}"


@pytest.mark.parametrize("order", [4, 12, 40])
def test_cornerjac_reduces_to_legendre_at_nu_one(order):
    """nu=1 (a straight edge, alpha=pi) means gamma=0 and no substitution, so the rule must
    collapse onto plain Gauss-Legendre. Not bit-for-bit: jacgauss routes through
    scipy.roots_jacobi and cached_leggauss through numpy.leggauss, which agree to ~1e-16."""
    tau, w = cached_cornerjacgauss(order, 1.0)
    u, wl = cached_leggauss(order)
    assert tau == pytest.approx(u, abs=1e-14)
    assert w == pytest.approx(wl, abs=1e-14)


@pytest.mark.parametrize("alpha_over_pi", CORNER_ALPHAS)
def test_cornerjac_nodes_ascending_and_interior(alpha_over_pi):
    tau, w = cached_cornerjacgauss(16, 1.0/alpha_over_pi)
    assert np.all(np.diff(tau) > 0)
    assert tau[0] > 0.0 and tau[-1] < 1.0
    assert np.all(w > 0.0)


@pytest.mark.parametrize("alpha_over_pi,integral_2_over_nu",
                         [(1.0, True), (1.5, True), (1.25, False),
                          (4/3, False), (1.6, False), (1.9, False)])
def test_cornerjac_weight_sum_is_exact_iff_two_over_nu_is_integral(alpha_over_pi,
                                                                  integral_2_over_nu):
    """sum(w)==1 says the rule integrates f==1 exactly, which needs the residual
    t**(2/nu - 2) to be polynomial -- true iff 2/nu is an integer (alpha = m*pi/2).
    This is a property to respect, not a bug: renormalizing w would destroy exactness on
    the singular class the rule is built for. See the docstring.

    Note how few spectral angles there are: 2/nu = 2*alpha/pi must be an integer AND
    nu > 1/2, so alpha is in (pi, 2pi) only for 2/nu = 3. **Within the admissible reentrant
    range the 270-degree corner is the ONLY fully spectral angle** -- 2/nu = 4 would need
    nu = 1/2 exactly, which is the divergent slit. Every other reentrant corner is
    high-order algebraic, so Stage 3's sizing cannot assume the spectral rate."""
    nu = 1.0/alpha_over_pi
    _, w = cached_cornerjacgauss(16, nu)
    dev = abs(np.sum(w) - 1.0)
    if integral_2_over_nu:
        assert dev < 1e-13
    else:
        assert dev > 1e-9, "expected a measurable residual, so this is a real signal"
        assert dev < 1e-2


@pytest.mark.parametrize("alpha_over_pi", CORNER_ALPHAS)
def test_cornerjac_weight_sum_residual_decreases_with_order(alpha_over_pi):
    """The sum(w) residual is the rule's own diagnostic of the non-polynomial remainder,
    so it must shrink as the order grows wherever there is a remainder to shrink. At
    alpha=1.5pi (the one spectral reentrant angle, 2/nu=3) it is already machine-zero at
    order 8 and the comparison is pure roundoff noise."""
    nu = 1.0/alpha_over_pi
    devs = [abs(np.sum(cached_cornerjacgauss(n, nu)[1]) - 1.0) for n in (8, 16, 32)]
    if devs[0] < 1e-13:
        assert max(devs) < 1e-13
    else:
        assert devs[1] < devs[0] and devs[2] < devs[1]


def test_cornerjac_rejects_slit_and_beyond():
    """nu <= 1/2 (alpha >= 2pi) puts the Jacobi exponent at or below -1; the integral itself
    diverges, so no rule can help and the failure must be loud."""
    for nu in (0.5, 0.4, 0.0, -1.0):
        with pytest.raises(ValueError, match="nu must exceed"):
            cached_cornerjacgauss(16, nu)
    # just above the threshold is admissible, if delicate
    cached_cornerjacgauss(8, _CORNER_NU_MIN + 0.01)


@pytest.mark.parametrize("alpha_over_pi", CORNER_ALPHAS + [1.99])
def test_cornerjac_order_cap_keeps_nodes_off_the_corner(alpha_over_pi):
    """Past the cap the innermost node falls below the float64 coordinate-collapse floor and
    (1-tau)*p0 + tau*pf rounds onto the corner, where a basis's 1/r terms are fatal."""
    nu = 1.0/alpha_over_pi
    cap = cornerjac_order_cap(nu)
    assert cap >= 8, f"nu={nu} should admit a usable order, got cap={cap}"
    assert cached_cornerjacgauss(cap, nu)[0].min() > _KRESS_TAU_FLOOR
    if cap < 128:
        assert cached_cornerjacgauss(cap + 1, nu)[0].min() <= _KRESS_TAU_FLOOR


def test_cornerjac_order_cap_tightens_as_nu_approaches_one_half():
    """The crowding tau_min ~ (c/n^2)**(1/nu) worsens as nu -> 1/2, so the cap must fall."""
    caps = [cornerjac_order_cap(1.0/a) for a in (1.5, 1.75, 1.9, 1.99)]
    assert caps == sorted(caps, reverse=True), caps
    assert caps[-1] < caps[0]


def test_cornerjac_order_cap_shrinks_on_a_shorter_panel():
    """`scale` is the panel's tau-length: a shorter panel puts its nodes physically closer to
    the corner, so it reaches the floor at a lower order."""
    nu = 1.0/1.9
    assert cornerjac_order_cap(nu, scale=0.1) < cornerjac_order_cap(nu, scale=1.0)


def test_cornerjac_beats_kress_on_the_corner_family():
    """The headline comparison, at the 270-degree corner: the same integrand both rules face
    in production, with r.N constant along the edge (x0 off the corner)."""
    nu = 1.0/1.5
    gamma = 2*nu - 2
    # tau**gamma * (series in tau**nu, tau**2), i.e. the real local structure
    def f(t):
        return t**gamma * (1.0 - 0.7*t**nu + 0.35*t**(2*nu)) * (1.0 + t**2/8)
    exact = sum(c/(gamma + e + 1.0)
                for c, e in [(1.0, 0.0), (-0.7, nu), (0.35, 2*nu),
                             (0.125, 2.0), (-0.0875, nu + 2), (0.04375, 2*nu + 2)])
    tau, w = cached_cornerjacgauss(16, nu)
    kt, kw = cached_kressgauss(64, 8)
    corner_err = abs(np.sum(w*f(tau))/exact - 1)
    kress_err = abs(np.sum(kw*f(kt))/exact - 1)
    assert corner_err < 1e-13, corner_err
    assert corner_err < kress_err/1e3, (corner_err, kress_err)


def test_cornerjac_is_cached():
    assert cached_cornerjacgauss(16, 2/3) is cached_cornerjacgauss(16, 2/3)


### Interpolatory corner rule for irrational nu

# Angles produced by an arc-arc corner (benchmarks/corner_quad/curved_domains.peanut): the
# angle is fixed by the circle geometry, so nu is irrational and no substitution rationalizes
# the exponent family.
IRRATIONAL_NUS = [0.657360, 0.730455, 0.587300]


def _curved_corner_integrand(nu):
    """r^gamma times a series in the curved family {j*nu + m}, including ODD integer powers
    of arclength -- what a curved edge actually produces, per the section comment in quad.py."""
    gamma = 2*nu - 2
    terms = [(1.0, 0, 0), (-0.7, 1, 0), (0.35, 2, 0), (0.5, 0, 1),
             (-0.2, 1, 1), (0.1, 0, 2), (-0.05, 1, 2)]
    def f(t):
        return t**gamma*sum(c*t**(k*nu + m) for c, k, m in terms)
    exact = sum(c/(gamma + k*nu + m + 1.0) for c, k, m in terms)
    return f, exact


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
def test_corner_substitution_reports_irrational_nu_as_inexact(nu):
    """These nu have no small-denominator rational form, so no substitution is exact and the
    caller must be told -- that flag is what selects the interpolatory rule."""
    sub, exact = corner_substitution(nu)
    assert exact is False
    assert sub == pytest.approx(nu)


@pytest.mark.parametrize("nu", [2/3, 0.8, 0.75, 5/8])
def test_corner_substitution_reports_rational_nu_as_exact(nu):
    sub, exact = corner_substitution(nu)
    assert exact is True
    assert (1.0/sub) == pytest.approx(round(1.0/sub), abs=1e-9)


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
def test_corner_exponents_are_sorted_and_start_at_gamma(nu):
    gamma = 2*nu - 2
    E = corner_exponents(nu, gamma, 12)
    assert len(E) == 12
    assert np.all(np.diff(E) > 0)
    assert E[0] == pytest.approx(gamma)
    assert np.all(E > -1.0), "every exponent must be integrable"


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
@pytest.mark.parametrize("order", [12, 16, 24])
def test_cornerinterp_is_exact_on_its_exponent_set(nu, order):
    """Defining property: it integrates every exponent it was built for, to 1/(e+1).

    Relative rather than absolute: the smallest exponent is gamma = 2nu-2, close to -1, so
    its moment 1/(e+1) is O(3) and an absolute 1e-13 would be a ~1e-14 relative demand on a
    least-squares-determined rule. The residuals sit near 1e-12 relative, which is what the
    conditioning of the (deliberately rectangular) Vandermonde allows."""
    gamma = 2*nu - 2
    tau, w = cached_cornerinterpgauss(order, nu)
    for e in corner_exponents(nu, gamma, max(2, order//2)):
        assert np.sum(w*tau**e) == pytest.approx(1.0/(e + 1.0), rel=1e-11), f"e={e}"


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
@pytest.mark.parametrize("order", [8, 12, 16, 24, 32])
def test_cornerinterp_weights_stay_well_conditioned(nu, order):
    """sum|w| is the factor by which the rule amplifies roundoff in the integrand. The
    square solve (n_exp == order) is exact but drives it to 1e3-1e4; taking n_exp < order
    keeps it at 1.0, which is why that is the default and why n_exp >= order is rejected."""
    _, w = cached_cornerinterpgauss(order, nu)
    assert np.abs(w).sum() < 1.5


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
def test_cornerinterp_beats_the_substitution_at_irrational_nu(nu):
    """The point of the rule. At order 24 the substitution stalls near 1e-8 (it leaves a
    t^(1/nu) residual, only C^1) while the interpolatory rule reaches ~1e-12."""
    f, exact = _curved_corner_integrand(nu)
    ti, wi = cached_cornerinterpgauss(24, nu)
    tj, wj = cached_cornerjacgauss(24, nu)
    err_i = abs(np.sum(wi*f(ti))/exact - 1)
    err_j = abs(np.sum(wj*f(tj))/exact - 1)
    assert err_i < 1e-11, err_i
    assert err_i < err_j/1e3, (err_i, err_j)


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
def test_cornerinterp_converges_with_order(nu):
    f, exact = _curved_corner_integrand(nu)
    errs = []
    for order in (8, 16, 24):
        tau, w = cached_cornerinterpgauss(order, nu)
        errs.append(abs(np.sum(w*f(tau))/exact - 1))
    assert errs[1] < errs[0]/1e3 and errs[2] < errs[1]/1e2, errs


def test_cornerinterp_rejects_square_solve():
    with pytest.raises(ValueError, match="must be <"):
        cached_cornerinterpgauss(16, 0.65736, None, 16)


def test_cornerinterp_is_cached():
    assert cached_cornerinterpgauss(16, 0.65736) is cached_cornerinterpgauss(16, 0.65736)


### Precision-driven sizing (self-certifying, no offline calibration table)

def test_corner_exponents_respect_edge_type():
    """A straight edge admits only EVEN integer powers (r.N constant, r == arclength); a
    curved edge admits odd ones too. Scoring a straight-edge rule against the curved set
    makes it look ~7 orders worse than it is."""
    nu, gamma = 2/3, 2*(2/3) - 2
    Es = corner_exponents(nu, gamma, 10, curved=False)
    Ec = corner_exponents(nu, gamma, 10, curved=True)
    allowed_straight = np.array([gamma + j*nu + 2*q
                                 for j in range(14) for q in range(14)])
    for e in Es:
        assert np.any(np.isclose(allowed_straight, e, atol=1e-12)), e
    # the curved set is a superset, so element-wise it is never larger, and strictly
    # smaller somewhere (the first odd integer power has no straight-edge counterpart)
    assert np.all(Ec <= Es + 1e-12)
    assert np.any(Ec < Es - 1e-12)
    # gamma + 1 is an ODD integer power: curved-only
    assert np.any(np.isclose(Ec, gamma + 1.0, atol=1e-12))
    assert not np.any(np.isclose(Es, gamma + 1.0, atol=1e-12))


def test_cornerjac_meets_machine_precision_on_a_straight_270_corner():
    """Regression on the edge-type fix: alpha=3pi/2 straight is the one angle where the plain
    substitution is exact, and it must be recognized as such at low order. Scored against the
    curved set it stalled at 1e-8 and the sizing rejected it, contradicting the measured
    2.4e-15 on a real sector."""
    nu = 2/3
    kind, sub = corner_rule_spec(nu, curved=False)
    assert kind == 'cornerjac'
    order, achieved = corner_order_for_precision(kind, nu, None, sub, False, 1e-14)
    assert order <= 12, order
    assert achieved < 1e-14, achieved


# Measured achievable residual per angle -- the capability degrades monotonically as
# nu -> 1/2 (the slit limit, where the Rellich integrand stops being integrable at all), so a
# flat target across all angles would be a wish rather than a test. alpha=1.5212pi is the
# arc-arc angle from benchmarks/corner_quad/curved_domains.peanut, i.e. irrational nu.
ACHIEVABLE = [(1.25, 1e-14), (4/3, 1e-14), (1.5, 1e-14), (1.521236, 1e-14),
              (1.6, 1e-13), (1.75, 2e-12), (1.9, 1e-9)]


@pytest.mark.parametrize("alpha_over_pi,bar", ACHIEVABLE)
@pytest.mark.parametrize("curved", [False, True])
def test_corner_order_for_precision_reaches_its_measured_capability(alpha_over_pi, bar,
                                                                   curved):
    nu = 1.0/alpha_over_pi
    kind, sub = corner_rule_spec(nu, curved=curved)
    order, achieved = corner_order_for_precision(kind, nu, None, sub, curved, bar)
    assert achieved <= bar, (kind, order, achieved)
    assert order <= 64


def test_corner_order_for_precision_reports_shortfall_near_the_slit():
    """nu -> 1/2 cannot reach 1e-14, and the shortfall must be RETURNED rather than hidden --
    boundary_quadrature turns it into a warning naming the corner."""
    nu = 1.0/1.9
    kind, sub = corner_rule_spec(nu, curved=True)
    order, achieved = corner_order_for_precision(kind, nu, None, sub, True, 1e-14)
    assert achieved > 1e-14
    assert order >= 4


@pytest.mark.parametrize("alpha_over_pi", [1.25, 1.5, 1.75])
def test_corner_order_grows_as_precision_tightens(alpha_over_pi):
    nu = 1.0/alpha_over_pi
    kind, sub = corner_rule_spec(nu, curved=True)
    orders = [corner_order_for_precision(kind, nu, None, sub, True, p)[0]
              for p in (1e-6, 1e-10, 1e-13)]
    assert orders[0] <= orders[1] <= orders[2], orders


def test_corner_rule_spec_never_returns_a_crushing_substitution():
    """sub = 1/q is exact but drives tau_min to 1e-29 at q=11, below the coordinate-collapse
    floor. corner_rule_spec must place nodes with sub = nu and get exactness from the
    interpolatory weights instead."""
    for alpha_over_pi in (1.25, 1.5, 1.6, 1.75, 11/6):
        nu = 1.0/alpha_over_pi
        kind, sub = corner_rule_spec(nu, curved=True)
        assert sub == pytest.approx(nu), (alpha_over_pi, kind, sub)
        tau, _ = (cached_cornerjacgauss(24, nu, None, sub) if kind == 'cornerjac'
                  else cached_cornerinterpgauss(24, nu))
        assert tau.min() > _KRESS_TAU_FLOOR, (alpha_over_pi, tau.min())


@pytest.mark.parametrize("k", [1.0, 10.0, 50.0, 100.0])
@pytest.mark.parametrize("precision", [1e-8, 1e-12])
def test_smooth_order_for_precision_meets_its_target(k, precision):
    order, achieved = smooth_order_for_precision(k, precision)
    assert achieved <= precision
    tau, w = cached_leggauss(order)
    exact = (np.exp(1j*k) - 1.0)/(1j*k)
    assert abs(np.sum(w*np.exp(1j*k*tau)) - exact) <= precision


def test_smooth_order_scales_with_wavenumber():
    orders = [smooth_order_for_precision(k, 1e-12)[0] for k in (10, 50, 100, 200)]
    assert orders == sorted(orders)
    assert orders[-1] < 2*200, "should be ~0.4k, not many points per wavelength"
