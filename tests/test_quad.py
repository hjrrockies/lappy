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
)

QS = [2, 3, 4, 6, 8, 10]


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
