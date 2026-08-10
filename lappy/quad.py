import numpy as np
from .utils import complex_form, real_form, polygon_edges, edge_lengths
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi
from scipy.interpolate import BSpline
from functools import cache
from .cubature_registry import get_cubature_rule


### Quadrature rules for domain boundaries
@cache
def cached_leggauss(order):
    x, w = leggauss(order)
    nodes = (x+1)/2 #adjust nodes to interval [0,1]
    weights = w/2 #adjust weights to interval of unit length
    return nodes, weights

@cache
def cached_chebgauss(order):
    x, w = chebgauss(order)
    # adjust the weights to cancel-out the Gauss-Cheb weighting function
    weights = w*np.sqrt(1-x**2)
    nodes = (x+1)/2 # adjust nodes to interval [0,1]
    weights = weights/2 # adjust weights to interval of unit length
    return nodes[::-1], weights[::-1]

@cache
def jacgauss(order, a=0, b=0):
    # reverse order so a is the singular exponent for the left, and b is the singular exponent for the right
    x, w = roots_jacobi(order, b, a)
    # adjust the weights to cancel out the Gauss-Jacobi weighting function
    weights = w/(((1+x)**a)*((1-x)**b))
    nodes = (x+1)/2 # adjust nodes to interval [0,1]
    weights = weights/2 # adjust weights to interval of unit length
    return nodes, weights

### Kress-style graded-mesh quadrature (docs/rellich_hadamard_mps.pdf Sec. 6.1).
#
# w(t) is a fixed, smooth sigmoid reparametrization of [0,1] with w(0)=0, w(1)=1, w(0.5)=0.5,
# and w^(k)(0) = 0 for k = 1, ..., q-1 (by construction w(t) = 1 - w(1-t), so the same holds at
# t=1). Composing a segment's arclength parametrization with w clusters Gauss-Legendre nodes
# geometrically near both of that segment's corner endpoints -- one shared rule per segment, no
# per-basis-function-pair grading.

def kress_v(t, q):
    """Cubic building block of the Kress sigmoid (eq. 11): v(0)=0, v(1)=1, v(0.5)=0.5."""
    return (1.0/q - 0.5)*(1 - 2*t)**3 + (2*t - 1)/q + 0.5

def kress_w(t, q):
    """Kress grading map w: [0,1] -> [0,1], order q >= 2. Flattens q-1 derivatives at each
    endpoint, clustering nodes there without introducing panel joints."""
    a, b = kress_v(t, q), kress_v(1 - t, q)
    aq, bq = a**q, b**q
    return aq/(aq + bq)

def kress_dw(t, q, h=1e-30):
    """dw/dt via complex-step differentiation (exact to machine precision for this smooth,
    polynomial-based map -- avoids a hand-derived closed form and its attendant algebra risk)."""
    return np.imag(kress_w(t + 1j*h, q))/h

_KRESS_TAU_FLOOR = 1e-9  # see cached_kressgauss docstring


@cache
def cached_kressgauss(order, q):
    """Kress-graded Gauss-Legendre rule on [0,1]: `order` base Legendre nodes, remapped through
    the grading map `kress_w` (order q), with weights adjusted by its Jacobian.

    For large `q`/`order`, `kress_w` can map the smallest base nodes to a `tau` many orders of
    magnitude below machine epsilon (e.g. 1e-23). Downstream, a segment's linear/arclength
    parametrization evaluates points as `(1-tau)*p0 + tau*pf`; once `tau` is below float64
    epsilon relative to `p0`, this rounds to *exactly* `p0` -- the quadrature node silently
    collapses onto the corner itself (r=0 in local polar coordinates), producing a division by
    zero (and NaN) in any basis whose corner-relative derivatives involve 1/r. No exact Gauss
    rule ever places a node at an endpoint, so this collapse is a pure floating-point artifact,
    not a real quadrature choice -- clamping `tau` comfortably above the epsilon that causes it
    (`_KRESS_TAU_FLOOR`, ~1e6x above float64 epsilon) removes it. The clamped points still carry
    a weight tiny enough (see module tests) that this has no detectable effect on the assembled
    integral -- the whole point of the grading is that near-endpoint nodes contribute
    vanishingly little individually."""
    u, wt = cached_leggauss(order)
    tau = kress_w(u, q)
    weights = wt*kress_dw(u, q)
    tau = np.clip(tau, _KRESS_TAU_FLOOR, 1 - _KRESS_TAU_FLOOR)
    return tau, weights

### Corner-adapted Gauss-Jacobi quadrature (docs/corner_quadrature.tex).
#
# At a corner of interior angle alpha, with nu = pi/alpha, a Dirichlet eigenfunction has the
# local expansion u = sum_k c_k J_{k nu}(sqrt(lam) r) sin(k nu theta), so on an edge leaving
# the corner du/dn = r^(nu-1) F(r) and (du/dn)^2 = r^gamma G(r) with gamma = 2nu-2. A boundary
# functional's integrand is that times a weight: r^gamma times a series in fractional powers
# of arclength. Substituting t = r^sub for a suitable `sub` maps those powers to INTEGER
# powers of t, after which Gauss-Jacobi in t is exact.
#
# WHICH POWERS APPEAR depends on the boundary, and tracking that is what `sub` is for:
#
# - Straight edges (a polygon corner). r.N = (x-x0).n is exactly CONSTANT along a straight
#   edge, and the wedge expansion contributes only r^(k nu) together with the Bessel factors'
#   r^(2q), so the family is {k nu + 2q}. Taking sub = nu maps it to {k + 2q/nu}, integral iff
#   2/nu is an integer. Since nu > 1/2 restricts alpha to (pi, 2pi), that allows only
#   2/nu = 3: under sub = nu the 270-degree corner is the ONLY spectral reentrant angle.
# - Curved edges. Two things change, both adding INTEGER powers of arclength s:
#     * r.N is no longer constant. With x0 AT the corner it is (kappa0/2) s^2 + O(s^3) --
#       vanishing to second order rather than identically, which still helps; with x0
#       anywhere else it is an analytic series in s with nonzero constant and linear terms.
#     * r = |x(s) - corner| = s - (kappa0^2/24) s^3 + ..., i.e. s times a series in s^2, and
#       the corner asymptotics carry curvature corrections at integer powers of r.
#   The family becomes {k nu + m} with m a non-negative integer. Under sub = nu that maps to
#   {k + m/nu}, whose leading residual t^(1/nu) has 1/nu in (1,2) -- only C^1, hence little
#   better than algebraic order 3. Under **sub = 1/q, where nu = p/q in lowest terms**, it
#   maps to {k p + m q}: every exponent integral, so the rule is exact again. sub = 1/q also
#   covers the straight family (as {k p + 2 q m}), making it the more general choice; its
#   cost is that the polynomial degree required scales with p.
#
# Unlike a Kress grading order (which need only be "large enough"), nu must be EXACT -- a
# 3e-4 relative error costs four digits -- so it comes from the geometry (pi/alpha) and is
# never padded or rounded.
#
# This rule is matched to *eigenfunction*-level Cauchy data, whose local structure is purely
# the corner family above. It is NOT appropriate for a basis-level Gram matrix, whose columns
# centred at other corners are plain-analytic here and would need the {k nu + m} treatment at
# every corner simultaneously, with O(1) amplitude.

_CORNER_NU_MIN = 0.5    # beta > -1 needs nu > 1/2; at nu = 1/2 (a slit) the integral diverges
_CORNER_MAX_Q = 12      # largest denominator q admitted when rationalizing nu = p/q
_CORNER_Q_RTOL = 1e-12  # |nu - p/q| must be this small (relative) to treat nu as rational


@cache
def corner_rule_spec(nu, curved=True, max_q=_CORNER_MAX_Q, rtol=_CORNER_Q_RTOL):
    """Which corner rule to use at a corner of exponent nu, and the node-placement exponent.

    Returns `(kind, sub)` with `kind` in {'cornerjac', 'cornerinterp'} and `sub` the
    substitution exponent used to PLACE the nodes. `sub` is always nu.

    Why not sub = 1/q. Substituting t = tau^(1/q) with nu = p/q rationalizes the full curved
    exponent family {j nu + m} and is exact to 2e-16 as an abstract rule on [0,1] -- but it is
    unusable on an actual boundary, because tau = t^q crushes the innermost node as
    tau_min ~ t_min^q:

        alpha      q    tau_min (order 24)
        3/2 pi     3      1.4e-08
        5/4 pi     5      1.1e-10   below the coordinate-collapse floor
        8/5 pi     8      1.4e-18   "
        7/4 pi     7      4.7e-19   "
        11/6 pi   11      1.6e-29   "

    Those nodes round onto the corner itself once mapped through a segment's
    parametrization, where a basis's 1/r terms are fatal (see `cached_kressgauss`). The
    interpolatory rule gets the same exactness from its WEIGHTS while taking nodes from the
    mild sub = nu placement, so tau_min stays at 3.4e-5 to 3.7e-7 and the accuracy survives:
    0 to 9.7e-14 at order 24 across the same angles. It therefore replaces the large-q
    substitution entirely.

    That leaves one case where the plain substitution still wins: a STRAIGHT edge whose
    2/nu is an integer -- among reentrant angles, only alpha = 3pi/2. There the family is the
    sparser {j nu + 2q}, sub = nu rationalizes it, and order 8 suffices where the
    interpolatory rule needs 24. `curved=False` selects it."""
    nu = float(nu)
    if not curved:
        two_over_nu = 2.0/nu
        if abs(two_over_nu - round(two_over_nu)) <= rtol*two_over_nu:
            return 'cornerjac', nu
    return 'cornerinterp', nu


@cache
def corner_substitution(nu, max_q=_CORNER_MAX_Q, rtol=_CORNER_Q_RTOL):
    """DIAGNOSTIC ONLY: the rationalizing substitution exponent 1/q for nu = p/q, and whether
    nu is usefully rational. NOT for node placement -- see `corner_rule_spec` on why t^q
    crushes the innermost node below the coordinate-collapse floor for q >= 4. Retained
    because `exact=False` identifies the irrational-nu corners (generic for arc-arc
    geometry) where no substitution could work even in principle."""
    from fractions import Fraction
    nu = float(nu)
    frac = Fraction(nu).limit_denominator(max_q)
    p, q = frac.numerator, frac.denominator
    if p > 0 and abs(nu - p/q) <= rtol*abs(nu):
        return 1.0/q, True
    return nu, False


@cache
def cached_cornerjacgauss(order, nu, gamma=None, sub=None):
    """Corner-adapted Gauss-Jacobi rule on [0,1], anchored at tau=0, for an integrand
    f(tau) = tau**gamma * (series in fractional powers of tau) -- see the section comment
    above. `gamma` defaults to 2*nu-2, the exponent of the Dirichlet 'NN' (and Neumann 'TT')
    kernel. `sub` is the substitution exponent, defaulting to `nu`; pass 1/q (see
    `corner_substitution`) on a curved edge, whose exponent family also contains integer
    powers of arclength.

    Exact for tau**gamma * (polynomial of degree <= 2*order-1 in tau**sub). Returns
    `(tau, w)` ascending on [0,1], matching the convention of every other boundary rule here.

    Derivation: with tau = t**(1/sub), dtau = (1/sub) t**(1/sub - 1) dt, the singular factor
    tau**gamma becomes t**(gamma/sub), and combining with the Jacobian gives total weight
    exponent beta = (gamma+1)/sub - 1. Feeding that to `jacgauss` (whose convention divides
    the Jacobi weight out, so it integrates a raw singular integrand directly) the
    gamma-dependence cancels from the returned weights and survives only in the exponent --
    the weight below is purely the substitution Jacobian.

    TWO PROPERTIES THAT MUST NOT BE "FIXED":

    - `sum(w) == 1` only when the constant function lies in the rule's exact class, i.e. when
      t**(-gamma/sub) is polynomial. Otherwise sum(w) tends to 1 only at the rule's own
      algebraic rate (8e-4 at order 8, 4.9e-6 at order 64 for nu=0.8, sub=nu). Renormalizing
      w would destroy exactness on the singular class this rule exists for; sum(w)-1 is
      instead the cheapest available diagnostic of the residual.
    - The substitution *amplifies* node crowding: tau.min() ~ (c/order^2)**(1/sub), so past an
      order-dependent threshold the innermost node falls below the float64
      coordinate-collapse floor that `cached_kressgauss` documents. Cap the order
      (`cornerjac_order_cap`); do NOT clamp tau, which would break exactness.

    Note that accuracy is not monotone in `order` -- see `cornerjac_order_cap`'s docstring."""
    if not nu > _CORNER_NU_MIN:
        raise ValueError(f"nu must exceed {_CORNER_NU_MIN} (got {nu}): the Jacobi exponent "
                         "would be <= -1, and the underlying integral diverges at nu = 1/2 "
                         "(a slit). Such a corner needs x0 placed on it so that r.N vanishes.")
    if gamma is None:
        gamma = 2.0*nu - 2.0
    if sub is None:
        sub = float(nu)
    if not sub > 0:
        raise ValueError(f"substitution exponent must be positive (got {sub})")
    beta = (gamma + 1.0)/sub - 1.0
    if not beta > -1.0:
        raise ValueError(f"gamma={gamma} with sub={sub} gives Jacobi exponent {beta} <= -1; "
                         "the underlying integral does not converge")
    t, W = jacgauss(order, beta, 0.0)
    return t**(1.0/sub), (W/sub)*t**(1.0/sub - 1.0)


def corner_exponents(nu, gamma, n, curved=True, j_max=None, m_max=None):
    """The integrand's exponent set at a corner, smallest `n` first.

    On an edge leaving a corner of exponent nu, a boundary functional's integrand is
    r^gamma * G(r) with G a series in r^(j nu) (the corner family) times an analytic factor.
    Which integer powers that analytic factor contributes depends on the edge, and getting
    it wrong is not a small matter:

    - `curved=True`:  {gamma + j nu + m}, m any non-negative integer. A curved edge admits ODD
      powers of arclength, because r.N is not constant along it and r is not arclength.
    - `curved=False`: {gamma + j nu + 2q}, EVEN powers only. On a straight edge r.N is exactly
      constant and r is exactly arclength, so the only integer powers come from the Bessel
      factors' r^(2q).

    Testing a straight-edge rule against the curved set makes it look far worse than it is:
    at alpha=3pi/2 the substitution rule's residual reads 1e-8 against the curved set while
    its measured error on a real straight-edge integrand is 2.4e-15.

    Every moment is closed-form, int_0^1 t^e dt = 1/(e+1), which is what makes an
    interpolatory rule on this set possible even when nu is irrational and no substitution
    rationalizes it. For irrational nu no two (j, m) coincide, so there are no resonances and
    hence no log terms; the set is then the complete exponent family."""
    j_max = j_max if j_max is not None else n + 2
    m_max = m_max if m_max is not None else n + 2
    step = 1 if curved else 2
    E = sorted({gamma + j*nu + m for j in range(j_max) for m in range(0, m_max, step)})
    return np.array(E[:n])


@cache
def cached_cornerinterpgauss(order, nu, gamma=None, n_exp=None, curved=True):
    """Interpolatory corner rule on [0,1], anchored at tau=0, exact on the corner's ACTUAL
    exponent set -- for use when `corner_substitution` reports no exact substitution exists,
    i.e. when nu is irrational. That is the generic case for a corner between two circular
    arcs, whose angle is fixed by the geometry rather than chosen.

    Nodes come from `cached_cornerjacgauss` (already correctly clustered); the weights are
    then solved so that sum_i w_i tau_i**e == 1/(e+1) for every e in
    `corner_exponents(nu, gamma, n_exp)`.

    `n_exp` defaults to `order//2` and MUST stay below `order`. Solving the square system
    (n_exp == order) is a trap: it is exact on the span but wildly ill-conditioned, with
    cond(V) reaching 1e13 at order 12 and 1e19 at order 16, and weights growing to -1e4 --
    whose total variation sum|w| then multiplies every roundoff error in the integrand.
    Taking n_exp < order makes it a minimum-norm least-squares solve instead, which keeps
    **sum|w| = 1.0** while retaining exactness on the exponent set. Measured at nu=0.65736
    on a realistic curved-corner integrand:

        order  n_exp   sum|w|   rel err
           12     12    2.1e2   6.6e-15     <- exact but ill-conditioned
           16     16    6.9e3   3.0e-13
           16     10    1.0e0   2.0e-15     <- exact AND well-conditioned
           24     14    1.0e0   7.8e-13

    Compare `cached_cornerjacgauss` at the same nu: 1.0e-6. Returns `(tau, w)` ascending on
    [0,1], the same contract as every other boundary rule here."""
    if gamma is None:
        gamma = 2.0*nu - 2.0
    if n_exp is None:
        n_exp = max(2, order//2)
    if n_exp >= order:
        raise ValueError(f"n_exp ({n_exp}) must be < order ({order}): the square solve is "
                         "ill-conditioned and inflates sum|w| by orders of magnitude (see "
                         "the docstring's table)")
    tau, _ = cached_cornerjacgauss(order, nu, gamma)
    E = corner_exponents(nu, gamma, n_exp, curved)
    V = tau[None, :]**E[:, None]
    w, *_ = np.linalg.lstsq(V, 1.0/(E + 1.0), rcond=None)
    return tau, w


@cache
def corner_rule_residual(kind, order, nu, gamma=None, sub=None, curved=True, n_terms=None):
    """A-priori error indicator for a corner rule: the largest RELATIVE error it makes on any
    single monomial of the integrand's actual exponent family (`corner_exponents`).

    Needs no reference solution and no offline calibration table, because every exponent's
    moment is closed-form: `int_0^1 t^e dt = 1/(e+1)`. That is what makes precision-driven
    sizing possible here without the calibration machinery `lappy.cubature` needs for its
    interior rule.

    `n_terms` must probe beyond the rule's own exactness claim, or the indicator reports
    machine precision for a rule that is merely being tested on what it integrates exactly:
    with a fixed 20 terms, order 8 at nu=2/3 read 4.2e-15 while its true error on a sector
    eigenfunction was 1.5e-10, because all 20 exponents lay inside the exact class. It must
    also not probe far past the claim, which measures error on high powers carrying negligible
    real coefficients. The default splits by rule -- see the code.

    It remains an indicator, not a bound on the assembled integral: it takes the max over
    exponents and weights none of them by the coefficient it actually carries. Empirically it
    tracks the achieved error to within one or two orders in either direction, so it is a
    sizing heuristic whose end-to-end accuracy has to be validated separately (see
    tests/test_eigfun_integrals.py, Leg 1) rather than a certificate."""
    if gamma is None:
        gamma = 2.0*nu - 2.0
    if kind == 'cornerjac':
        tau, w = cached_cornerjacgauss(order, nu, gamma, sub)
    elif kind == 'cornerinterp':
        tau, w = cached_cornerinterpgauss(order, nu, gamma, None, curved)
    else:
        raise ValueError(f"unknown corner rule {kind!r}")
    if n_terms is None:
        # Probe just PAST each rule's own claim, not far past. cached_cornerjacgauss claims
        # exactness to degree 2*order-1 in tau**sub; cached_cornerinterpgauss claims only its
        # n_exp = order//2 exponents. Probing a rule far beyond its claim measures its error on
        # high powers that carry negligible coefficients in the real integrand and makes it look
        # uselessly bad (cornerinterp read 2.5e-5 at an order that measures 9e-14 end to end).
        n_terms = (2*order + 8) if kind == 'cornerjac' else (max(2, order//2) + 8)
    E = corner_exponents(nu, gamma, n_terms, curved)
    approx = (w[None, :]*tau[None, :]**E[:, None]).sum(axis=1)
    return float(np.abs(approx*(E + 1.0) - 1.0).max())


@cache
def corner_model_error(kind, order, nu, gamma=None, sub=None, curved=True, k=1.0, n_j=6,
                       n_m=8):
    """Relative error of a corner rule on a REPRESENTATIVE model integrand of the corner's
    class, whose exact integral is closed-form.

    This is the sizing signal, and it replaces a max over unweighted monomials
    (`corner_rule_residual`), which cannot size `cornerinterp`: that rule's exactness claim
    grows with its order, so probing a set that also grows with the order measures a moving
    target and its argmin lands in the wrong place (order 40 where the true error keeps
    improving to order 64).

    The model mirrors the real structure. On an edge leaving the corner,
    du/dn = r^(nu-1) F(r) with F a series whose Bessel factors contribute (k r/2)^(2q)/(q!)^2,
    so the coefficients decay factorially and the high powers that an unweighted max
    over-weights are in fact negligible. The model integrand is

        s^gamma * (sum_j a_j s^(j nu))^2 * (sum_m b_m s^m)

    with a_j, b_m carrying that decay at wavenumber `k = sqrt(lam_max)*panel_arclength`, and
    m restricted to even powers on a straight edge. Every term integrates in closed form, so
    the returned error is a genuine relative error on a member of the class -- not a bound over
    it, but a far better predictor of achieved accuracy than the unweighted max."""
    if gamma is None:
        gamma = 2.0*nu - 2.0
    if kind == 'cornerjac':
        tau, w = cached_cornerjacgauss(order, nu, gamma, sub)
    elif kind == 'cornerinterp':
        tau, w = cached_cornerinterpgauss(order, nu, gamma, None, curved)
    else:
        raise ValueError(f"unknown corner rule {kind!r}")

    from math import factorial
    # radial coefficients: the J_nu series, squared below by the product over (j1, j2)
    a = np.array([(0.5*k)**(2*q)/factorial(q)**2 for q in range(n_j)])
    j_exp = np.array([2.0*q for q in range(n_j)])          # powers of s in F beyond s^(nu-1)
    step = 1 if curved else 2
    b = np.array([(0.5*k)**m/factorial(m) for m in range(0, n_m, step)])
    m_exp = np.array([float(m) for m in range(0, n_m, step)])

    terms = {}
    for a1, e1 in zip(a, j_exp):
        for a2, e2 in zip(a, j_exp):
            for bm, em in zip(b, m_exp):
                e = gamma + e1 + e2 + em
                terms[e] = terms.get(e, 0.0) + a1*a2*bm
    E = np.array(sorted(terms))
    C = np.array([terms[e] for e in E])
    exact = float(np.sum(C/(E + 1.0)))
    approx = float(np.sum(C*(w[None, :]*tau[None, :]**E[:, None]).sum(axis=1)))
    return abs(approx/exact - 1.0)


@cache
def corner_order_for_precision(kind, nu, gamma=None, sub=None, curved=True, precision=1e-14,
                               scale=1.0, k=1.0, order_min=4, order_max=64, step=2):
    """Smallest order whose `corner_model_error` meets `precision`, subject to the
    coordinate-collapse cap. Returns `(order, achieved)`.

    (It scans `corner_model_error`, NOT `corner_rule_residual`, which this docstring claimed
    for a while -- long enough to nearly send a calibration effort at the wrong function.
    `corner_rule_residual` is kept as a diagnostic only.)

    Scans rather than extrapolating, because accuracy is NOT monotone in order: past a
    nu-dependent threshold the integrand's dynamic range (tau**(2nu-2) is ~6e8 at tau~1e-9
    against a correspondingly tiny weight) makes roundoff dominate, so "raise the order until
    the target is met" walks past the optimum and returns a WORSE rule. When no order meets
    `precision`, the argmin is returned together with what it actually achieves, so the caller
    can report the shortfall instead of silently missing the target."""
    cap = cornerjac_order_cap(nu, gamma, sub, scale=scale, order_max=order_max)
    best = (np.inf, order_min)
    for order in range(order_min, max(order_min, min(cap, order_max)) + 1, step):
        try:
            r = corner_model_error(kind, order, nu, gamma, sub, curved, k)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if r <= precision:
            return order, r
        if r < best[0]:
            best = (r, order)
    return best[1], best[0]


@cache
def smooth_power_error(gamma, order):
    """Relative error of an `order`-point Gauss-Legendre rule on `int_0^1 tau^gamma dtau`.

    This is the question "does a corner actually need the corner-adapted rule?", asked
    directly. The Rellich integrand near a corner of exponent `nu` carries `tau^(2nu-2)`, and
    whether Gauss-Legendre can integrate that has nothing to do with `nu < 1`:

        gamma = 2nu-2     nu       order 16     order 64
             0.333      1.167       8.0e-05      2.1e-06     needs the corner rule
             0.667      1.333       1.2e-05      1.2e-07     needs the corner rule
             1.000      1.500       0.0e+00      0.0e+00     exact: gamma is an integer
             2.500      2.250       1.7e-09      1.2e-13     borderline
            11.550      6.775       8.7e-16      3.5e-16     smooth enough already

    An integer `gamma` is a polynomial and integrates exactly; a large fractional one has so
    many continuous derivatives that the algebraic rate is indistinguishable from spectral at
    any usable order. Only small fractional powers -- corners near 135 degrees -- genuinely
    defeat a smooth rule, and they are exactly the ones the reentrant-only criterion missed.
    """
    if gamma <= 0:
        return float('inf') if gamma < 0 else 0.0
    tau, w = cached_leggauss(order)
    exact = 1.0/(gamma + 1.0)
    return float(abs(np.sum(w*tau**gamma) - exact)/exact)


@cache
def smooth_order_for_precision(k, precision=1e-14, order_min=4, order_max=512, step=2):
    """Smallest Gauss-Legendre order integrating exp(i k tau) on [0,1] to `precision`.

    Self-certifying in the same spirit as `corner_rule_residual`: the model integrand's
    integral is closed-form, so the rule's error on it is computable directly rather than
    estimated from an asymptotic (kL/4n)^(2n) formula. `k` is the integrand's wavenumber --
    for a product of two eigenfunctions with eigenvalue up to lam_max that is
    2*sqrt(lam_max)*arclength, since the product oscillates at twice the wavenumber of
    either factor.

    The error is measured ABSOLUTELY against a unit-magnitude integrand on a unit-measure
    interval, not relative to the integral's own value. That value is ~2/k for large k, so a
    relative criterion would demand ~1e-16 absolute at k=100 -- below the roundoff floor of
    the sum -- and the scan would run to `order_max` chasing something unattainable. What
    matters for a panel whose contribution is summed with others is its absolute error against
    the total measure, which is what this returns."""
    if k <= 0:
        return order_min, 0.0
    exact = (np.exp(1j*k) - 1.0)/(1j*k)
    for order in range(order_min, order_max + 1, step):
        tau, w = cached_leggauss(order)
        err = abs(np.sum(w*np.exp(1j*k*tau)) - exact)
        if err <= precision:
            return order, float(err)
    tau, w = cached_leggauss(order_max)
    return order_max, float(abs(np.sum(w*np.exp(1j*k*tau)) - exact))


@cache
def cornerjac_order_cap(nu, gamma=None, sub=None, scale=1.0, tau_floor=_KRESS_TAU_FLOOR,
                        order_max=128):
    """Largest order for which `cached_cornerjacgauss(order, nu, gamma, sub)`'s innermost node
    stays above `tau_floor/scale`, i.e. for which a segment's parametrization
    `(1-tau)*p0 + tau*pf` does not round onto the corner itself (see `cached_kressgauss` for
    the underlying float64 defect and why a node at r=0 is fatal). `scale` is the panel's
    tau-length, so a shorter panel reaches the floor at a lower order.

    This cap is necessary but NOT sufficient for choosing an order. Measured accuracy is
    non-monotone in order well *before* the floor binds, because the integrand
    tau**(2nu-2) is ~6e8 at tau~1e-9 while its weight is correspondingly tiny and the terms
    must sum to O(1) -- roundoff, not truncation. At nu=0.526 the best order is 16 (1.2e-11)
    while this cap is 75, and using the cap gives 4.2e-9. Order selection against a target
    precision therefore needs a calibrated error curve, not a cap alone."""
    cap = 0
    for order in range(2, order_max + 1):
        tau, _ = cached_cornerjacgauss(order, nu, gamma, sub)
        if tau.min()*scale <= tau_floor:
            break
        cap = order
    return cap


def boundary_nodes_polygon(vertices,n_pts=20,rule='legendre',skip=None):
    """Computes boundary nodes and weights using Chebyshev or Gauss-Legendre
    quadrature rules. Transforms the nodes to lie along the edges of the polygon with
    the given vertices."""
    vertices = np.asarray(vertices)
    if vertices.ndim > 1:
        vertices = complex_form(vertices)

    # select quadrature rule
    if rule == 'chebyshev': quadfunc = cached_chebgauss
    elif rule == 'legendre': quadfunc = cached_leggauss
    elif rule == 'even': quadfunc = lambda n: (np.linspace(0,1,n+2)[1:-1], np.ones(n)/n)
    else: raise(NotImplementedError(f"quadrature rule {rule} is not implemented"))

    # build array of n_pts (number of nodes/weights) for each edge
    if isinstance(n_pts,(int,np.integer)):
        n_pts = n_pts*np.ones(len(vertices),dtype='int')
        if skip is not None:
            n_pts[skip] = 0
    elif len(n_pts) != len(vertices):
        raise ValueError("quadrature n_pts do not match number of polygon edges")
    else:
        if skip is not None:
            raise ValueError("skip must be 'None' if n_pts are provided for each edge")

    # set up arrays for nodes and weights
    n_nodes = int(np.sum(n_pts))
    nodes = np.empty(n_nodes,dtype='complex')
    weights = np.empty(n_nodes,dtype='float')

    # get polygon edges and lengths
    edges = polygon_edges(vertices)
    lens = edge_lengths(vertices)
    for i in range(len(vertices)):
        if n_pts[i] > 0:
            start = np.sum(n_pts[:i])
            end = np.sum(n_pts[:i+1])
            # get quadrature nodes and weights for interval [0,1]
            qnodes,qweights = quadfunc(n_pts[i])
            # space nodes along edge, adjust weights for edge length
            nodes[start:end] = edges[i]*qnodes + vertices[i]
            weights[start:end] = qweights*lens[i]
    return nodes, weights

### Triangular meshes and cubature rules
def triangle_areas(mesh_vertices,triangles):
    """Computes the areas of triangles in a triangular mesh"""
    v = mesh_vertices[triangles]
    return 0.5*np.abs((v[:,0,0]-v[:,2,0])*(v[:,1,1]-v[:,0,1])-(v[:,0,0]-v[:,1,0])*(v[:,2,1]-v[:,0,1]))

def tri_quad_rule(mesh_vertices, triangles, kind, deg):
    """Applies one cubature rule to a given set of triangles (rows of vertex indices
    into mesh_vertices), in complex form. Core of tri_quad, factored out so callers
    (e.g. lappy.cubature's per-triangle rule assignment) can apply different rules
    to different subsets of the same mesh and concatenate the results."""
    # get triangle vertices in complex form
    tri_vertices = mesh_vertices[triangles]

    # get cubature nodes and weights in barycentric form
    # convert to array of nodes in complex form
    bary_coords, bary_weights = get_cubature_rule(kind,deg)
    nodes = (tri_vertices[:,:,0]@(bary_coords.T) + 1j*(tri_vertices[:,:,1]@(bary_coords.T))).flatten()

    # get areas of triangles, scale weights appropriately
    areas = triangle_areas(mesh_vertices,triangles)
    weights = np.outer(areas,bary_weights).flatten()
    return nodes, weights

def tri_quad(mesh, kind='dunavant', deg=4):
    """"Sets up a cubature rule for a given mesh, in complex form"""
    # extract mesh vertices and triangle-to-vertex array
    mesh_vertices = mesh.points[:,:2]
    triangles = mesh.cells[1].data
    return tri_quad_rule(mesh_vertices, triangles, kind, deg)

# mesh building
def polygon_triangular_mesh(vertices, mesh_size, mesh_size_min=0.05, mesh_size_max=0.5):
    """Builds a triangular mesh on a polygon with pygmsh"""
    import gmsh
    from pygmsh.geo import Geometry
    vertices = np.asarray(vertices)
    if vertices.dtype == 'complex128':
        vertices = real_form(vertices)
    if vertices.shape[0] == 2:
        vertices = vertices.T
    if vertices.shape[1] != 2 or vertices.ndim != 2:
        raise ValueError('vertices must be a 2-dimensional array of x & y coordinates')

    # build triangular mesh with pygmsh
    with Geometry() as geom:
        geom.add_polygon(vertices, mesh_size)

        # Set meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)  # Use point sizes
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Extend to interior
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        mesh = geom.generate_mesh()
    return mesh

def corner_fan_triangles(z0, theta0, omega, R0, L=2, n_theta=8, sigma=0.17):
    """Explicit (non-gmsh) triangulation of a wedge at a reentrant corner, geometrically
    graded toward the corner point, in complex form.

    gmsh's background-field meshing does not reliably realize an extreme geometric size
    field near a single point (verified empirically: even when a size field demands
    elements ~1e-4 in size at a corner, the realized mesh's smallest edge stays ~0.2-0.4),
    which caps achievable accuracy on singular corner integrands at ~1e-11 regardless of
    requested precision. Building the graded layers directly, with exact straight-sided
    geometry matching the polygon's real local edges (no circular-arc approximation),
    removes that ceiling -- see lappy.cubature._choose_corner_rule for how the per-layer
    cubature rule degree is chosen based on target precision.

    Parameters
    ----------
    z0 : complex
        Corner vertex location.
    theta0 : float
        Direction (radians) of the wedge's starting edge from z0.
    omega : float
        Interior angle of the wedge (the corner's reentrant interior angle).
    R0 : float
        Outer radius of the fan (transition to the rest of the domain).
    L : int, optional
        Number of geometric radial layers (default 2; accuracy is governed by the
        cubature rule's degree, not by L -- see module docstring in lappy.cubature).
    n_theta : int, optional
        Angular subdivisions of the wedge.
    sigma : float, optional
        Geometric grading ratio between successive layers.

    Returns
    -------
    (n, 3, 2) ndarray
        Triangle vertices in real (x, y) coordinates, corner-relative offsets added to z0.
    """
    thetas = theta0 + np.linspace(0.0, omega, n_theta + 1)
    radii = R0 * sigma**np.arange(L + 1)   # radii[0]=R0 (outer) ... radii[L] (innermost ring)

    ring_pts = radii[:, None] * np.exp(1j * thetas)[None, :]   # (L+1, n_theta+1) complex, corner-relative

    tris = []
    for k in range(L):
        p_out, p_in = ring_pts[k], ring_pts[k + 1]
        for j in range(n_theta):
            tris.append((p_out[j], p_out[j + 1], p_in[j]))
            tris.append((p_out[j + 1], p_in[j + 1], p_in[j]))
    # apex fan: innermost ring down to the corner point itself
    p_in = ring_pts[-1]
    for j in range(n_theta):
        tris.append((0.0 + 0.0j, p_in[j], p_in[j + 1]))

    tris = np.array(tris) + z0   # (n, 3) complex
    return np.stack([tris.real, tris.imag], axis=-1)   # (n, 3, 2) real

def curvature_sampling(spline, t0, tf, pts_per_2pi=20):
    """Gets samples from a SciPy BSpline with density based on curvature."""
    if not isinstance(spline, BSpline):
        raise TypeError("'spline' must be a SciPy BSpline.")
    
    t_fine = np.linspace(t0, tf, 1000)
    dr = spline.derivative(nu=1)(t_fine)
    ddr = spline.derivative(nu=2)(t_fine)

    speed = np.abs(dr)
    cross = (dr.conj() * ddr).imag
    curvature = np.abs(cross) / (speed**3 + 1e-10)
    
    # Point density: higher curvature = more points
    # Scale by arc length element (speed * dt) to account for parameterization
    dt = np.diff(t_fine, prepend=t_fine[0])
    arc_element = speed * dt
    
    # Points needed per segment based on curvature
    points_per_segment = (curvature / (2 * np.pi)) * pts_per_2pi * arc_element
    
    # Add minimum to handle straight sections
    points_per_segment = np.maximum(points_per_segment, 0.02)
    
    cumulative_points = np.cumsum(points_per_segment)
    total_points = max(2, int(np.ceil(cumulative_points[-1])))
    
    target_positions = np.linspace(0, cumulative_points[-1], total_points)
    t_samples = np.interp(target_positions, cumulative_points, t_fine)
    
    return t_samples

def spline_mesh_with_curvature(segments, pts_per_2pi=20, mesh_size_min=0.05, mesh_size_max=0.5):
    """
    Creates a mesh from a list of SciPy BSpline objects with curvature-adaptive sampling.

    Parameters
    ----------
    splines : list of BSpline
        Closed loop of splines defining the boundary
    pts_per_2pi : float
        Points per 2π radians of curvature for boundary sampling
    mesh_size_min : float
        Minimum mesh size (at high-curvature regions)
    mesh_size_max : float
        Maximum mesh size (in interior/low-curvature regions)

    Returns
    -------
    mesh : meshio.Mesh
        Generated mesh
    """
    import gmsh
    from pygmsh.geo import Geometry
    # Sample each spline with curvature-adaptive spacing
    boundary_points = []
    boundary_curvatures = []
    
    for seg in segments:
        # Get curvature-sampled points
        t_samples = curvature_sampling(seg.spline, seg.t0, seg.tf, pts_per_2pi)
        pts = seg.spline(t_samples)
        
        # Also get curvature at these points for mesh sizing
        dr = seg.spline.derivative(nu=1)(t_samples)
        ddr = seg.spline.derivative(nu=2)(t_samples)
        speed = np.abs(dr)
        cross = (dr.conj() * ddr).imag
        curvature = np.abs(cross) / (speed**3 + 1e-10)
        
        boundary_points.append(pts)
        boundary_curvatures.append(curvature)

    # Concatenate all boundary points (remove duplicate endpoints between splines)
    all_points = []
    all_curvatures = []
    for i, (pts, curv) in enumerate(zip(boundary_points, boundary_curvatures)):
        if i == 0:
            all_points.append(pts)
            all_curvatures.append(curv)
        else:
            # Skip first point (duplicate of previous spline's last point)
            all_points.append(pts[1:])
            all_curvatures.append(curv[1:])
            
    all_points = np.concatenate(all_points)[:-1]
    all_curvatures = np.concatenate(all_curvatures)[:-1]
    
    # Remove last point if it duplicates the first (closing the loop)
    if np.allclose(all_points[-1], all_points[0]):
        all_points = all_points[:-1]
        all_curvatures = all_curvatures[:-1]
    
    # Compute mesh sizes based on curvature
    # Higher curvature -> smaller mesh size
    max_curv = np.percentile(all_curvatures, 95)  # Use 95th percentile to avoid outliers
    normalized_curv = np.clip(all_curvatures / (max_curv + 1e-10), 0, 1)
    
    # Interpolate between min and max size (inverse relationship with curvature)
    mesh_sizes = mesh_size_max - normalized_curv * (mesh_size_max - mesh_size_min)
    
    with Geometry() as geom: 
        # Create gmsh points with prescribed mesh sizes
        gmsh_points = []
        for pt, size in zip(all_points, mesh_sizes):
            gmsh_points.append(geom.add_point([pt.real, pt.imag, 0], mesh_size=size))
        
        # Create line segments connecting consecutive points
        lines = []
        n_pts = len(gmsh_points)
        for i in range(n_pts):
            lines.append(geom.add_line(gmsh_points[i], gmsh_points[(i + 1) % n_pts]))
        
        # Create curve loop and surface
        curve_loop = geom.add_curve_loop(lines)
        surface = geom.add_plane_surface(curve_loop)
        
        # Set meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)  # Use point sizes
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Extend to interior
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        
        # Generate mesh
        mesh = geom.generate_mesh(dim=2)
    
    return mesh

### Quadrilateral meshes and quadrature rules
def quadrilateral_mesh(vertices,mesh_size):
    """Builds a quadrilateral mesh using pygmsh. NOTE: This function does not always
    give purely quadrilateral meshes. It is retained only for convenience, and should
    not be relied on in general."""
    import gmsh
    from pygmsh.geo import Geometry
    vertices = np.array(vertices)
    if vertices.shape[0] == 2:
        vertices = vertices.T
    if vertices.shape[1] != 2 or vertices.ndim != 2:
        raise ValueError('vertices must be a 2-dimensional array of x & y coordinates')

    # build quadrilateral mesh with pygmsh
    with Geometry() as geom:
        polygon = geom.add_polygon(vertices,mesh_size)
        geom.set_recombined_surfaces([polygon.surface])
        mesh = geom.generate_mesh(dim=2,algorithm=8)
    return mesh

def transform_quad(xi,eta,x_v,y_v):
    """Computes a transformation from the  reference square [-1,1]^2 to a
    quadrilateral with given vertices. Also computes the Jacobian determinant"""
    a,b = x_v[2]-x_v[3],x_v[2]+x_v[3]
    c,d = x_v[1]-x_v[0],x_v[1]+x_v[0]
    e,f = y_v[2]-y_v[3],y_v[2]+y_v[3]
    g,h = y_v[1]-y_v[0],y_v[1]+y_v[0]
    etap1 = eta+1
    etam1 = eta-1
    dx_dxi = ((a-c)*eta+a+c)/4
    dx_deta = ((a-c)*xi+b-d)/4
    dy_dxi = ((e-g)*eta+e+g)/4
    dy_deta = ((e-g)*xi+f-h)/4
    x = (etap1*(a*xi+b) - etam1*(c*xi+d))/4
    y = (etap1*(e*xi+f) - etam1*(g*xi+h))/4
    detJ = dx_dxi*dy_deta-dx_deta*dy_dxi
    return  x,y,detJ

def gauss_quad_nodes(mesh_vertices,quads,order=5):
    """Tensor-product Gauss-Legendre quadrature for a quadrilateral mesh"""
    # get Gauss-Legendre points and weights for [-1,1]^2
    pts,wts = cached_leggauss(order)
    Wts = np.outer(wts,wts)
    Xi,Eta = np.meshgrid(pts,pts,indexing='ij')

    # set up data structures
    k = order**2
    n_nodes = k*len(quads)
    nodes = np.empty((2,n_nodes))
    weights = np.empty(n_nodes)

    for i,quad in enumerate(quads):
        x,y = mesh_vertices[quad].T
        x_nodes, y_nodes, detJ = transform_quad(Xi,Eta,x,y)
        quad_weights = detJ*Wts
        nodes[:,i*k:(i+1)*k] = x_nodes.flatten(),y_nodes.flatten()
        weights[i*k:(i+1)*k] = quad_weights.flatten()

    return nodes.T,weights

def quadrilateral_quad(mesh,order=5):
    """Sets up a quadrature rule for a quadrilateral mesh"""
    mesh_vertices = mesh.points[:,:2]
    quads = mesh.cells[1].data
    return gauss_quad_nodes(mesh_vertices,quads,order)