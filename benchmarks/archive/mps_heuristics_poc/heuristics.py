"""Closed-form basis-selection heuristics from ``docs/mps_heuristics.pdf`` (polygon-only,
testing-purposes proof of concept).

Translates the paper's geometry -> basis recipe (Secs 1-5) into ``lappy``'s existing
``FourierBesselBasis``/``FundamentalBasis`` primitives. Additive module: no changes to
``bases.py``, ``geometry.py``, or ``mps.py``.

Deliberately out of scope for this pass:

* Sec 6 (replacing interior collocation with a Rellich-identity ``A_I`` block) -- a
  separate, already-tracked initiative (``docs/rellich.md``, the ``rellich`` branch); a
  different scope (eigenfunction normalization, not the eigenvalue-finding pencil).
* Sec 7 (budget/conditioning diagnostics), Sec 8 (lambda-sweep window freezing), Sec 9
  (adaptive correction loop) -- solver/search orchestration and self-certification, set
  aside for a later pass. :func:`polygon_default_basis` is a pure, static, one-shot
  function: it builds a basis from geometry alone and does not measure or rebuild it.
  (:func:`plan_basis` reports what it *would* build without a loop -- see below.)
* Curved-arc obstructions (tangential curvature-jump junctions) -- polygon-only.

History (why this module looks the way it does). A first version of this module
implemented Secs 1-5 fairly literally (a crude "2x nearest non-adjacent edge" proxy for
the obstruction distance d_c, a taper-stop-only handoff between the ambient MFS curve and
a singular corner's own Fourier-Bessel terms) and testing it on ``geo.L_shape()`` surfaced
real problems: near-duplicate MFS columns from an under-specified bridge tail
destabilizing the GSVD's SVD-rtol truncation and corrupting the tension background, and
(separately) an ambient MFS layer that covered almost the whole boundary regardless of
how well a singular corner's own Fourier-Bessel terms already covered it -- because
nothing in that first pass actually ceded boundary responsibility to Fourier-Bessel the
way Design Principle 1 (Sec 2) intends.

This version is a substantial rewrite matching a more complete, independently-written
implementation of the same paper (``~/claude_basis_heuristics.py`, not itself wired into
lappy) much more closely in its mathematics:

* A real geometric obstruction set (:func:`build_obstruction_set`): actual point
  reflections of non-regular corners across every edge, with a per-obstruction algebraic
  order and near-integer amplitude, and an amplitude-decay cutoff that retires an
  obstruction once it is too weak to matter within Lambda digits -- replacing the old
  proxy.
* A continuation term solved from the actual series-convergence condition
  (:func:`_nu_required`, root-finding) instead of a closed-form approximation.
* ``handover_frac`` (default 0.80): ambient MFS sources within ``handover_frac * R_c`` of
  an FB-equipped singular corner are dropped outright -- this, not the obstruction-set
  refinement, is what actually shrinks the ambient layer on domains like the L-shape
  (a lone singular corner's own reflected image across the domain's far edges is still a
  genuine, finite obstruction even under the more careful math -- there is no free
  ``d_c = infinity`` here).
* A log-uniform (``geomspace``) bridge tail for FB-equipped corners, replacing sqrt
  clustering there (kept only for the no-FB-terms "lightning" case, Sec 5.4 eq 4).
* A floored ``delta(s)`` and global arclength reparametrization (pre-sample the boundary,
  place sources at equal intervals of ``integral ds/h``) replacing greedy local marching.
* Optional regular/weakly-singular-corner Fourier-Bessel terms, on by default.
* Sec 5.5 far multipoles, implemented but off by default (a single global far centre's
  conditioning cap makes it useless; see :func:`_far_multipoles`).
* :func:`plan_basis`, a diagnostic companion exposing the per-corner planning table
  (:class:`BasisPlan`) without building the lappy basis objects.
"""
from dataclasses import dataclass, field

from .core import BaseDomain
from .bases import FourierBesselBasis, FundamentalBasis

import numpy as np
import warnings
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

__all__ = [
    'HeuristicConfig', 'Obstruction', 'CornerPlan', 'BasisPlan',
    'target_lambda', 'build_obstruction_set', 'polygon_default_basis', 'plan_basis',
]

_WEAK_SINGULAR_DENOM = 2*np.log(15)  # Sec 3 cutoff denominator, 2 ln 15


@dataclass
class HeuristicConfig:
    """Tuned constants (Sec 11, "tuned versus derived"). Field-for-field match to
    ``~/claude_basis_heuristics.py``'s ``PlanConfig``."""
    C_omega: float = 10.0        # Moler-Payne calibration constant, Sec 1
    gamma: float = 0.40          # Sec 2, corner reach as a fraction of d_c
    eta: float = 0.30            # Sec 5.1, source offset as a fraction of dist to S*
    nyquist_ppw: float = 3.0     # Sec 5.2, minimum points per wavelength on the source curve
    delta_frac_D: float = 0.25   # Sec 5.1, ambient offset as a fraction of the diameter
    mu_cliff: float = 0.05       # Sec 3, near-integer amplitude discount scale
    airy_margin: float = 2.0     # Sec 4, coefficient of (kappa R)^(1/3)
    order_margin: float = 5.0    # Sec 4, additive safety margin on nu_osc
    n_bridge: int = 10           # Sec 5.4, bridge poles at an FB-bearing singular corner
    sigma_light: float = 4.0     # Sec 5.4, Gopal-Trefethen taper parameter (lightning case)
    s_min_frac: float = 0.05     # Sec 5.4, innermost bridge radius, as a fraction of R_c
    handover_frac: float = 0.80  # FB/MFS handover radius, as a fraction of R_c
    eps_mach: float = float(np.finfo(float).eps)
    max_reflections: int = 1     # obstruction-set reflection depth; 2 costs O(n_edges^2)
    n_far_centers: int = 0       # Sec 5.5 far multipole centres; off by default
    far_radius_frac: float = 1.5  # far-centre circle radius, in diameters
    include_regular_fb: bool = True
    n_boundary_samples: int = 4000


def target_lambda(precision, C_Omega=10.0):
    """Sec 1, eq 1: converts a target eigenvalue relative precision into
    ``Lambda = ln(1/eps_hat)``, the single "how many digits" quantity used by every count
    below.

    ``C_Omega`` is the Moler-Payne calibration constant ("O(1)-O(10) for non-pathological
    domains... calibrate it once per domain family"). No calibration loop here (Sec 7/9
    are out of scope for this pass) -- it is a caller-supplied constant, default 10.0.
    """
    if not (0 < precision < 1):
        raise ValueError("'precision' must be in (0, 1)")
    eps_hat = precision/(10*C_Omega)
    if precision < 1e-12:
        warnings.warn(
            "target_lambda: precision below ~1e-12 is past the paper's documented double "
            "precision floor (Sec 1, 'Hard floor': GSVD sigma_min plateaus around "
            "1e-13-1e-14 even with column equilibration); more basis functions will not "
            "achieve this, extended precision is the only remedy.")
    return np.log(1.0/eps_hat)


def _corner_data(domain, cfg):
    """``alpha, mu, amp`` for every corner: exponent, distance to the nearest integer,
    and a near-integer amplitude discount ``amp = min(1, mu/mu_cliff)`` (Sec 3)."""
    omega = domain.corner_int_angles
    alpha = np.pi/omega
    mu = np.abs(alpha - np.round(alpha))
    amp = np.minimum(1.0, mu/cfg.mu_cliff)
    return alpha, mu, amp


def _validate_polygon_domain(domain):
    if not isinstance(domain, BaseDomain):
        raise TypeError("'domain' must be a Domain object")
    if not domain.bdry.is_polyline:
        raise TypeError("this function is restricted to polygon domains "
                        "(domain.bdry.is_polyline must be True)")
    if len(domain.corners) == 0:
        raise ValueError("domain must have at least one corner")


# --------------------------------------------------------------------------------- #
# obstruction set S*  (Sec 2, Design Principle 1)
# --------------------------------------------------------------------------------- #

@dataclass
class Obstruction:
    point: complex
    p: float                # algebraic index: r^p or r^p log r (here, the corner exponent)
    amp: float               # relative amplitude in (0, 1]
    kind: str                # 'corner' | 'image'
    parent: int = -1         # generating corner index
    depth: int = 0           # number of reflections applied


def _reflect_across_edge(p, a, b):
    """Reflect complex point ``p`` across the infinite line through complex points
    ``a, b``. Returns ``(image, t)``, where ``t`` is the parameter of the perpendicular
    foot along the segment (``t=0`` at ``a``, ``t=1`` at ``b``)."""
    d = b - a
    L2 = (d*np.conj(d)).real
    t = ((p - a)*np.conj(d)).real/L2
    foot = a + t*d
    return 2*foot - p, t


def build_obstruction_set(domain, cfg, Lambda):
    """Sec 2: the obstruction set S* -- every non-regular corner, plus its reflected
    images across every edge (up to ``cfg.max_reflections`` deep), with an
    amplitude-decay cutoff that stops reflecting an obstruction once it is too weak to
    matter within ``Lambda`` digits (``Lambda + log(amp) <= 0``).

    Ported from ``~/claude_basis_heuristics.py``'s ``build_obstruction_set``, adapted to
    lappy's complex-plane geometry (``LineSegment.p0``/``.pf`` in place of raw ``(n,2)``
    edge arrays). A corner reflected across one of its own two adjacent edges maps back
    to itself exactly (the foot of the perpendicular from a vertex onto a line through
    that same vertex is the vertex), so no separate adjacency bookkeeping is needed --
    those degenerate reflections are filtered by the "maps to nearly the same point"
    check below.
    """
    corners = domain.corners
    n = len(corners)
    alpha, mu, amp = _corner_data(domain, cfg)
    nonregular = mu > 1e-9

    obs = [Obstruction(corners[i], p=alpha[i], amp=amp[i], kind='corner', parent=i)
           for i in range(n) if nonregular[i]]

    edges = [(seg.p0, seg.pf) for seg in domain.bdry.segments]
    frontier = list(obs)
    for depth in range(1, cfg.max_reflections + 1):
        new = []
        for o in frontier:
            if Lambda + np.log(o.amp) <= 0.0:
                continue
            for a, b in edges:
                img, t = _reflect_across_edge(o.point, a, b)
                if not (-0.05 <= t <= 1.05):      # not reachable through this edge
                    continue
                if abs(img - o.point) < 1e-12:
                    continue                      # point lies on the mirror
                new.append(Obstruction(img, p=o.p, amp=o.amp, kind='image',
                                       parent=o.parent, depth=depth))
        obs += new
        frontier = new
    return obs


# --------------------------------------------------------------------------------- #
# Fourier-Bessel corner plans (Secs 3-4)
# --------------------------------------------------------------------------------- #

def _nu_required(d, R, p, amp, Lambda):
    """Smallest ``nu`` with ``amp * nu^-(p+1) * (R/d)^nu <= exp(-Lambda)`` -- the actual
    series-convergence condition for an obstruction at distance ``d`` with algebraic
    order ``p`` and amplitude ``amp``, solved by root-finding rather than approximated
    in closed form. Ported near-verbatim from the reference file."""
    Ls = Lambda + np.log(amp)
    if Ls <= 0.0:
        return 0.0
    q = np.log(d/R)
    if q <= 0.0:
        return np.inf
    f = lambda nu: nu*q + (p + 1.0)*np.log(nu) - Ls
    hi = Ls/q + 10.0
    while f(hi) < 0.0:
        hi *= 2.0
    return brentq(f, 1.0, hi, xtol=1e-8) if f(1.0) < 0.0 else 1.0


@dataclass
class CornerPlan:
    index: int
    vertex: complex
    omega: float
    alpha: float
    kind: str                  # 'singular' | 'weak' | 'regular'
    M: int                      # Fourier-Bessel terms j = 1..M, orders j*alpha
    R_c: float = 0.0
    d_c: float = np.inf
    nu_osc: float = 0.0
    nu_cont: float = 0.0
    nu_cap: float = np.inf
    capped: bool = False
    binding: str = ""


def _corner_plans(domain, cfg, Lambda, kappa, obs, sample):
    """One consolidated per-corner planning pass (Secs 3-4), replacing separate
    classification/obstruction-distance/count functions with a single loop, matching
    the reference file's ``_corner_plans``.

    Regular corners (``mu <= 1e-9``) and weakly-singular ones (``alpha > Lambda/(2 ln
    15)``, Sec 3) get no continuation term; when ``cfg.include_regular_fb`` they instead
    get a small angle-and-local-reach-sized Fourier-Bessel budget (Sec 3's "optional"
    treatment, applied here to both regular and weakly-singular corners alike, matching
    the reference).

    Singular corners get the full treatment: ``d_c`` from the real obstruction set
    (nearest surviving obstruction, i.e. one not filtered by the amplitude-decay cutoff),
    capped by the exact max-boundary-reach/gamma bound (exact for a polygon: distance
    from a vertex to a point on a straight edge is convex along the edge, so the max over
    the whole boundary is attained at a vertex); ``R_c = gamma*d_c``; the oscillatory term
    ``nu_osc`` (eq 2, Airy-margin form); the continuation term ``nu_cont`` (the max, over
    surviving obstructions, of :func:`_nu_required`); the Sec 4 conditioning cap
    ``nu_cap`` (using a dense boundary sample's median radius over the owned arc, correct
    even when ``R_c`` reaches past the two immediately adjacent edges); and
    ``M = max(1, ceil(min(nu_osc+nu_cont, nu_cap) / alpha))``.
    """
    corners = domain.corners
    n = len(corners)
    omegas = domain.corner_int_angles
    alpha, mu, _ = _corner_data(domain, cfg)
    weak_cut = Lambda/_WEAK_SINGULAR_DENOM

    _, Xb, _, _ = sample
    obs_xy = np.array([o.point for o in obs]) if obs else np.empty(0, dtype=complex)
    plans = []

    for i in range(n):
        v = corners[i]
        w, al = omegas[i], alpha[i]

        if mu[i] <= 1e-9 or al > weak_cut:
            kind = 'regular' if mu[i] <= 1e-9 else 'weak'
            M = 0
            if cfg.include_regular_fb:
                nb1, nb2 = corners[(i + 1) % n], corners[i - 1]
                R_loc = 0.5*min(abs(nb1 - v), abs(nb2 - v))
                M = int(np.ceil((w/np.pi)*(kappa*R_loc + cfg.order_margin)))
            plans.append(CornerPlan(i, v, w, al, kind, M))
            continue

        d = np.abs(obs_xy - v) if len(obs_xy) else np.empty(0)
        keep = [k for k in range(len(obs))
                if d[k] > 1e-9 and Lambda + np.log(obs[k].amp) > 0.0]
        if keep:
            d_c = min(d[k] for k in keep)
            near = min(keep, key=lambda k: d[k])
            binding = f"{obs[near].kind}@corner{obs[near].parent}"
        else:
            d_c, binding = np.inf, "none"

        reach_cap = np.max(np.abs(corners - v))/cfg.gamma
        d_c = min(d_c, reach_cap)
        R_c = cfg.gamma*d_c

        kR = kappa*R_c
        nu_osc = kR + cfg.airy_margin*kR**(1.0/3.0) + cfg.order_margin
        nu_cont = max((_nu_required(d[k], R_c, obs[k].p, obs[k].amp, Lambda)
                       for k in keep), default=0.0)

        arc = np.abs(Xb - v)
        arc = arc[arc <= R_c]
        r_mid = np.median(arc) if arc.size else 0.5*R_c
        nu_cap = (kR + np.log(1.0/cfg.eps_mach)/np.log(R_c/max(r_mid, 1e-12))
                  if r_mid < R_c else np.inf)

        nu = nu_osc + nu_cont
        capped = nu > nu_cap
        nu = min(nu, nu_cap)
        plans.append(CornerPlan(i, v, w, al, 'singular',
                                M=max(1, int(np.ceil(nu/al))),
                                R_c=R_c, d_c=d_c, nu_osc=nu_osc, nu_cont=nu_cont,
                                nu_cap=nu_cap, capped=capped, binding=binding))
    return plans


# --------------------------------------------------------------------------------- #
# fundamental-solution sources (Sec 5)
# --------------------------------------------------------------------------------- #

def _lightning_count(alpha, Lambda):
    """Sec 5.4 eq 4: pole count for the full Gopal-Trefethen tapered cluster at a
    corner with no Fourier-Bessel terms."""
    return max(4, int(np.ceil(Lambda**2/(2.0*np.pi**2*alpha))))


def _inner_radius(pl, cfg, Lambda, delta_amb):
    """The radius inside which a corner's own basis takes over from the ambient MFS
    curve. For an FB-equipped corner that is the handover radius
    (``handover_frac*R_c``) -- the Fourier-Bessel series already resolves the arc within
    ``R_c`` to ``Lambda`` digits, so ambient MFS coverage there is pure duplication. For
    a no-FB ("lightning") corner the ambient curve must taper all the way in, and the
    cluster is its continuation -- the floor is the cluster's own innermost reach."""
    if pl.kind == 'singular' and pl.M > 0:
        return cfg.handover_frac*pl.R_c
    n = _lightning_count(pl.alpha, Lambda)
    return delta_amb*np.exp(-cfg.sigma_light*(np.sqrt(n) - 1.0))


def _sample_boundary(domain, n_total):
    """Arclength samples with outward unit normals, roughly ``n_total`` in all,
    allocated proportionally to edge length. Adapted from the reference file's
    ``_sample_boundary``, built from lappy ``LineSegment.p``/``.N``."""
    segments = domain.bdry.segments
    lengths = np.array([seg.len for seg in segments])
    total = lengths.sum()
    counts = np.maximum(2, np.round(n_total*lengths/total).astype(int))

    S, X, N = [], [], []
    s0 = 0.0
    for seg, Li, ci in zip(segments, lengths, counts):
        u = (np.arange(ci) + 0.5)/ci
        X.append(seg.p(u))
        N.append(seg.N(u))
        S.append(s0 + u*Li)
        s0 += Li
    return np.concatenate(S), np.concatenate(X), np.concatenate(N), s0


def _graded_curve_sources(domain, cfg, Lambda, kappa, obs, delta_amb, plans, sample):
    """Sec 5.1-5.3: the ambient MFS source curve, built by global arclength
    reparametrization rather than greedy local marching -- pre-sample the boundary,
    form ``delta(s)`` (floored at the nearest corner's own inner radius, preventing the
    ``integral ds/h`` reparametrization from diverging near a singular corner) and
    ``h(s)``, then place sources at equal intervals of ``Phi = cumtrapz(1/h, s)`` via
    inverse interpolation. Finally drops any source within ``handover_frac*R_c`` of an
    FB-equipped singular corner's vertex -- ceding that arc to Fourier-Bessel entirely
    (Design Principle 1), not merely thinning it.
    """
    s, X, N, L = sample

    obs_xy = np.array([o.point for o in obs]) if obs else np.empty(0, dtype=complex)
    if len(obs_xy):
        dS = np.min(np.abs(X[:, None] - obs_xy[None, :]), axis=1)
    else:
        dS = np.full(len(X), np.inf)

    floors = [_inner_radius(pl, cfg, Lambda, delta_amb) for pl in plans if pl.kind != 'regular']
    if floors:
        dS = np.maximum(dS, min(floors))

    delta = np.minimum(delta_amb, cfg.eta*dS)
    h = np.minimum(2.0*np.pi/(cfg.nyquist_ppw*kappa), np.pi*delta/Lambda)

    Phi = np.concatenate([[0.0], cumulative_trapezoid(1.0/h, s)])
    n_src = max(8, int(np.floor(Phi[-1])))
    targets = Phi[-1]*(np.arange(n_src) + 0.5)/n_src

    idx = np.interp(targets, Phi, np.arange(len(s)))
    lo = np.clip(np.floor(idx).astype(int), 0, len(s) - 2)
    frac = idx - lo
    P = X[lo] + frac*(X[lo + 1] - X[lo])
    Nn = N[lo] + frac*(N[lo + 1] - N[lo])
    Nn = Nn/np.abs(Nn)
    dd = np.interp(targets, Phi, delta)
    pts = P + dd*Nn

    drop = np.zeros(len(pts), dtype=bool)
    for pl in plans:
        if pl.kind == 'singular' and pl.M > 0:
            drop |= np.abs(pts - pl.vertex) < cfg.handover_frac*pl.R_c
    return pts[~drop]


def _corner_cluster_sources(domain, cfg, Lambda, delta_amb, plans):
    """Sec 5.4: per-corner cluster/bridge sources on the outward angle bisector.

    FB-equipped singular corners get a short, log-uniformly-spaced (``geomspace``)
    bridge from ``s_min = s_min_frac*R_c`` out to ``delta_amb`` -- simpler than
    sqrt-clustering, and its closest pole is exactly ``s_min`` by construction (no
    near-machine-epsilon poles). No-FB corners (weakly-singular, Sec 3) get the full
    Gopal-Trefethen tapered cluster (eq 4), unchanged from earlier versions of this
    module and matching the reference file's "lightning" case.
    """
    phi0, _phi1 = domain.corner_angles
    psi = domain.corner_int_angles

    pts, kinds = [], []
    for pl in plans:
        if pl.kind == 'regular':
            continue
        out_angle = phi0[pl.index] + psi[pl.index]/2
        ray = -np.exp(1j*out_angle)

        if pl.kind == 'singular' and pl.M > 0:
            s_min = max(cfg.s_min_frac*pl.R_c, 1e-14)
            t = np.geomspace(s_min, delta_amb, cfg.n_bridge)
            tag = 'bridge'
        else:
            n = _lightning_count(pl.alpha, Lambda)
            j = np.arange(1, n + 1)
            t = delta_amb*np.exp(-cfg.sigma_light*(np.sqrt(n) - np.sqrt(j)))
            tag = 'lightning'

        pts.append(pl.vertex + t*ray)
        kinds += [tag]*len(t)

    if not pts:
        return np.empty(0, dtype=complex), []
    return np.concatenate(pts), kinds


def _far_multipoles(domain, cfg, Lambda, kappa, D, centroid):
    """Sec 5.5: multipole sources at a handful of far exterior centres. Off by default
    (``n_far_centers=0``): a centre at distance ``rho`` serving an arc of half-extent
    ``a`` needs ``K ~ Lambda/ln(sqrt(rho^2+a^2)/rho)``; for a centre far enough to see
    the whole boundary, ``a/rho`` is small, the log is ``~a^2/(2 rho^2)``, and ``K``
    blows up like ``2 Lambda rho^2/a^2`` -- the conditioning cap then truncates it and
    the columns carry no information. Implemented (matching the reference file) for
    completeness/parity, not because a single global centre is recommended; multipoles
    only pay when each centre serves an arc comparable to its own standoff.
    """
    if cfg.n_far_centers <= 0:
        return np.empty(0, dtype=complex), np.empty(0, dtype=int)
    R_far = cfg.far_radius_frac*D
    th = 2.0*np.pi*np.arange(cfg.n_far_centers)/cfg.n_far_centers
    C = centroid + R_far*np.exp(1j*th)

    a = 0.5*D
    rho = max(R_far - 0.5*D, 1e-12)
    reach = np.hypot(rho, a)
    K = kappa*reach + Lambda/np.log(max(reach/rho, 1.0 + 1e-12))
    K_cap = np.log(1.0/cfg.eps_mach)/np.log(1.0 + D/rho)
    K = int(max(0, np.floor(min(K, K_cap))))
    return C, np.full(cfg.n_far_centers, K, dtype=int)


def _drop_interior_sources(domain, sources, orders, where):
    """Like ``bases._exterior_sources_only``, extended to keep a parallel ``orders``
    array in sync (needed once far-multipole sources, Sec 5.5, can carry orders > 1)."""
    inside = np.asarray(domain.contains(sources), dtype=bool)
    if not inside.any():
        return sources, orders
    if inside.all():
        raise ValueError(f"{where}: every source lies inside the domain; none is a "
                         "particular solution there.")
    warnings.warn(f"{where}: dropped {int(inside.sum())} of {len(sources)} sources that "
                  "lie inside the domain (see bases._exterior_sources_only's docstring "
                  "for why this invalidates both the tension and any certified bound).")
    return sources[~inside], orders[~inside]


# --------------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------------- #

def _plan(domain, lam_max, precision, cfg):
    """Shared computation behind :func:`polygon_default_basis` and :func:`plan_basis`."""
    _validate_polygon_domain(domain)
    kappa = np.sqrt(lam_max)
    Lambda = target_lambda(precision, cfg.C_omega)
    D = domain.diameter
    delta_amb = cfg.delta_frac_D*D  # no curvature term: polygon-only (see module docstring)

    obs = build_obstruction_set(domain, cfg, Lambda)
    sample = _sample_boundary(domain, cfg.n_boundary_samples)
    plans = _corner_plans(domain, cfg, Lambda, kappa, obs, sample)

    P_curve = _graded_curve_sources(domain, cfg, Lambda, kappa, obs, delta_amb, plans, sample)
    P_clust, kinds_clust = _corner_cluster_sources(domain, cfg, Lambda, delta_amb, plans)
    if cfg.n_far_centers > 0:
        centroid = np.mean(domain.corners)
        C_far, K_far = _far_multipoles(domain, cfg, Lambda, kappa, D, centroid)
    else:
        C_far, K_far = np.empty(0, dtype=complex), np.empty(0, dtype=int)

    return dict(kappa=kappa, Lambda=Lambda, delta_amb=delta_amb, obs=obs, plans=plans,
               P_curve=P_curve, P_clust=P_clust, kinds_clust=kinds_clust,
               C_far=C_far, K_far=K_far)


def polygon_default_basis(domain, lam_max, precision=1e-10, cfg=None):
    """Secs 1-5: the paper's closed-form basis-selection recipe, polygon-only, with no
    certify/adapt loop (Secs 7-9 deferred, see module docstring).

    Matches ``benchmarks/basis_lab/bench.py``'s ``build_basis(domain, lam_max)``
    convention -- ``precision`` and ``cfg`` both default, so this drops straight into
    ``bench.evaluate`` alongside ``pure_fb``/``mixed``/``fixed_n``.

    Parameters
    ----------
    domain : Polygon
        Must be a straight-edged polygon (``domain.bdry.is_polyline``).
    lam_max : float
        Top of the spectral window; used as ``kappa_max = sqrt(lam_max)`` throughout.
    precision : float
        Target eigenvalue relative precision (Sec 1).
    cfg : HeuristicConfig, optional
        Tuned constants; defaults to ``HeuristicConfig()``.

    See Also
    --------
    plan_basis : returns the per-corner planning diagnostics without building a basis.
    """
    cfg = cfg or HeuristicConfig()
    info = _plan(domain, lam_max, precision, cfg)

    fs_sources = np.concatenate([info['P_curve'], info['P_clust'], info['C_far']])
    fs_orders = np.concatenate([
        np.ones(len(info['P_curve']), dtype=int),
        np.ones(len(info['P_clust']), dtype=int),
        info['K_far'] + 1,
    ])
    fs_sources, fs_orders = _drop_interior_sources(domain, fs_sources, fs_orders,
                                                    'polygon_default_basis')
    fs_basis = FundamentalBasis(fs_sources, fs_orders)

    fb_orders = np.array([pl.M for pl in info['plans']], dtype=int)
    if fb_orders.sum() == 0:
        return fs_basis
    fb_basis = FourierBesselBasis.from_domain(domain, fb_orders)
    return fb_basis + fs_basis


@dataclass
class BasisPlan:
    """Diagnostic summary of what :func:`polygon_default_basis` would build, without
    building it. Ported from the reference file's ``BasisPlan``."""
    plans: list
    Lambda: float
    kappa_max: float
    delta_amb: float
    obstructions: list
    n_curve: int
    n_bridge_lightning: int
    kinds_clust: list = field(default_factory=list)
    n_far: int = 0
    n_far_functions: int = 0

    @property
    def n_fb(self):
        return sum(c.M for c in self.plans)

    @property
    def n_fs_points(self):
        return self.n_curve + self.n_bridge_lightning + self.n_far

    @property
    def n_fs_functions(self):
        return self.n_curve + self.n_bridge_lightning + self.n_far_functions

    @property
    def n_total(self):
        return self.n_fb + self.n_fs_functions

    def summary(self):
        out = [f"Lambda = {self.Lambda:.1f}   kappa_max = {self.kappa_max:.3f}   "
              f"delta_amb = {self.delta_amb:.4f}", "",
              f"{'corner':>6} {'omega/pi':>9} {'alpha':>7} {'kind':>9} "
              f"{'d_c':>8} {'R_c':>8} {'nu_osc':>8} {'nu_cont':>8} {'M':>5}  binding"]
        for c in self.plans:
            out.append(f"{c.index:>6} {c.omega/np.pi:>9.3f} {c.alpha:>7.3f} "
                       f"{c.kind:>9} {c.d_c:>8.3f} {c.R_c:>8.3f} "
                       f"{c.nu_osc:>8.1f} {c.nu_cont:>8.1f} {c.M:>5}"
                       f"{'*' if c.capped else ' '} {c.binding}")
        tally = {t: self.kinds_clust.count(t) for t in sorted(set(self.kinds_clust))}
        out += ["",
               f"FB terms          : {self.n_fb}",
               f"FS source points  : {self.n_fs_points}  curve={self.n_curve} "
               f"clust={tally} far={self.n_far}",
               f"FS functions      : {self.n_fs_functions}",
               f"TOTAL basis size  : {self.n_total}"]
        if self.n_total > 600:
            out.append("WARNING: > 600 functions; expect conditioning-limited GSVD (Sec 7).")
        return "\n".join(out)


def plan_basis(domain, lam_max, precision=1e-10, cfg=None):
    """Diagnostic companion to :func:`polygon_default_basis`: runs the same planning
    pipeline and returns a :class:`BasisPlan` (per-corner table, FB/FS/total counts)
    without constructing any ``FourierBesselBasis``/``FundamentalBasis`` objects.
    """
    cfg = cfg or HeuristicConfig()
    info = _plan(domain, lam_max, precision, cfg)
    n_far_functions = int(np.sum(2*info['K_far'] + 1)) if len(info['K_far']) else 0
    return BasisPlan(plans=info['plans'], Lambda=info['Lambda'], kappa_max=info['kappa'],
                     delta_amb=info['delta_amb'], obstructions=info['obs'],
                     n_curve=len(info['P_curve']),
                     n_bridge_lightning=len(info['P_clust']),
                     kinds_clust=info['kinds_clust'],
                     n_far=len(info['C_far']), n_far_functions=n_far_functions)
