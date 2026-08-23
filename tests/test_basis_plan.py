"""Tests for `lappy.basis_plan`.

The interesting ones are not the formulas but the INVARIANTS that make a plan usable inside a
shape-optimization loop: a frozen plan realizes to the same basis size on a perturbed domain,
sources stay outside, ownership is a partition, and refinement can never return a worse plan than
it started with. Those are the properties the old `heuristics.py` lacked, and each failure they
guard against was measured (`benchmarks/basis_lab/HEURISTICS.md`, `PLAN_LAB.md`).
"""
import warnings

import numpy as np
import pytest

from lappy import basis_plan as BP
from lappy import geometry as geo
from lappy.asymp import weyl_est
from lappy.bases import FourierBesselBasis, FundamentalBasis, MultiBasis


def _lam_max(dom):
    return float(weyl_est(6, dom))


@pytest.fixture
def L():
    return geo.L_shape()


@pytest.fixture
def sq():
    return geo.rect(1.0, 1.0)


# ── planning basics ──────────────────────────────────────────────────────────

def test_plan_is_frozen_and_hashable(L):
    """`BasisPlan` is the thing a caller holds across an optimization run, so it must not be
    mutable by accident and must be usable as a cache key."""
    p = BP.plan_basis(L, _lam_max(L), 1e-7)
    with pytest.raises(Exception):
        p.target = 1e-3
    hash(p.corners)
    hash(p.arcs)


def test_every_corner_gets_fourier_bessel_terms(L):
    """No singular/regular branch: the measured reason is that turning the regular-corner block
    off cost -8.7 digits on `square` once combined with a sparser source layer, while a
    single-factor screen had scored it at -0.1."""
    p = BP.plan_basis(L, _lam_max(L), 1e-7)
    assert len(p.corners) == len(L.corners)
    assert all(c.M >= 1 for c in p.corners)


def test_reentrant_corner_gets_the_largest_budget(L):
    """L_shape's corner 0 is the reentrant one (alpha = 2/3)."""
    p = BP.plan_basis(L, _lam_max(L), 1e-7)
    reentrant = min(p.corners, key=lambda c: c.alpha)
    assert reentrant.corner == 0
    assert reentrant.M == max(c.M for c in p.corners)


def test_size_grows_monotonically_with_target(L):
    """The old recipe's `n` was NOT monotone in requested precision -- `iso_tri_h4` went 343
    columns at 1e-2 to 261 at 1e-4, because a threshold reclassified a corner and deleted a hole
    in the source layer. Asking for more accuracy must never hand back a smaller basis."""
    ns = [BP.plan_basis(L, _lam_max(L), t).n_total
          for t in (1e-3, 1e-5, 1e-7, 1e-9, 1e-11)]
    assert ns == sorted(ns), ns


@pytest.mark.parametrize('build', [geo.L_shape, lambda: geo.rect(1, 1), lambda: geo.eq_tri(1),
                                   lambda: geo.iso_tri(4.0), lambda: geo.chevron(1, 2),
                                   lambda: geo.reg_ngon(6), lambda: geo.right_trapezoid(1, 2)])
def test_plans_and_realizes_on_every_polygon_family(build):
    dom = build()
    p = BP.plan_basis(dom, _lam_max(dom), 1e-7)
    basis = BP.realize(p, dom)
    assert len(basis) == p.n_total


def test_rejects_curved_and_non_domains():
    with pytest.raises(TypeError):
        BP.plan_basis(geo.disk(1), 10.0, 1e-7)
    with pytest.raises(TypeError):
        BP.plan_basis('not a domain', 10.0, 1e-7)
    with pytest.raises(ValueError):
        BP.plan_basis(geo.L_shape(), 10.0, 0.0)


# ── ownership is a partition ─────────────────────────────────────────────────

def test_arcs_are_disjoint_and_inside_their_segments(L):
    p = BP.plan_basis(L, _lam_max(L), 1e-7)
    by_seg = {}
    for a in p.arcs:
        assert 0.0 <= a.tau0 < a.tau1 <= 1.0
        by_seg.setdefault(a.seg, []).append((a.tau0, a.tau1))
    for seg, spans in by_seg.items():
        spans.sort()
        for (_, e), (s, _) in zip(spans, spans[1:]):
            assert e <= s + 1e-12, f'overlapping arcs on segment {seg}'


def test_no_sources_in_the_corner_owned_arcs(L):
    """Ownership is a partition, not a subtraction. The old recipe generated a global source curve
    and then deleted the sources near an FB corner, which left near-duplicate columns; here they
    are never generated."""
    p = BP.plan_basis(L, _lam_max(L), 1e-7)
    segs = L.bdry.segments
    for a in p.arcs:
        seg = segs[a.seg]
        u = a.tau0 + (a.tau1 - a.tau0)*(np.arange(a.n_src) + 0.5)/a.n_src
        pts = seg.p(u)
        for c in p.corners:
            v = L.corners[c.corner]
            assert np.all(np.abs(pts - v) > 0.99*c.R*0.5), 'a source landed in an owned arc'


# ── sources stay exterior ────────────────────────────────────────────────────

@pytest.mark.parametrize('build', [geo.L_shape, lambda: geo.iso_tri(4.0), lambda: geo.iso_tri(16.0),
                                   lambda: geo.chevron(2, 4), lambda: geo.H_shape(),
                                   lambda: geo.parallelogram(1, 1, np.pi/6.5)])
def test_sources_never_land_inside_the_domain(build):
    """`iso_tri_h4` under the old recipe placed 6 of 621 sources inside the domain at
    `precision=1e-10`, which voids both the tension and any certified bound. The offset here is
    bounded by `exterior_clearance` at plan time, so it cannot happen -- and `realize` raises
    rather than dropping sources if it ever does."""
    dom = build()
    for target in (1e-4, 1e-7, 1e-10):
        p = BP.plan_basis(dom, _lam_max(dom), target)
        basis = BP.realize(p, dom, check_exterior=True)      # raises if any source is interior
        assert len(basis) == p.n_total


def test_exterior_clearance_matches_a_hand_computation():
    """A U with a notch `x in [1,2], y in [1,3]`: the two notch walls face each other across a gap
    of exactly 1, so a source pushed further than that off either wall re-enters the domain.

    Note the L-shape is NOT a witness for this, though it looks like one: an outward ray from
    either edge of its notch escapes to infinity rather than re-entering, because the notch opens
    onto the exterior. That is why the offset bound needs a real ray cast and not a
    reentrant-corner heuristic.
    """
    U = geo.Polygon([0, 3, 3 + 3j, 2 + 3j, 2 + 1j, 1 + 1j, 1 + 3j, 3j], val_simple=False)
    walls = {3: -1.0 + 0j, 5: 1.0 + 0j}          # the two facing notch walls, and their normals
    for seg_idx, normal in walls.items():
        mid = U.bdry.segments[seg_idx].p(np.array([0.5]))
        d = BP.exterior_clearance(U, mid, np.array([normal]))
        assert d[0] == pytest.approx(1.0, abs=1e-9), seg_idx

    # and an outward ray from the L-shape's notch escapes
    assert np.isinf(BP.exterior_clearance(geo.L_shape(), np.array([0.5j]),
                                          np.array([1.0 + 0j]))[0])


def test_exterior_clearance_is_infinite_outward_from_a_convex_domain():
    sq = geo.rect(1.0, 1.0)
    seg = sq.bdry.segments[0]
    p = seg.p(np.array([0.5]))
    n = seg.N(np.array([0.5]))
    assert np.isinf(BP.exterior_clearance(sq, p, n)[0])


def test_local_thickness_reads_the_width_of_a_thin_rectangle():
    r = geo.rect(1.0, 8.0)
    seg = r.bdry.segments[1]                     # a long edge
    p = seg.p(np.array([0.5]))
    n = seg.N(np.array([0.5]))
    assert BP.local_thickness(r, p, n)[0] == pytest.approx(1.0, rel=2e-2)


# ── the conditioning ceilings ────────────────────────────────────────────────

def test_fb_ceiling_bites_hardest_at_sharp_corners():
    cfg = BP.PlanConfig()
    assert BP._fb_ceiling(0.667, cfg) > BP._fb_ceiling(2.0, cfg) > BP._fb_ceiling(9.76, cfg)
    assert BP._fb_ceiling(25.0, cfg) >= 1


def test_source_ceiling_scales_with_the_offset():
    cfg = BP.PlanConfig()
    lo = BP._src_ceiling(1.0, 0.01, 1.0, cfg)
    hi = BP._src_ceiling(1.0, 0.05, 1.0, cfg)
    assert lo > hi >= cfg.min_src_per_arc, (lo, hi)


def test_planned_arcs_respect_the_source_ceiling(L):
    p = BP.plan_basis(L, _lam_max(L), 1e-11)
    for a in p.arcs:
        assert a.n_src <= BP._src_ceiling(a.arclen, a.delta_rel, p.diameter, p.cfg)


def test_cap_is_reported_not_silently_exceeded():
    """A domain this module cannot serve at a target should say so. `spiral` has 24 corners and its
    Fourier-Bessel budget alone exceeds any sane cap."""
    p = BP.plan_basis(geo.spiral(), _lam_max(geo.spiral()), 1e-10)
    assert p.capped and p.shortfall


# ── the inner-loop invariant ─────────────────────────────────────────────────

def _l_family(t):
    a = 1.0 + t
    return geo.Polygon([0, 1j, -a + 1j, -a - 1j, 1 - 1j, 1], bc='dir', val_simple=False)


def test_a_frozen_plan_realizes_to_a_constant_size_along_a_shape_family():
    """THE property the plan/realize split exists for. If `n_basis` moved as the optimizer stepped,
    a change in lambda could not be attributed to the shape."""
    plan = BP.plan_basis(_l_family(0.0), _lam_max(_l_family(0.0)), 1e-7)
    sizes = {len(BP.realize(plan, _l_family(t))) for t in np.linspace(-0.05, 0.05, 11)}
    assert sizes == {plan.n_total}, sizes


def test_realization_is_continuous_in_the_shape():
    """Source positions must move smoothly with the geometry, or a finite-difference gradient of
    lambda is differencing two different bases."""
    plan = BP.plan_basis(_l_family(0.0), _lam_max(_l_family(0.0)), 1e-7)

    def sources(t):
        b = BP.realize(plan, _l_family(t))
        fs = [s for s in b.bases if isinstance(s, FundamentalBasis)] if isinstance(b, MultiBasis) else [b]
        return np.concatenate([s.sources for s in fs])

    s0, s1 = sources(0.0), sources(1e-6)
    assert s0.shape == s1.shape
    assert np.max(np.abs(s1 - s0)) < 1e-4        # O(dt), not O(1)


def test_realize_is_deterministic(L):
    plan = BP.plan_basis(L, _lam_max(L), 1e-7)
    a, b = BP.realize(plan, L), BP.realize(plan, L)
    fb_a = a.bases[0] if isinstance(a, MultiBasis) else a
    fb_b = b.bases[0] if isinstance(b, MultiBasis) else b
    assert isinstance(fb_a, FourierBesselBasis)
    np.testing.assert_array_equal(fb_a.orders, fb_b.orders)


# ── refinement ───────────────────────────────────────────────────────────────

def test_refinement_never_returns_a_worse_plan_than_it_started_with():
    """Growth is not monotone in accuracy: adding columns to a block already at its conditioning
    limit degrades the pencil, measured on `chevron_2_4` as a fall to 0.8 certified digits. The
    loop keeps the best plan it measured, so it can only improve on where it began."""
    from lappy.mps import MPSEigensolver
    from lappy import reference as ref

    dom = geo.chevron(1, 2)
    lam = float(ref.chevron_eigs(1, 1.0, 2.0)[0])

    def factory(basis):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return MPSEigensolver.from_domain(dom, basis=basis, rng=7, prec=1e-14)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p0 = BP.plan_basis(dom, _lam_max(dom), 1e-7)
        r0 = BP._residual_by_arc(p0, dom, factory(BP.realize(p0, dom)), lam)
        p1 = BP.refine_plan(p0, dom, factory, lam)
        r1 = BP._residual_by_arc(p1, dom, factory(BP.realize(p1, dom)), lam)

    worst0 = max(r0[0].max(initial=0), r0[1].max(initial=0))
    worst1 = max(r1[0].max(initial=0), r1[1].max(initial=0))
    assert worst1 <= worst0*(1 + 1e-9), (worst0, worst1)


@pytest.mark.parametrize('build,key', [(geo.L_shape, 'L_shape'),
                                       (lambda: geo.iso_tri(16.0), 'iso_tri_h16'),
                                       (lambda: geo.right_trapezoid(1, 2), 'right_trapezoid')])
def test_achieved_accuracy_is_monotone_in_the_target(build, key):
    """Asking for more accuracy must not deliver less. This guards `_indep_digits`: with the
    ceiling constant at machine epsilon instead of the solver's `rtol`, `iso_tri_h16` went 4.2
    digits at 1e-7, 5.2 at 1e-10 and back to 4.2 at 1e-13 -- the extra columns were redundant to
    the rank truncation and actively harmful.

    Uses `_residual_by_arc` rather than a full certification: they agree to 0.1 digits and this is
    ~100x cheaper.
    """
    from lappy.mps import MPSEigensolver
    from benchmarks.basis_lab.heur import reference_eigs

    dom = build()
    lam = float(np.asarray(reference_eigs(key, 2)[0], dtype=float)[0])
    digits = []
    for target in (1e-7, 1e-10, 1e-13):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = BP.plan_basis(dom, _lam_max(dom), target)
            solver = MPSEigensolver.from_domain(dom, basis=BP.realize(p, dom), rng=7, prec=1e-14)
            arc, cor = BP._residual_by_arc(p, dom, solver, lam)
        worst = max(arc.max(initial=0.0), cor.max(initial=0.0))
        digits.append(-np.log10(max(worst, 1e-16)))
    # 0.5 digits of slack for domains already sitting on the double-precision floor
    for a, b in zip(digits, digits[1:]):
        assert b >= a - 0.5, digits


def test_the_residual_diagnostic_is_the_certified_bound_per_arc():
    """`_residual_by_arc` reports each block's contribution to Moler--Payne's `eps`, so that
    "residual <= target" means "this block would certify the target". Measured agreement with the
    full certification is 0.1 digits; an earlier uniform-sampling version ran two digits
    optimistic and would have let refinement stop early."""
    from lappy.mps import MPSEigensolver
    from lappy import reference as ref
    from benchmarks.reference.certify import moler_payne

    dom = geo.L_shape()
    lam = float(ref.L_shape_eigs(1)[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = BP.plan_basis(dom, _lam_max(dom), 1e-7)
        solver = MPSEigensolver.from_domain(dom, basis=BP.realize(p, dom), rng=7, prec=1e-14)
        arc, cor = BP._residual_by_arc(p, dom, solver, lam)
        u = solver.eigenfunction(lam, mult=1)
        mp = moler_payne(dom, lambda z, u=u: u(z)[:, 0], lam)

    worst = max(arc.max(initial=0), cor.max(initial=0))
    assert abs(-np.log10(worst) - mp['digits']) < 0.5, (worst, mp['digits'])
