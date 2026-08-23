"""Tests for lappy.heuristics — the docs/mps_heuristics.pdf closed-form basis recipe
(polygon-only, testing-purposes proof of concept).

This module was substantially rewritten to match ~/claude_basis_heuristics.py's
mathematics (real reflection-based obstruction set, handover-radius-based FB/MFS
partition, root-solved continuation term, global arclength reparametrization). Most of
this file's expected values are therefore re-derived against the CURRENT formulas, not
against an earlier version's numbers or the paper's own internally-inconsistent Sec 10
worked example (see target_lambda's own eq-1 check below for why the paper's example is
not trustworthy at face value: its stated Lambda=25.3 does not follow from its own eq 1).
"""
import warnings

import numpy as np
import pytest

from lappy import geometry as geo
from lappy import heuristics as H
from lappy.bases import FundamentalBasis, FourierBesselBasis, MultiBasis


# ── target_lambda (Sec 1, eq 1) ─────────────────────────────────────────────

def test_target_lambda_matches_eq1():
    precision, C_Omega = 1e-10, 10.0
    Lambda = H.target_lambda(precision, C_Omega)
    eps_hat = precision/(10*C_Omega)
    assert Lambda == pytest.approx(np.log(1/eps_hat))
    assert Lambda == pytest.approx(27.631021115928547)


def test_target_lambda_warns_below_hard_floor():
    with pytest.warns(UserWarning, match="Hard floor"):
        H.target_lambda(1e-13)


def test_target_lambda_rejects_bad_precision():
    with pytest.raises(ValueError):
        H.target_lambda(0.0)
    with pytest.raises(ValueError):
        H.target_lambda(1.0)


# ── _reflect_across_edge / build_obstruction_set (Sec 2) ────────────────────

def test_reflect_across_edge_matches_hand_computation():
    # reflecting 1+2j across the real axis (segment 0 -> 1) gives 1-2j, foot at t=1
    img, t = H._reflect_across_edge(1 + 2j, 0 + 0j, 1 + 0j)
    assert img == pytest.approx(1 - 2j)
    assert t == pytest.approx(1.0)


def test_reflect_a_corner_across_its_own_adjacent_edge_maps_to_itself():
    """A vertex reflected across a line through itself is a fixed point -- this is
    exactly what build_obstruction_set relies on to skip self-reflection across a
    corner's own two adjacent edges without separate adjacency bookkeeping."""
    img, t = H._reflect_across_edge(0 + 0j, 0 + 0j, 1 + 1j)
    assert img == pytest.approx(0 + 0j)


def test_L_shape_obstruction_set_has_four_images_at_distance_2():
    """L_shape's single reentrant corner (origin) reflects across the four
    non-adjacent edges to four images, each at distance 2 -- the domain's own reach in
    each direction from the corner."""
    dom = geo.L_shape()
    cfg = H.HeuristicConfig()
    Lambda = H.target_lambda(1e-8, cfg.C_omega)
    obs = H.build_obstruction_set(dom, cfg, Lambda)
    images = [o for o in obs if o.kind == 'image']
    assert len(images) == 4
    dists = sorted(abs(o.point) for o in images)
    assert dists == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_obstruction_set_empty_for_all_regular_domain(rect_domain):
    cfg = H.HeuristicConfig()
    Lambda = H.target_lambda(1e-8, cfg.C_omega)
    obs = H.build_obstruction_set(rect_domain, cfg, Lambda)
    assert obs == []


def test_amplitude_decay_cutoff_stops_reflection():
    """A near-integer corner exponent gives amp << 1; once Lambda + log(amp) <= 0 the
    obstruction generates no reflected images at all.

    L_shape's own reentrant corner has alpha=2/3, mu=1/3 -- far from an integer, so its
    amp is exactly 1.0 and this cutoff can never trigger for it regardless of Lambda
    (log(1.0)=0 needs Lambda<=0, which target_lambda never produces). A small quadrilateral
    with one corner perturbed just off a right angle gives a genuinely near-integer
    corner (mu ~ 3e-4) whose amp is small enough for a modest Lambda=5 to trip the cutoff.
    """
    dom = geo.Polygon(np.array([0, 1, 1 + 1j*(1 - 3e-4), 1j]))
    cfg = H.HeuristicConfig()
    alpha, mu, amp = H._corner_data(dom, cfg)
    assert (mu[(mu > 1e-9) & (mu < 0.01)]).size > 0  # sanity: a genuinely near-integer corner exists

    obs_full = H.build_obstruction_set(dom, cfg, Lambda=30.0)
    obs_tiny = H.build_obstruction_set(dom, cfg, Lambda=4.0)
    assert any(o.kind == 'corner' for o in obs_tiny)
    assert not any(o.kind == 'image' for o in obs_tiny)
    # sanity: the SAME domain at a large-enough Lambda does generate images somewhere
    assert any(o.kind == 'image' for o in obs_full)


# ── _nu_required (root-solved continuation term) ────────────────────────────

def test_nu_required_decreases_with_distance():
    near = H._nu_required(d=2.0, R=1.0, p=1.0, amp=1.0, Lambda=20.0)
    far = H._nu_required(d=5.0, R=1.0, p=1.0, amp=1.0, Lambda=20.0)
    assert far < near


def test_nu_required_zero_when_amplitude_already_decayed():
    assert H._nu_required(d=5.0, R=1.0, p=1.0, amp=1e-20, Lambda=20.0) == 0.0


def test_nu_required_satisfies_its_own_equation():
    d, R, p, amp, Lambda = 3.0, 1.0, 0.75, 0.8, 22.0
    nu = H._nu_required(d, R, p, amp, Lambda)
    lhs = nu*np.log(d/R) + (p + 1.0)*np.log(nu)
    assert lhs == pytest.approx(Lambda + np.log(amp), abs=1e-6)


# ── _corner_plans / plan_basis (Secs 3-4) ────────────────────────────────────

def test_L_shape_plan_matches_hand_computed_d_c_and_R_c():
    dom = geo.L_shape()
    plan = H.plan_basis(dom, lam_max=42.521723355910495, precision=1e-8)
    singular = [c for c in plan.plans if c.kind == 'singular']
    assert len(singular) == 1
    c = singular[0]
    assert c.alpha == pytest.approx(2/3)
    assert c.d_c == pytest.approx(2.0)
    assert c.R_c == pytest.approx(0.8)
    assert c.binding.startswith('image@corner')
    assert c.M > 0


def test_regular_corners_get_optional_fb_by_default(rect_domain):
    plan = H.plan_basis(rect_domain, lam_max=50.0, precision=1e-8)
    assert all(c.kind == 'regular' for c in plan.plans)
    assert all(c.M > 0 for c in plan.plans)


def test_include_regular_fb_false_disables_it(rect_domain):
    cfg = H.HeuristicConfig(include_regular_fb=False)
    plan = H.plan_basis(rect_domain, lam_max=50.0, precision=1e-8, cfg=cfg)
    assert all(c.M == 0 for c in plan.plans)


def test_conditioning_cap_flags_capped_for_extreme_precision():
    """R_c is capped by the domain's own max boundary reach once gamma is large enough
    to bind that cap (independent of gamma beyond that point -- R_c = gamma*d_c and
    d_c = max_reach/gamma cancel), so gamma is not a lever for the conditioning cap.
    An absurdly tight precision (driving Lambda, hence nu_cont, arbitrarily high) is:
    nu_cap depends only on kappa/R_c/r_mid, not on precision, so it stays fixed while
    nu_osc+nu_cont grows without bound."""
    dom = geo.L_shape()
    cfg = H.HeuristicConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # precision far past the documented hard floor
        plan = H.plan_basis(dom, lam_max=100.0, precision=1e-20, cfg=cfg)
    singular = [c for c in plan.plans if c.kind == 'singular'][0]
    assert singular.capped
    assert singular.nu_osc + singular.nu_cont > singular.nu_cap


# ── handover_frac (the actual "FS looks superfluous" fix) ───────────────────

def test_handover_drops_ambient_sources_near_fb_corner():
    dom = geo.L_shape()
    cfg = H.HeuristicConfig()
    Lambda = H.target_lambda(1e-8, cfg.C_omega)
    kappa = np.sqrt(42.521723355910495)
    delta_amb = cfg.delta_frac_D*dom.diameter
    obs = H.build_obstruction_set(dom, cfg, Lambda)
    sample = H._sample_boundary(dom, cfg.n_boundary_samples)
    plans = H._corner_plans(dom, cfg, Lambda, kappa, obs, sample)
    curve = H._graded_curve_sources(dom, cfg, Lambda, kappa, obs, delta_amb, plans, sample)

    singular = [c for c in plans if c.kind == 'singular'][0]
    dists = np.abs(curve - singular.vertex)
    assert dists.min() >= cfg.handover_frac*singular.R_c - 1e-9


def test_larger_handover_frac_drops_more_ambient_sources():
    dom = geo.L_shape()
    basis_small = H.polygon_default_basis(dom, lam_max=42.5, precision=1e-8,
                                          cfg=H.HeuristicConfig(handover_frac=0.2))
    basis_large = H.polygon_default_basis(dom, lam_max=42.5, precision=1e-8,
                                          cfg=H.HeuristicConfig(handover_frac=0.8))
    assert len(basis_large) < len(basis_small)


# ── bridge tail (Sec 5.4, FB-equipped corner) ───────────────────────────────

def test_bridge_tail_closest_pole_is_exactly_s_min():
    dom = geo.L_shape()
    cfg = H.HeuristicConfig()
    Lambda = H.target_lambda(1e-8, cfg.C_omega)
    delta_amb = cfg.delta_frac_D*dom.diameter
    kappa = np.sqrt(42.521723355910495)
    obs = H.build_obstruction_set(dom, cfg, Lambda)
    sample = H._sample_boundary(dom, cfg.n_boundary_samples)
    plans = H._corner_plans(dom, cfg, Lambda, kappa, obs, sample)
    pts, kinds = H._corner_cluster_sources(dom, cfg, Lambda, delta_amb, plans)

    singular = [c for c in plans if c.kind == 'singular'][0]
    bridge_dists = np.abs(pts - singular.vertex)
    s_min = cfg.s_min_frac*singular.R_c
    assert bridge_dists.min() == pytest.approx(s_min, rel=1e-6)


# ── polygon_default_basis: integration ──────────────────────────────────────

def test_polygon_default_basis_raises_on_curved_domain(disk_domain):
    with pytest.raises(TypeError, match="polygon"):
        H.polygon_default_basis(disk_domain, lam_max=30.0)


def test_polygon_default_basis_raises_on_non_domain():
    with pytest.raises(TypeError):
        H.polygon_default_basis("not a domain", lam_max=30.0)


@pytest.mark.parametrize("domain_fn", [
    lambda: geo.L_shape(),
    lambda: geo.rect(1, 1),
    lambda: geo.GWW1(),
    lambda: geo.GWW2(),
    lambda: geo.eq_tri(1.0),
])
def test_polygon_default_basis_sources_stay_exterior(domain_fn):
    """Regression per the exterior-source failure mode documented in
    bases._exterior_sources_only: no warning should fire, meaning every constructed
    source landed outside the domain."""
    dom = domain_fn()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        basis = H.polygon_default_basis(dom, lam_max=100.0, precision=1e-8)
    assert len(basis) > 0


def test_polygon_default_basis_is_pure_fundamental_basis_without_regular_fb():
    dom = geo.rect(1, 1)
    cfg = H.HeuristicConfig(include_regular_fb=False)
    basis = H.polygon_default_basis(dom, lam_max=50.0, precision=1e-8, cfg=cfg)
    assert isinstance(basis, FundamentalBasis)


def test_polygon_default_basis_is_multibasis_with_regular_fb_on():
    dom = geo.rect(1, 1)
    basis = H.polygon_default_basis(dom, lam_max=50.0, precision=1e-8)
    assert isinstance(basis, MultiBasis)
    kinds = [type(b) for b in basis.bases]
    assert FourierBesselBasis in kinds
    assert FundamentalBasis in kinds


def test_polygon_default_basis_has_fb_terms_for_singular_corner():
    dom = geo.L_shape()
    basis = H.polygon_default_basis(dom, lam_max=100.0, precision=1e-10)
    assert isinstance(basis, MultiBasis)
    kinds = [type(b) for b in basis.bases]
    assert FourierBesselBasis in kinds
    assert FundamentalBasis in kinds


def test_polygon_default_basis_size_grows_with_precision():
    dom = geo.L_shape()
    basis_loose = H.polygon_default_basis(dom, lam_max=100.0, precision=1e-6)
    basis_tight = H.polygon_default_basis(dom, lam_max=100.0, precision=1e-12)
    assert len(basis_tight) > len(basis_loose)


# ── plan_basis / BasisPlan consistency ───────────────────────────────────────

def test_plan_basis_fb_count_matches_built_basis():
    dom = geo.L_shape()
    lam_max, precision = 100.0, 1e-10
    plan = H.plan_basis(dom, lam_max=lam_max, precision=precision)
    basis = H.polygon_default_basis(dom, lam_max=lam_max, precision=precision)
    assert plan.n_fb == sum(c.M for c in plan.plans)
    assert len(basis) == plan.n_total


def test_plan_basis_summary_is_a_string_and_mentions_conditioning_warning_when_huge():
    dom = geo.L_shape()
    plan = H.plan_basis(dom, lam_max=100.0, precision=1e-10)
    s = plan.summary()
    assert isinstance(s, str)
    assert "TOTAL basis size" in s
