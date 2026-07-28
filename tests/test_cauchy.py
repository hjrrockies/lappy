"""Tests for lappy.cauchy — the generic Cauchy-data evaluation and
boundary-integral kernel-assembler API that lappy.rellich (and, potentially,
code outside lappy building Hadamard-type quantities) is built on, plus the
Kress-style graded-mesh boundary quadrature (docs/rellich_hadamard_mps.pdf
Sec. 6.1) that replaced the earlier SS/SR/RS/RR singularity-subtraction
machinery."""

import numpy as np
import pytest

from lappy import Polygon
from lappy.bases import FourierBesselBasis, FundamentalBasis
from lappy.reference import rect_eig
from lappy.geometry import PointSet, L_shape
from lappy.utils import complex_dot
from lappy.cauchy import (
    basis_cauchy_data, assemble_kernel, default_x0, graded_pts_per_seg,
    corner_grading_orders, build_boundary_quadrature,
)

L, H = 2.0, 1.0
RECT_VERTS = np.array([0, L, L + 1j*H, 1j*H])


def _rect_domain_basis():
    domain = Polygon(RECT_VERTS, bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[6, 6, 6, 6])
    return domain, basis


def _plain_bdry_cauchy_data(domain, basis, lam, n_per_seg=30):
    """Cauchy data on an ordinary (ungraded) Legendre boundary node set --
    the kind of node set an external caller with its own weight function
    might build, independent of anything Rellich-specific."""
    pts = domain.bdry_pts(n_per_seg, kind='legendre', weights=True)
    normals = domain.bdry_normals(n_per_seg, kind='legendre')
    tangents = domain.bdry_tangents(n_per_seg, kind='legendre')
    return basis_cauchy_data(basis, lam, pts, normals, tangents, pts.wts)


def test_basis_cauchy_data_shapes():
    domain, basis = _rect_domain_basis()
    cd = _plain_bdry_cauchy_data(domain, basis, lam=20.0)
    n_pts, n_basis = len(cd.pts), len(basis)
    assert cd.Phi.shape == (n_pts, n_basis)
    assert cd.Phi_N.shape == (n_pts, n_basis)
    assert cd.Phi_T.shape == (n_pts, n_basis)
    assert cd.wts.shape == (n_pts,)


def test_assemble_kernel_uv_matches_direct_matrix_algebra():
    domain, basis = _rect_domain_basis()
    cd = _plain_bdry_cauchy_data(domain, basis, lam=20.0)
    weight = np.ones(len(cd.pts))
    A = assemble_kernel(cd, 'uv', weight)
    expected = (cd.Phi*(cd.wts*weight)[:, np.newaxis]).T@cd.Phi
    assert np.allclose(A, expected)
    assert np.allclose(A, A.T)  # symmetric for weight >= 0 / kernel 'uv'


def test_assemble_kernel_NN_TT_cr_shapes_and_symmetry():
    domain, basis = _rect_domain_basis()
    cd = _plain_bdry_cauchy_data(domain, basis, lam=20.0)
    weight = np.linspace(0.5, 1.5, len(cd.pts))  # arbitrary, not Rellich-specific
    n = len(basis)
    for kernel in ('NN', 'TT', 'cr'):
        A = assemble_kernel(cd, kernel, weight)
        assert A.shape == (n, n)
        assert np.all(np.isfinite(A))
        assert np.allclose(A, A.T)  # all four kernels are symmetric bilinear forms


def test_assemble_kernel_invalid_kernel_raises():
    domain, basis = _rect_domain_basis()
    cd = _plain_bdry_cauchy_data(domain, basis, lam=20.0)
    with pytest.raises(ValueError):
        assemble_kernel(cd, 'bogus', np.ones(len(cd.pts)))


def test_assemble_kernel_empty_cauchy_data_is_zero_matrix():
    domain, basis = _rect_domain_basis()
    empty_pts = PointSet(np.zeros(0, dtype=complex))
    cd = basis_cauchy_data(basis, 20.0, empty_pts, empty_pts, empty_pts, np.zeros(0))
    A = assemble_kernel(cd, 'uv', np.zeros(0))
    assert A.shape == (len(basis), len(basis))
    assert np.all(A == 0)


def test_external_style_custom_weight_hadamard_like():
    """Simulates an external (non-Rellich) consumer: its own node set, its
    own scalar weight standing in for a Hadamard-type boundary velocity V(s),
    built purely from the public basis_cauchy_data/assemble_kernel API."""
    domain, basis = _rect_domain_basis()
    lam = 25.0
    pts = domain.bdry_pts(40, kind='legendre', weights=True)
    normals = domain.bdry_normals(40, kind='legendre')
    tangents = domain.bdry_tangents(40, kind='legendre')
    cd = basis_cauchy_data(basis, lam, pts, normals, tangents, pts.wts)

    V = np.sin(np.linspace(0, 2*np.pi, len(cd.pts)))  # synthetic velocity field
    M = assemble_kernel(cd, 'NN', V)
    assert M.shape == (len(basis), len(basis))
    assert np.allclose(M, M.T)


def test_default_x0_is_bounding_box_center():
    domain, _ = _rect_domain_basis()
    x0 = default_x0(domain)
    assert x0 == pytest.approx(L/2 + 1j*H/2)


def test_graded_pts_per_seg_scales_with_basis_size():
    domain, basis = _rect_domain_basis()
    small = graded_pts_per_seg(domain, basis, mult=1, min_per_seg=1)
    large = graded_pts_per_seg(domain, basis, mult=4, min_per_seg=1)
    assert np.all(large >= small)


def test_graded_pts_per_seg_scales_with_lam_max():
    """The lam term (c_lam*sqrt(lam_max)*seg_len) should dominate a small basis's point count
    at high lam_max, tracking sqrt(lam_max) -- the gap this term fixes: point count used to be
    blind to how oscillatory the boundary Cauchy data gets at high eigenvalues."""
    domain, basis = _rect_domain_basis()
    low = graded_pts_per_seg(domain, basis, lam_max=1.0, mult=1, min_per_seg=1)
    high = graded_pts_per_seg(domain, basis, lam_max=1e4, mult=1, min_per_seg=1)
    assert np.all(high >= low)
    assert np.any(high > low)

    seg_lens = domain.seg_lens
    predicted = np.round(1.0*np.sqrt(1e4)*seg_lens).astype(int)  # c_lam default is 1.0
    assert np.array_equal(high, predicted)


def test_graded_pts_per_seg_scales_with_q():
    """A segment's Kress grading order should inflate its point count (correcting for grading
    clustering nodes near the corner instead of resolving the smooth mid-segment), even at
    fixed lam_max/mult."""
    domain, basis = _rect_domain_basis()
    n_segs = len(domain.bdry.segments)
    q0 = np.zeros(n_segs, dtype=int)
    q_max = np.full(n_segs, 12, dtype=int)
    low = graded_pts_per_seg(domain, basis, lam_max=1.0, q_seg=q0, mult=4, min_per_seg=1)
    high = graded_pts_per_seg(domain, basis, lam_max=1.0, q_seg=q_max, mult=4, min_per_seg=1)
    assert np.all(high >= low)
    assert np.any(high > low)

    beta = 0.2  # default
    expected_ratio = 1.0 + beta*(12 - 1)
    assert np.allclose(high/low, expected_ratio, atol=0.05)


def test_assemble_kernel_cols_subsets_match_full_matrix_slices():
    domain, basis = _rect_domain_basis()
    cd = _plain_bdry_cauchy_data(domain, basis, lam=20.0)
    weight = np.linspace(0.5, 1.5, len(cd.pts))
    full = assemble_kernel(cd, 'NN', weight)

    cols1 = np.array([0, 2, 5])
    cols2 = np.array([1, 3])
    block = assemble_kernel(cd, 'NN', weight, cols1=cols1, cols2=cols2)
    assert np.allclose(block, full[np.ix_(cols1, cols2)])

    sq = assemble_kernel(cd, 'NN', weight, cols1=cols1, cols2=cols1)
    assert np.allclose(sq, full[np.ix_(cols1, cols1)])


def _gram_from_quadrature(basis, lam, pts, normals, tangents, wts, x0):
    cd = basis_cauchy_data(basis, lam, pts, normals, tangents, wts)
    rN = complex_dot(pts - x0, normals)
    return assemble_kernel(cd, 'NN', rN)


# ── corner_grading_orders / build_boundary_quadrature (Kress graded mesh) ────

def test_corner_grading_orders_zero_for_rectangle():
    """A rectangle's corners are all right angles (integer exponents, entire) --
    no singularity to grade for anywhere."""
    domain, basis = _rect_domain_basis()
    q = corner_grading_orders(basis, domain)
    assert q.shape == (len(domain.corners),)
    assert np.all(q == 0)


def test_corner_grading_orders_nonzero_for_reentrant_corner():
    """L-shape has one genuinely singular (non-integer alpha=2/3) reentrant corner --
    that corner (and only that one) should get a positive grading order."""
    domain = L_shape(bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[6]*len(domain.corners))
    q = corner_grading_orders(basis, domain)
    assert np.any(q > 0)
    assert np.all(q <= 12)


def test_build_boundary_quadrature_shapes_and_masks():
    domain, basis = _rect_domain_basis()
    pts, normals, tangents, wts, dir_mask, neu_mask = build_boundary_quadrature(domain, basis)
    n = len(pts)
    assert normals.shape == (n,) and tangents.shape == (n,) and wts.shape == (n,)
    assert dir_mask.shape == (n,) and neu_mask.shape == (n,)
    assert np.all(np.isfinite(wts)) and np.all(wts > 0)
    # pure-Dirichlet domain: every point is on a Dirichlet segment, none Neumann
    assert np.all(dir_mask == 1.0)
    assert np.all(neu_mask == 0.0)


def test_build_boundary_quadrature_mixed_bc_masks_partition_points():
    domain = Polygon(np.array([0, L, L + 1j*H, 1j*H]), bc='dir')
    for seg, bc in zip(domain.bdry.segments, ['dir', 'neu', 'dir', 'neu']):
        seg.bc = bc
    basis = FourierBesselBasis.from_domain(domain, orders=[6, 6, 6, 6])
    _, _, _, _, dir_mask, neu_mask = build_boundary_quadrature(domain, basis)
    assert np.all(dir_mask + neu_mask == 1.0)
    assert dir_mask.sum() > 0 and neu_mask.sum() > 0


def test_gram_from_boundary_quadrature_node_doubling_stable_rectangle():
    """Certification test (doc Sec. 10.4): doubling the quadrature density should leave the
    assembled Gram matrix essentially unchanged once already reasonably resolved. No corner
    singularity here, so this exercises the plain-Gauss-Legendre path."""
    domain, basis = _rect_domain_basis()
    lam = rect_eig(1, 2, L, H)
    x0 = default_x0(domain)

    # mult=2/min_per_seg=6 (48 pts) vs mult=4/min_per_seg=12 (96 pts) is NOT yet past this
    # basis's convergence knee for a smooth entire integrand (diff ~3.5e-7, still actively
    # converging) -- comparing those would only show "didn't blow up," not "converged." Using
    # mult=4/8 and min_per_seg=12/24 (96 vs 192 pts) lands solidly past the knee (diff ~1e-14
    # empirically), so a tight tolerance here is a meaningful check.
    q1 = build_boundary_quadrature(domain, basis, mult=4, min_per_seg=12)
    q2 = build_boundary_quadrature(domain, basis, mult=8, min_per_seg=24)
    G1 = _gram_from_quadrature(basis, lam, *q1[:4], x0)
    G2 = _gram_from_quadrature(basis, lam, *q2[:4], x0)
    assert np.max(np.abs(G1 - G2)) < 1e-10


def test_gram_from_boundary_quadrature_node_doubling_stable_reentrant_corner():
    """Same certification, but on the L-shape's genuinely singular reentrant corner --
    exercises the Kress-graded path. Doubling both the grading margin and the point
    density should leave the Gram matrix essentially unchanged."""
    domain = L_shape(bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[6]*len(domain.corners))
    lam = 5.0
    x0 = default_x0(domain)

    # Empirically this pair is already deep in the converged regime (diff ~3e-13) --
    # tight tolerance here is a meaningful check, not a coincidence of a loose bound.
    q1 = build_boundary_quadrature(domain, basis, mult=2, min_per_seg=6, margin=2.0)
    q2 = build_boundary_quadrature(domain, basis, mult=4, min_per_seg=12, margin=4.0)
    G1 = _gram_from_quadrature(basis, lam, *q1[:4], x0)
    G2 = _gram_from_quadrature(basis, lam, *q2[:4], x0)
    assert np.max(np.abs(G1 - G2)) < 1e-9


def test_gram_from_boundary_quadrature_lam_aware_beats_lam_blind_at_high_lam():
    """The gap the lam term fixes: build_boundary_quadrature's node set is built once per basis
    and reused for every lam tried across a solve_interval search (see
    rellich.build_rellich_data's docstring), so it must stay accurate at high lam even though
    the basis itself hasn't grown. Sizing the quadrature for the correct (high) lam_max should
    land much closer to a doubled-density reference than deliberately under-stating lam_max --
    the failure mode the old, lam-blind heuristic was exposed to."""
    domain, basis = _rect_domain_basis()
    lam = rect_eig(8, 10, L, H)  # high eigenvalue relative to this basis's modest order
    x0 = default_x0(domain)

    q_aware = build_boundary_quadrature(domain, basis, lam_max=lam, mult=2, min_per_seg=6)
    q_blind = build_boundary_quadrature(domain, basis, lam_max=1.0, mult=2, min_per_seg=6)
    q_ref = build_boundary_quadrature(domain, basis, lam_max=lam, mult=8, min_per_seg=24)

    G_aware = _gram_from_quadrature(basis, lam, *q_aware[:4], x0)
    G_blind = _gram_from_quadrature(basis, lam, *q_blind[:4], x0)
    G_ref = _gram_from_quadrature(basis, lam, *q_ref[:4], x0)

    err_aware = np.max(np.abs(G_aware - G_ref))
    err_blind = np.max(np.abs(G_blind - G_ref))
    assert err_aware < err_blind


def test_gram_from_boundary_quadrature_node_doubling_stable_reentrant_corner_high_lam():
    """Same certification as the lam=5.0 reentrant-corner test above, but at a much higher lam
    -- exercises the lam term and the q-inflation term together on the genuinely singular
    corner."""
    domain = L_shape(bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[6]*len(domain.corners))
    lam = 300.0
    x0 = default_x0(domain)

    q1 = build_boundary_quadrature(domain, basis, lam_max=lam, mult=2, min_per_seg=6, margin=2.0)
    q2 = build_boundary_quadrature(domain, basis, lam_max=lam, mult=4, min_per_seg=12, margin=4.0)
    G1 = _gram_from_quadrature(basis, lam, *q1[:4], x0)
    G2 = _gram_from_quadrature(basis, lam, *q2[:4], x0)
    assert np.max(np.abs(G1 - G2)) < 1e-6


def test_gram_from_boundary_quadrature_stable_large_q():
    """Isolates the q-effect from the lam-effect: forces the reentrant corner's grading order
    to q_max at a low, unremarkable lam, and checks node-doubling stability still holds."""
    domain = L_shape(bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[6]*len(domain.corners))
    lam = 5.0
    x0 = default_x0(domain)

    q1 = build_boundary_quadrature(domain, basis, lam_max=lam, mult=2, min_per_seg=6,
                                   margin=2.0, q_min=12, q_max=12)
    q2 = build_boundary_quadrature(domain, basis, lam_max=lam, mult=4, min_per_seg=12,
                                   margin=4.0, q_min=12, q_max=12)
    G1 = _gram_from_quadrature(basis, lam, *q1[:4], x0)
    G2 = _gram_from_quadrature(basis, lam, *q2[:4], x0)
    assert np.max(np.abs(G1 - G2)) < 1e-8


def test_gram_from_boundary_quadrature_finite_at_high_margin_and_density():
    """Regression test for a node-collapse bug: at high margin/point density, the Kress
    grading map used to be able to push a quadrature node's tau far enough below float64
    epsilon that a segment's linear parametrization (1-tau)*p0 + tau*pf rounded to exactly
    p0 -- silently landing a node ON the corner (r=0), producing NaN in the assembled Gram
    matrix via the corner-relative 1/r terms in FourierBesselBasis's derivatives. Fixed by
    cached_kressgauss's tau clamp (lappy.quad); this exercises margin/mult/min_per_seg
    values well past what corner_grading_orders' defaults would produce, to confirm the fix
    holds generally rather than only at the specific values that first exposed the bug."""
    domain = L_shape(bc='dir')
    basis = FourierBesselBasis.from_domain(domain, orders=[6]*len(domain.corners))
    lam = 5.0
    x0 = default_x0(domain)

    for mult, min_per_seg, margin, lam_max in [(8, 24, 6.0, lam), (16, 48, 8.0, lam),
                                               (32, 96, 8.0, lam), (16, 48, 8.0, 5000.0)]:
        pts, normals, tangents, wts, dir_mask, neu_mask = build_boundary_quadrature(
            domain, basis, lam_max=lam_max, mult=mult, min_per_seg=min_per_seg, margin=margin)
        G = _gram_from_quadrature(basis, lam, pts, normals, tangents, wts, x0)
        assert np.all(np.isfinite(G)), (mult, min_per_seg, margin, lam_max)


def test_build_boundary_quadrature_multibasis_fb_plus_fs():
    """FourierBesselBasis + FundamentalBasis combination: FS columns are regular everywhere and
    should compose with no special-casing (corner_terms() marks them -1, contributing no
    grading order anywhere)."""
    domain, basis_fb = _rect_domain_basis()
    basis_fs = FundamentalBasis(np.array([10+10j, -10-10j, 10-10j]), orders=2)
    basis = basis_fb + basis_fs
    lam = rect_eig(1, 2, L, H)
    x0 = default_x0(domain)

    pts, normals, tangents, wts, dir_mask, neu_mask = build_boundary_quadrature(domain, basis)
    G = _gram_from_quadrature(basis, lam, pts, normals, tangents, wts, x0)
    n = len(basis)
    assert G.shape == (n, n)
    assert np.all(np.isfinite(G))
    assert np.allclose(G, G.T)


def test_build_boundary_quadrature_single_basis_evaluation_per_call():
    """The whole point of the Kress-graded rewrite: one basis_cauchy_data call serves the
    entire boundary (no more per-exponent-group re-evaluation), confirmed by instrumenting
    basis_cauchy_data and checking it's called exactly once for a full Gram build."""
    import lappy.rellich as rellich_mod
    from lappy.rellich import build_rellich_data, rellich_gram_basis

    domain, basis = _rect_domain_basis()
    lam = rect_eig(1, 2, L, H)
    rellich_data = build_rellich_data(domain, basis)

    calls = []
    original = rellich_mod.basis_cauchy_data
    def record_call(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    rellich_mod.basis_cauchy_data = record_call
    try:
        rellich_gram_basis(basis, lam, rellich_data)
    finally:
        rellich_mod.basis_cauchy_data = original

    assert len(calls) == 1
