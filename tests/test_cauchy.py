"""Tests for lappy.cauchy — the generic Cauchy-data evaluation and
boundary-integral kernel-assembler API that lappy.rellich (and, potentially,
code outside lappy building Hadamard-type quantities) is built on."""

import numpy as np
import pytest

from lappy import Polygon
from lappy.bases import FourierBesselBasis, FundamentalBasis
from lappy.reference import rect_eig
from lappy.geometry import PointSet
from lappy.utils import complex_dot
from lappy.cauchy import (
    basis_cauchy_data, assemble_kernel, default_x0, graded_pts_per_seg,
    panel_radius, singular_gram,
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


def test_panel_radius_positive_and_bounded_by_adjacent_segments():
    domain, _ = _rect_domain_basis()
    for c in range(len(domain.corners)):
        R = panel_radius(domain, c, frac=0.4)
        seg_idx = domain.corner_idx[c]
        n_segs = len(domain.bdry.segments)
        adj_lens = [domain.seg_lens[seg_idx], domain.seg_lens[(seg_idx - 1) % n_segs]]
        assert 0 < R <= 0.4*min(adj_lens) + 1e-12


# ── singular_gram: SS/SR/RS/RR quadrature (docs/rellich_hadamard_mps.pdf) ────

def _weight_rN(x0):
    def weight_fn(pts, normals, tangents):
        return complex_dot(pts - x0, normals)
    return weight_fn


def test_singular_gram_node_doubling_stable():
    """Certification test (Sec. 9.4-inspired): doubling the quadrature density should leave the
    assembled Gram matrix essentially unchanged once already reasonably resolved."""
    domain, basis = _rect_domain_basis()
    lam = rect_eig(1, 2, L, H)
    x0 = default_x0(domain)
    seg_mask = np.array([True]*4)
    weight_fn = _weight_rN(x0)

    G1 = singular_gram(basis, domain, lam, 'NN', weight_fn, seg_mask=seg_mask,
                       group_pts=24, bulk_mult=3, bulk_min_per_seg=6)
    G2 = singular_gram(basis, domain, lam, 'NN', weight_fn, seg_mask=seg_mask,
                       group_pts=48, bulk_mult=6, bulk_min_per_seg=12)
    assert np.max(np.abs(G1 - G2)) < 1e-8


def test_singular_gram_multibasis_fb_plus_fs():
    """FourierBesselBasis + FundamentalBasis combination: FS columns are regular everywhere and
    should compose into the RR path with no special-casing (corner_terms() marks them -1)."""
    domain, basis_fb = _rect_domain_basis()
    basis_fs = FundamentalBasis(np.array([10+10j, -10-10j, 10-10j]), orders=2)
    basis = basis_fb + basis_fs
    lam = rect_eig(1, 2, L, H)
    x0 = default_x0(domain)
    seg_mask = np.array([True]*4)

    G = singular_gram(basis, domain, lam, 'NN', _weight_rN(x0), seg_mask=seg_mask,
                      group_pts=24, bulk_mult=3, bulk_min_per_seg=6)
    n = len(basis)
    assert G.shape == (n, n)
    assert np.all(np.isfinite(G))
    assert np.allclose(G, G.T)
