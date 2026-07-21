"""Unit tests for lappy/geometry.py."""
import pytest
import numpy as np
from lappy import PointSet, Domain, Polygon, ParametricSegment, LineSegment, MultiSegment
from lappy.geometry import SplineSegment
from lappy.geometry import (
    rect, disk, L_shape, GWW1, GWW2, H_shape, reg_ngon,
    disk_sector, iso_right_tri, iso_tri, mushroom, cut_square, chevron,
    corner_branch_cut_rays, segment_intersection, spiral,
    free_ray_from_point, corner_branch_cut_polyline,
    ellipse, adaptive_arclength_table, adaptive_polyline, _estimate_length,
)


# ---------------------------------------------------------------------------
# TestPointSet
# ---------------------------------------------------------------------------

class TestPointSet:
    def test_creation_no_weights(self):
        pts_arr = np.array([1+0j, 2+1j, 3+2j])
        ps = PointSet(pts_arr)
        assert np.allclose(ps.pts, pts_arr)
        assert len(ps) == 3

    def test_creation_with_weights(self):
        pts_arr = np.array([1+0j, 2+1j])
        wts_arr = np.array([0.3, 0.7])
        ps = PointSet(pts_arr, wts_arr)
        assert np.allclose(ps.wts, wts_arr)
        assert ps.wts.shape == pts_arr.shape
        assert np.allclose(ps.sqrt_wts.flatten(), np.sqrt(wts_arr))

    def test_weight_shape_mismatch_raises(self):
        pts = np.array([1+0j, 2+0j])
        weights = np.array([0.5])  # wrong length
        with pytest.raises(ValueError):
            PointSet(pts, weights)

    def test_immutability(self):
        ps = PointSet(np.array([1+0j]))
        assert ps.pts.flags.writeable == False

        ps_w = PointSet(np.array([1+0j]), weights=np.array([0.5]))
        assert ps_w.pts.flags.writeable == False
        assert ps_w.wts.flags.writeable == False

    def test_x_y_properties(self):
        pts_arr = np.array([1+2j, 3+4j])
        ps = PointSet(pts_arr)
        assert np.allclose(ps.x, pts_arr.real)
        assert np.allclose(ps.y, pts_arr.imag)

    def test_hash_stability(self):
        ps = PointSet(np.array([1+0j, 2+0j]))
        assert hash(ps) == hash(ps)

    def test_add_no_weights(self):
        pts1 = PointSet(np.array([1+0j, 2+0j]))
        pts2 = PointSet(np.array([3+0j]))
        result = pts1 + pts2
        assert not hasattr(result, 'wts')
        assert len(result) == 3

    def test_add_both_weights(self):
        pts1 = PointSet(np.array([1+0j]), weights=np.array([0.5]))
        pts2 = PointSet(np.array([2+0j, 3+0j]), weights=np.array([0.3, 0.2]))
        result = pts1 + pts2
        assert hasattr(result, 'wts')
        assert np.allclose(result.wts, [0.5, 0.3, 0.2])

    def test_add_one_sided_weights(self):
        # self has weights, other doesn't
        pts1 = PointSet(np.array([1+0j]), weights=np.array([0.5]))
        pts2 = PointSet(np.array([2+0j]))
        result = pts1 + pts2
        assert hasattr(result, 'wts')
        assert np.allclose(result.wts, [0.5, 1.0])

        # self doesn't have weights, other does
        pts3 = PointSet(np.array([3+0j]))
        pts4 = PointSet(np.array([4+0j]), weights=np.array([0.7]))
        result2 = pts3 + pts4
        assert hasattr(result2, 'wts')
        assert np.allclose(result2.wts, [1.0, 0.7])


# ---------------------------------------------------------------------------
# TestLineSegment
# ---------------------------------------------------------------------------

class TestLineSegment:
    def test_identical_points_raises(self):
        with pytest.raises(ValueError):
            LineSegment(0, 0)

    def test_len(self):
        assert np.isclose(LineSegment(0, 1).len, 1.0)
        assert np.isclose(LineSegment(0, 1j).len, 1.0)
        assert np.isclose(LineSegment(0, 1+1j).len, np.sqrt(2))

    def test_p_endpoints(self):
        seg = LineSegment(0, 1+1j)
        assert np.isclose(seg.p(0), 0+0j)
        assert np.isclose(seg.p(1), 1+1j)
        assert np.isclose(seg.p(0.5), 0.5+0.5j)

    def test_dp_constant(self):
        seg = LineSegment(0, 1+1j)
        tau = np.linspace(0, 1, 10)
        dp_vals = seg.dp(tau)
        assert np.allclose(dp_vals, 1+1j)

    def test_tangent_unit_norm(self):
        seg = LineSegment(0, 1+1j)
        tau = np.linspace(0, 1, 10)
        assert np.allclose(np.abs(seg.T(tau)), 1.0, atol=1e-12)

    def test_normal_unit_norm(self):
        seg = LineSegment(0, 1+1j)
        tau = np.linspace(0, 1, 10)
        assert np.allclose(np.abs(seg.N(tau)), 1.0, atol=1e-12)

    def test_tangent_normal_orthogonal(self):
        seg = LineSegment(0, 1+1j)
        tau = np.linspace(0, 1, 10)
        T = seg.T(tau)
        N = seg.N(tau)
        assert np.allclose((T * np.conj(N)).real, 0.0, atol=1e-12)

    def test_normal_points_right(self):
        # For seg 0→1 (rightward), outward normal for CCW boundary is downward = -1j
        seg = LineSegment(0, 1)
        tau = np.array([0.5])
        assert np.allclose(seg.N(tau), -1j)

    def test_pts_length(self):
        seg = LineSegment(0, 1)
        ps = seg.pts(10)
        assert len(ps) == 10

    def test_pts_kinds(self):
        seg = LineSegment(0, 1)
        for kind in ('legendre', 'chebyshev', 'even'):
            ps = seg.pts(8, kind=kind)
            assert len(ps) == 8

    def test_pts_with_weights(self):
        seg = LineSegment(0, 1)
        ps = seg.pts(10, weights=True)
        assert hasattr(ps, 'wts')
        assert len(ps.wts) == 10

    def test_pts_jacobi_default_exponents(self):
        # a=b=0 reduces to a non-singular weight; should still behave like a valid quadrature rule
        seg = LineSegment(0, 1)
        ps = seg.pts(8, kind='jacobi')
        assert len(ps) == 8

    def test_pts_jacobi_integral_accuracy(self):
        # Gauss-Jacobi quadrature integrates tau^a*(1-tau)^b exactly (it's the weight
        # function itself): int_0^1 tau^-0.5 (1-tau)^0 dtau = B(0.5, 1) = 2
        seg = LineSegment(0, 1)
        ps = seg.pts(20, kind='jacobi', weights=True, a=-0.5, b=0)
        tau = ps.x
        integral = np.sum(ps.wts * tau**(-0.5))
        assert np.isclose(integral, 2.0, rtol=1e-10)

    def test_tangents_normals_jacobi_kind(self):
        seg = LineSegment(0, 1)
        assert len(seg.tangents(5, kind='jacobi', a=0.5, b=-0.5)) == 5
        assert len(seg.normals(5, kind='jacobi', a=0.5, b=-0.5)) == 5

    def test_tangents_normals_length(self):
        seg = LineSegment(0, 1)
        assert len(seg.tangents(5)) == 5
        assert len(seg.normals(5)) == 5

    def test_is_simple(self):
        assert LineSegment(0, 1).is_simple == True

    def test_is_closed(self):
        assert LineSegment(0, 1).is_closed == False

    def test_dist_to_midpoint(self):
        seg = LineSegment(0, 1)
        assert np.isclose(seg.dist(0.5+0j), 0.0, atol=1e-6)

    def test_dist_to_perpendicular(self):
        seg = LineSegment(0, 1)
        assert np.isclose(seg.dist(0.5+2j), 2.0, rtol=1e-4)

    def test_to_splineseg(self):
        seg = LineSegment(0, 1)
        ss = seg.to_splineseg()
        assert isinstance(ss, SplineSegment)
        assert np.isclose(ss.p0, 0+0j, atol=1e-10)
        assert np.isclose(ss.pf, 1+0j, atol=1e-10)

    def test_add_lineseg(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        assert isinstance(seg1 + seg2, MultiSegment)

# ---------------------------------------------------------------------------
# TestParametricSegment
# ---------------------------------------------------------------------------

class TestParametricSegment:
    def test_tf_le_t0_raises(self):
        with pytest.raises(ValueError):
            ParametricSegment(
                lambda t: t+0j, lambda t: np.ones_like(t)+0j, 1, 0,
                val_simple=False
            )

    def test_circle_len(self, unit_disk_seg):
        assert np.isclose(unit_disk_seg.len, 2*np.pi, rtol=1e-5)

    def test_disk_p_endpoints(self, unit_disk_seg):
        assert np.isclose(unit_disk_seg.p(0), 1+0j, atol=1e-6)
        assert np.isclose(unit_disk_seg.p(1), 1+0j, atol=1e-6)

    def test_tangent_unit_norm(self, unit_disk_seg):
        tau = np.linspace(0.05, 0.95, 10)
        assert np.allclose(np.abs(unit_disk_seg.T(tau)), 1.0, atol=1e-12)

    def test_normal_unit_norm(self, unit_disk_seg):
        tau = np.linspace(0.05, 0.95, 10)
        assert np.allclose(np.abs(unit_disk_seg.N(tau)), 1.0, atol=1e-12)

    def test_tangent_normal_orthogonal(self, unit_disk_seg):
        tau = np.linspace(0.05, 0.95, 10)
        T = unit_disk_seg.T(tau)
        N = unit_disk_seg.N(tau)
        assert np.allclose((T * np.conj(N)).real, 0.0, atol=1e-12)

    def test_arc_length_parametrization(self, unit_disk_seg):
        # After reparameterization, |dp/dtau| == len (constant speed)
        tau = np.linspace(0.05, 0.95, 10)
        dp_mag = np.abs(unit_disk_seg.dp(tau))
        assert np.allclose(dp_mag, unit_disk_seg.len, rtol=1e-4)

    def test_is_closed(self, unit_disk_seg):
        assert unit_disk_seg.is_closed == True

    def test_dist(self, unit_disk_seg):
        # Distance from origin to unit disk == 1.0
        assert np.isclose(unit_disk_seg.dist(0+0j), 1.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# TestSplineSegment
# ---------------------------------------------------------------------------

class TestSplineSegment:
    def test_interp_from_pts_endpoints(self):
        pts = np.array([0+0j, 0.5+0.3j, 1+0j])
        seg = SplineSegment.interp_from_pts(pts)
        assert np.isclose(seg.p0, pts[0], atol=1e-8)
        assert np.isclose(seg.pf, pts[-1], atol=1e-8)

    def test_interp_from_pts_len(self):
        # Collinear points: natural-BC cubic spline reduces to line, len == 1
        pts = np.array([0+0j, 0.5+0j, 1+0j])
        seg = SplineSegment.interp_from_pts(pts)
        assert np.isclose(seg.len, 1.0, rtol=1e-4)

    def test_to_splineseg_identity(self):
        pts = np.array([0+0j, 0.5+0.3j, 1+0j])
        seg = SplineSegment.interp_from_pts(pts)
        assert seg.to_splineseg() is seg

    def test_from_lineseg(self):
        seg = LineSegment(0, 1)
        ss = seg.to_splineseg()
        assert isinstance(ss, SplineSegment)
        assert np.isclose(ss.p0, 0+0j, atol=1e-10)
        assert np.isclose(ss.pf, 1+0j, atol=1e-10)
        assert np.isclose(ss.len, 1.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestMultiSegment
# ---------------------------------------------------------------------------

class TestMultiSegment:
    def test_non_segment_raises(self):
        with pytest.raises(TypeError):
            MultiSegment([LineSegment(0, 1), "not a segment"])

    def test_from_vertices_count(self):
        vertices = np.array([0, 1, 1+1j, 1j])
        ms = MultiSegment.from_vertices(vertices, make_closed=True)
        assert len(ms.segments) == 4

    def test_from_vertices_closed(self):
        vertices = np.array([0, 1, 1+1j, 1j])
        ms = MultiSegment.from_vertices(vertices, make_closed=True)
        assert ms.is_closed == True

    def test_len(self, unit_square_domain):
        ms = unit_square_domain.bdry
        assert np.isclose(ms.len, 4.0)
        seg_sum = sum(seg.len for seg in ms.segments)
        assert np.isclose(ms.len, seg_sum)

    def test_is_polyline(self):
        vertices = np.array([0, 1, 1+1j, 1j])
        ms = MultiSegment.from_vertices(vertices)
        assert ms.is_polyline == True

        # Mixed: one ParametricSegment → not a polyline
        half_disk = ParametricSegment(
            lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t),
            0, np.pi, val_simple=False
        )
        ms2 = MultiSegment([LineSegment(0, 1), half_disk])
        assert ms2.is_polyline == False

    def test_corners_unit_square(self, unit_square_domain):
        corners = unit_square_domain.bdry.corners
        expected = np.array([0, 1, 1+1j, 1j])
        assert len(corners) == 4
        for c in corners:
            assert any(np.isclose(c, e) for e in expected)

    def test_corner_angles_unit_square(self, unit_square_domain):
        angle0, angle1 = unit_square_domain.bdry.corner_angles
        # Wedge angle: (angle1 - angle0) mod 2π == π/2 for all corners
        wedge = (angle1 - angle0) % (2*np.pi)
        assert np.allclose(wedge, np.pi/2, atol=1e-10)

    def test_pts_scalar_n(self, unit_square_domain):
        ps = unit_square_domain.bdry.pts(5)
        assert len(ps) == 4 * 5

    def test_pts_array_n(self, unit_square_domain):
        ns = np.array([3, 5, 7, 2])
        ps = unit_square_domain.bdry.pts(ns)
        assert len(ps) == ns.sum()

    def test_pts_with_weights(self, unit_square_domain):
        ps = unit_square_domain.bdry.pts(5, weights=True)
        assert hasattr(ps, 'wts')

    def test_pts_jacobi_scalar_exponents_broadcast(self, unit_square_domain):
        # scalar a, b should broadcast to every segment, same as scalar N
        ps = unit_square_domain.bdry.pts(5, kind='jacobi', weights=True, a=-0.5, b=0)
        assert len(ps) == 4 * 5

    def test_pts_jacobi_per_segment_exponents(self, unit_square_domain):
        bdry = unit_square_domain.bdry
        n_seg = len(bdry.segments)
        a = np.zeros(n_seg)
        a[0] = -0.5  # only the first segment gets a singular left endpoint
        ps_mixed = bdry.pts(6, kind='jacobi', weights=True, a=a, b=0)
        ps_uniform = bdry.pts(6, kind='jacobi', weights=True, a=0, b=0)
        # weights on the first segment should differ from the uniform (a=0) case,
        # while later segments (unaffected by 'a') should match
        assert not np.allclose(ps_mixed.wts[:6], ps_uniform.wts[:6])
        assert np.allclose(ps_mixed.wts[6:], ps_uniform.wts[6:])

    def test_dist_from_interior_point(self, unit_square_domain):
        assert np.isclose(unit_square_domain.bdry.dist(0.5+0.5j), 0.5, rtol=1e-3)

    def test_dist_from_corner(self, unit_square_domain):
        assert np.isclose(unit_square_domain.bdry.dist(0+0j), 0.0, atol=1e-5)

    def test_bcs(self, unit_square_domain):
        bcs = unit_square_domain.bdry.bcs
        assert len(bcs) == 4
        assert all(bc == 0.0 for bc in bcs)  # 'dir' → 0.0

    def test_validate_closed_raises(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        with pytest.raises(ValueError):
            MultiSegment([seg1, seg2], val_closed=True)


# ---------------------------------------------------------------------------
# TestDomain
# ---------------------------------------------------------------------------

class TestDomain:
    def test_non_multiseg_raises(self):
        with pytest.raises(TypeError):
            Domain("not a multisegment")

    def test_open_boundary_raises(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])
        with pytest.raises(ValueError):
            Domain(ms)

    def test_area_unit_square(self, unit_square_domain):
        assert np.isclose(unit_square_domain.area, 1.0, rtol=1e-8)

    def test_area_rectangle(self, rect_domain):
        assert np.isclose(rect_domain.area, 2.0, rtol=1e-8)

    def test_perimeter_unit_square(self, unit_square_domain):
        assert np.isclose(unit_square_domain.perimeter, 4.0, rtol=1e-8)

    def test_diameter_unit_square(self, unit_square_domain):
        assert np.isclose(unit_square_domain.diameter, np.sqrt(2), rtol=1e-5)

    def test_area_cached(self, unit_square_domain):
        a1 = unit_square_domain.area
        a2 = unit_square_domain.area
        assert a1 is a2

    def test_contains_interior(self, unit_square_domain):
        pts = np.array([0.5+0.5j])
        assert unit_square_domain.contains(pts)[0]

    def test_contains_exterior(self, unit_square_domain):
        pts = np.array([2+2j])
        assert not unit_square_domain.contains(pts)[0]

    def test_contains_array(self, unit_square_domain):
        interior = np.array([0.3+0.3j, 0.7+0.7j, 0.5+0.5j])
        exterior = np.array([2+0j, -1+0j, 0.5+2j])
        all_pts = np.concatenate([interior, exterior])
        result = unit_square_domain.contains(all_pts)
        assert np.all(result[:3])
        assert np.all(~result[3:])

    def test_int_pts_random_count(self, unit_square_domain):
        pts = unit_square_domain.int_pts(method='random', npts_rand=20)
        assert len(pts) == 20

    def test_int_pts_random_interior(self, unit_square_domain):
        pts = unit_square_domain.int_pts(method='random', npts_rand=10)
        assert np.all(unit_square_domain.contains(pts.pts))

    def test_int_pts_random_weights(self, unit_square_domain):
        pts = unit_square_domain.int_pts(method='random', weights=True, npts_rand=20)
        assert hasattr(pts, 'wts')
        assert np.isclose(pts.wts.sum(), unit_square_domain.area, rtol=1e-6)

    def test_int_pts_mesh_weights(self, unit_square_domain):
        pts = unit_square_domain.int_pts(method='mesh', weights=True)
        assert hasattr(pts, 'wts')
        assert np.isclose(pts.wts.sum(), unit_square_domain.area, rtol=1e-2)

    def test_bdry_pts_count(self, unit_square_domain):
        ps = unit_square_domain.bdry_pts(5)
        assert len(ps) == 4 * 5

    def test_bdry_data_shapes(self, unit_square_domain):
        pts, normals, bc_param = unit_square_domain.bdry_data(5)
        assert len(pts) == len(normals)
        assert len(pts) == len(bc_param)

    def test_max_dist_unit_square(self, unit_square_domain):
        # Distance from center (0.5+0.5j) to farthest boundary point (corners) = sqrt(2)/2
        d = unit_square_domain.max_dist(0.5+0.5j)
        assert np.isclose(d, np.sqrt(2)/2, rtol=1e-5)

    def test_bc_type(self, unit_square_domain):
        assert unit_square_domain.bc_type == 'dir'

    def test_inradius_unit_square(self, unit_square_domain):
        assert np.isclose(unit_square_domain.inradius, 0.5, rtol=1e-4)

    def test_inradius_cached(self, unit_square_domain):
        r1 = unit_square_domain.inradius
        r2 = unit_square_domain.inradius
        assert r1 is r2


# ---------------------------------------------------------------------------
# TestPolygon
# ---------------------------------------------------------------------------

class TestPolygon:
    def test_both_args_raises(self):
        vertices = np.array([0, 1, 1+1j, 1j])
        bdry = MultiSegment.from_vertices(vertices)
        with pytest.raises(ValueError):
            Polygon(vertices=vertices, bdry=bdry)

    def test_neither_arg_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Polygon()

    def test_area_unit_square_exact(self, unit_square_domain):
        assert unit_square_domain.area == 1.0

    def test_area_rectangle_exact(self, rect_domain):
        assert rect_domain.area == 2.0

    def test_area_triangle_exact(self, right_triangle):
        assert right_triangle.area == 6.0

    def test_diameter_unit_square(self, unit_square_domain):
        assert np.isclose(unit_square_domain.diameter, np.sqrt(2), rtol=1e-12)

    def test_diameter_triangle(self, right_triangle):
        assert right_triangle.diameter == 5.0

    def test_interior_angles_square(self, unit_square_domain):
        angles = unit_square_domain.int_angles
        assert np.allclose(angles, np.pi/2, atol=1e-12)

    def test_interior_angles_sum(self, unit_square_domain, rect_domain, right_triangle):
        for poly in [unit_square_domain, rect_domain, right_triangle]:
            n = poly.n_vertices
            assert np.isclose(poly.int_angles.sum(), (n - 2)*np.pi, rtol=1e-10)

    def test_edge_lengths_square(self, unit_square_domain):
        assert np.allclose(unit_square_domain.edge_lengths, 1.0, atol=1e-12)

    def test_edge_lengths_triangle(self, right_triangle):
        # vertices [0, 3, 3+4j] → edges 0→3, 3→3+4j, 3+4j→0 → lengths [3, 4, 5]
        expected = [3.0, 4.0, 5.0]
        assert np.allclose(right_triangle.edge_lengths, expected, atol=1e-12)

    def test_corner_idx_all_vertices(self, unit_square_domain):
        corner_idx = unit_square_domain.corner_idx
        n = unit_square_domain.n_vertices
        assert np.array_equal(corner_idx, np.arange(n))

    def test_int_pts_random_interior(self, unit_square_domain):
        pts = unit_square_domain.int_pts(method='random', npts_rand=15)
        assert np.all(unit_square_domain.contains(pts.pts))

    def test_int_pts_random_weights_sum(self, unit_square_domain):
        pts = unit_square_domain.int_pts(method='random', weights=True, npts_rand=20)
        assert np.isclose(pts.wts.sum(), unit_square_domain.area, rtol=1e-6)

    def test_int_pts_mesh_weights_sum(self, unit_square_domain):
        pts = unit_square_domain.int_pts(method='mesh', weights=True)
        assert np.isclose(pts.wts.sum(), unit_square_domain.area, rtol=1e-2)

    def test_n_vertices(self, unit_square_domain):
        assert unit_square_domain.n_vertices == 4
        assert unit_square_domain.n_sides == 4
        assert unit_square_domain.n_vertices == len(unit_square_domain.vertices)

    def test_inradius_unit_square(self, unit_square_domain):
        assert np.isclose(unit_square_domain.inradius, 0.5, rtol=1e-8)

    def test_inradius_rectangle(self, rect_domain):
        # 2×1 rectangle: inradius = 0.5
        assert np.isclose(rect_domain.inradius, 0.5, rtol=1e-8)

    def test_inradius_right_triangle(self, right_triangle):
        # 3-4-5 triangle: r = Area / semi_perimeter = 6 / 6 = 1.0
        assert np.isclose(right_triangle.inradius, 1.0, rtol=1e-8)

    def test_translate_polygon(self):
        p = Polygon([0, 1, 1+1j, 1j])
        shifted = p + (3+4j)
        assert isinstance(shifted, Polygon)
        assert np.isclose(shifted.area, p.area)

    def test_scale_polygon(self):
        p = Polygon([0, 1, 1+1j, 1j])
        scaled = p * 2
        assert isinstance(scaled, Polygon)
        assert np.isclose(scaled.area, 4 * p.area)

    def test_radd_polygon(self):
        p = Polygon([0, 1, 1+1j, 1j])
        shifted1 = p + (3+4j)
        shifted2 = (3+4j) + p
        assert np.isclose(shifted1.area, shifted2.area)
        assert np.allclose(shifted1.vertices, shifted2.vertices)

    def test_rmul_polygon(self):
        p = Polygon([0, 1, 1+1j, 1j])
        scaled1 = p * 2
        scaled2 = 2 * p
        assert isinstance(scaled2, Polygon)
        assert np.allclose(scaled1.vertices, scaled2.vertices)


# ---------------------------------------------------------------------------
# TestGeometryFixes -- design-spec review follow-up fixes
# ---------------------------------------------------------------------------

class TestGeometryFixes:
    # --- Polygon(vertices=..., val_simple=True) validation ---

    def test_bowtie_vertices_raises_with_val_simple(self):
        # self-intersecting "bowtie" quadrilateral
        bowtie = np.array([0, 1, 1j, 1+1j])
        with pytest.raises(ValueError):
            Polygon(bowtie, val_simple=True)

    def test_bowtie_vertices_allowed_without_val_simple(self):
        bowtie = np.array([0, 1, 1j, 1+1j])
        p = Polygon(bowtie, val_simple=False, val_orientation=False)
        assert isinstance(p, Polygon)

    def test_bowtie_bdry_raises_with_val_simple(self):
        bowtie = np.array([0, 1, 1j, 1+1j])
        bdry = MultiSegment.from_vertices(bowtie)
        with pytest.raises(ValueError):
            Polygon(bdry=bdry, val_simple=True)

    # --- CCW / positive orientation validation ---

    def test_cw_polygon_vertices_raises(self):
        cw_square = np.array([0, -1j, -1-1j, -1])
        with pytest.raises(ValueError):
            Polygon(cw_square)

    def test_cw_polygon_val_orientation_false_bypasses(self):
        cw_square = np.array([0, -1j, -1-1j, -1])
        p = Polygon(cw_square, val_orientation=False)
        assert p.area == pytest.approx(1.0)

    def test_cw_domain_raises(self):
        cw_square = np.array([0, -1j, -1-1j, -1])
        bdry = MultiSegment.from_vertices(cw_square)
        with pytest.raises(ValueError):
            Domain(bdry)

    def test_cw_domain_val_orientation_false_bypasses(self):
        cw_square = np.array([0, -1j, -1-1j, -1])
        bdry = MultiSegment.from_vertices(cw_square)
        d = Domain(bdry, val_orientation=False)
        assert isinstance(d, Domain)

    def test_polyline_signed_area_matches_gl_sign(self, disk_domain, sector_domain):
        # sign of the cheap adaptive-polyline shoelace estimate must agree with
        # the composite-quadrature integral used for .area
        for d in (disk_domain, sector_domain):
            assert np.sign(d._polyline_signed_area()) == np.sign(d._signed_area())

    def test_signed_area_composite_quadrature_resolves_oscillation(self):
        # a boundary with far more oscillations than a single fixed-order GL
        # rule per segment can resolve. _signed_area uses composite 5-point
        # GL quadrature over the adaptive polyline partition (rather than one
        # rule spanning the whole segment), so it should stay accurate
        # regardless of oscillation frequency -- unlike the old flat-order
        # rule, which was off by >2x on this exact case (see git history).
        # The polyline/shoelace estimate (linear between polyline nodes,
        # ignoring curvature within each panel) is a cruder, cheaper
        # approximation -- accurate enough for a sign check, not for .area.
        k, amp = 39, 0.95
        r = lambda t: 1 + amp*np.cos(k*t)
        dr = lambda t: -amp*k*np.sin(k*t)
        p = lambda t: r(t)*np.exp(1j*t)
        dp = lambda t: (dr(t) + 1j*r(t))*np.exp(1j*t)
        seg = ParametricSegment(p, dp, 0, 2*np.pi, val_closed=True)
        d = Domain(MultiSegment([seg], val_simple=False), val_simple=False, val_orientation=False)

        # analytic area of a polar curve r(theta): (1/2) int r(theta)^2 dtheta
        analytic_area = np.pi*(1 + amp**2/2)
        composite_area = abs(d._signed_area())
        poly_area = abs(d._polyline_signed_area())

        assert abs(composite_area - analytic_area)/analytic_area < 1e-2
        assert abs(poly_area - analytic_area)/analytic_area > 5e-2

    def test_polygon_area_always_positive(self):
        # Polygon._compute_area must not leak a negative (signed) shoelace area
        cw_square = np.array([0, -1j, -1-1j, -1])
        p = Polygon(cw_square, val_orientation=False)
        assert p.area > 0

    # --- Domain.bdry_data weights propagation ---

    def test_bdry_data_weights_propagate_to_normals(self, unit_square_domain):
        bdry_pts, bdry_normals, bc_param = unit_square_domain.bdry_data(10, weights=True)
        assert hasattr(bdry_pts, 'wts') and bdry_pts.wts is not None
        assert hasattr(bdry_normals, 'wts') and bdry_normals.wts is not None

    def test_bdry_data_weights_false_by_default(self, unit_square_domain):
        bdry_pts, bdry_normals, bc_param = unit_square_domain.bdry_data(10)
        assert not hasattr(bdry_pts, 'wts')
        assert not hasattr(bdry_normals, 'wts')

    # --- ParametricSegment laziness ---

    def test_parametric_segment_construction_is_lazy(self):
        seg = ParametricSegment(lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t), 0, 2*np.pi)
        assert seg._len is None

    def test_parametric_segment_len_populates_lazily(self):
        seg = ParametricSegment(lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t), 0, 2*np.pi)
        assert seg.len == pytest.approx(2*np.pi, rel=1e-4)
        assert seg._len is not None

    def test_parametric_segment_polyline_populates_lazily(self):
        seg = ParametricSegment(lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t), 0, 2*np.pi)
        assert seg._len is None
        pts = seg.polyline_pts
        assert seg._len is not None
        assert len(pts) > 2

    # --- MultiSegment.to_splinesegs ---

    def test_to_splinesegs_all_spline(self, unit_square_domain):
        splinesegs = unit_square_domain.bdry.to_splinesegs()
        assert isinstance(splinesegs, MultiSegment)
        assert all(isinstance(seg, SplineSegment) for seg in splinesegs.segments)
        assert len(splinesegs.segments) == len(unit_square_domain.bdry.segments)

    def test_to_splinesegs_matches_geometry(self, unit_square_domain):
        splinesegs = unit_square_domain.bdry.to_splinesegs()
        tau = np.linspace(0, 1, 9)
        for orig, spline in zip(unit_square_domain.bdry.segments, splinesegs.segments):
            assert np.allclose(orig.p(tau), spline.p(tau), atol=1e-8)


# ---------------------------------------------------------------------------
# New TestMultiSegment tests
# ---------------------------------------------------------------------------

class TestMultiSegmentNew:
    # --- polyline ---

    def test_polyline_polygon_subseg_count(self):
        ms = MultiSegment.from_vertices([0, 1, 1+1j, 1j])
        b0, b1, owner = ms.polyline()
        # LineSegments contribute one sub-segment each
        assert len(b0) == len(ms.segments)
        assert len(b1) == len(b0) == len(owner)

    def test_polyline_curved_subseg_count(self):
        seg = ParametricSegment(lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t),
                                0, 2*np.pi, 'dir', tol=1e-3)
        ms = MultiSegment([seg], val_simple=False)
        b0, b1, owner = ms.polyline()
        assert len(b0) == len(seg.polyline_tau) - 1
        assert np.all(owner == 0)

    def test_polyline_contiguous_within_segment(self):
        ms = MultiSegment.from_vertices([0, 1, 1+1j, 1j])
        b0, b1, owner = ms.polyline()
        for j in range(len(ms.segments)):
            mask = owner == j
            assert np.allclose(b1[mask][:-1], b0[mask][1:])

    # --- contiguity validation ---

    def test_contiguous_segments_ok(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])  # should not raise
        assert len(ms.segments) == 2

    def test_non_contiguous_raises(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(2, 2+1j)  # gap: seg1 ends at 1, seg2 starts at 2
        with pytest.raises(ValueError):
            MultiSegment([seg1, seg2])

    def test_single_segment_is_contiguous(self):
        seg = LineSegment(0, 1)
        ms = MultiSegment([seg])  # single segment always ok
        assert len(ms.segments) == 1

    def test_val_contiguous_false_skips_check(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(2, 2+1j)  # gap
        ms = MultiSegment([seg1, seg2], val_contiguous=False)  # should not raise
        assert len(ms.segments) == 2

    # --- flattening ---

    def test_flatten_multisegment_input(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        seg3 = LineSegment(1+1j, 1j)
        inner = MultiSegment([seg2, seg3])
        ms = MultiSegment([seg1, inner])
        assert len(ms.segments) == 3

    def test_flatten_preserves_order(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        seg3 = LineSegment(1+1j, 1j)
        inner = MultiSegment([seg2, seg3])
        ms = MultiSegment([seg1, inner])
        assert np.isclose(ms.segments[0].p0, 0)
        assert np.isclose(ms.segments[1].p0, 1)
        assert np.isclose(ms.segments[2].p0, 1+1j)

    def test_nested_multisegment_flattening(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        seg3 = LineSegment(1+1j, 1j)
        seg4 = LineSegment(1j, 0)
        inner = MultiSegment([seg2, seg3])
        outer = MultiSegment([seg1, inner])
        ms = MultiSegment([outer, seg4])
        assert len(ms.segments) == 4

    # --- __add__ join with BaseSegment ---

    def test_add_lineseg_to_multiseg(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1])
        result = ms + seg2
        assert isinstance(result, MultiSegment)
        assert len(result.segments) == 2

    def test_add_non_contiguous_raises(self):
        seg1 = LineSegment(0, 1)
        ms = MultiSegment([seg1])
        seg2 = LineSegment(5, 5+1j)  # gap
        with pytest.raises(ValueError):
            ms + seg2

    # --- __add__ / __radd__ translation with scalar ---

    def test_translate_multiseg(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])
        shift = 1+2j
        result = ms + shift
        assert np.isclose(result.segments[0].p0, ms.segments[0].p0 + shift)
        assert np.isclose(result.segments[0].pf, ms.segments[0].pf + shift)

    def test_translate_preserves_contiguity(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])
        result = ms + (3+4j)
        # result is contiguous if pf of seg[0] == p0 of seg[1]
        assert np.isclose(result.segments[0].pf, result.segments[1].p0)

    def test_radd_scalar_multiseg(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])
        shift = 1+2j
        result1 = ms + shift
        result2 = shift + ms
        assert np.isclose(result1.segments[0].p0, result2.segments[0].p0)

    # --- __mul__ / __rmul__ scaling with scalar ---

    def test_scale_multiseg(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])
        result = ms * 2
        assert np.isclose(result.len, 2 * ms.len)

    def test_rmul_scalar_multiseg(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])
        result1 = ms * 2
        result2 = 2 * ms
        assert np.isclose(result1.segments[0].p0, result2.segments[0].p0)
        assert np.isclose(result1.segments[0].pf, result2.segments[0].pf)

    def test_scale_preserves_contiguity(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        ms = MultiSegment([seg1, seg2])
        result = ms * 3
        assert np.isclose(result.segments[0].pf, result.segments[1].p0)

    # --- error cases ---

    def test_add_invalid_type_raises(self):
        seg = LineSegment(0, 1)
        ms = MultiSegment([seg])
        with pytest.raises(TypeError):
            ms + "string"

    def test_mul_non_scalar_raises(self):
        seg = LineSegment(0, 1)
        ms = MultiSegment([seg])
        with pytest.raises(ValueError):
            ms * np.array([1, 2])


# ---------------------------------------------------------------------------
# New TestLineSegment tests
# ---------------------------------------------------------------------------

class TestLineSegmentNew:
    def test_mul_scalar(self):
        seg = LineSegment(0, 1+1j)
        s = seg * 2
        assert np.isclose(s.p0, 0)
        assert np.isclose(s.pf, 2+2j)

    def test_rmul_scalar(self):
        seg = LineSegment(0, 1+1j)
        s1 = seg * 2
        s2 = 2 * seg
        assert np.isclose(s1.p0, s2.p0)
        assert np.isclose(s1.pf, s2.pf)

    def test_translate_scalar(self):
        seg = LineSegment(0, 1)
        s = seg + (1+2j)
        assert np.isclose(s.p0, 1+2j)
        assert np.isclose(s.pf, 2+2j)

    def test_mul_non_scalar_raises(self):
        seg = LineSegment(0, 1)
        with pytest.raises(ValueError):
            seg * np.array([1, 2])


# ---------------------------------------------------------------------------
# New TestParametricSegment tests
# ---------------------------------------------------------------------------

class TestParametricSegmentNew:
    def test_mul_scalar_scales_length(self, unit_disk_seg):
        scaled = unit_disk_seg * 3
        assert np.isclose(scaled.len, 3 * unit_disk_seg.len, rtol=1e-4)

    def test_rmul_scalar(self, unit_disk_seg):
        scaled1 = unit_disk_seg * 3
        scaled2 = 3 * unit_disk_seg
        assert np.isclose(scaled1.len, scaled2.len, rtol=1e-4)

    def test_translate_scalar(self, unit_disk_seg):
        shift = 1+2j
        shifted = unit_disk_seg + shift
        assert np.isclose(shifted.p0, unit_disk_seg.p0 + shift, atol=1e-6)

    def test_translate_preserves_derivative(self, unit_disk_seg):
        shift = 1+2j
        shifted = unit_disk_seg + shift
        tau = np.linspace(0.1, 0.9, 5)
        assert np.allclose(shifted.dp(tau), unit_disk_seg.dp(tau), rtol=1e-4)


# ---------------------------------------------------------------------------
# New TestDomain tests
# ---------------------------------------------------------------------------

class TestDomainNew:
    def test_translate_domain(self, rect_domain):
        shifted = rect_domain + (1+2j)
        assert np.isclose(shifted.area, rect_domain.area, rtol=1e-5)

    def test_scale_domain(self, rect_domain):
        scaled = rect_domain * 3
        assert np.isclose(scaled.area, 9 * rect_domain.area, rtol=1e-5)

    def test_radd_domain(self, rect_domain):
        shift = 1+2j
        r1 = rect_domain + shift
        r2 = shift + rect_domain
        assert np.isclose(r1.area, r2.area, rtol=1e-5)

    def test_rmul_domain(self, rect_domain):
        r1 = rect_domain * 3
        r2 = 3 * rect_domain
        assert np.isclose(r1.area, r2.area, rtol=1e-5)

    def test_translate_non_scalar_raises(self, rect_domain):
        with pytest.raises(TypeError):
            rect_domain + np.array([1, 2])


# ---------------------------------------------------------------------------
# TestFactoryFunctions
# ---------------------------------------------------------------------------

class TestFactoryFunctions:
    # --- basic construction ---

    def test_rect_is_polygon(self):
        assert isinstance(rect(2, 1), Polygon)

    def test_disk_is_domain(self):
        assert isinstance(disk(), Domain)

    def test_L_shape_is_domain(self):
        assert isinstance(L_shape(), Polygon)

    def test_GWW1_is_domain(self):
        assert isinstance(GWW1(), Polygon)

    def test_GWW2_is_domain(self):
        assert isinstance(GWW2(), Polygon)

    def test_H_shape_is_domain(self):
        assert isinstance(H_shape(), Polygon)

    def test_reg_ngon_is_domain(self):
        assert isinstance(reg_ngon(6), Polygon)

    def test_disk_sector_is_domain(self):
        assert isinstance(disk_sector(), Domain)

    def test_iso_right_tri_is_domain(self):
        assert isinstance(iso_right_tri(), Polygon)

    def test_iso_tri_is_domain(self):
        assert isinstance(iso_tri(), Polygon)

    def test_mushroom_is_domain(self):
        assert isinstance(mushroom(), Domain)

    def test_cut_square_is_domain(self):
        assert isinstance(cut_square(), Domain)

    def test_chevron_is_domain(self):
        assert isinstance(chevron(), Polygon)

    # --- area spot-checks ---

    def test_rect_area(self):
        assert np.isclose(rect(2, 1).area, 2.0)

    def test_disk_area(self):
        assert np.isclose(disk(1).area, np.pi, rtol=1e-4)

    def test_reg_ngon_hexagon_area(self):
        # Regular hexagon with circumradius 1: area = 3*sqrt(3)/2
        assert np.isclose(reg_ngon(6).area, 3*np.sqrt(3)/2, rtol=1e-6)

    # --- boundary contiguity ---

    def test_cut_square_contiguous(self):
        ms = cut_square().bdry
        for i in range(len(ms.segments) - 1):
            assert np.isclose(ms.segments[i].pf, ms.segments[i+1].p0, atol=1e-12)

    def test_mushroom_contiguous(self):
        ms = mushroom().bdry
        for i in range(len(ms.segments) - 1):
            assert np.isclose(ms.segments[i].pf, ms.segments[i+1].p0, atol=1e-12)

    def test_disk_sector_contiguous(self):
        ms = disk_sector().bdry
        for i in range(len(ms.segments) - 1):
            assert np.isclose(ms.segments[i].pf, ms.segments[i+1].p0, atol=1e-12)

    # --- parameter validation ---

    def test_chevron_h1_ge_h2_raises(self):
        with pytest.raises(ValueError):
            chevron(2, 1)

    def test_chevron_negative_raises(self):
        with pytest.raises(ValueError):
            chevron(-1, 1)

    def test_cut_square_r_out_of_range_raises(self):
        with pytest.raises(ValueError):
            cut_square(0)
        with pytest.raises(ValueError):
            cut_square(1)

    def test_mushroom_b_ge_r_raises(self):
        with pytest.raises(ValueError):
            mushroom(r=0.5, b=1)

    def test_disk_sector_theta_out_of_range_raises(self):
        with pytest.raises(ValueError):
            disk_sector(theta=0)
        with pytest.raises(ValueError):
            disk_sector(theta=2*np.pi)

    # --- BC propagation ---

    def test_rect_neumann_bc(self):
        r = rect(2, 1, bc='neu')
        assert all(seg.bc == 1.0 for seg in r.bdry.segments)

    def test_disk_neumann_bc(self):
        c = disk(bc='neu')
        assert all(seg.bc == 1.0 for seg in c.bdry.segments)


class TestSegmentIntersects:
    # LineSegment.intersects agrees with .intersection existence

    def test_line_line_crossing(self):
        a = LineSegment(0, 2+2j)
        b = LineSegment(0+2j, 2+0j)
        assert a.intersects(b)
        assert b.intersects(a)

    def test_line_line_disjoint(self):
        a = LineSegment(0, 1)
        b = LineSegment(0+1j, 1+1j)
        assert not a.intersects(b)

    def test_line_self_is_true(self):
        a = LineSegment(0, 1+1j)
        assert a.intersects(a)

    def test_line_parametric_crossing(self):
        # a unit-circle arc crosses a chord through the disk
        arc = ParametricSegment(lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t),
                                0, np.pi, 'dir', tol=1e-3)
        chord = LineSegment(-2, 2)            # x-axis, crosses the arc endpoints region
        assert chord.intersects(arc)
        assert arc.intersects(chord)

    def test_line_parametric_disjoint(self):
        arc = ParametricSegment(lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t),
                                0, np.pi/2, 'dir', tol=1e-3)
        far = LineSegment(5, 5+1j)
        assert not far.intersects(arc)
        assert not arc.intersects(far)

    def test_parametric_parametric_crossing(self):
        arc1 = ParametricSegment(lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t),
                                 0, np.pi, 'dir', tol=1e-3)
        arc2 = ParametricSegment(lambda t: 1 + np.exp(1j*t), lambda t: 1j*np.exp(1j*t),
                                 0, np.pi, 'dir', tol=1e-3)
        assert arc1.intersects(arc2)

    def test_intersects_matches_intersection_existence(self):
        # boolean must agree with whether .intersection() found points
        a = LineSegment(0, 2+2j)
        for other, expect in [(LineSegment(0+2j, 2+0j), True),
                              (LineSegment(0+3j, 1+3j), False)]:
            assert a.intersects(other) == (len(a.intersection(other)) > 0) == expect


class TestCornerBranchCutRays:

    def test_returns_float_array_correct_shape(self):
        dom = Polygon([0, 1, 1+1j, 1j])
        result = corner_branch_cut_rays(dom)
        assert result.shape == (4,)
        assert result.dtype == float

    def test_no_nans_convex_square(self):
        dom = Polygon([0, 1, 1+1j, 1j])
        assert not np.any(np.isnan(corner_branch_cut_rays(dom)))

    def test_no_nans_convex_triangle(self):
        dom = Polygon([0, 2, 1j])
        assert not np.any(np.isnan(corner_branch_cut_rays(dom)))

    def test_no_nans_regular_hexagon(self):
        dom = reg_ngon(6)
        assert not np.any(np.isnan(corner_branch_cut_rays(dom)))

    def test_no_nans_L_shape(self):
        dom = L_shape()
        assert not np.any(np.isnan(corner_branch_cut_rays(dom)))

    def test_no_nans_H_shape(self):
        dom = H_shape()
        assert not np.any(np.isnan(corner_branch_cut_rays(dom)))

    def test_ray_does_not_cross_boundary_L_shape(self):
        dom = L_shape()
        angles = corner_branch_cut_rays(dom)
        R = 10 * dom.diameter
        bdry = dom.bdry
        n_segs = len(bdry.segments)
        corner_idx = bdry.corner_idx
        b0, b1, owner = bdry.polyline()

        for i, (c, ci, theta) in enumerate(zip(bdry.corners, corner_idx, angles)):
            adj = {ci, (ci - 1) % n_segs}
            ray_end = c + R * np.exp(1j * theta)
            for z0, z1, o in zip(b0, b1, owner):
                if o in adj:
                    continue
                hit = segment_intersection(c, ray_end, z0, z1)
                assert hit is None, (
                    f"corner {i}: ray at theta={theta:.4f} rad hits segment {o}"
                )

    def test_angle_in_exterior_sector(self):
        dom = L_shape()
        angles = corner_branch_cut_rays(dom)
        phi0, phi1 = dom.bdry.corner_angles
        corner_idx = dom.bdry.corner_idx

        for i, (ci, theta) in enumerate(zip(corner_idx, angles)):
            angle_out    = phi0[ci]
            angle_in_rev = phi1[ci]
            ext_span = (angle_out - angle_in_rev) % (2 * np.pi)
            dist = (theta - angle_in_rev) % (2 * np.pi)
            assert 0 < dist < ext_span, (
                f"corner {i}: theta={theta:.4f} not in exterior sector "
                f"[{angle_in_rev:.4f}, {angle_in_rev+ext_span:.4f}]"
            )

    def test_bisector_returned_for_convex_square(self):
        dom = Polygon([0, 1, 1+1j, 1j])
        angles = corner_branch_cut_rays(dom)
        phi0, phi1 = dom.bdry.corner_angles
        corner_idx = dom.bdry.corner_idx

        for ci, theta in zip(corner_idx, angles):
            ext_span = (phi0[ci] - phi1[ci]) % (2 * np.pi)
            theta_bisect = (phi1[ci] + ext_span / 2.0) % (2 * np.pi)
            assert np.isclose(theta, theta_bisect, atol=1e-12), (
                f"segment {ci}: expected bisector {theta_bisect:.6f}, got {theta:.6f}"
            )

    def test_empty_result_for_smooth_disk(self):
        dom = disk()
        result = corner_branch_cut_rays(dom)
        assert result.shape == (0,)

    def test_mixed_boundary_cut_square(self):
        dom = cut_square()
        result = corner_branch_cut_rays(dom)
        assert not np.any(np.isnan(result))

    def test_deterministic_and_exact(self):
        dom = L_shape()
        a1 = corner_branch_cut_rays(dom)
        a2 = corner_branch_cut_rays(dom)
        np.testing.assert_array_equal(a1, a2)

    def test_domain_branch_cut_rays_method(self):
        dom = L_shape()
        np.testing.assert_array_equal(dom.branch_cut_rays(), corner_branch_cut_rays(dom))

    def test_max_clearance_on_reflex_corner(self):
        # the chosen ray should sit in the interior of a free gap, with appreciable
        # angular clearance from every subtended arc (not hugging the boundary)
        dom = L_shape()
        angles = corner_branch_cut_rays(dom)
        bdry = dom.bdry
        n_segs = len(bdry.segments)
        b0, b1, owner = bdry.polyline()
        corner_idx = bdry.corner_idx
        int_angles = dom.int_angles

        reflex = np.where(int_angles[corner_idx] > np.pi)[0]
        assert len(reflex) > 0, "L_shape should have at least one reflex corner"

        for i in reflex:
            c, ci, theta = bdry.corners[i], corner_idx[i], angles[i]
            keep = (owner != ci) & (owner != (ci - 1) % n_segs)
            a0 = np.angle(b0[keep] - c)
            a1 = np.angle(b1[keep] - c)
            d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            lo = np.where(d >= 0, a0, a1)
            hi = lo + np.abs(d)
            # signed angular distance from theta to each arc edge
            edges = np.concatenate([lo, hi])
            gaps = np.abs((theta - edges + np.pi) % (2 * np.pi) - np.pi)
            assert gaps.min() > 1e-3, (
                f"reflex corner {i}: ray hugs an arc edge (clearance {gaps.min():.2e})"
            )


class TestPolylineBranchCuts:

    def test_spiral_is_ccw_simple_with_surrounded_corners(self):
        sp = spiral()
        assert sp.area > 0                      # CCW
        rays = corner_branch_cut_rays(sp)
        assert np.any(np.isnan(rays))           # has surrounded corners (no straight ray)

    def test_free_ray_from_exterior_point(self):
        sp = spiral()
        # a point far outside the spiral always has a clear sightline
        p = 100 + 100j
        beta = free_ray_from_point(sp, p)
        assert not np.isnan(beta)
        R = 10*sp.diameter
        b0, b1, _ = sp.bdry.polyline()
        ray_end = p + R*np.exp(1j*beta)
        for z0, z1 in zip(b0, b1):
            assert segment_intersection(p, ray_end, z0, z1) is None

    def test_free_ray_returns_nan_in_pocket(self):
        # center of a near-closed C: surrounded, no ray to infinity
        # build a thick C (annulus with a small mouth) via spiral with >1 turn
        sp = spiral(turns=1.2, width=0.5)
        # the geometric center is enclosed by the coil
        p = 0 + 0j
        # may or may not be NaN depending on mouth; assert it is consistent with contains
        beta = free_ray_from_point(sp, p)
        if not sp.contains(np.array([p]))[0]:
            # exterior pocket point: if NaN, no sightline; if not, ray must be clear
            if not np.isnan(beta):
                R = 10*sp.diameter
                b0, b1, _ = sp.bdry.polyline()
                end = p + R*np.exp(1j*beta)
                for z0, z1 in zip(b0, b1):
                    assert segment_intersection(p, end, z0, z1) is None

    def test_polyline_cut_valid_for_all_surrounded_corners(self):
        sp = spiral()
        rays = corner_branch_cut_rays(sp)
        b0, b1, _ = sp.bdry.polyline()
        R = 10*sp.diameter
        for i in np.where(np.isnan(rays))[0]:
            verts, beta = corner_branch_cut_polyline(sp, int(i))
            c = sp.corners[i]
            path = np.concatenate(([c], verts, [verts[-1] + R*np.exp(1j*beta)]))
            # no segment of the cut crosses the boundary (touches at the corner allowed)
            for a, b in zip(path[:-1], path[1:]):
                for z0, z1 in zip(b0, b1):
                    hit = segment_intersection(a, b, z0, z1)
                    assert hit is None or np.isclose(hit, c, atol=1e-9), (
                        f"corner {i}: cut crosses boundary"
                    )
            # cut interior samples are outside the domain
            samp = np.concatenate([np.linspace(a, b, 15)[1:] for a, b in zip(path[:-1], path[1:])])
            assert not np.any(sp.contains(samp))


# ---------------------------------------------------------------------------
# TestAdaptiveSampling -- the adaptive curve-sampling helpers
# ---------------------------------------------------------------------------

class TestAdaptiveSampling:

    def _circle(self, r=1.0):
        p = lambda t: r * np.exp(1j * t)
        dp = lambda t: 1j * r * np.exp(1j * t)
        speed = lambda t: np.abs(dp(t))
        return p, dp, speed

    def test_estimate_length_circle(self):
        p, dp, speed = self._circle(2.0)
        L0 = _estimate_length(p, 0, 2 * np.pi)
        # chord-sum slightly under-estimates the true 2*pi*r = 4*pi
        assert L0 == pytest.approx(4 * np.pi, rel=2e-3)

    def test_arclength_table_length_accurate(self):
        p, dp, speed = self._circle(1.0)
        eps = 1e-6
        t_nodes, s_nodes = adaptive_arclength_table(speed, 0, 2 * np.pi, eps,
                                                    eps * 2 * np.pi)
        assert s_nodes[-1] == pytest.approx(2 * np.pi, rel=eps)

    def test_arclength_table_monotone(self):
        p, dp, speed = self._circle(1.0)
        eps = 1e-5
        t_nodes, s_nodes = adaptive_arclength_table(speed, 0, 2 * np.pi, eps,
                                                    eps * 2 * np.pi)
        assert np.all(np.diff(t_nodes) > 0)
        assert np.all(np.diff(s_nodes) > 0)

    def test_arclength_table_nonuniform_speed(self):
        # ellipse: arc length matches the analytic perimeter
        from scipy.special import ellipe
        a, b = 2.0, 1.0
        dp = lambda t: -a * np.sin(t) + 1j * b * np.cos(t)
        speed = lambda t: np.abs(dp(t))
        eps = 1e-7
        _, s_nodes = adaptive_arclength_table(speed, 0, 2 * np.pi, eps, eps * 12)
        exact = 4 * a * ellipe(1 - (b / a) ** 2)
        assert s_nodes[-1] == pytest.approx(exact, rel=1e-5)

    def test_polyline_within_tolerance(self):
        # every chord midpoint stays within eps_abs of the circle
        p, dp, speed = self._circle(1.0)
        eps_abs = 1e-4
        t = adaptive_polyline(p, 0, 2 * np.pi, eps_abs=eps_abs)
        mids = p(0.5 * (t[:-1] + t[1:]))
        chord_mid = 0.5 * (p(t[:-1]) + p(t[1:]))
        assert np.max(np.abs(mids - chord_mid)) <= eps_abs

    def test_polyline_sorted_and_spans_interval(self):
        p, dp, speed = self._circle(1.0)
        t = adaptive_polyline(p, 0, 2 * np.pi, eps_abs=1e-3)
        assert np.all(np.diff(t) > 0)
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(2 * np.pi)

    def test_tighter_tol_more_nodes(self):
        p, dp, speed = self._circle(1.0)
        coarse = adaptive_polyline(p, 0, 2 * np.pi, eps_abs=1e-2)
        fine = adaptive_polyline(p, 0, 2 * np.pi, eps_abs=1e-4)
        assert len(fine) > len(coarse)

    def test_segment_len_matches_analytic(self):
        # ParametricSegment.len uses the adaptive table
        seg = disk(1.0).bdry.segments[0]
        assert seg.len == pytest.approx(2 * np.pi, rel=1e-4)
        es = ellipse(2, 1).bdry.segments[0]
        from scipy.special import ellipe
        exact = 4 * 2 * ellipe(1 - (1 / 2) ** 2)
        assert es.len == pytest.approx(exact, rel=1e-4)

    def test_segment_tighter_tol_more_polyline_nodes(self):
        coarse = ParametricSegment(lambda t: np.exp(1j * t),
                                   lambda t: 1j * np.exp(1j * t),
                                   0, 2 * np.pi, 'dir', tol=1e-2)
        fine = ParametricSegment(lambda t: np.exp(1j * t),
                                 lambda t: 1j * np.exp(1j * t),
                                 0, 2 * np.pi, 'dir', tol=1e-4)
        assert len(fine.polyline_tau) > len(coarse.polyline_tau)

    def test_constant_speed_table_is_small(self):
        # constant/low-degree speed is integrated exactly by Gauss-Legendre; the
        # table must NOT explode (regression: a quadrature-error or piecewise-
        # linear criterion gives 2 or ~1e4 nodes respectively)
        _, s_nodes = adaptive_arclength_table(
            lambda t: np.full_like(np.asarray(t, float), 2.0),
            0, 1, 1e-4, 1e-4)
        assert 2 < len(s_nodes) < 50
        assert s_nodes[-1] == pytest.approx(2.0, rel=1e-6)

    def test_smooth_curve_table_is_modest(self):
        # a smooth ellipse should need only a handful of table nodes, not 1e4
        dp = lambda t: -2 * np.sin(t) + 1j * np.cos(t)
        t_nodes, _ = adaptive_arclength_table(lambda t: np.abs(dp(t)),
                                              0, 2 * np.pi, 1e-4, 1e-4 * 9.69)
        assert len(t_nodes) < 500

    def test_zero_speed_cusp_reparam_uniform(self):
        # p(t) = exp(i*pi*t^2) has |p'| = 2*pi*t, a zero-speed point at t=0.
        # After arc-length reparameterization, equal tau steps must map to equal
        # arc length (regression: a quadrature-only table left t(s) linear).
        seg = ParametricSegment(lambda t: np.exp(1j * np.pi * t ** 2),
                                lambda t: 2j * np.pi * t * np.exp(1j * np.pi * t ** 2),
                                0, 1, 'dir', tol=1e-4)
        assert seg.len == pytest.approx(np.pi, rel=1e-4)
        tau = np.linspace(0, 1, 17)
        ang = np.unwrap(np.angle(seg.p(tau)))     # radius 1, so arc length == angle
        darc = np.diff(ang)
        assert np.allclose(darc, darc.mean(), rtol=5e-3)
        assert not np.any(np.isnan(seg.T(tau)))

    def test_polyline_chord_guard_scales_with_tolerance(self):
        # regression: the chord-length guard in adaptive_polyline used to be
        # linear in eps_abs, which shrinks faster than the legitimate
        # sqrt(eps_abs) sagitta scaling and became the sole binding constraint
        # at tight tol, forcing node count to blow up ~100x for a 100x tighter
        # tol instead of the expected ~10x (sqrt(100)).
        p, dp, speed = self._circle(1.0)
        coarse = adaptive_polyline(p, 0, 2 * np.pi, eps_abs=1e-4 * 2 * np.pi,
                                   L=2 * np.pi)
        fine = adaptive_polyline(p, 0, 2 * np.pi, eps_abs=1e-6 * 2 * np.pi,
                                 L=2 * np.pi)
        assert len(fine) / len(coarse) < 20

    def test_polyline_chord_guard_catches_aliased_chord(self):
        # adversarial curve: sin(6*pi*t) is exactly zero at t=0, 1/3, 1/2,
        # 2/3, 1, so the top-level chord's midpoint/tercile deviation tests
        # are all exactly zero even though the curve bulges by amplitude A
        # in between -- exactly the case the chord-length guard exists for.
        A = 0.05
        p = lambda t: t + 1j * A * np.sin(6 * np.pi * t)
        t_dense = np.linspace(0, 1, 20000)
        L = np.sum(np.abs(np.diff(p(t_dense))))
        eps_abs = 1e-4 * L
        t = adaptive_polyline(p, 0, 1, eps_abs=eps_abs, L=L)

        # true deviation of the curve from the resulting polyline, checked
        # densely (not just at the same sample points used to build it)
        t_check = np.linspace(0, 1, 20000)
        idx = np.clip(np.searchsorted(t, t_check, side='right') - 1, 0, len(t) - 2)
        t_l, t_r = t[idx], t[idx + 1]
        frac = (t_check - t_l) / (t_r - t_l)
        chord_pt = p(t_l) + frac * (p(t_r) - p(t_l))
        max_dev = np.max(np.abs(p(t_check) - chord_pt))
        assert max_dev < 3 * eps_abs
