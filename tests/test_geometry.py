"""Unit tests for lappy/geometry.py."""
import pytest
import numpy as np
from lappy import PointSet, Domain, Polygon, ParametricSegment, LineSegment, MultiSegment
from lappy.geometry import SplineSegment
from lappy.geometry import (
    rect, disk, L_shape, GWW1, GWW2, H_shape, reg_ngon,
    disk_sector, iso_right_tri, iso_tri, mushroom, cut_square, chevron,
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
# New TestMultiSegment tests
# ---------------------------------------------------------------------------

class TestMultiSegmentNew:
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
