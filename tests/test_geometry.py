"""Unit tests for lappy/geometry.py."""
import pytest
import numpy as np
from lappy import PointSet, Domain, Polygon, ParametricSegment, LineSegment, MultiSegment
from lappy.geometry import SplineSegment


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
        # LineSegment has no dist(); use to_splineseg() which inherits from ParametricSegment
        seg = LineSegment(0, 1).to_splineseg()
        assert np.isclose(seg.dist(0.5+0j), 0.0, atol=1e-6)

    def test_dist_to_perpendicular(self):
        seg = LineSegment(0, 1).to_splineseg()
        assert np.isclose(seg.dist(0.5+2j), 2.0, rtol=1e-4)

    def test_to_splineseg(self):
        seg = LineSegment(0, 1)
        ss = seg.to_splineseg()
        assert isinstance(ss, SplineSegment)
        assert np.isclose(ss.p0, 0+0j, atol=1e-10)
        assert np.isclose(ss.pf, 1+0j, atol=1e-10)


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

    def test_circle_len(self, unit_circle_seg):
        assert np.isclose(unit_circle_seg.len, 2*np.pi, rtol=1e-5)

    def test_circle_p_endpoints(self, unit_circle_seg):
        assert np.isclose(unit_circle_seg.p(0), 1+0j, atol=1e-6)
        assert np.isclose(unit_circle_seg.p(1), 1+0j, atol=1e-6)

    def test_tangent_unit_norm(self, unit_circle_seg):
        tau = np.linspace(0.05, 0.95, 10)
        assert np.allclose(np.abs(unit_circle_seg.T(tau)), 1.0, atol=1e-12)

    def test_normal_unit_norm(self, unit_circle_seg):
        tau = np.linspace(0.05, 0.95, 10)
        assert np.allclose(np.abs(unit_circle_seg.N(tau)), 1.0, atol=1e-12)

    def test_tangent_normal_orthogonal(self, unit_circle_seg):
        tau = np.linspace(0.05, 0.95, 10)
        T = unit_circle_seg.T(tau)
        N = unit_circle_seg.N(tau)
        assert np.allclose((T * np.conj(N)).real, 0.0, atol=1e-12)

    def test_arc_length_parametrization(self, unit_circle_seg):
        # After reparameterization, |dp/dtau| == len (constant speed)
        tau = np.linspace(0.05, 0.95, 10)
        dp_mag = np.abs(unit_circle_seg.dp(tau))
        assert np.allclose(dp_mag, unit_circle_seg.len, rtol=1e-4)

    def test_is_closed(self, unit_circle_seg):
        assert unit_circle_seg.is_closed == True

    def test_dist(self, unit_circle_seg):
        # Distance from origin to unit circle == 1.0
        assert np.isclose(unit_circle_seg.dist(0+0j), 1.0, rtol=1e-4)


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
        half_circle = ParametricSegment(
            lambda t: np.exp(1j*t), lambda t: 1j*np.exp(1j*t),
            0, np.pi, val_simple=False
        )
        ms2 = MultiSegment([LineSegment(0, 1), half_circle])
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

    def test_dist_from_interior_point(self):
        # Build boundary with SplineSegments so that dist() works
        verts = [0+0j, 1+0j, 1+1j, 0+1j]
        segs = [LineSegment(verts[i], verts[(i+1) % 4]).to_splineseg()
                for i in range(4)]
        ms = MultiSegment(segs)
        assert np.isclose(ms.dist(0.5+0.5j), 0.5, rtol=1e-3)

    def test_dist_from_corner(self):
        verts = [0+0j, 1+0j, 1+1j, 0+1j]
        segs = [LineSegment(verts[i], verts[(i+1) % 4]).to_splineseg()
                for i in range(4)]
        ms = MultiSegment(segs)
        assert np.isclose(ms.dist(0+0j), 0.0, atol=1e-5)

    def test_validate_closed_raises(self):
        seg1 = LineSegment(0, 1)
        seg2 = LineSegment(1, 1+1j)
        with pytest.raises(ValueError):
            MultiSegment([seg1, seg2], val_closed=True)

    def test_bcs(self, unit_square_domain):
        bcs = unit_square_domain.bdry.bcs
        assert len(bcs) == 4
        assert all(bc == 0.0 for bc in bcs)  # 'dir' → 0.0


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
        # Use higher n for accurate winding number near boundary
        assert np.all(unit_square_domain.contains(pts.pts, n=500))

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
