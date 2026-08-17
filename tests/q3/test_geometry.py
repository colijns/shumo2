"""M1: geometry primitives — interval, tangent, detour, arc, visibility graph."""

import math

import numpy as np
import pytest

from Q3.geometry import (
    arc_intersection_with_circle,
    detour_distance,
    detour_path,
    path_length,
    point_inside,
    segment_circle_interval,
    segment_clear,
    tangent_points,
    visibility_path,
)

TAU = 2.0 * math.pi


# ---------- segment_circle_interval ----------

def test_interval_no_intersection():
    assert segment_circle_interval((0, 0), (2, 0), (1, 1), 0.5) is None


def test_interval_tangent_is_single_point():
    t0, t1 = segment_circle_interval((0, 0), (2, 0), (1, 1), 1.0)
    assert t0 == pytest.approx(0.5, abs=1e-9)
    assert t1 - t0 < 1e-9


def test_interval_crossing():
    t0, t1 = segment_circle_interval((0, 0), (2, 0), (1, 0), 0.5)
    assert t0 == pytest.approx(0.25)
    assert t1 == pytest.approx(0.75)


def test_interval_segment_fully_inside_disk():
    t0, t1 = segment_circle_interval((0, 0), (1, 0), (0.5, 0), 2.0)
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(1.0)


def test_interval_disk_fully_inside_segment():
    t0, t1 = segment_circle_interval((0, 0), (10, 0), (5, 0), 1.0)
    assert t0 == pytest.approx(0.4)
    assert t1 == pytest.approx(0.6)


# ---------- segment_clear ----------

def test_segment_clear_tangent_is_legal():
    disks = [((0.5, 0), 0.21)]
    assert segment_clear((0, 0), (1, 0), disks) is False
    # tangent from a point on the circle
    assert segment_clear((0.5, 0.21), (1.0, 0.21), disks) is True


def test_segment_clear_starting_inside_disk_is_illegal():
    disks = [((0.5, 0), 0.3)]
    assert segment_clear((0.5, 0.1), (2, 0.1), disks) is False


# ---------- tangent_points ----------

def test_tangent_points_properties():
    tangents = tangent_points((0, 0), (1, 0), 0.3)
    assert tangents is not None
    for point in tangents:
        assert np.linalg.norm(point - np.array([1.0, 0.0])) == pytest.approx(0.3)
        # radius orthogonal to tangent line
        assert np.dot(point - np.array([1.0, 0.0]), -point) == pytest.approx(0.0, abs=1e-9)


def test_tangent_points_none_when_inside():
    assert tangent_points((0.9, 0), (1, 0), 0.3) is None


# ---------- detour_path ----------

def test_detour_matches_hand_formula():
    """base->T1 tangent + arc + T2->T1 case: compare to the closed form."""
    from tests.q3.conftest import DETOUR_BASE_TO_T1_KM

    result = detour_path((0, 0), (1, 0), (0.5, 0), 0.21)
    assert result is not None
    points, arc_angle, direction = result
    assert len(points) == 4
    tangent_len = np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[3] - points[2])
    assert tangent_len + 0.21 * arc_angle == pytest.approx(DETOUR_BASE_TO_T1_KM, rel=1e-9)
    assert detour_distance((0, 0), (1, 0), (0.5, 0), 0.21) == pytest.approx(DETOUR_BASE_TO_T1_KM, rel=1e-9)
    assert direction in (-1, 1)
    assert 0 < arc_angle < math.pi


def test_detour_none_when_endpoint_inside():
    assert detour_path((0.45, 0), (1, 0), (0.5, 0), 0.21) is None


def test_detour_tangent_points_lie_on_circle():
    points, _, _ = detour_path((0, 0), (1, 0), (0.5, 0), 0.21)
    for point in (points[1], points[2]):
        assert np.linalg.norm(point - np.array([0.5, 0.0])) == pytest.approx(0.21, abs=1e-9)


# ---------- arc_intersection_with_circle ----------

def test_arc_circle_intersection_interval():
    """Arc (0,0) r=1 over [0, pi] vs disk (1,0) r=1: overlap is [0, pi/3]."""
    intervals = arc_intersection_with_circle((0, 0), 1.0, 0.0, math.pi, 1, (1, 0), 1.0)
    assert len(intervals) == 1
    lo, hi = intervals[0]
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(math.pi / 3, abs=1e-9)


def test_arc_circle_no_intersection():
    assert arc_intersection_with_circle((0, 0), 1.0, 0.0, math.pi, 1, (0, 3), 0.5) == []


def test_arc_circle_fully_inside():
    intervals = arc_intersection_with_circle((0, 0), 0.5, 0.0, math.pi, 1, (0, 0), 2.0)
    assert intervals == [(0.0, math.pi)]


# ---------- visibility_path ----------

def _check_straight_legs(path, disks):
    arc_indices = set()
    for start, end in path.arc_pairs:
        arc_indices.update((start, end))
    for index, (start, end) in enumerate(zip(path.points, path.points[1:])):
        if (index, index + 1) not in path.arc_pairs:
            assert segment_clear(start, end, disks) is True


def test_visibility_direct_when_clear():
    path = visibility_path((0, 0), (1, 0), [])
    assert path is not None
    assert len(path.points) == 2
    assert path.length_km == pytest.approx(1.0)
    assert path.arc_pairs == ()


def test_visibility_single_disk_detours():
    from tests.q3.conftest import DETOUR_BASE_TO_T1_KM

    disks = [((0.5, 0), 0.21)]
    path = visibility_path((0, 0), (1, 0), disks)
    assert path is not None
    assert len(path.points) >= 3
    assert path.length_km > 1.0
    # single-disk visibility route == tangent + arc + tangent (the detour)
    assert path.length_km == pytest.approx(DETOUR_BASE_TO_T1_KM, rel=1e-6)
    _check_straight_legs(path, disks)


def test_visibility_dual_overlapping_disks():
    """Simulates Case2 Z1/Z2 overlap: union must be bypassed, not crossed."""
    disks = [((0.4, 0), 0.3), ((0.8, 0), 0.3)]
    path = visibility_path((0, 0), (1.2, 0), disks)
    assert path is not None
    _check_straight_legs(path, disks)
    assert path.length_km < 2.0


def test_visibility_three_chain_disks():
    disks = [((0.35, 0), 0.3), ((0.7, 0), 0.3), ((1.05, 0), 0.3)]
    path = visibility_path((0, 0), (1.4, 0), disks)
    assert path is not None
    _check_straight_legs(path, disks)
    assert path.length_km > 1.4


def test_visibility_start_inside_disk_is_infeasible():
    disks = [((0.5, 0), 0.3)]
    assert visibility_path((0.5, 0.1), (1.0, 0.1), disks) is None


def test_visibility_picks_shorter_of_two_equivalent_routes():
    """Symmetric setup: shortest route must go around the nearer side."""
    disks = [((0.5, 0), 0.3)]
    below = visibility_path((0, 0.6), (1, 0.6), disks)
    above = visibility_path((0, -0.6), (1, -0.6), disks)
    assert below is not None and above is not None
    assert below.length_km == pytest.approx(above.length_km, rel=1e-6)


def test_path_length_plain():
    assert path_length([(0, 0), (3, 0), (3, 4)]) == pytest.approx(7.0)
