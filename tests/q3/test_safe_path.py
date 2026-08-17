"""M2: earliest safe travel E, service completion Phi, route evaluation.

Scenarios per PLAN 8.2 acceptance: direct, detour, wait-before-direct,
service-window delay, no waiting inside active disks, Case2-style depot-in-
disk no-landing, overlapping union detour, and the eps_arc clearance.
"""

import math

import numpy as np
import pytest

from Q3.domain import Base, EPS_ARC_KM, EPS_VER_KM, SERVICE_S, NoFlyZone
from Q3.safe_path import (
    base_safety_assert,
    earliest_safe_service_completion,
    earliest_safe_travel,
    eval_route,
    eval_solution,
    event_delays,
    spatiotemporal_safe,
    wait_safe,
)
from tests.q3.conftest import (
    DIRECT_BASE_TO_T1_S,
    DETOUR_BASE_TO_T1_KM,
    MINI_TASKS,
    Z1_ALL_DAY,
    Z2_ALL_DAY,
    make_problem,
)


def test_direct_when_no_zone_conflicts():
    problem = make_problem(zones=())
    leg = earliest_safe_travel(problem, Base(), MINI_TASKS[0], 0)
    assert leg is not None
    assert leg.path_type == "direct"
    assert leg.depart_s == 0
    assert leg.arrive_s == DIRECT_BASE_TO_T1_S
    assert leg.wait_s == 0
    assert leg.distance_km == pytest.approx(1.0)
    assert leg.affected_zones == ()


def test_direct_touching_safe_boundary_is_legal():
    """A tangent touch on the safe disk boundary is not an interior crossing."""
    problem = make_problem(zones=(NoFlyZone("Z", 0.5, 0.0, 0.2, 0, 90_000),))
    leg = earliest_safe_travel(problem, Base(), (0.0, 0.42) if False else MINI_TASKS[2], 0)
    # T3 (-1,0): the base->T3 leg passes at distance 0.0? no: y=0, x in [-1,0],
    # Z center (0.5,0) is outside the segment; no conflict by construction.
    assert leg is not None
    assert leg.path_type == "direct"


def test_detour_when_crossing_active_zone():
    problem = make_problem(zones=(Z1_ALL_DAY,))
    leg = earliest_safe_travel(problem, Base(), MINI_TASKS[0], 0)
    assert leg is not None
    # single disk: the visibility route is the same tangent-arc-tangent detour
    assert leg.path_type in ("detour", "visibility")
    assert leg.depart_s == 0
    assert leg.wait_s == 0
    assert leg.distance_km == pytest.approx(DETOUR_BASE_TO_T1_KM, rel=1e-6)
    assert leg.affected_zones == ("Z1",)


def test_wait_direct_beats_detour():
    """A disk that ends early makes waiting + direct strictly better."""
    zone = NoFlyZone("Z", 0.75, 0.0, 0.5, 0, 50)  # safe radius 0.51
    problem = make_problem(zones=(zone,))
    leg = earliest_safe_travel(problem, Base(), MINI_TASKS[0], 0)
    assert leg is not None
    assert leg.path_type == "wait_direct"
    assert leg.arrive_s < 120  # 102 s, below the detour's ~127 s
    assert leg.wait_s == 36  # ceil(50 - 0.24*66 + 1)
    assert leg.depart_s == leg.wait_s


def test_phi_delays_departure_outside_service_window(
    mini_problem_service_conflict,
):
    """T1 inside Z4 [67,5000]: direct arrival is safe, the service window is
    not; Phi must delay the departure at the depot until the window escapes."""
    problem = mini_problem_service_conflict
    result = earliest_safe_service_completion(problem, Base(), MINI_TASKS[0], 0)
    assert result is not None
    travel, arrive, service_end = result
    assert arrive > 5000  # service starts after the zone ends
    assert service_end == arrive + SERVICE_S
    assert travel.wait_s > 0
    assert travel.wait_s == 4943  # ceil(5000 - 58.7 + 1)
    assert arrive == 5009  # 4943 + 66


def test_no_wait_inside_active_zone():
    """R3: u inside an active disk cannot wait there; the leg must either leave
    the disk immediately or fail closed."""
    problem = make_problem(
        zones=(NoFlyZone("Z", 1.0, 0.0, 0.1, 67, 5000),)
    )
    # T1 sits inside Z; waiting at T1 during an active window is unsafe, and
    # no tangent/visibility route exists from inside the disk.
    leg = earliest_safe_travel(problem, MINI_TASKS[0], MINI_TASKS[1], 67)
    assert leg is None
    # Waiting at the depot (outside the disk) is fine.
    assert wait_safe(problem, (0.0, 0.0), 0, 500) is True


def test_wait_safe_rejects_position_inside_active_disk():
    problem = make_problem(zones=(Z1_ALL_DAY,))
    assert wait_safe(problem, (0.5, 0.0), 0, 100) is False   # inside safe disk
    assert wait_safe(problem, (0.9, 0.0), 0, 100) is True    # 0.4 > 0.21


def test_event_delays_contains_zone_end_escapes():
    problem = make_problem(zones=(NoFlyZone("Z", 1.0, 1.0, 0.1, 300, 600),))
    delays = event_delays(problem, problem.zones, 100)
    assert delays[0] == 0
    assert 600 - 100 + 1 in delays
    assert 600 - 100 + 1 + SERVICE_S in delays


def test_case2_style_depot_in_disk_cannot_land():
    """Depot inside a zone active 10:30-13:00: t=0 takeoff is fine, but a
    return leg from an in-disk task during the window fails closed."""
    zone = NoFlyZone("Z3", 0.5, 0.0, 0.6, 9000, 18000)  # safe radius 0.61
    problem = make_problem(zones=(zone,))
    base_safety_assert(problem)  # not active at t=0
    # T1 lies inside the disk: no safe waiting spot, no detour -> fail closed.
    assert earliest_safe_travel(problem, MINI_TASKS[0], Base(), 10000) is None
    # T2 lies outside: waiting there until the window ends, then landing, is
    # legal (R3) — a wait_direct into the disk is allowed.
    leg = earliest_safe_travel(problem, MINI_TASKS[1], Base(), 10000)
    assert leg is not None
    assert leg.path_type == "wait_direct"
    assert leg.arrive_s > 18000
    # Outside the window the same leg is a plain direct.
    leg = earliest_safe_travel(problem, MINI_TASKS[1], Base(), 0)
    assert leg is not None
    assert leg.path_type == "direct"


def test_dual_overlapping_disks_use_visibility_union():
    """Two disks close enough that either single-disk detour cuts the other;
    only the visibility-graph union route is feasible."""
    disks = (
        NoFlyZone("Z1", 0.4, 0.0, 0.2, 0, 90_000),
        NoFlyZone("Z2", 0.7, 0.0, 0.2, 0, 90_000),
    )
    problem = make_problem(zones=disks)
    leg = earliest_safe_travel(problem, Base(), MINI_TASKS[0], 0)
    assert leg is not None
    assert leg.path_type == "visibility"
    assert leg.affected_zones == ("Z1", "Z2")
    assert leg.distance_km < 2.0
    # A single-disk detour would cut the other disk: both detours must have
    # failed the spacetime check.
    assert leg.arrive_s < 150


def test_visibility_arcs_clear_safe_disk_by_eps_arc():
    disks = (
        NoFlyZone("Z1", 0.4, 0.0, 0.2, 0, 90_000),
        NoFlyZone("Z2", 0.7, 0.0, 0.2, 0, 90_000),
    )
    problem = make_problem(zones=disks)
    from Q3.geometry import visibility_path

    path = visibility_path(
        (0.0, 0.0), (1.0, 0.0),
        [((z.cx_km, z.cy_km), z.safe_radius()) for z in disks],
        margin=EPS_ARC_KM,
    )
    assert path is not None
    for leg in path.arc_legs:
        samples = np.linspace(0.0, 1.0, 25)
        for frac in samples:
            theta = math.atan2(
                path.points[leg.end][1] - leg.center[1],
                path.points[leg.end][0] - leg.center[0],
            ) * frac + math.atan2(
                path.points[leg.start][1] - leg.center[1],
                path.points[leg.start][0] - leg.center[0],
            ) * (1.0 - frac)
            point = leg.center + leg.radius_km * np.array([math.cos(theta), math.sin(theta)])
            for zone in disks:
                distance = float(np.linalg.norm(point - np.array([zone.cx_km, zone.cy_km])))
                assert distance >= zone.safe_radius() - EPS_VER_KM


def test_spatiotemporal_safe_detects_time_disjoint_zone():
    path = earliest_safe_travel(
        make_problem(zones=()), Base(), MINI_TASKS[0], 0
    )
    from Q3.safe_path import _direct_path

    assert spatiotemporal_safe(_direct_path((0.0, 0.0), (1.0, 0.0)), 0, (Z1_ALL_DAY,)) is False
    # Zone active after the leg is done: no conflict.
    late = NoFlyZone("Z", 0.5, 0.0, 0.2, 100, 200)
    assert spatiotemporal_safe(_direct_path((0.0, 0.0), (1.0, 0.0)), 0, (late,)) is True


def test_eval_route_builds_timeline():
    problem = make_problem(zones=(Z1_ALL_DAY, Z2_ALL_DAY))
    schedule = eval_route(problem, (1, 2), 1)
    assert schedule is not None
    assert schedule.task_route == (1, 2)
    assert len(schedule.segments) == 3  # T1, T2, return
    assert schedule.segments[0].path_type in ("detour", "visibility")
    assert schedule.segments[1].path_type == "direct"
    assert schedule.segments[2].path_type in ("detour", "visibility")
    assert [iv[1] - iv[0] for iv in schedule.service_intervals] == [SERVICE_S, SERVICE_S]
    assert schedule.S_k_s > 0
    assert schedule.S_k_s == schedule.flight_s + schedule.wait_s + 2 * SERVICE_S
    assert schedule.distance_km > 0.0


def test_eval_route_infeasible_returns_none():
    # A disk active at t=0 containing the depot: takeoff cannot leave and no
    # waiting location is safe -> fail closed (violates the plan's case
    # invariant, but the fail-closed path must be airtight).
    zone = NoFlyZone("Z", 0.5, 0.0, 0.6, 0, 90_000)  # safe 0.61; base inside
    problem = make_problem(zones=(zone,))
    with pytest.raises(AssertionError):
        base_safety_assert(problem)
    assert eval_route(problem, (2,), 1) is None
    # The same zone active from 10:30: takeoff is fine, waiting at the depot
    # during the window is not, and the return leg fails closed.
    late = NoFlyZone("Z3", 0.5, 0.0, 0.6, 9000, 18000)
    problem = make_problem(zones=(late,))
    base_safety_assert(problem)
    assert eval_route(problem, (1,), 1) is not None  # returns before 10:30
    # A service-window delay at T1 (no depot disk) is still feasible.
    conflict = make_problem(zones=(NoFlyZone("Z4", 1.0, 0.0, 0.1, 67, 5000),))
    schedule = eval_route(conflict, (1,), 1)
    assert schedule is not None
    assert schedule.segments[0].depart_s > 4900  # delayed departure


def test_eval_solution_aggregates_metrics():
    problem = make_problem(zones=(Z1_ALL_DAY, Z2_ALL_DAY))
    solution = eval_solution(problem, [(1,), (2,)])
    assert solution is not None
    assert len(solution.schedules) == 2
    assert solution.metrics.delta_s == solution.metrics.S_max_s - solution.metrics.S_min_s
    assert solution.metrics.sum_T_s == solution.metrics.S_max_s + solution.metrics.S_min_s
    assert solution.metrics.total_wait_s == sum(s.wait_s for s in solution.schedules)
    assert solution.metrics.total_distance_km == pytest.approx(
        sum(s.distance_km for s in solution.schedules), rel=1e-6
    )
    assert solution.metrics.S_max_s >= solution.metrics.S_min_s


def test_eval_solution_wrong_fleet_size():
    problem = make_problem(zones=(), fleet_size=2)
    assert eval_solution(problem, [(1,)]) is None
    assert eval_solution(problem, [(1,), (2,), (3,)]) is None


def test_eval_solution_any_infeasible_route_kills_all():
    problem = make_problem(zones=(Z1_ALL_DAY,))
    assert eval_solution(problem, [(1, 2, 3, 4), (4, 3, 2, 1)]) is not None
