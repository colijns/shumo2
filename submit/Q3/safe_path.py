







import math

import numpy as np

from dataclasses import replace

from .domain import (
    EPS_ARC_KM,
    Base,
    NoFlyZone,
    ProblemData,
    SERVICE_S,
    SPEED_KMH,
    SegmentRecord,
    Solution,
    SolutionMetrics,
    Task,
    UAVSchedule,
)
from .geometry import (
    TAU,
    ArcLeg,
    VisibilityPath,
    arc_intersection_with_circle,
    detour_path,
    segment_circle_interval,
    visibility_path,
)

KM_PER_S = SPEED_KMH / 3600.0


def _duration(distance_km: float) -> int:
    return int(math.ceil(distance_km / KM_PER_S))


def _distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _path_duration(path: VisibilityPath) -> int:

    total = 0
    for i in range(len(path.points) - 1):
        total += _duration(_leg_length(path, i))
    return total



def _leg_length(path: VisibilityPath, index: int) -> float:
    start, end = path.points[index], path.points[index + 1]
    for leg in path.arc_legs:
        if leg.start == index:
            forward = (math.atan2(end[1] - leg.center[1], end[0] - leg.center[0])
                       - math.atan2(start[1] - leg.center[1], start[0] - leg.center[0])) % TAU
            return leg.radius_km * min(forward, TAU - forward)
    return _distance(start, end)


def _arc_bounds(path: VisibilityPath, index: int) -> tuple:

    for leg in path.arc_legs:
        if leg.start == index:
            start, end = path.points[index], path.points[index + 1]
            theta_a = math.atan2(start[1] - leg.center[1], start[0] - leg.center[0])
            theta_b = math.atan2(end[1] - leg.center[1], end[0] - leg.center[0])
            forward = (theta_b - theta_a) % TAU
            arc_angle = min(forward, TAU - forward)
            direction = 1 if forward <= TAU - forward else -1
            return theta_a, theta_b, direction, leg.radius_km
    return None


def spatiotemporal_safe(path: VisibilityPath, depart_s: float, zones) -> bool:






    t0 = float(depart_s)
    for i in range(len(path.points) - 1):
        start, end = path.points[i], path.points[i + 1]
        arc = _arc_bounds(path, i)
        duration = _leg_length(path, i) / KM_PER_S
        t1 = t0 + duration
        for zone in zones:
            if t1 < zone.start_s or t0 > zone.end_s:
                continue
            if arc is None:
                interval = segment_circle_interval(
                    start, end, (zone.cx_km, zone.cy_km), zone.safe_radius()
                )
                if interval is None:
                    continue


                if interval[1] - interval[0] <= 1e-6:
                    continue
                ta = t0 + interval[0] * duration
                tb = t0 + interval[1] * duration
                if tb >= zone.start_s and ta <= zone.end_s:
                    return False
            else:
                theta_a, theta_b, direction, radius = arc
                intervals = arc_intersection_with_circle(
                    _vec_center(path, i), radius, theta_a, theta_b, direction,
                    (zone.cx_km, zone.cy_km), zone.safe_radius(),
                )
                if not intervals:
                    continue
                if _arc_leg_conflicts(intervals, theta_a, direction, _arc_angle(path, i), t0, duration, zone):
                    return False
        t0 = t1
    return True


def _vec_center(path: VisibilityPath, index: int):
    for leg in path.arc_legs:
        if leg.start == index:
            return leg.center
    raise KeyError(index)


def _arc_angle(path: VisibilityPath, index: int) -> float:
    start, end = path.points[index], path.points[index + 1]
    for leg in path.arc_legs:
        if leg.start == index:
            theta_a = math.atan2(start[1] - leg.center[1], start[0] - leg.center[0])
            theta_b = math.atan2(end[1] - leg.center[1], end[0] - leg.center[0])
            forward = (theta_b - theta_a) % TAU
            return min(forward, TAU - forward)
    raise KeyError(index)


def _arc_leg_conflicts(intervals, theta_start, direction, arc_angle, t0, duration, zone) -> bool:

    sweep = direction * arc_angle
    arc_lo = min(theta_start, theta_start + sweep)
    arc_hi = max(theta_start, theta_start + sweep)
    for lo, hi in intervals:
        for k in (-1, 0, 1):
            shifted_lo = lo + k * TAU
            shifted_hi = hi + k * TAU
            overlap_lo = max(shifted_lo, arc_lo)
            overlap_hi = min(shifted_hi, arc_hi)
            if overlap_hi < overlap_lo - 1e-9:
                continue
            ta = t0 + (overlap_lo - arc_lo) / (arc_hi - arc_lo) * duration
            tb = t0 + (overlap_hi - arc_lo) / (arc_hi - arc_lo) * duration
            if tb >= zone.start_s and ta <= zone.end_s:
                return True
    return False


def wait_safe(problem: ProblemData, position, t0_s: float, t1_s: float) -> bool:


    for zone in problem.zones:
        if t1_s < zone.start_s or t0_s > zone.end_s:
            continue
        if _distance(position, (zone.cx_km, zone.cy_km)) <= zone.safe_radius():
            return False
    return True


def event_delays(problem: ProblemData, zones, now_s: int) -> list[int]:



    candidates = {0}
    for zone in zones:
        if zone.end_s >= now_s:
            candidates.add(int(math.ceil(zone.end_s - now_s + 1)))
            candidates.add(int(math.ceil(zone.end_s - now_s + 1 + SERVICE_S)))
    return sorted(w for w in candidates if w >= 0)


def _service_conflicts(problem: ProblemData, u: Task, arrive_s: int) -> tuple[NoFlyZone, ...]:


    conflicts = []
    for zone in problem.zones:
        if arrive_s > zone.end_s or arrive_s + SERVICE_S < zone.start_s:
            continue
        if _distance((u.x_km, u.y_km), (zone.cx_km, zone.cy_km)) <= zone.safe_radius():
            conflicts.append(zone)
    return tuple(conflicts)


def _direct_path(p, q) -> VisibilityPath:
    return VisibilityPath(
        (np.asarray(p, dtype=float), np.asarray(q, dtype=float)), (), _distance(p, q)
    )


def _detour_path(p, q, center, radius, points, arc_angle, direction) -> VisibilityPath:
    start, end = points[1], points[2]
    theta_a = math.atan2(start[1] - center[1], start[0] - center[0])
    theta_b = math.atan2(end[1] - center[1], end[0] - center[0])
    tangent_len = _distance(points[0], points[1]) + _distance(points[2], points[3])
    return VisibilityPath(
        tuple(points),
        ((1, 2),),
        tangent_len + radius * arc_angle,
        (ArcLeg(1, 2, np.asarray(center, dtype=float), radius, direction),),
    )


def earliest_safe_travel(problem: ProblemData, u, v, depart_s: int) -> SegmentRecord | None:






    p = (u.x_km, u.y_km)
    q = (v.x_km, v.y_km)
    zones = problem.zones
    direct = _direct_path(p, q)
    direct_duration = _path_duration(direct)
    if spatiotemporal_safe(direct, depart_s, zones):
        return SegmentRecord(
            u.task_id, v.task_id, "direct", depart_s,
            depart_s + direct_duration, 0, direct.length_km, (),
        )

    candidates: list[SegmentRecord] = []


    for zone in zones:
        if zone.end_s < depart_s or zone.start_s > depart_s + direct_duration:
            continue
        interval = segment_circle_interval(
            p, q, (zone.cx_km, zone.cy_km), zone.safe_radius()
        )
        if interval is None or interval[1] - interval[0] <= 1e-6:
            continue
        t_entry = depart_s + interval[0] * direct_duration
        wait_candidates = sorted(
            set(event_delays(problem, (zone,), depart_s + direct_duration))
            | set(event_delays(problem, (zone,), t_entry))
        )
        for w in wait_candidates:
            if w <= 0:
                continue
            t0 = depart_s + w
            if not wait_safe(problem, p, depart_s, t0):
                continue
            leg = _direct_path(p, q)
            if spatiotemporal_safe(leg, t0, zones):
                candidates.append(
                    SegmentRecord(
                        u.task_id, v.task_id, "wait_direct", t0,
                        t0 + _path_duration(leg), w, leg.length_km, (),
                    )
                )


    for zone in zones:
        if zone.end_s < depart_s or zone.start_s > depart_s + direct_duration:
            continue
        interval = segment_circle_interval(
            p, q, (zone.cx_km, zone.cy_km), zone.safe_radius()
        )
        if interval is None or interval[1] - interval[0] <= 1e-6:
            continue
        arc_radius = zone.safe_radius() + EPS_ARC_KM
        result = detour_path(p, q, (zone.cx_km, zone.cy_km), arc_radius)
        if result is None:
            continue
        points, arc_angle, direction = result
        leg = _detour_path(p, q, (zone.cx_km, zone.cy_km), arc_radius, points, arc_angle, direction)
        if spatiotemporal_safe(leg, depart_s, zones):
            candidates.append(
                SegmentRecord(
                    u.task_id, v.task_id, "detour", depart_s,
                    depart_s + _path_duration(leg), 0, leg.length_km, (zone.zone_id,),
                )
            )


    bound = direct_duration * 4 + 3600
    obstacles = [
        (zone, zone.cx_km, zone.cy_km, zone.safe_radius())
        for zone in zones
        if zone.start_s <= depart_s + bound and zone.end_s >= depart_s
    ]
    if obstacles:
        result = visibility_path(
            p, q, [((cx, cy), radius) for _, cx, cy, radius in obstacles], margin=EPS_ARC_KM
        )
        if result is not None and spatiotemporal_safe(result, depart_s, zones):
            candidates.append(
                SegmentRecord(
                    u.task_id, v.task_id, "visibility", depart_s,
                    depart_s + _path_duration(result), 0, result.length_km,
                    tuple(zone.zone_id for zone, _, _, _ in obstacles),
                )
            )

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda record: (record.arrive_s, record.wait_s, record.distance_km),
    )


def earliest_safe_service_completion(
    problem: ProblemData, prev, u: Task, depart_s: int
) -> tuple[SegmentRecord, int, int] | None:





    travel = earliest_safe_travel(problem, prev, u, depart_s)
    if travel is None:
        return None
    arrive = travel.arrive_s
    conflicts = _service_conflicts(problem, u, arrive)
    if not conflicts:
        return travel, arrive, arrive + SERVICE_S

    p = (prev.x_km, prev.y_km)
    q = (u.x_km, u.y_km)
    direct_duration = _duration(_distance(p, q))
    candidates = {0}
    for zone in conflicts:
        interval = segment_circle_interval(
            p, q, (zone.cx_km, zone.cy_km), zone.safe_radius()
        )
        if interval is None:
            continue
        t_entry = depart_s + interval[0] * direct_duration
        candidates.add(int(math.ceil(zone.end_s - t_entry + 1)))

    for wait in sorted(w for w in candidates if w >= 0):
        if wait == 0:
            continue
        t0 = depart_s + wait
        if not wait_safe(problem, p, depart_s, t0):
            continue
        travel_w = earliest_safe_travel(problem, prev, u, t0)
        if travel_w is None:
            continue
        arrive_w = travel_w.arrive_s
        if not _service_conflicts(problem, u, arrive_w):
            merged = replace(travel_w, wait_s=travel_w.wait_s + wait)
            return merged, arrive_w, arrive_w + SERVICE_S
    return None


def eval_route(problem: ProblemData, route: tuple[int, ...], uav_id: int) -> UAVSchedule | None:

    prev: Task | Base = Base()
    depart = 0
    segments: list[SegmentRecord] = []
    service_intervals: list[tuple[int, int]] = []
    flight = 0
    wait = 0
    distance = 0.0
    for task_id in route:
        task = problem.tasks[task_id - 1]
        result = earliest_safe_service_completion(problem, prev, task, depart)
        if result is None:
            return None
        travel, arrive, service_end = result
        segments.append(travel)
        service_intervals.append((arrive, service_end))
        flight += travel.arrive_s - travel.depart_s
        wait += travel.wait_s
        distance += travel.distance_km
        prev = task
        depart = service_end
    ret = earliest_safe_travel(problem, prev, Base(), depart)
    if ret is None:
        return None
    segments.append(ret)
    flight += ret.arrive_s - ret.depart_s
    wait += ret.wait_s
    distance += ret.distance_km
    return UAVSchedule(
        uav_id, route, tuple(segments), tuple(service_intervals),
        ret.arrive_s, flight, wait, distance,
    )


def eval_solution(problem: ProblemData, task_routes: list[tuple[int, ...]]) -> Solution | None:

    if len(task_routes) != problem.fleet_size:
        return None
    schedules = []
    for index, route in enumerate(task_routes, 1):
        schedule = eval_route(problem, tuple(route), index)
        if schedule is None:
            return None
        schedules.append(schedule)
    s_max = max(schedule.S_k_s for schedule in schedules)
    s_min = min(schedule.S_k_s for schedule in schedules)
    return Solution(
        problem,
        tuple(schedules),
        SolutionMetrics(
            s_max,
            s_min,
            s_max - s_min,
            sum(schedule.S_k_s for schedule in schedules),
            sum(schedule.wait_s for schedule in schedules),
            round(sum(schedule.distance_km for schedule in schedules), 6),
        ),
    )


def base_safety_assert(problem: ProblemData) -> None:


    for zone in problem.zones:
        if zone.start_s <= 0 <= zone.end_s and _distance(
            (0.0, 0.0), (zone.cx_km, zone.cy_km)
        ) <= zone.safe_radius():
            raise AssertionError(
                f"base inside active zone {zone.zone_id} at t=0 (start {zone.start_s})"
            )
