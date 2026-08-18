

import math
from dataclasses import dataclass

import numpy as np

Point = np.ndarray

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class ArcLeg:





    start: int
    end: int
    center: Point
    radius_km: float
    direction: int


@dataclass(frozen=True)
class VisibilityPath:









    points: tuple[Point, ...]
    arc_pairs: tuple[tuple[int, int], ...]
    length_km: float
    arc_legs: tuple[ArcLeg, ...] = ()


def _vec(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def segment_circle_interval(p, q, c, r) -> tuple[float, float] | None:





    p, q, c = _vec(p), _vec(q), _vec(c)
    d = q - p
    f = p - c
    a = float(d @ d)
    if a <= 1e-18:
        return None
    b = 2.0 * float(f @ d)
    cc = float(f @ f) - r * r
    disc = b * b - 4.0 * a * cc
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    t0 = max((-b - s) / (2.0 * a), 0.0)
    t1 = min((-b + s) / (2.0 * a), 1.0)
    if t0 > t1:
        return None
    return (t0, t1)


def segment_clear(p, q, disks, margin: float = 0.0) -> bool:




    p, q = _vec(p), _vec(q)
    for center, radius in disks:
        radius = max(radius - margin, 0.0)
        interval = segment_circle_interval(p, q, center, radius)
        if interval is None:
            continue
        t0, t1 = interval


        if t1 - t0 > 1e-6:
            return False
    return True


def point_inside(point, center, radius) -> bool:
    return float(np.linalg.norm(_vec(point) - _vec(center))) <= radius


def tangent_points(p, c, r) -> tuple[Point, Point] | None:

    p, c = _vec(p), _vec(c)
    d = float(np.linalg.norm(p - c))
    if d <= r:
        return None
    base = math.atan2(p[1] - c[1], p[0] - c[0])
    beta = math.acos(r / d)
    t1 = c + r * np.array([math.cos(base + beta), math.sin(base + beta)])
    t2 = c + r * np.array([math.cos(base - beta), math.sin(base - beta)])
    return t1, t2


def detour_path(p, q, c, r) -> tuple[list[Point], float, int] | None:





    p, q, c = _vec(p), _vec(q), _vec(c)
    tp = tangent_points(p, c, r)
    tq = tangent_points(q, c, r)
    if tp is None or tq is None:
        return None
    best_arc = None
    best = None
    for a in tp:
        for b in tq:
            ang_a = math.atan2(a[1] - c[1], a[0] - c[0])
            ang_b = math.atan2(b[1] - c[1], b[0] - c[0])
            forward = (ang_b - ang_a) % TAU
            arc = min(forward, TAU - forward)
            if best is None or arc < best_arc:
                best_arc = arc
                direction = 1 if forward <= TAU - forward else -1
                best = (a, b, direction)
    a, b, direction = best
    return [p, a, b, q], best_arc, direction


def detour_distance(p, q, c, r) -> float | None:

    result = detour_path(p, q, c, r)
    if result is None:
        return None
    points, arc_angle, _ = result
    tangent_len = float(np.linalg.norm(points[1] - points[0])) + float(
        np.linalg.norm(points[3] - points[2])
    )
    return tangent_len + r * arc_angle


def _normalized_intervals(lo: float, hi: float) -> list[tuple[float, float]]:

    if hi - lo <= 1e-12:
        return []
    start = lo % TAU
    span = hi - lo
    if start + span <= TAU + 1e-12:
        return [(start, min(start + span, TAU))]
    return [(start, TAU), (0.0, start + span - TAU)]


def arc_intersection_with_circle(
    c1, r1, theta_start, theta_end, direction, c2, r2
) -> list[tuple[float, float]]:






    c1, c2 = _vec(c1), _vec(c2)
    center_distance = float(np.linalg.norm(c2 - c1))
    arc_sweep = _arc_step(theta_start, theta_end, direction)
    arc_lo = min(theta_start, theta_start + arc_sweep)
    arc_hi = max(theta_start, theta_start + arc_sweep)

    if center_distance >= r1 + r2 or center_distance <= abs(r1 - r2):

        mid = (theta_start + arc_sweep / 2.0) % TAU
        probe = c1 + r1 * np.array([math.cos(mid), math.sin(mid)])
        if float(np.linalg.norm(probe - c2)) <= r2:
            return _normalized_intervals(arc_lo, arc_hi)
        return []


    angle = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
    offset = math.acos(
        (center_distance * center_distance + r1 * r1 - r2 * r2)
        / (2.0 * center_distance * r1)
    )
    t1 = (angle + offset) % TAU
    t2 = (angle - offset) % TAU
    sweep = (t2 - t1) % TAU
    probe_mid = t1 + sweep / 2.0
    probe = c1 + r1 * np.array([math.cos(probe_mid), math.sin(probe_mid)])
    if float(np.linalg.norm(probe - c2)) <= r2:
        inside_parts = _normalized_intervals(t1, t1 + sweep)
    else:
        inside_parts = _normalized_intervals(t2, t2 + TAU - sweep)
    arc_parts = _normalized_intervals(arc_lo, arc_hi)

    result = []
    for in_lo, in_hi in inside_parts:
        for ar_lo, ar_hi in arc_parts:
            lo = max(in_lo, ar_lo)
            hi = min(in_hi, ar_hi)
            if lo < hi - 1e-12:
                result.append((lo, hi))
    return result


def _arc_step(start, end, direction) -> float:

    if direction > 0:
        return (end - start) % TAU
    return -((start - end) % TAU)


def arc_intersects_disk(c1, r1, theta_start, theta_end, direction, c2, r2) -> bool:
    return bool(arc_intersection_with_circle(c1, r1, theta_start, theta_end, direction, c2, r2))


def visibility_path(p, q, disks, margin: float = 0.0) -> VisibilityPath | None:







    p, q = _vec(p), _vec(q)
    disks = [(_vec(center), float(radius)) for center, radius in disks]
    if segment_clear(p, q, disks, margin):
        return VisibilityPath((p, q), (), float(np.linalg.norm(q - p)))

    nodes: list[Point] = [p, q]
    for center, radius in disks:
        for origin in (p, q):
            tangents = tangent_points(origin, center, radius)
            if tangents is not None:
                nodes.extend(tangents)
    for index, (c1, r1) in enumerate(disks):
        for c2, r2 in disks[index + 1:]:
            center_distance = float(np.linalg.norm(c2 - c1))
            if center_distance >= r1 + r2 or center_distance <= abs(r1 - r2):
                continue
            angle = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
            offset = math.acos(
                (center_distance * center_distance + r1 * r1 - r2 * r2)
                / (2.0 * center_distance * r1)
            )
            nodes.append(c1 + r1 * np.array([math.cos(angle + offset), math.sin(angle + offset)]))
            nodes.append(c1 + r1 * np.array([math.cos(angle - offset), math.sin(angle - offset)]))

    unique: list[Point] = []
    for node in nodes:
        if all(float(np.linalg.norm(node - existing)) > 1e-9 for existing in unique):
            unique.append(node)
    nodes = unique

    size = len(nodes)
    dist = [math.inf] * size
    prev: list[tuple[int, float, tuple | None] | None] = [None] * size
    visited = [False] * size
    dist[0] = 0.0

    def relax(a: int, b: int, weight: float, edge_info: tuple | None) -> None:
        if visited[b]:
            return
        candidate = dist[a] + weight
        if candidate < dist[b]:
            dist[b] = candidate
            prev[b] = (a, weight, edge_info)

    for _ in range(size):
        current = -1
        best = math.inf
        for index in range(size):
            if not visited[index] and dist[index] < best:
                best = dist[index]
                current = index
        if current < 0 or current == 1:
            break
        visited[current] = True

        for other in range(size):
            if visited[other]:
                continue
            if segment_clear(nodes[current], nodes[other], disks, margin):
                relax(current, other, float(np.linalg.norm(nodes[other] - nodes[current])), None)



        for disk_index, (center, radius) in enumerate(disks):
            arc_radius = radius + margin
            on_ring = sorted(
                (
                    index,
                    math.atan2(nodes[index][1] - center[1], nodes[index][0] - center[0]),
                )
                for index in range(size)
                if abs(float(np.linalg.norm(nodes[index] - center)) - radius) < 1e-6
            )
            for position, (a, theta_a) in enumerate(on_ring):
                b, theta_b = on_ring[(position + 1) % len(on_ring)]
                forward = (theta_b - theta_a) % TAU
                arc_angle = min(forward, TAU - forward)
                if arc_angle < 1e-9 or a == b:
                    continue
                direction = 1 if forward <= TAU - forward else -1
                others = [
                    (c2, r2)
                    for index, (c2, r2) in enumerate(disks)
                    if index != disk_index
                ]
                if any(
                    arc_intersects_disk(center, arc_radius, theta_a, theta_b, direction, c2, r2)
                    for c2, r2 in others
                ):
                    continue
                arc_info = ("arc", center, arc_radius, theta_a, theta_b, direction)
                if a == current:
                    relax(a, b, arc_radius * arc_angle, arc_info)
                elif b == current:
                    relax(b, a, arc_radius * arc_angle, arc_info)
    if math.isinf(dist[1]):
        return None

    reversed_nodes: list[Point] = [nodes[1]]
    reversed_edge_infos: list[tuple | None] = []
    cursor = 1
    while cursor != 0:
        entry = prev[cursor]
        if entry is None:
            return None
        parent, _, edge_info = entry
        reversed_edge_infos.append(edge_info)
        reversed_nodes.append(nodes[parent])
        cursor = parent
    points = tuple(reversed(reversed_nodes))
    edge_infos = tuple(reversed(reversed_edge_infos))
    arc_pairs: list[tuple[int, int]] = []
    arc_legs: list[ArcLeg] = []
    for index, edge_info in enumerate(edge_infos):
        if edge_info is not None:
            _, center, radius, theta_a, theta_b, direction = edge_info
            arc_pairs.append((index, index + 1))
            arc_legs.append(ArcLeg(index, index + 1, center, radius, direction))
    return VisibilityPath(points, tuple(arc_pairs), dist[1], tuple(arc_legs))


def path_length(points) -> float:

    points = [_vec(point) for point in points]
    return sum(float(np.linalg.norm(points[i + 1] - points[i])) for i in range(len(points) - 1))
