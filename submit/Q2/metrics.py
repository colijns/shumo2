

import math
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from fractions import Fraction
from typing import Iterable

from .domain import CAP_S, SERVICE_S, RouteMetrics, SolutionMetrics, Task

Routes = tuple[tuple[int, ...], ...]
ObjectiveKey = tuple[int, int, int, Routes]


def normalize_routes(routes: Iterable[Iterable[int]]) -> Routes:

    try:
        canonical = tuple(tuple(route) for route in routes)
    except TypeError as exc:
        raise ValueError("routes must be iterables of exact integers") from exc
    if any(type(task_id) is not int for route in canonical for task_id in route):
        raise ValueError("routes must contain exact integers")
    return canonical


def _validate_candidate_task_ids(routes: Routes, task_count: int) -> None:

    task_ids = tuple(task_id for route in routes for task_id in route)
    expected = set(range(1, task_count + 1))
    if any(task_id not in expected for task_id in task_ids):
        raise ValueError(f"candidate task IDs must be in 1..{task_count}")
    if len(task_ids) != task_count or set(task_ids) != expected:
        raise ValueError(f"candidate task IDs must cover 1..{task_count} exactly once")


def canonical_route_signature(routes: Iterable[Iterable[int]]) -> Routes:

    return tuple(sorted(normalize_routes(routes)))


def strict_key(Tmax_s: int, delta_s: int, sum_T_s: int, routes: Iterable[Iterable[int]]) -> ObjectiveKey:

    return Tmax_s, delta_s, sum_T_s, canonical_route_signature(routes)


def balanced_key(Tmax_s: int, delta_s: int, sum_T_s: int, routes: Iterable[Iterable[int]]) -> ObjectiveKey:

    return delta_s, Tmax_s, sum_T_s, canonical_route_signature(routes)


def _epsilon_decimal(epsilon: object) -> Decimal:
    try:
        value = Decimal(str(epsilon))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("epsilon must be a finite nonnegative number") from exc
    if not value.is_finite() or value < 0:
        raise ValueError("epsilon must be a finite nonnegative number")
    return value


def epsilon_bound(Tmax_star: int, epsilon: object) -> int:

    if isinstance(Tmax_star, bool) or not isinstance(Tmax_star, int):
        raise ValueError("Tmax_star must be an integer number of seconds")
    if not 1 <= Tmax_star <= CAP_S:
        raise ValueError(f"Tmax_star must be in 1..{CAP_S}")
    product = (Decimal(Tmax_star) * (Decimal(1) + _epsilon_decimal(epsilon))).to_integral_value(
        rounding=ROUND_FLOOR
    )
    return min(CAP_S, int(product))


def _task_lookup(tasks: tuple[Task, ...]) -> dict[int, Task]:
    lookup = {task.task_id: task for task in tasks}
    expected = set(range(1, len(tasks) + 1))
    if set(lookup) != expected:
        raise ValueError("task IDs must be continuous from 1 through M")
    return lookup


def _replay_route(
    route: tuple[int, ...],
    lookup: dict[int, Task],
    distance_km: tuple[tuple[float, ...], ...],
    time_s: tuple[tuple[int, ...], ...],
) -> RouteMetrics:
    path = (0,) + route + (0,)
    flight_s = sum(time_s[left][right] for left, right in zip(path, path[1:]))
    distance = sum(distance_km[left][right] for left, right in zip(path, path[1:]))
    points = tuple(lookup[task_id].point_id for task_id in route)
    return RouteMetrics(route, points, int(flight_s), float(distance), int(flight_s + SERVICE_S * len(route)))


def replay_metrics(
    tasks: tuple[Task, ...],
    distance_km: tuple[tuple[float, ...], ...],
    time_s: tuple[tuple[int, ...], ...],
    routes: Iterable[Iterable[int]],
) -> SolutionMetrics:

    canonical = normalize_routes(routes)
    if not canonical or any(not route for route in canonical):
        raise ValueError("all fleet routes must be nonempty")
    lookup = _task_lookup(tasks)
    _validate_candidate_task_ids(canonical, len(lookup))
    route_metrics = tuple(_replay_route(route, lookup, distance_km, time_s) for route in canonical)
    work = tuple(route.work_s for route in route_metrics)
    mean = Fraction(sum(work), len(work))
    std = math.sqrt(sum((value - float(mean)) ** 2 for value in work) / len(work))
    return SolutionMetrics(
        routes=route_metrics,
        work_s=work,
        flight_s=tuple(route.flight_s for route in route_metrics),
        distance_km=tuple(route.distance_km for route in route_metrics),
        Tmax_s=max(work),
        Tmin_s=min(work),
        delta_s=max(work) - min(work),
        sum_T_s=sum(work),
        mean_T_s=mean,
        std_T_s=std,
    )
