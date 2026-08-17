"""Immutable legal route moves and deterministic route shortening."""

from dataclasses import dataclass
from typing import Iterable

from .domain import CAP_S, SERVICE_S, ProblemData, SolutionMetrics, Task
from .metrics import ObjectiveKey, Routes, normalize_routes, replay_metrics, strict_key


def _point_lookup(tasks: tuple[Task, ...]) -> dict[int, int]:
    lookup = {task.task_id: task.point_id for task in tasks}
    if set(lookup) != set(range(1, len(tasks) + 1)):
        raise ValueError("task IDs must be continuous from 1 through M")
    return lookup



@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One full-replayed candidate with immutable routes and strict key."""

    routes: Routes
    metrics: SolutionMetrics
    strict_key: ObjectiveKey


def score_candidate(problem: ProblemData, routes: Iterable[Iterable[int]]) -> ScoredCandidate | None:
    """Replay and score a complete fixed-fleet candidate or reject it."""
    try:
        canonical = normalize_routes(routes)
    except ValueError:
        return None
    if len(canonical) != problem.fleet_size or not is_legal_routes(canonical, problem.tasks):
        return None
    metrics = replay_metrics(problem.tasks, problem.distance_km, problem.time_s, canonical)
    if metrics.Tmax_s > CAP_S:
        return None
    return ScoredCandidate(canonical, metrics, strict_key(metrics.Tmax_s, metrics.delta_s, metrics.sum_T_s, canonical))


def accept_strict_improvement(
    incumbent: ScoredCandidate, problem: ProblemData, routes: Iterable[Iterable[int]]
) -> ScoredCandidate | None:
    """Return only a fully replayed strict-key improvement over incumbent."""
    candidate = score_candidate(problem, routes)
    if candidate is None or candidate.strict_key >= incumbent.strict_key:
        return None
    return candidate


def route_work_s(route: Iterable[int], time_s: tuple[tuple[int, ...], ...]) -> int:
    """Replay one depot-closed route in integer seconds."""
    canonical = tuple(route)
    path = (0,) + canonical + (0,)
    return sum(time_s[left][right] for left, right in zip(path, path[1:])) + SERVICE_S * len(canonical)


def _route_is_legal(route: tuple[int, ...], point_by_task: dict[int, int]) -> bool:
    if not route or any(task_id not in point_by_task for task_id in route):
        return False
    points = tuple(point_by_task[task_id] for task_id in route)
    return not any(left == right for left, right in zip(points, points[1:]))


def is_legal_routes(routes: Iterable[Iterable[int]], tasks: tuple[Task, ...]) -> bool:
    """Check Q2 hard routing constraints without mutating caller state."""
    try:
        canonical = normalize_routes(routes)
    except ValueError:
        return False
    point_by_task = _point_lookup(tasks)
    task_ids = tuple(task_id for route in canonical for task_id in route)
    expected = tuple(range(1, len(tasks) + 1))
    return (
        bool(canonical)
        and len(task_ids) == len(expected)
        and set(task_ids) == set(expected)
        and all(_route_is_legal(route, point_by_task) for route in canonical)
    )


def _replace_routes(routes: Routes, replacements: dict[int, tuple[int, ...]]) -> Routes:
    return tuple(replacements.get(index, route) for index, route in enumerate(routes))


def relocate(
    routes: Iterable[Iterable[int]],
    source_route: int,
    source_index: int,
    target_route: int,
    target_index: int,
    tasks: tuple[Task, ...],
) -> Routes | None:
    """Move one task between routes when resulting state remains legal."""
    try:
        canonical = normalize_routes(routes)
    except ValueError:
        return None
    if source_route == target_route or not _valid_route_index(canonical, source_route, target_route):
        return None
    source, target = canonical[source_route], canonical[target_route]
    if len(source) <= 1 or not 0 <= source_index < len(source) or not 0 <= target_index <= len(target):
        return None
    task_id = source[source_index]
    new_source = source[:source_index] + source[source_index + 1 :]
    new_target = target[:target_index] + (task_id,) + target[target_index:]
    candidate = _replace_routes(canonical, {source_route: new_source, target_route: new_target})
    return candidate if is_legal_routes(candidate, tasks) else None


def swap(
    routes: Iterable[Iterable[int]],
    left_route: int,
    left_index: int,
    right_route: int,
    right_index: int,
    tasks: tuple[Task, ...],
) -> Routes | None:
    """Swap one task between distinct routes when resulting state remains legal."""
    try:
        canonical = normalize_routes(routes)
    except ValueError:
        return None
    if left_route == right_route or not _valid_route_index(canonical, left_route, right_route):
        return None
    left, right = canonical[left_route], canonical[right_route]
    if not 0 <= left_index < len(left) or not 0 <= right_index < len(right):
        return None
    new_left = left[:left_index] + (right[right_index],) + left[left_index + 1 :]
    new_right = right[:right_index] + (left[left_index],) + right[right_index + 1 :]
    candidate = _replace_routes(canonical, {left_route: new_left, right_route: new_right})
    return candidate if is_legal_routes(candidate, tasks) else None


def _valid_route_index(routes: Routes, *indices: int) -> bool:
    return all(0 <= index < len(routes) for index in indices)


def deterministic_two_opt(
    route: Iterable[int], tasks: tuple[Task, ...], time_s: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    """Repeatedly apply first strict legal 2-opt improvement in fixed order."""
    current = tuple(route)
    point_by_task = _point_lookup(tasks)
    while True:
        improved = _first_two_opt_improvement(current, point_by_task, time_s)
        if improved is None:
            return current
        current = improved


def _first_two_opt_improvement(
    route: tuple[int, ...], point_by_task: dict[int, int], time_s: tuple[tuple[int, ...], ...]
) -> tuple[int, ...] | None:
    baseline = route_work_s(route, time_s)
    for start in range(len(route) - 1):
        for end in range(start + 2, len(route) + 1):
            candidate = route[:start] + tuple(reversed(route[start:end])) + route[end:]
            if _route_is_legal(candidate, point_by_task) and route_work_s(candidate, time_s) < baseline:
                return candidate
    return None


def routes_within_capacity(routes: Iterable[Iterable[int]], time_s: tuple[tuple[int, ...], ...]) -> bool:
    """Check depot-closed route workloads against Q2 nine-hour ceiling."""
    return all(route_work_s(route, time_s) <= CAP_S for route in normalize_routes(routes))
