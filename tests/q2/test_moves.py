import math

import pytest

from Q2.balance_core import (
    deterministic_two_opt,
    is_legal_routes,
    relocate,
    route_work_s,
    swap,
)
from Q2.domain import Task


def _tasks() -> tuple[Task, ...]:
    return (
        Task(1, 10, 1.0, 0.0),
        Task(2, 20, 8.0, 0.0),
        Task(3, 30, 2.0, 0.0),
        Task(4, 40, 9.0, 0.0),
    )


def _time_s(tasks: tuple[Task, ...]) -> tuple[tuple[int, ...], ...]:
    coords = ((0.0, 0.0),) + tuple((task.x, task.y) for task in tasks)
    return tuple(
        tuple(math.ceil(0.1 * math.dist(left, right) / 55.0 * 3600.0) for right in coords)
        for left in coords
    )


def test_is_legal_routes_requires_nonempty_complete_nonadjacent_partition():
    tasks = _tasks()

    assert is_legal_routes(((1, 2), (3, 4)), tasks)
    assert not is_legal_routes(((1, 1), (3, 4)), tasks)
    assert not is_legal_routes(((1, 2), (3,), ()), tasks)
    assert not is_legal_routes(((1, 2), (3, 5)), tasks)
    assert not is_legal_routes(((True, 2), (3, 4)), tasks)


def test_relocate_keeps_source_nonempty_and_rejects_adjacent_same_point():
    tasks = _tasks()
    routes = ((1, 2), (3, 4))
    repeated_point_tasks = (Task(1, 10, 1.0, 0.0), Task(2, 20, 8.0, 0.0), Task(3, 10, 2.0, 0.0), Task(4, 40, 9.0, 0.0))

    assert relocate(routes, 0, 0, 1, 1, tasks) == ((2,), (3, 1, 4))
    assert relocate(routes, 0, 0, 1, 0, repeated_point_tasks) is None
    assert relocate(((1,), (2, 3, 4)), 0, 0, 1, 0, tasks) is None
    assert relocate(((1.0, 2), (3, 4)), 0, 0, 1, 0, tasks) is None


def test_swap_returns_only_legal_cross_route_exchange():
    tasks = _tasks()
    repeated_point_tasks = (Task(1, 10, 1.0, 0.0), Task(2, 10, 8.0, 0.0), Task(3, 30, 2.0, 0.0), Task(4, 40, 9.0, 0.0))

    assert swap(((1, 2), (3, 4)), 0, 0, 1, 0, tasks) == ((3, 2), (1, 4))
    assert swap(((1, 3), (2, 4, 5)), 0, 0, 1, 1, repeated_point_tasks + (Task(5, 10, 10.0, 0.0),)) is None
    assert swap(((1, 2), (3, 4)), 0, 0, 0, 1, tasks) is None


def test_deterministic_two_opt_returns_only_strict_legal_shortening():
    tasks = _tasks()
    time_s = _time_s(tasks)
    route = (1, 4, 3, 2)

    optimized = deterministic_two_opt(route, tasks, time_s)

    assert optimized == (1, 3, 4, 2)
    assert route_work_s(optimized, time_s) < route_work_s(route, time_s)


@pytest.mark.parametrize("route", ((), (1,), (1, 2)))
def test_deterministic_two_opt_preserves_short_routes(route):
    tasks = _tasks()
    time_s = _time_s(tasks)

    assert deterministic_two_opt(route, tasks, time_s) == route
