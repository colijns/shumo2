from types import MappingProxyType

from Q2.balance_core import accept_strict_improvement, score_candidate
from Q2.domain import ProblemData, Task
from Q2.metrics import replay_metrics, strict_key


def _tasks() -> tuple[Task, ...]:
    return (
        Task(1, 10, 0.0, 0.0),
        Task(2, 20, 0.0, 0.0),
        Task(3, 30, 0.0, 0.0),
        Task(4, 40, 0.0, 0.0),
    )


def _time_s() -> tuple[tuple[int, ...], ...]:
    return (
        (0, 1, 10, 1, 10),
        (1, 0, 20, 1, 20),
        (10, 20, 0, 20, 1),
        (1, 1, 20, 0, 20),
        (10, 20, 1, 20, 0),
    )


def _problem(time_s: tuple[tuple[int, ...], ...] | None = None) -> ProblemData:
    tasks = _tasks()
    matrix = time_s or _time_s()
    routes = ((1, 2), (3, 4))
    metrics = replay_metrics(tasks, tuple(tuple(float(value) for value in row) for row in matrix), matrix, routes)
    return ProblemData(
        "CaseX", 2, tasks, routes, tuple(tuple(float(value) for value in row) for row in matrix), matrix,
        metrics, MappingProxyType({}), "attachment", "archive", "a" * 64, "b" * 64, "c" * 64, True,
    )


def test_score_candidate_replays_complete_fixed_fleet_solution():
    problem = _problem()
    routes = ((1, 3), (2, 4))

    candidate = score_candidate(problem, routes)

    assert candidate is not None
    assert candidate.routes == routes
    assert candidate.metrics == replay_metrics(problem.tasks, problem.distance_km, problem.time_s, routes)
    assert candidate.strict_key == strict_key(
        candidate.metrics.Tmax_s, candidate.metrics.delta_s, candidate.metrics.sum_T_s, routes
    )


def test_score_candidate_rejects_wrong_fleet_and_over_capacity():
    problem = _problem()
    expensive = tuple(tuple(20_000 if left != right else 0 for right in range(5)) for left in range(5))

    assert score_candidate(problem, ((1, 2, 3, 4),)) is None
    assert score_candidate(problem, ((1,), (2,), (3, 4))) is None
    assert score_candidate(_problem(expensive), ((1, 2), (3, 4))) is None


def test_accept_strict_improvement_replays_before_accepting():
    problem = _problem()
    incumbent = score_candidate(problem, problem.routes)

    assert incumbent is not None
    assert accept_strict_improvement(incumbent, problem, problem.routes) is None
    improved = accept_strict_improvement(incumbent, problem, ((1, 3), (2, 4)))

    assert improved is not None
    assert improved.strict_key < incumbent.strict_key
    assert improved.metrics == replay_metrics(problem.tasks, problem.distance_km, problem.time_s, improved.routes)
