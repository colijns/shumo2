from types import MappingProxyType

from Q2.domain import ProblemData, Task
from Q2.metrics import balanced_key, epsilon_bound, replay_metrics
from Q2.search import (
    _normalize_changed_routes,
    _prioritized_route_pairs,
    derive_seed,
    run_epsilon_search,
    run_strict_search,
)


def _problem() -> ProblemData:
    tasks = (
        Task(1, 10, 0.0, 0.0),
        Task(2, 20, 0.0, 0.0),
        Task(3, 30, 0.0, 0.0),
        Task(4, 40, 0.0, 0.0),
    )
    time_s = (
        (0, 1, 10, 1, 10),
        (1, 0, 20, 1, 20),
        (10, 20, 0, 20, 1),
        (1, 1, 20, 0, 20),
        (10, 20, 1, 20, 0),
    )
    distance = tuple(tuple(float(value) for value in row) for row in time_s)
    routes = ((1, 2), (3, 4))
    return ProblemData(
        "CaseX", 2, tasks, routes, distance, time_s, replay_metrics(tasks, distance, time_s, routes),
        MappingProxyType({}), "attachment", "archive", "a" * 64, "b" * 64, "c" * 64, True,
    )


def test_derive_seed_is_stable_and_namespace_sensitive():
    assert derive_seed(42, "Case1", "strict", 3) == derive_seed(42, "Case1", "strict", 3)
    assert derive_seed(42, "Case1", "strict", 3) != derive_seed(42, "Case1", "epsilon", 3)


def test_strict_search_is_deterministic_and_strictly_improves_when_possible():
    problem = _problem()

    first = run_strict_search(problem, evaluation_limit=100, seed=7)
    second = run_strict_search(problem, evaluation_limit=100, seed=7)

    assert first == second
    assert first.incumbent.strict_key <= first.initial.strict_key
    assert first.improvements >= 0
    assert first.evaluations <= 100
    assert first.incumbent.metrics.Tmax_s <= problem.metrics.Tmax_s


def test_strict_search_respects_zero_budget_and_keeps_valid_initial_solution():
    problem = _problem()

    result = run_strict_search(problem, evaluation_limit=0, seed=7)

    assert result.evaluations == 0
    assert result.incumbent.routes == result.initial.routes
    assert result.incumbent.metrics == result.initial.metrics


def test_route_normalization_only_replays_changed_routes(monkeypatch):
    problem = _problem()
    calls = []

    def record(route, tasks, time_s):
        calls.append(route)
        return route

    monkeypatch.setattr("Q2.search.deterministic_two_opt", record)
    before = ((1, 2), (3,), (4,))
    after = ((1,), (3, 2), (4,))

    assert _normalize_changed_routes(problem, before, after) == after
    assert calls == [(1,), (3, 2)]


def test_route_pairs_prioritize_largest_workload_imbalance():
    problem = _problem()
    routes = ((1, 2), (3,), (4,))

    relocate_pairs, swap_pairs = _prioritized_route_pairs(problem, routes)

    workloads = [sum(problem.time_s[left][right] for left, right in zip((0,) + route, route + (0,))) + 300 * len(route) for route in routes]
    assert relocate_pairs[0] == (workloads.index(max(workloads)), workloads.index(min(workloads)))
    assert abs(workloads[swap_pairs[0][0]] - workloads[swap_pairs[0][1]]) == max(
        abs(workloads[left] - workloads[right]) for left, right in swap_pairs
    )


def _balanced_key(candidate):
    metrics = candidate.metrics
    return balanced_key(metrics.Tmax_s, metrics.delta_s, metrics.sum_T_s, candidate.routes)


def test_epsilon_search_is_deterministic_respects_bound_and_improves_balance():
    problem = _problem()
    strict = run_strict_search(problem, evaluation_limit=100, seed=7)
    bound_s = epsilon_bound(strict.incumbent.metrics.Tmax_s, "0.02")

    first = run_epsilon_search(problem, bound_s=bound_s, evaluation_limit=100, seed=7)
    second = run_epsilon_search(problem, bound_s=bound_s, evaluation_limit=100, seed=7)

    assert first == second
    assert first.incumbent.metrics.Tmax_s <= bound_s
    assert first.evaluations <= 100
    assert _balanced_key(first.incumbent) <= _balanced_key(first.initial)


def test_epsilon_search_uses_explicit_strict_warm_start():
    problem = _problem()
    warm_start = ((1, 3), (2, 4))

    result = run_epsilon_search(
        problem,
        bound_s=32_400,
        evaluation_limit=0,
        seed=7,
        initial_routes=warm_start,
    )

    assert result.initial.routes == warm_start
    assert result.incumbent.routes == warm_start


def test_epsilon_search_rejects_malformed_bound():
    problem = _problem()

    for bad in (None, 0, 32_401, True, "100"):
        try:
            run_epsilon_search(problem, bound_s=bad, evaluation_limit=1, seed=7)
        except ValueError:
            continue
        raise AssertionError(f"bound_s={bad!r} must be rejected")
