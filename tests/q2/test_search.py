from types import MappingProxyType

from Q2.domain import ProblemData, Task
from Q2.metrics import balanced_key, epsilon_bound, replay_metrics
from Q2.search import derive_seed, run_epsilon_search, run_strict_search


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


def test_epsilon_search_rejects_malformed_bound():
    problem = _problem()

    for bad in (None, 0, 32_401, True, "100"):
        try:
            run_epsilon_search(problem, bound_s=bad, evaluation_limit=1, seed=7)
        except ValueError:
            continue
        raise AssertionError(f"bound_s={bad!r} must be rejected")
