from types import MappingProxyType

from Q2.balance_core import score_candidate
from Q2.domain import ProblemData, Task
from Q2.epsilon_state import EPSILON_VALUES, create_dual_track_state, propagate_candidate
from Q2.metrics import replay_metrics


def _problem() -> ProblemData:
    tasks = (Task(1, 10, 0.0, 0.0), Task(2, 20, 0.0, 0.0), Task(3, 30, 0.0, 0.0), Task(4, 40, 0.0, 0.0))
    time_s = ((0, 1, 10, 1, 10), (1, 0, 20, 1, 20), (10, 20, 0, 20, 1), (1, 1, 20, 0, 20), (10, 20, 1, 20, 0))
    distance = tuple(tuple(float(value) for value in row) for row in time_s)
    routes = ((1, 2), (3, 4))
    return ProblemData("CaseX", 2, tasks, routes, distance, time_s, replay_metrics(tasks, distance, time_s, routes), MappingProxyType({}), "attachment", "archive", "a" * 64, "b" * 64, "c" * 64, True)


def test_dual_track_state_initializes_exact_published_epsilon_grid():
    problem = _problem()
    initial = score_candidate(problem, problem.routes)
    assert initial is not None

    state = create_dual_track_state(initial)

    assert EPSILON_VALUES == tuple(track.epsilon for track in state.epsilon_incumbents)
    assert all(track.candidate == initial for track in state.epsilon_incumbents)


def test_propagation_keeps_strict_and_selects_balanced_eligible_candidate():
    problem = _problem()
    initial = score_candidate(problem, problem.routes)
    balanced = score_candidate(problem, ((1, 3), (2, 4)))
    assert initial is not None and balanced is not None
    state = create_dual_track_state(initial)

    updated = propagate_candidate(state, balanced)

    assert updated.strict_incumbent == min((initial, balanced), key=lambda candidate: candidate.strict_key)
    assert all(track.candidate.metrics.Tmax_s <= track.bound_s for track in updated.epsilon_incumbents)


def test_strict_improvement_creates_epoch_and_rebuilds_all_bounds():
    problem = _problem()
    initial = score_candidate(problem, problem.routes)
    improved = score_candidate(problem, ((1, 3), (2, 4)))
    assert initial is not None and improved is not None
    state = create_dual_track_state(initial)

    updated = propagate_candidate(state, improved)

    expected_epoch = int(improved.strict_key < initial.strict_key)
    assert updated.strict_epoch == expected_epoch
    assert all(track.bound_s >= updated.strict_incumbent.metrics.Tmax_s for track in updated.epsilon_incumbents)
