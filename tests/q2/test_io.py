import json
from types import MappingProxyType

import pytest

from Q2.balance_core import score_candidate
from Q2.domain import ProblemData, Task
from Q2.io import CheckpointError, atomic_write_json, load_checkpoint, validate_solution_archive
from Q2.metrics import replay_metrics


def _problem() -> ProblemData:
    tasks = (Task(1, 10, 0.0, 0.0), Task(2, 20, 0.0, 0.0), Task(3, 30, 0.0, 0.0), Task(4, 40, 0.0, 0.0))
    time_s = ((0, 1, 10, 1, 10), (1, 0, 20, 1, 20), (10, 20, 0, 20, 1), (1, 1, 20, 0, 20), (10, 20, 1, 20, 0))
    distance = tuple(tuple(float(value) for value in row) for row in time_s)
    routes = ((1, 2), (3, 4))
    return ProblemData("CaseX", 2, tasks, routes, distance, time_s, replay_metrics(tasks, distance, time_s, routes), MappingProxyType({}), "attachment", "archive", "a" * 64, "b" * 64, "c" * 64, True)


def _archive(problem: ProblemData) -> dict:
    candidate = score_candidate(problem, problem.routes)
    assert candidate is not None
    return {
        "schema_version": "q2-solution-v1",
        "case": problem.case_name,
        "fleet_size": problem.fleet_size,
        "parent_archive_sha256": problem.archive_sha256,
        "task_routes": [list(route) for route in candidate.routes],
        "metrics": {"Tmax_s": candidate.metrics.Tmax_s, "Tmin_s": candidate.metrics.Tmin_s, "delta_s": candidate.metrics.delta_s, "sum_T_s": candidate.metrics.sum_T_s},
    }


def test_atomic_json_write_round_trips_exact_data(tmp_path):
    path = tmp_path / "state.json"
    payload = {"alpha": [1, 2], "nested": {"ok": True}}

    atomic_write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_checkpoint_rejects_problem_contract_hash_drift(tmp_path):
    path = tmp_path / "checkpoint.json"
    atomic_write_json(path, {"schema_version": "q2-checkpoint-v1", "problem_contract_sha256": "a" * 64})

    assert load_checkpoint(path, "a" * 64)["schema_version"] == "q2-checkpoint-v1"
    with pytest.raises(CheckpointError, match="contract"):
        load_checkpoint(path, "b" * 64)
    with pytest.raises(CheckpointError, match="SHA-256"):
        load_checkpoint(path, True)


def test_solution_validator_replays_routes_and_rejects_tampered_metrics():
    problem = _problem()
    archive = _archive(problem)

    validated = validate_solution_archive(problem, archive)
    assert validated.metrics == problem.metrics

    archive["metrics"]["delta_s"] += 1
    with pytest.raises(ValueError, match="delta_s"):
        validate_solution_archive(problem, archive)
    archive = _archive(problem)
    archive["metrics"]["Tmin_s"] = True
    with pytest.raises(ValueError, match="exact integer"):
        validate_solution_archive(problem, archive)
