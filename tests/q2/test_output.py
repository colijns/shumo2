from types import MappingProxyType

import pandas as pd
import pytest

from Q2.balance_core import score_candidate
from Q2.domain import ProblemData, Task
from Q2.io import build_solution_archive, validate_solution_archive, write_result_workbook
from Q2.metrics import epsilon_bound
from Q2.metrics import replay_metrics


def _problem(case_name: str = "Case1") -> ProblemData:
    tasks = (Task(1, 10, 0.0, 0.0), Task(2, 20, 0.0, 0.0), Task(3, 30, 0.0, 0.0), Task(4, 40, 0.0, 0.0))
    time_s = ((0, 1, 10, 1, 10), (1, 0, 20, 1, 20), (10, 20, 0, 20, 1), (1, 1, 20, 0, 20), (10, 20, 1, 20, 0))
    distance = tuple(tuple(float(value) for value in row) for row in time_s)
    routes = ((1, 2), (3, 4))
    return ProblemData(case_name, 2, tasks, routes, distance, time_s, replay_metrics(tasks, distance, time_s, routes), MappingProxyType({}), "attachment", "archive", "a" * 64, "b" * 64, "c" * 64, True)


def test_build_solution_archive_contains_replayable_provenance():
    problem = _problem()
    candidate = score_candidate(problem, problem.routes)
    assert candidate is not None

    archive = build_solution_archive(problem, candidate)

    assert archive["input"]["problem_contract_sha256"] == problem.problem_contract_sha256
    assert archive["task_routes"] == [[1, 2], [3, 4]]
    assert archive["metrics"]["route_work_s"] == list(candidate.metrics.work_s)


def test_epsilon_archive_must_replay_with_declared_bound():
    problem = _problem()
    candidate = score_candidate(problem, problem.routes)
    assert candidate is not None

    archive = build_solution_archive(problem, candidate, track="epsilon_formal", epsilon="0.005")

    assert validate_solution_archive(problem, archive) == candidate
    assert candidate.metrics.Tmax_s <= epsilon_bound(candidate.metrics.Tmax_s, "0.005")


def test_epsilon_archive_rejects_invalid_epsilon_label():
    problem = _problem()
    candidate = score_candidate(problem, problem.routes)
    assert candidate is not None
    archive = build_solution_archive(problem, candidate, track="epsilon_formal", epsilon="bad")

    with pytest.raises(ValueError, match="epsilon"):
        validate_solution_archive(problem, archive)


def test_result_workbook_contains_only_case_sheets_with_point_sequences(tmp_path):
    problem = _problem()
    candidate = score_candidate(problem, problem.routes)
    assert candidate is not None
    path = tmp_path / "result2.xlsx"

    write_result_workbook(path, {problem.case_name: (problem, candidate)})

    workbook = pd.ExcelFile(path)
    assert workbook.sheet_names == ["Case1"]
    frame = pd.read_excel(path, sheet_name="Case1")
    assert frame.iloc[:, 0].tolist() == [1, 2]
    assert frame.columns.tolist() == ["UAV ID", "1th Inspection Point", "2th Inspection Point"]
    assert frame.iloc[0, 1:3].tolist() == [10, 20]
    assert 0 not in frame.fillna(-1).values
