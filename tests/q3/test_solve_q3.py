"""M4: solve_q3 CLI publication chain — archive, timetable, manifest, workbook."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from Q3.domain import Config
from Q3.safe_path import eval_solution
from tests.q3.conftest import make_problem


@pytest.fixture
def mini_solution():
    solution = eval_solution(make_problem(zones=()), [(1, 2), (3, 4)])
    assert solution is not None
    return solution


def _solution_for_case(case: str, config=None, repository_root=None) -> object:
    """Mirror of solve_case: a feasible mini plan whose problem.case matches."""
    base = make_problem(zones=())
    problem = replace(base, case=case)
    solution = eval_solution(problem, [(1, 2), (3, 4)])
    assert solution is not None
    return solution


def test_solve_and_publish_writes_archive_and_timetable(tmp_path, monkeypatch, mini_solution):
    import Q3.solve_q3 as solve

    root = tmp_path / "repo"
    (root / "outputs" / "workbooks" / "q3").mkdir(parents=True)
    monkeypatch.setattr(solve, "solve_case",
                        lambda case, config, repository_root=None: mini_solution)
    result = solve._solve_and_publish("Mini", Config(), root)

    archive_path = root / "outputs" / "workbooks" / "q3" / "strict" / "Mini.json"
    timetable = root / "outputs" / "workbooks" / "q3" / "timetables" / "Mini_timetable.csv"
    assert archive_path.is_file()
    assert timetable.is_file()
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    assert archive["schema_version"] == "q3-solution-v1"
    assert archive["case"] == "Mini"
    assert archive["metrics"]["S_max_s"] == mini_solution.metrics.S_max_s
    assert archive["hot_start"]["source"] == "none"  # no Q2 strict / Q1 baseline here
    assert result["hot_start"]["fallback"] == "greedy"

    frame = pd.read_csv(timetable, encoding="utf-8")
    assert list(frame.columns) == [
        "case", "uav_id", "segment_id", "from_id", "to_id", "path_type",
        "depart_time", "arrive_time", "service_start", "service_end",
        "wait_seconds", "distance_km", "affected_zones",
    ]
    assert len(frame) == 6  # 2 uavs x 3 legs


def test_publish_workbook_and_manifest_round_trip(tmp_path, monkeypatch, mini_solution):
    import Q3.solve_q3 as solve

    root = tmp_path / "repo"
    (root / "outputs" / "workbooks" / "q3").mkdir(parents=True)
    monkeypatch.setattr(solve, "solve_case", _solution_for_case)
    solve.main(["--all", "--time-budget", "1", "--repository-root", str(root)])

    # the four pieces exist
    strict = root / "outputs" / "workbooks" / "q3" / "strict" / "Case1.json"
    timetable = root / "outputs" / "workbooks" / "q3" / "timetables" / "Case1_timetable.csv"
    workbook = root / "outputs" / "workbooks" / "result3.xlsx"
    manifest_path = root / "outputs" / "workbooks" / "q3" / "reports" / "run_manifest.json"
    assert strict.is_file() and timetable.is_file() and workbook.is_file() and manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "q3-manifest-v1"
    entry = manifest["cases"]["Case1"]
    assert entry["verified"] is True
    assert (root / entry["archive"]).is_file()
    assert (root / entry["timetable"]).is_file()
    # hot-start metrics baseline recorded (fallback because no Q2 strict archive
    # exists under the tmp repo)
    assert entry["hot_start"]["source"] == "none"
    assert entry["hot_start_metrics"]["S_max_s"] == mini_solution.metrics.S_max_s

    # every sheet holds the same Point_ID routes as the archive
    archive = json.loads(strict.read_text(encoding="utf-8"))
    point_by_task = {task.task_id: task.point_id for task in mini_solution.problem.tasks}
    expected = [
        [point_by_task[task_id] for task_id in route]
        for route in archive["task_routes"]
    ]
    frame = pd.read_excel(workbook, sheet_name="Case1")
    got = [
        [int(value) for column, value in row.items() if column != "UAV ID" and pd.notna(value)]
        for _, row in frame.iterrows()
    ]
    assert got == expected


def test_verify_mode_exit_code(tmp_path, monkeypatch, mini_solution):
    """--verify over a clean publication exits 0; tampering yields a non-zero code."""
    import Q3.solve_q3 as solve
    import Q3.verify as verify_module

    root = tmp_path / "repo"
    (root / "outputs" / "workbooks" / "q3").mkdir(parents=True)
    monkeypatch.setattr(solve, "solve_case", _solution_for_case)
    problem = mini_solution.problem
    monkeypatch.setattr(verify_module, "load_problem",
                        lambda case, repository_root=None: replace(problem, case=case))
    assert solve.main(["--all", "--time-budget", "1", "--repository-root", str(root)]) == 0
    assert solve.main(["--verify", "--repository-root", str(root)]) == 0

    strict = root / "outputs" / "workbooks" / "q3" / "strict" / "Case1.json"
    body = json.loads(strict.read_text(encoding="utf-8"))
    body["metrics"]["sum_T_s"] += 1
    strict.write_text(json.dumps(body), encoding="utf-8")
    assert solve.main(["--verify", "--repository-root", str(root)]) == 1


def test_failed_verification_blocks_publication(tmp_path, monkeypatch, mini_solution):
    """A solution that fails independent verification is never published."""
    import Q3.solve_q3 as solve
    from Q3.verify import verify_archive

    root = tmp_path / "repo"
    (root / "outputs" / "workbooks" / "q3").mkdir(parents=True)
    monkeypatch.setattr(solve, "solve_case",
                        lambda case, config, repository_root=None: mini_solution)
    monkeypatch.setattr(solve, "verify_archive", lambda problem, archive: ["1: tampered"])
    with pytest.raises(RuntimeError, match="verification failed"):
        solve._solve_and_publish("Case1", Config(), root)
    assert not (root / "outputs" / "workbooks" / "q3" / "strict" / "Case1.json").exists()
