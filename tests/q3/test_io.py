"""M0/M4: attachment parsing, atomic JSON, timetable CSV, result workbook."""

import json
from pathlib import Path

import pandas as pd
import pytest

from Q3.domain import (
    Base,
    SegmentRecord,
    Solution,
    SolutionMetrics,
    UAVSchedule,
)
from Q3.io import (
    atomic_write_json,
    load_archive,
    load_hot_start_archive,
    load_problem,
    write_archive_with_hot_start,
    write_result_workbook,
    write_timetable_csv,
)
from tests.q3.conftest import MINI_TASKS, Z1_ALL_DAY, SHA, make_problem

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TASK_COUNT = {"Case1": 72, "Case2": 139, "Case3": 140, "Case4": 182}
EXPECTED_ZONE_COUNT = {"Case1": 3, "Case2": 4, "Case3": 6, "Case4": 7}


def _mini_solution() -> Solution:
    problem = make_problem()
    base = Base()
    leg1 = SegmentRecord(0, 1, "direct", 0, 66, 0, 1.0, ())
    leg2 = SegmentRecord(1, 0, "direct", 366, 432, 0, 1.0, ())
    sched1 = UAVSchedule(1, (1,), (leg1, leg2), ((66, 366),), 432, 132, 0, 2.0)
    leg3 = SegmentRecord(0, 2, "direct", 0, 66, 0, 1.0, ())
    leg4 = SegmentRecord(2, 0, "direct", 366, 432, 0, 1.0, ())
    sched2 = UAVSchedule(2, (2,), (leg3, leg4), ((66, 366),), 432, 132, 0, 2.0)
    return Solution(problem, (sched1, sched2), SolutionMetrics(432, 432, 0, 864, 0, 4.0))


@pytest.mark.parametrize("case_name", ["Case1", "Case2", "Case3", "Case4"])
def test_load_problem_real_attachment(case_name):
    problem = load_problem(case_name, repository_root=ROOT)
    assert problem.case == case_name
    assert problem.fleet_size == {"Case1": 4, "Case2": 2, "Case3": 5, "Case4": 4}[case_name]
    assert len(problem.tasks) == EXPECTED_TASK_COUNT[case_name]
    assert len(problem.zones) == EXPECTED_ZONE_COUNT[case_name]
    assert all(task.task_id == index for index, task in enumerate(problem.tasks, 1))
    assert all(task.level in {"I", "II", "III"} for task in problem.tasks)


def test_load_problem_case4_skips_zero_duration_z8():
    problem = load_problem("Case4", repository_root=ROOT)
    assert problem.warnings == ("Case4-Z8: zero-duration zone skipped",)
    assert all(zone.zone_id != "Z8" for zone in problem.zones)


def test_load_problem_case2_base_inside_z3_disk():
    """Case2 Z3 center ~ (4.515, 3.133) km r=5.5: base (0,0) is inside safe disk."""
    problem = load_problem("Case2", repository_root=ROOT)
    z3 = next(zone for zone in problem.zones if zone.zone_id == "Z3")
    distance = (z3.cx_km**2 + z3.cy_km**2) ** 0.5
    assert distance <= z3.safe_radius()
    assert z3.start_s == 9000 and z3.end_s == 18000  # 10:30-13:00


def test_atomic_write_json_roundtrip(tmp_path):
    payload = {"schema_version": "q3-solution-v1", "case": "Mini", "list": [1, 2], "nested": {"a": "b"}}
    target = tmp_path / "nested" / "archive.json"
    atomic_write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(target.parent.glob(".*.tmp"))


def test_atomic_write_json_leaves_no_partial_on_failure(tmp_path):
    target = tmp_path / "archive.json"

    class Boom:
        def __str__(self):
            raise RuntimeError("serialization failure")

    with pytest.raises((TypeError, ValueError)):
        atomic_write_json(target, {"bad": Boom()})
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_load_archive_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_archive(bad)


def test_hot_start_contract_mismatch_rejected(tmp_path):
    problem = make_problem()
    archive = {
        "schema_version": "q2-solution-v1",
        "case": "Mini",
        "fleet_size": 2,
        "input": {"problem_contract_sha256": "0" * 64},
        "task_routes": [[1], [2]],
    }
    path = tmp_path / "warm.json"
    path.write_text(json.dumps(archive), encoding="utf-8")
    with pytest.raises(ValueError, match="contract hash mismatch"):
        load_hot_start_archive(problem, path)
    archive["input"]["problem_contract_sha256"] = SHA
    path.write_text(json.dumps(archive), encoding="utf-8")
    assert load_hot_start_archive(problem, path)["task_routes"] == [[1], [2]]


def test_archive_with_hot_start_roundtrip(tmp_path):
    solution = _mini_solution()
    target = tmp_path / "Case_Mini.json"
    write_archive_with_hot_start(
        target,
        solution,
        hot_start={"source": "q2-strict", "archive_path": "outputs/workbooks/q2/strict/Case1.json",
                   "archive_sha256": "a" * 64, "fallback": None},
        config={"eta_km": 0.01, "seed": 42, "time_budget_s_per_case": 1800, "nine_hour_cap_s": None},
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "q3-solution-v1"
    assert payload["case"] == "Mini"
    assert payload["hot_start"]["source"] == "q2-strict"
    assert payload["config"]["time_budget_s"] == 1800
    assert payload["task_routes"] == [[1], [2]]
    assert payload["metrics"]["S_max_s"] == 432


def test_timetable_csv_has_13_fields(tmp_path):
    solution = _mini_solution()
    target = tmp_path / "Mini_timetable.csv"
    write_timetable_csv(target, solution)
    frame = pd.read_csv(target)
    assert list(frame.columns) == [
        "case", "uav_id", "segment_id", "from_id", "to_id", "path_type",
        "depart_time", "arrive_time", "service_start", "service_end",
        "wait_seconds", "distance_km", "affected_zones",
    ]
    assert len(frame) == 4
    return_leg = frame[frame["to_id"] == 0]
    assert return_leg["service_start"].isna().all()
    assert return_leg["service_end"].isna().all()


def test_result_workbook_roundtrip(tmp_path):
    solution = _mini_solution()
    target = tmp_path / "result3.xlsx"
    write_result_workbook(target, {"Mini": solution})
    workbook = pd.ExcelFile(target)
    assert workbook.sheet_names == ["Mini"]
    frame = pd.read_excel(target, sheet_name="Mini")
    assert frame.columns.tolist()[:2] == ["UAV ID", "1th Inspection Point"]
    assert frame["UAV ID"].tolist() == [1, 2]
    assert frame["1th Inspection Point"].tolist() == [101, 102]


def test_write_result_workbook_atomic(tmp_path):
    target = tmp_path / "result3.xlsx"
    write_result_workbook(target, {"Mini": _mini_solution()})
    assert list(target.parent.glob(".*.tmp*")) == []
