"""M4: independent verifier — legal mini passes all nine checks; every
injected violation class is detected."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Q3.domain import Config, NoFlyZone, Task
from Q3.io import write_result_workbook, write_timetable_csv
from Q3.safe_path import eval_solution
from Q3.verify import verify_all, verify_archive
from tests.q3.conftest import MINI_TASKS, make_problem

ROOT = Path(__file__).resolve().parents[2]


def _valid(problem, routes=((1, 2), (3, 4))):
    solution = eval_solution(problem, list(routes))
    assert solution is not None, "fixture route set must stay feasible"
    return solution.freeze()


def _violations(problem, archive):
    return verify_archive(problem, archive)


def test_legal_mini_passes_all_nine_checks():
    """With both zones active all day, the solver's detour routes must replay
    cleanly (R1/R4 sampled, R2/R3 windows, R5 continuity, R7 metrics)."""
    problem = make_problem()  # Z1/Z2 active [0, 90000]
    archive = _valid(problem)
    assert _violations(problem, archive) == []


def test_legal_mini_without_zones_passes():
    problem = make_problem(zones=())
    archive = _valid(problem)
    assert _violations(problem, archive) == []


def test_arc_samples_cw_sweep_not_complementary_arc():
    """Regression: the old inline sweep `direction * (-((theta_a - theta_b) % TAU))`
    double-negated for direction=-1 and sampled the complementary arc, so a
    safe cw arc flew through the mirror arc and triggered phantom R1 hits
    (Case2 101->2: d=6.049 inside Z2 while the real arc sits 9.8 km away)."""
    import math

    from Q3.verify import _arc

    center = np.asarray((0.0, 0.0), dtype=float)
    radius = 1.0
    theta_a = math.radians(10.0)
    theta_b = math.radians(-80.0)  # 280 deg in atan2 terms
    samples = _arc(center, radius, theta_a, theta_b, -1, 0.0, 1.0)
    mid = samples[len(samples) // 2]
    theta_mid = math.degrees(math.atan2(mid.y_km, mid.x_km))
    # real cw sweep runs 10 deg down to -80 deg; the buggy code sampled the
    # complementary arc sweeping 10 deg up to +100 deg instead
    assert -80.0 < theta_mid < 10.0, f"mid sample at theta={theta_mid:.2f} deg"
    assert all(
        -80.0 - 1e-6 <= math.degrees(math.atan2(s.y_km, s.x_km)) <= 10.0 + 1e-6
        for s in samples
    ), "every sample must stay on the cw arc, not the mirror arc"


def test_wait_direct_segments_replay_cleanly():
    """Regression: wait_direct legs (wait_s > 0) must replay on the real flight
    axis [depart_s, arrive_s]. The old code started the flight at depart - wait,
    so its samples landed inside the active window and every wait_direct leg
    was flagged as R1 crossing / flight-time / sum mismatches; the first-leg
    check also compared the take-off instant against 0 instead of the chain
    start (depart_s - wait_s)."""
    from Q3.domain import NoFlyZone

    windowed = make_problem(
        zones=(NoFlyZone("Z1", 0.5, 0.0, 0.2, 0, 20),
               NoFlyZone("Z2", 0.0, 0.5, 0.2, 0, 20)),
    )
    solution = eval_solution(windowed, [(1, 2), (3, 4)])
    assert solution is not None
    assert any(seg.wait_s > 0 for s in solution.schedules for seg in s.segments)
    assert verify_archive(windowed, solution.freeze()) == []


def test_injected_zone_crossing_is_detected():
    """Swapping a detour leg to direct puts samples inside the active disk."""
    problem = make_problem()
    archive = _valid(problem)
    first = archive["schedules"][0]["segments"][0]
    assert first["to_id"] == 1 and first["path_type"] != "direct"
    first["path_type"] = "direct"  # now crosses Z1 for its whole length
    assert any(item.startswith("4:") for item in _violations(problem, archive))


def test_injected_service_window_conflict_is_detected():
    """Task 1 sits inside Z4; widening the active window to all day makes the
    [A, A+300] service interval overlap it (R2)."""
    problem = make_problem()  # plain layout: T1 at (1,0)
    z4_all_day = NoFlyZone("Z4", 1.0, 0.0, 0.1, 0, 90_000)
    hostile = make_problem(zones=(z4_all_day,))
    archive = _valid(problem)
    # solver served T1 at some time; under the all-day window that service
    # interval now collides with the disk containing T1
    assert any(item.startswith("5:") for item in _violations(hostile, archive))


def test_injected_wait_inside_disk_is_detected():
    """Hand-built archive: the UAV waits at task 1, which lies inside Z4 (R3)."""
    problem = make_problem(
        tasks=(Task(1, 101, 1.0, 0.0, "I"),),
        zones=(NoFlyZone("Z4", 1.0, 0.0, 0.1, 0, 90_000),),
        fleet_size=1,
    )
    archive = {
        "schema_version": "q3-solution-v1",
        "case": "Mini",
        "fleet_size": 1,
        "task_routes": [[1]],
        "schedules": [
            {
                "uav_id": 1,
                "segments": [
                    {"from_id": 0, "to_id": 1, "path_type": "direct", "depart_s": 5000,
                     "arrive_s": 5100, "wait_s": 4900, "distance_km": 1.0, "affected_zones": []},
                    {"from_id": 1, "to_id": 0, "path_type": "direct", "depart_s": 5400,
                     "arrive_s": 5500, "wait_s": 100, "distance_km": 1.0, "affected_zones": []},
                ],
                "service_intervals": [[5100, 5400]],
                "S_k_s": 5500,
                "flight_s": 200,
                "wait_s": 5000,
                "distance_km": 2.0,
            }
        ],
        "metrics": {"S_max_s": 5500, "S_min_s": 5500, "delta_s": 0, "sum_T_s": 5500,
                    "total_wait_s": 5000, "total_distance_km": 2.0},
        "config": {},
    }
    violations = _violations(problem, archive)
    assert any(item.startswith("5:") and "waiting at node 1" in item for item in violations)


def test_injected_timeline_break_is_detected():
    problem = make_problem(zones=())
    archive = _valid(problem)
    archive["schedules"][0]["segments"][1]["depart_s"] += 1
    assert any(item.startswith("6:") for item in _violations(problem, archive))


def test_injected_metrics_tampering_is_detected():
    problem = make_problem(zones=())
    archive = _valid(problem)
    archive["metrics"]["sum_T_s"] += 1
    assert any("7:" in item for item in _violations(problem, archive))


def test_injected_served_mismatch_is_detected():
    problem = make_problem(zones=())
    archive = _valid(problem)
    archive["task_routes"][0] = [1]  # task 2 now unserved
    assert any(item.startswith("1:") for item in _violations(problem, archive))


def test_injected_adjacent_same_point_is_detected():
    five = (MINI_TASKS[0], MINI_TASKS[1], MINI_TASKS[2], MINI_TASKS[3],
            Task(5, 103, -1.0, 0.0, "III"))  # shares point 103 with task 3
    problem = make_problem(tasks=five, zones=(), fleet_size=2)
    archive = _valid(problem, ((1, 2), (3, 4, 5)))
    archive["task_routes"] = [[1, 2], [3, 5, 4]]  # 3 -> 5 adjacent same point
    assert any(item.startswith("3:") for item in _violations(problem, archive))


def test_injected_nine_hour_cap_is_detected():
    problem = make_problem(zones=())
    archive = _valid(problem)
    archive["config"] = {"nine_hour_cap_s": 1000}
    archive["schedules"][0]["S_k_s"] = 5000
    assert any(item.startswith("9:") for item in _violations(problem, archive))


# ---------------------------------------------------------------------------
# verify_all: cross-file consistency (check 8) through the manifest
# ---------------------------------------------------------------------------

def _publish_tree(tmp_path, problem, solution):
    root = tmp_path / "repo"
    out = root / "outputs" / "workbooks"
    archive_path = out / "q3" / "strict" / "Mini.json"
    timetable = out / "q3" / "timetables" / "Mini_timetable.csv"
    report_dir = out / "q3" / "reports"
    archive_path.parent.mkdir(parents=True)
    timetable.parent.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    archive_path.write_text(json.dumps(solution.freeze()), encoding="utf-8")
    write_timetable_csv(timetable, solution)
    write_result_workbook(out / "result3.xlsx", {"Mini": solution})
    manifest = {
        "schema_version": "q3-manifest-v1",
        "cases": {"Mini": {
            "archive": str(archive_path.relative_to(root)),
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "timetable": str(timetable.relative_to(root)),
            "verified": True,
        }},
        "config": {},
    }
    (report_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, timetable, out / "result3.xlsx"


@pytest.fixture
def published_mini(tmp_path, monkeypatch):
    import Q3.verify as verify_module

    problem = make_problem(zones=())
    solution = eval_solution(problem, [(1, 2), (3, 4)])
    assert solution is not None
    root, timetable, workbook = _publish_tree(tmp_path, problem, solution)
    monkeypatch.setattr(verify_module, "load_problem",
                        lambda case, repository_root=None: problem)
    return root, problem, timetable, workbook


def test_verify_all_passes_clean_publication(published_mini):
    root, problem, timetable, workbook = published_mini
    assert verify_all(("Mini",), Config(), repository_root=root) == []
    report = json.loads((root / "outputs" / "workbooks" / "q3" / "reports" / "verify_report.json").read_text())
    assert report["cases"]["Mini"]["verified"] is True


def test_verify_all_detects_timetable_mismatch(published_mini):
    root, problem, timetable, workbook = published_mini
    frame = pd.read_csv(timetable, encoding="utf-8")
    rows = frame[(frame["uav_id"] == 1) & (frame["to_id"] != 0)].index.tolist()
    frame.loc[rows[0], "to_id"], frame.loc[rows[1], "to_id"] = 2, 1
    frame.to_csv(timetable, index=False, encoding="utf-8")
    violations = verify_all(("Mini",), Config(), repository_root=root)
    assert any(item.startswith("Mini: 8: timetable") for item in violations)


def test_verify_all_detects_workbook_mismatch(published_mini):
    root, problem, timetable, workbook = published_mini
    frame = pd.read_excel(workbook, sheet_name="Mini")
    frame.iloc[0, 1], frame.iloc[0, 2] = 102, 101  # swap the first two points
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Mini", index=False)
    violations = verify_all(("Mini",), Config(), repository_root=root)
    assert any(item.startswith("Mini: 8: result3.xlsx") for item in violations)


def test_verify_all_accepts_same_point_served_twice_out_of_task_order(tmp_path, monkeypatch):
    """Regression: result3 sheets carry Point_IDs only, and a UAV serving two
    expanded tasks of the same point visits that point twice in whatever task
    order the search found (Case1 route 0 served point-43 tasks as 57 then 56).
    The old verifier recovered task IDs in canonical ascending order per point,
    so an out-of-order same-point pair was flagged as a workbook mismatch.
    verify_all must compare point-level sequences instead."""
    import Q3.verify as verify_module
    from Q3.domain import Task

    # point 101 is Level I expanded to tasks 1,2; point 102 is task 3.
    # The route serves point 101 twice as task 2 then task 1 (not canonical).
    problem = make_problem(
        tasks=(Task(1, 101, 1.0, 0.0, "I"), Task(2, 101, 1.0, 0.0, "I"),
               Task(3, 102, 0.0, 1.0, "III")),
        zones=(), fleet_size=1,
    )
    solution = eval_solution(problem, [(2, 3, 1)])
    assert solution is not None
    assert [list(sched.task_route) for sched in solution.schedules] == [[2, 3, 1]]
    root, timetable, workbook = _publish_tree(tmp_path, problem, solution)
    monkeypatch.setattr(verify_module, "load_problem",
                        lambda case, repository_root=None: problem)
    assert verify_all(("Mini",), Config(), repository_root=root) == []


def test_verify_all_detects_archive_sha_mismatch(published_mini):
    root, problem, timetable, workbook = published_mini
    archive_path = root / "outputs" / "workbooks" / "q3" / "strict" / "Mini.json"
    archive_path.write_text(
        json.dumps(json.loads(archive_path.read_text()) | {"tampered": True}), encoding="utf-8"
    )
    violations = verify_all(("Mini",), Config(), repository_root=root)
    assert any("8: archive sha256 mismatch" in item for item in violations)


def test_verify_all_reports_missing_manifest(tmp_path):
    violations = verify_all(("Case1",), Config(), repository_root=tmp_path)
    assert any("manifest missing" in item for item in violations)
