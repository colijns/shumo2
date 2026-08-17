from collections import Counter
import copy
import hashlib
import json
import math
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from Q2.domain import Task
from Q2.metrics import (
    balanced_key,
    canonical_route_signature,
    epsilon_bound,
    normalize_routes,
    replay_metrics,
    strict_key,
)
from Q2.q1_adapter import ParentArchiveError, load_problem


def _write_attachment(path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {"Point_ID": 10, "X_Coordinate": 3, "Y_Coordinate": 4, "Inspection_Level": "II"},
            {"Point_ID": 20, "X_Coordinate": 6, "Y_Coordinate": 8, "Inspection_Level": "I"},
            {"Point_ID": 30, "X_Coordinate": 0, "Y_Coordinate": 11, "Inspection_Level": "III"},
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Case1", index=False)
    return path


def _tasks() -> tuple[Task, ...]:
    return (
        Task(1, 10, 3.0, 4.0),
        Task(2, 10, 3.0, 4.0),
        Task(3, 20, 6.0, 8.0),
        Task(4, 20, 6.0, 8.0),
        Task(5, 20, 6.0, 8.0),
        Task(6, 30, 0.0, 11.0),
    )


def _routes() -> tuple[tuple[int, ...], ...]:
    return ((1, 3), (2, 4), (5,), (6,))


def _matrices(tasks: tuple[Task, ...]):
    coords = ((0.0, 0.0),) + tuple((task.x, task.y) for task in tasks)
    distance = tuple(
        tuple(0.1 * math.hypot(ax - bx, ay - by) for bx, by in coords)
        for ax, ay in coords
    )
    times = tuple(
        tuple(math.ceil(value / 55.0 * 3600.0) for value in row)
        for row in distance
    )
    return distance, times


def _archive() -> dict:
    tasks = _tasks()
    distance, times = _matrices(tasks)
    replay = replay_metrics(tasks, distance, times, _routes())
    uavs = []
    for uav_id, route in enumerate(_routes(), 1):
        route_metrics = replay.routes[uav_id - 1]
        uavs.append(
            {
                "uav_id": uav_id,
                "n_tasks": len(route),
                "fly_s": route_metrics.flight_s,
                "dist_km": round(route_metrics.distance_km, 6),
                "work_s": route_metrics.work_s,
                "work_h": route_metrics.work_s / 3600.0,
                "task_seq": list(route),
                "point_seq": [tasks[task_id - 1].point_id for task_id in route],
            }
        )
    return {
        "case": "Case1",
        "status": "SOLVED",
        "complete": True,
        "N": 4,
        "Tmax_s": replay.Tmax_s,
        "Tmin_s": replay.Tmin_s,
        "uavs": uavs,
    }


@pytest.fixture
def synthetic_parent(tmp_path):
    attachment = _write_attachment(tmp_path / "fixture.xlsx")
    archive = _archive()
    archive_path = tmp_path / "q1_solution_Case1.json"
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    return attachment, archive_path, archive


@pytest.mark.parametrize("case_name", ("Case1", "Case2", "Case3", "Case4"))
def test_load_problem_validates_real_parent_archives(case_name):
    root = Path(__file__).resolve().parents[2]
    attachment = root.parents[2] / "attachment" / "附件1.xlsx"
    archive = root / "outputs" / "workbooks" / "baseline_20260816" / f"q1_solution_{case_name}.json"
    if not attachment.is_file():
        pytest.fail(f"required real attachment is unavailable: {attachment}")

    problem = load_problem(case_name, attachment_path=attachment, archive_path=archive)

    assert problem.fleet_size == len(problem.routes)
    assert set(task_id for route in problem.routes for task_id in route) == set(
        range(1, len(problem.tasks) + 1)
    )
    assert problem.metrics.Tmax_s <= 32_400


def test_load_problem_returns_immutable_canonical_state(synthetic_parent):
    attachment, archive_path, _ = synthetic_parent
    problem = load_problem("Case1", attachment_path=attachment, archive_path=archive_path)
    before = archive_path.read_bytes()

    assert problem.routes == _routes()
    assert isinstance(problem.routes, tuple)
    assert all(isinstance(route, tuple) for route in problem.routes)
    assert problem.point_sequences == ((10, 20), (10, 20), (20,), (30,))
    assert problem.fleet_size == 4
    assert problem.metrics.mean_T_s == Fraction(problem.metrics.sum_T_s, 4)
    with pytest.raises(FrozenInstanceError):
        problem.fleet_size = 9
    with pytest.raises(TypeError):
        problem.parent_archive["N"] = 9
    assert archive_path.read_bytes() == before


def test_hashes_cover_exact_attachment_archive_and_problem_contract(synthetic_parent):
    attachment, archive_path, _ = synthetic_parent
    problem = load_problem("Case1", attachment_path=attachment, archive_path=archive_path)

    assert problem.attachment_sha256 == hashlib.sha256(attachment.read_bytes()).hexdigest()
    assert problem.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert len(problem.problem_contract_sha256) == 64

    with pytest.raises(ParentArchiveError, match="attachment SHA-256"):
        load_problem(
            "Case1",
            attachment_path=attachment,
            archive_path=archive_path,
            expected_attachment_sha256="0" * 64,
        )
    with pytest.raises(ParentArchiveError, match="archive SHA-256"):
        load_problem(
            "Case1",
            attachment_path=attachment,
            archive_path=archive_path,
            expected_archive_sha256="0" * 64,
        )
    with pytest.raises(ParentArchiveError, match="problem-contract SHA-256"):
        load_problem(
            "Case1",
            attachment_path=attachment,
            archive_path=archive_path,
            expected_problem_contract_sha256="0" * 64,
        )


def _make_adjacent_same_point(data):
    data["uavs"][0]["task_seq"] = [1, 2]
    data["uavs"][1]["task_seq"] = [3, 4]
    data["uavs"][2]["task_seq"] = [5]
    data["uavs"][3]["task_seq"] = [6]
    data["uavs"][0]["point_seq"] = [10, 10]
    data["uavs"][1]["point_seq"] = [20, 20]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.update(case="Case2"), "case"),
        (lambda data: data.update(status="FAILED"), "status"),
        (lambda data: data.update(complete=False), "complete"),
        (lambda data: data.update(N=3), "fixed fleet"),
        (lambda data: data["uavs"].__setitem__(0, {**data["uavs"][0], "uav_id": 2}), "UAV IDs"),
        (lambda data: data["uavs"][0].update(task_seq=[]), "nonempty"),
        (lambda data: data["uavs"][0]["task_seq"].__setitem__(0, 2), "cover exactly"),
        (lambda data: data["uavs"][0]["task_seq"].__setitem__(0, 7), "cover exactly"),
        (lambda data: data["uavs"][0]["point_seq"].__setitem__(0, 20), "point_seq"),
        (_make_adjacent_same_point, "adjacent same Point_ID"),
        (lambda data: data["uavs"][0].update(fly_s=data["uavs"][0]["fly_s"] + 1), "fly_s"),
        (lambda data: data["uavs"][0].update(work_s=32401), "work_s"),
    ],
    ids=[
        "wrong-case",
        "bad-status",
        "incomplete",
        "wrong-fixed-n",
        "noncontinuous-uav-ids",
        "empty-route",
        "duplicate-and-missing-task",
        "out-of-range-task",
        "mapping-mismatch",
        "adjacent-same-point",
        "bad-flight-time",
        "bad-work-time",
    ],
)
def test_malformed_parent_archive_is_rejected(synthetic_parent, mutate, message):
    attachment, archive_path, original = synthetic_parent
    malformed = copy.deepcopy(original)
    mutate(malformed)
    archive_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ParentArchiveError, match=message):
        load_problem("Case1", attachment_path=attachment, archive_path=archive_path)


def test_replayed_over_capacity_route_is_rejected(synthetic_parent, monkeypatch):
    attachment, archive_path, _ = synthetic_parent
    import Q2.q1_adapter as adapter

    original = adapter.replay_metrics

    def over_capacity(*args, **kwargs):
        metrics = original(*args, **kwargs)
        route = metrics.routes[0]
        inflated = type(route)(
            route.task_sequence,
            route.point_sequence,
            route.flight_s,
            route.distance_km,
            32_401,
        )
        return type(metrics)(
            (inflated,) + metrics.routes[1:],
            (32_401,) + metrics.work_s[1:],
            metrics.flight_s,
            metrics.distance_km,
            32_401,
            metrics.Tmin_s,
            32_401 - metrics.Tmin_s,
            metrics.sum_T_s - metrics.work_s[0] + 32_401,
            metrics.mean_T_s,
            metrics.std_T_s,
        )

    monkeypatch.setattr(adapter, "replay_metrics", over_capacity)
    with pytest.raises(ParentArchiveError, match="capacity exceeded"):
        load_problem("Case1", attachment_path=attachment, archive_path=archive_path)


def test_visit_counts_are_checked_against_task_expansion():
    expected_tasks = (
        Task(1, 10, 3.0, 4.0),
        Task(2, 10, 3.0, 4.0),
        Task(3, 20, 6.0, 8.0),
    )
    routes = ((1, 3), (2,))
    uavs = [{}, {}]
    mismapped_tasks = (
        Task(1, 10, 3.0, 4.0),
        Task(2, 10, 3.0, 4.0),
        Task(3, 30, 6.0, 8.0),
    )
    with pytest.raises(ParentArchiveError, match="visitation counts"):
        import Q2.q1_adapter as adapter

        adapter._validate_mapping(
            mismapped_tasks,
            routes,
            uavs,
            expected_point_counts=Counter(task.point_id for task in expected_tasks),
        )


def test_replay_uses_task_mapping_integer_leg_seconds_and_q1_cross_check(synthetic_parent):
    attachment, archive_path, _ = synthetic_parent
    problem = load_problem("Case1", attachment_path=attachment, archive_path=archive_path)

    assert all(isinstance(value, int) for row in problem.time_s for value in row)
    assert problem.time_s[0][1] == math.ceil(0.5 / 55 * 3600)
    assert problem.metrics.routes[0].work_s == (
        problem.time_s[0][1] + problem.time_s[1][3] + problem.time_s[3][0] + 600
    )
    assert problem.q1_compatibility_checked is True
    assert problem.metrics.std_T_s == pytest.approx(
        math.sqrt(sum((value - float(problem.metrics.mean_T_s)) ** 2 for value in problem.metrics.work_s) / 4)
    )


def test_canonical_signature_sorts_routes_without_changing_working_order():
    working = ((5, 4), (1, 3), (2,))

    signature = canonical_route_signature(working)

    assert signature == ((1, 3), (2,), (5, 4))
    assert working == ((5, 4), (1, 3), (2,))


def test_objective_keys_have_required_lexicographic_order():
    routes = ((2,), (1,))

    assert strict_key(12, 4, 20, routes) == (12, 4, 20, ((1,), (2,)))
    assert balanced_key(12, 4, 20, routes) == (4, 12, 20, ((1,), (2,)))
    assert strict_key(12, 4, 20, routes) < strict_key(13, 0, 1, routes)
    assert balanced_key(99, 3, 999, routes) < balanced_key(1, 4, 1, routes)


@pytest.mark.parametrize(
    "tmax, epsilon, expected",
    [(10_000, 0.05, 10_500), (10_001, 0.05, 10_501), (32_000, 0.02, 32_400), (32_400, 0, 32_400)],
)
def test_epsilon_bound_uses_exact_floor_and_capacity_cap(tmax, epsilon, expected):
    assert epsilon_bound(tmax, epsilon) == expected


@pytest.mark.parametrize(
    "tmax, epsilon",
    [(0, 0.05), (-1, 0.05), (32_401, 0.05), (100, -0.01), (True, 0.05), (100, float("nan"))],
)
def test_epsilon_bound_rejects_invalid_inputs(tmax, epsilon):
    with pytest.raises(ValueError):
        epsilon_bound(tmax, epsilon)


def test_epsilon_boundary_accepts_exact_value_and_rejects_plus_one():
    bound = epsilon_bound(10_001, 0.05)

    assert bound == 10_501
    assert 10_501 <= bound
    assert not 10_502 <= bound


def test_load_problem_sorts_shuffled_valid_uav_records(synthetic_parent):
    attachment, archive_path, original = synthetic_parent
    shuffled = copy.deepcopy(original)
    shuffled["uavs"] = [shuffled["uavs"][2], shuffled["uavs"][0], shuffled["uavs"][3], shuffled["uavs"][1]]
    archive_path.write_text(json.dumps(shuffled), encoding="utf-8")

    problem = load_problem("Case1", attachment_path=attachment, archive_path=archive_path)

    assert problem.routes == _routes()
    assert tuple(record["uav_id"] for record in problem.parent_archive["uavs"]) == (3, 1, 4, 2)


def test_load_problem_rejects_non_object_uav_record(synthetic_parent):
    attachment, archive_path, original = synthetic_parent
    malformed = copy.deepcopy(original)
    malformed["uavs"][0] = None
    archive_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ParentArchiveError, match="JSON object"):
        load_problem("Case1", attachment_path=attachment, archive_path=archive_path)


@pytest.mark.parametrize("uav_id", (True, 1.0))
def test_load_problem_rejects_non_exact_uav_id(synthetic_parent, uav_id):
    attachment, archive_path, original = synthetic_parent
    malformed = copy.deepcopy(original)
    malformed["uavs"][0]["uav_id"] = uav_id
    archive_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ParentArchiveError, match="exact integers"):
        load_problem("Case1", attachment_path=attachment, archive_path=archive_path)


@pytest.mark.parametrize("field, value", (("N", 4.0), ("N", True), ("n_tasks", 2.0), ("fly_s", True), ("work_s", 1.0)))
def test_load_problem_rejects_non_exact_integer_archive_fields(synthetic_parent, field, value):
    attachment, archive_path, original = synthetic_parent
    malformed = copy.deepcopy(original)
    target = malformed if field == "N" else malformed["uavs"][0]
    target[field] = value
    archive_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ParentArchiveError, match="exact integer"):
        load_problem("Case1", attachment_path=attachment, archive_path=archive_path)


@pytest.mark.parametrize("task_id", (1.9, "2", True, None))
def test_normalize_routes_rejects_non_exact_task_ids(task_id):
    with pytest.raises(ValueError, match="exact integers"):
        normalize_routes(((task_id,),))


def test_replay_metrics_rejects_out_of_range_before_matrix_access():
    tasks = _tasks()
    distance, times = _matrices(tasks)

    with pytest.raises(ValueError, match="1..6"):
        replay_metrics(tasks, distance, times, ((1, 3), (2, 4), (5,), (7,)))


def test_replay_metrics_rejects_duplicate_or_missing_task_ids():
    tasks = _tasks()
    distance, times = _matrices(tasks)

    with pytest.raises(ValueError, match="exactly once"):
        replay_metrics(tasks, distance, times, ((1, 3), (2, 4), (5,), (5,)))


def test_replay_metrics_preserves_empty_route_error():
    tasks = _tasks()
    distance, times = _matrices(tasks)

    with pytest.raises(ValueError, match="nonempty"):
        replay_metrics(tasks, distance, times, ((1, 3), (2, 4), (5,), ()))
