import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from Q2.q1_adapter import load_problem as q2_load_problem
from .domain import (
    START_CLOCK_S,
    UNIT_KM,
    NoFlyZone,
    ProblemData,
    Solution,
    Task,
    hhmm_to_s,
)
OUT_RELATIVE = Path("outputs") / "workbooks"
Q3_OUT_RELATIVE = OUT_RELATIVE / "q3"
def _repository_root(repository_root: str | Path | None) -> Path:
    return Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[1]
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _attachment_paths(root: Path) -> tuple[Path, Path]:
    attachment1 = root / "attachment" / "附件1.xlsx"
    attachment2 = root / "attachment" / "附件2.xlsx"
    if not attachment1.is_file() or not attachment2.is_file():
        raise FileNotFoundError(f"Q3 attachments missing: {attachment1} / {attachment2}")
    return attachment1, attachment2
def load_problem(
    case: str,
    *,
    repository_root: str | Path | None = None,
) -> ProblemData:
    root = _repository_root(repository_root)
    attachment1, attachment2 = _attachment_paths(root)
    q2 = q2_load_problem(case, repository_root=root)
    frame1 = pd.read_excel(attachment1, sheet_name=case)
    level_by_point = {
        int(row.Point_ID): str(row.Inspection_Level).strip().upper()
        for row in frame1.itertuples(index=False)
    }
    tasks = tuple(
        Task(task.task_id, task.point_id, task.x * UNIT_KM, task.y * UNIT_KM, level_by_point[task.point_id])
        for task in q2.tasks
    )
    frame2 = pd.read_excel(attachment2, sheet_name=case)
    zones: list[NoFlyZone] = []
    warnings: list[str] = []
    for row in frame2.itertuples(index=False):
        start_s = hhmm_to_s(row.Start_Time) - START_CLOCK_S
        end_s = hhmm_to_s(row.End_Time) - START_CLOCK_S
        zone_id = str(row.Zone_ID).strip()
        if end_s <= start_s:
            warnings.append(f"{case}-{zone_id}: zero-duration zone skipped")
            continue
        zones.append(
            NoFlyZone(
                zone_id,
                float(row.Center_X) * UNIT_KM,
                float(row.Center_Y) * UNIT_KM,
                float(row.Radius) * UNIT_KM,
                start_s,
                end_s,
            )
        )
    return ProblemData(
        case=case,
        tasks=tasks,
        zones=tuple(zones),
        fleet_size=q2.fleet_size,
        warnings=tuple(warnings),
        input_sha256=q2.attachment_sha256,
        problem_contract_sha256=q2.problem_contract_sha256,
        parent_archive_sha256=q2.archive_sha256,
    )
def load_archive(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"archive cannot be read: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"archive root must be an object: {path}")
    return payload
def archive_sha256(path: str | Path) -> str:
    return _sha256(Path(path))
def load_hot_start_archive(problem: ProblemData, path: str | Path) -> dict[str, Any]:
    archive = load_archive(path)
    if archive.get("schema_version") != "q2-solution-v1":
        raise ValueError(f"hot-start archive schema unsupported: {archive.get('schema_version')!r}")
    if archive.get("case") != problem.case:
        raise ValueError("hot-start archive case mismatch")
    if archive.get("fleet_size") != problem.fleet_size:
        raise ValueError("hot-start archive fleet size mismatch")
    stored_contract = archive.get("input", {}).get("problem_contract_sha256")
    if not isinstance(stored_contract, str) or stored_contract != problem.problem_contract_sha256:
        raise ValueError("hot-start problem contract hash mismatch")
    routes = archive.get("task_routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("hot-start task_routes must be a nonempty list")
    return archive
def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
def write_archive_with_hot_start(
    path: str | Path,
    solution: Solution,
    *,
    hot_start: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> None:
    payload = dict(solution.freeze())
    payload["hot_start"] = {
        "source": hot_start.get("source", "none"),
        "archive_path": hot_start.get("archive_path"),
        "archive_sha256": hot_start.get("archive_sha256"),
        "fallback": hot_start.get("fallback"),
    } if hot_start else {"source": "none", "archive_path": None, "archive_sha256": None, "fallback": None}
    payload["config"] = {
        "eta_km": config["eta_km"],
        "seed": config["seed"],
        "time_budget_s": config["time_budget_s_per_case"],
        "nine_hour_cap_s": config["nine_hour_cap_s"],
    }
    atomic_write_json(path, payload)
TIMETABLE_FIELDS = [
    "case",
    "uav_id",
    "segment_id",
    "from_id",
    "to_id",
    "path_type",
    "depart_time",
    "arrive_time",
    "service_start",
    "service_end",
    "wait_seconds",
    "distance_km",
    "affected_zones",
]
def write_timetable_csv(path: str | Path, solution: Solution) -> None:
    rows = []
    for schedule in solution.schedules:
        service_by_from = dict(schedule.service_intervals)
        for index, segment in enumerate(schedule.segments, 1):
            interval = service_by_from.get(segment.from_id)
            rows.append(
                {
                    "case": solution.problem.case,
                    "uav_id": schedule.uav_id,
                    "segment_id": index,
                    "from_id": segment.from_id,
                    "to_id": segment.to_id,
                    "path_type": segment.path_type,
                    "depart_time": segment.depart_s,
                    "arrive_time": segment.arrive_s,
                    "service_start": interval[0] if interval else "",
                    "service_end": interval[1] if interval else "",
                    "wait_seconds": segment.wait_s,
                    "distance_km": segment.distance_km,
                    "affected_zones": ",".join(segment.affected_zones),
                }
            )
    frame = pd.DataFrame(rows, columns=TIMETABLE_FIELDS)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8")
def write_result_workbook(path: str | Path, solutions: Mapping[str, Solution]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".xlsx", dir=target.parent)
    os.close(descriptor)
    try:
        with pd.ExcelWriter(temporary_name, engine="openpyxl") as writer:
            for case_name, solution in solutions.items():
                if case_name != solution.problem.case:
                    raise ValueError("solution map case key mismatch")
                _result_frame(solution).to_excel(writer, sheet_name=case_name, index=False)
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
def _result_frame(solution: Solution) -> pd.DataFrame:
    point_by_task = {task.task_id: task.point_id for task in solution.problem.tasks}
    point_routes = [
        tuple(point_by_task[task_id] for task_id in schedule.task_route)
        for schedule in solution.schedules
    ]
    width = max(len(route) for route in point_routes)
    rows = [
        {
            "UAV ID": index,
            **{
                f"{column}th Inspection Point": route[column - 1] if column <= len(route) else None
                for column in range(1, width + 1)
            },
        }
        for index, route in enumerate(point_routes, 1)
    ]
    return pd.DataFrame(rows)
