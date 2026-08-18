import hashlib
import json
import math
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from Q1.solve_q1 import build_matrices, evaluate, load_case
from .domain import CAP_S, SERVICE_S, SPEED_KMH, UNIT_KM, ProblemData, Task, freeze_json
from .metrics import replay_metrics
FLEET_SIZE_BY_CASE = {
    "Case1": 4,
    "Case2": 2,
    "Case3": 5,
    "Case4": 4,
}
DEFAULT_ARCHIVE_RELATIVE = Path("outputs") / "workbooks" / "baseline_20260816"
CONTRACT_VERSION = "q2-domain-v1"
_LOAD_CASE_LOCK = threading.Lock()
class ParentArchiveError(ValueError):
    pass
def _repository_root(repository_root: str | Path | None) -> Path:
    return Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[1]
def _resolve_paths(
    case_name: str,
    repository_root: str | Path | None,
    attachment_path: str | Path | None,
    archive_path: str | Path | None,
) -> tuple[Path, Path]:
    root = _repository_root(repository_root)
    attachment = Path(attachment_path).resolve() if attachment_path else root / "attachment" / "附件1.xlsx"
    archive = Path(archive_path).resolve() if archive_path else root / DEFAULT_ARCHIVE_RELATIVE / f"q1_solution_{case_name}.json"
    if not attachment.is_file():
        raise FileNotFoundError(f"Q2 attachment not found: {attachment}")
    if not archive.is_file():
        raise FileNotFoundError(f"Q1 parent archive not found: {archive}")
    return attachment, archive
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _require_hash(label: str, actual: str, expected: str | None) -> None:
    if expected is not None:
        if not isinstance(expected, str) or len(expected) != 64:
            raise ParentArchiveError(f"expected {label} SHA-256 must be a 64-character hexadecimal string")
        try:
            int(expected, 16)
        except ValueError as exc:
            raise ParentArchiveError(f"expected {label} SHA-256 must be hexadecimal") from exc
        if actual.lower() != expected.lower():
            raise ParentArchiveError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
def _load_case_at(case_name: str, attachment: Path) -> dict[str, Any]:
    namespace = load_case.__globals__
    with _LOAD_CASE_LOCK:
        previous = namespace["ATTACH_XLSX"]
        try:
            namespace["ATTACH_XLSX"] = attachment
            return load_case(case_name)
        finally:
            namespace["ATTACH_XLSX"] = previous
def _immutable_matrices(case: dict[str, Any]) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...]]:
    distance, time_s = build_matrices(case)
    immutable_distance = tuple(tuple(float(value) for value in row) for row in distance)
    immutable_time = tuple(tuple(int(value) for value in row) for row in time_s)
    return immutable_distance, immutable_time
def _tasks(case: dict[str, Any]) -> tuple[Task, ...]:
    return tuple(
        Task(int(item["task_id"]), int(item["pid"]), float(item["x"]), float(item["y"]))
        for item in case["tasks"]
    )
def _validate_header(case_name: str, archive: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(archive, dict):
        raise ParentArchiveError("parent archive root must be a JSON object")
    if archive.get("case") != case_name:
        raise ParentArchiveError(f"parent case must be {case_name}, got {archive.get('case')!r}")
    if archive.get("status") != "SOLVED":
        raise ParentArchiveError("parent status must be SOLVED")
    if archive.get("complete") is not True:
        raise ParentArchiveError("parent complete must be true")
    expected = FLEET_SIZE_BY_CASE.get(case_name)
    if expected is None:
        raise ParentArchiveError(f"unsupported case {case_name!r}")
    if type(archive.get("N")) is not int or archive["N"] != expected:
        raise ParentArchiveError(f"parent fixed fleet N must be exact integer {expected}")
    uavs = archive.get("uavs")
    if not isinstance(uavs, list) or len(uavs) != expected:
        raise ParentArchiveError(f"parent must contain fixed fleet of {expected} UAV records")
    return uavs
def _extract_routes(
    uavs: list[dict[str, Any]], task_count: int
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, Any]]]:
    for record in uavs:
        if not isinstance(record, dict):
            raise ParentArchiveError("each parent UAV record must be a JSON object")
    ids = [record.get("uav_id") for record in uavs]
    if any(type(uav_id) is not int for uav_id in ids):
        raise ParentArchiveError("parent UAV IDs must be exact integers")
    expected_ids = set(range(1, len(uavs) + 1))
    if set(ids) != expected_ids:
        raise ParentArchiveError("parent UAV IDs must be continuous and unique from 1")
    ordered_uavs = sorted(uavs, key=lambda record: record["uav_id"])
    routes: list[tuple[int, ...]] = []
    for record in ordered_uavs:
        raw_route = record.get("task_seq")
        if not isinstance(raw_route, list) or not raw_route:
            raise ParentArchiveError("every parent route must be nonempty and loaded by task_seq")
        if any(type(value) is not int for value in raw_route):
            raise ParentArchiveError("parent task_seq entries must be exact integers")
        routes.append(tuple(raw_route))
    flat = [task_id for route in routes for task_id in route]
    if len(flat) != task_count or set(flat) != set(range(1, task_count + 1)):
        raise ParentArchiveError(f"parent tasks must cover exactly 1..{task_count} once")
    return tuple(routes), ordered_uavs
def _validate_mapping(
    tasks: tuple[Task, ...],
    routes: tuple[tuple[int, ...], ...],
    uavs: list[dict[str, Any]],
    expected_point_counts: Counter[int] | None = None,
) -> None:
    lookup = {task.task_id: task.point_id for task in tasks}
    derived = tuple(tuple(lookup[task_id] for task_id in route) for route in routes)
    for index, (points, record) in enumerate(zip(derived, uavs), 1):
        archived = record.get("point_seq")
        if archived is not None and tuple(archived) != points:
            raise ParentArchiveError(f"UAV {index} archived point_seq does not match task mapping")
        if any(left == right for left, right in zip(points, points[1:])):
            raise ParentArchiveError(f"UAV {index} has adjacent same Point_ID")
    expected = expected_point_counts or Counter(task.point_id for task in tasks)
    actual = Counter(point_id for route in derived for point_id in route)
    if actual != expected:
        raise ParentArchiveError("parent per-point visitation counts do not match task expansion")
def _close_float(actual: float, expected: float, tolerance: float = 5e-7) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
def _validate_route_metrics(uavs: list[dict[str, Any]], metrics: Any) -> None:
    for index, (record, replayed) in enumerate(zip(uavs, metrics.routes), 1):
        if replayed.work_s > CAP_S:
            raise ParentArchiveError(f"UAV {index} route capacity exceeded: work_s {replayed.work_s} > {CAP_S}")
        checks = {
            "n_tasks": len(replayed.task_sequence),
            "fly_s": replayed.flight_s,
            "work_s": replayed.work_s,
        }
        for field, expected in checks.items():
            if field in record:
                if type(record[field]) is not int:
                    raise ParentArchiveError(f"UAV {index} archived {field} must be an exact integer")
                if record[field] != expected:
                    raise ParentArchiveError(f"UAV {index} archived {field} mismatch: {record[field]} != {expected}")
        if "dist_km" in record and not _close_float(float(record["dist_km"]), round(replayed.distance_km, 6)):
            raise ParentArchiveError(f"UAV {index} archived dist_km mismatch")
        if "work_h" in record and not _close_float(float(record["work_h"]), replayed.work_s / 3600.0):
            raise ParentArchiveError(f"UAV {index} archived work_h mismatch")
def _validate_parent_metrics(archive: dict[str, Any], metrics: Any) -> None:
    expected = {"Tmax_s": metrics.Tmax_s, "Tmin_s": metrics.Tmin_s}
    for field, value in expected.items():
        if field in archive and archive[field] != value:
            raise ParentArchiveError(f"parent archived {field} mismatch: {archive[field]} != {value}")
    hour_values = {"Tmax_h": metrics.Tmax_s / 3600.0, "Tmin_h": metrics.Tmin_s / 3600.0}
    for field, value in hour_values.items():
        if field in archive and not _close_float(float(archive[field]), value):
            raise ParentArchiveError(f"parent archived {field} mismatch")
def _validate_compatibility(case: dict[str, Any], distance: tuple[tuple[float, ...], ...], time_s: tuple[tuple[int, ...], ...], routes: tuple[tuple[int, ...], ...], metrics: Any) -> None:
    compatibility = evaluate(case, distance, time_s, [list(route) for route in routes])
    for index, (q1_stats, replayed) in enumerate(zip(compatibility, metrics.routes), 1):
        if q1_stats["fly_s"] != replayed.flight_s or q1_stats["work_s"] != replayed.work_s:
            raise ParentArchiveError(f"Q1 compatibility evaluation mismatch for UAV {index}")
        if not _close_float(float(q1_stats["dist_km"]), round(replayed.distance_km, 6)):
            raise ParentArchiveError(f"Q1 compatibility distance mismatch for UAV {index}")
        if tuple(q1_stats["point_seq"]) != replayed.point_sequence:
            raise ParentArchiveError(f"Q1 compatibility point mapping mismatch for UAV {index}")
def _contract_sha(case_name: str, tasks: tuple[Task, ...], routes: tuple[tuple[int, ...], ...], attachment_sha: str, archive_sha: str) -> str:
    payload = {
        "version": CONTRACT_VERSION,
        "case": case_name,
        "fleet_size": FLEET_SIZE_BY_CASE[case_name],
        "constants": {"speed_kmh": SPEED_KMH, "unit_km": UNIT_KM, "service_s": SERVICE_S, "cap_s": CAP_S},
        "tasks": [[task.task_id, task.point_id, task.x, task.y] for task in tasks],
        "routes": [list(route) for route in routes],
        "attachment_sha256": attachment_sha,
        "archive_sha256": archive_sha,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
def load_problem(
    case_name: str,
    *,
    repository_root: str | Path | None = None,
    attachment_path: str | Path | None = None,
    archive_path: str | Path | None = None,
    expected_attachment_sha256: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_problem_contract_sha256: str | None = None,
) -> ProblemData:
    attachment, archive_file = _resolve_paths(case_name, repository_root, attachment_path, archive_path)
    attachment_sha, archive_sha = _sha256(attachment), _sha256(archive_file)
    _require_hash("attachment", attachment_sha, expected_attachment_sha256)
    _require_hash("archive", archive_sha, expected_archive_sha256)
    archive = json.loads(archive_file.read_text(encoding="utf-8"))
    uavs = _validate_header(case_name, archive)
    case = _load_case_at(case_name, attachment)
    tasks = _tasks(case)
    routes, ordered_uavs = _extract_routes(uavs, len(tasks))
    _validate_mapping(tasks, routes, ordered_uavs)
    distance, time_s = _immutable_matrices(case)
    metrics = replay_metrics(tasks, distance, time_s, routes)
    _validate_route_metrics(ordered_uavs, metrics)
    _validate_parent_metrics(archive, metrics)
    _validate_compatibility(case, distance, time_s, routes, metrics)
    contract_sha = _contract_sha(case_name, tasks, routes, attachment_sha, archive_sha)
    _require_hash("problem-contract", contract_sha, expected_problem_contract_sha256)
    return ProblemData(
        case_name, FLEET_SIZE_BY_CASE[case_name], tasks, routes, distance, time_s, metrics,
        freeze_json(archive), str(attachment), str(archive_file), attachment_sha, archive_sha,
        contract_sha, True,
    )
def replay_archive(
    case_name: str,
    archive: dict[str, Any],
    *,
    attachment_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    attachment = Path(attachment_path).resolve() if attachment_path else root / "attachment" / "附件1.xlsx"
    case = _load_case_at(case_name, attachment)
    distance, time_s = _immutable_matrices(case)
    routes = tuple(tuple(record["task_seq"]) for record in archive["uavs"])
    stats = evaluate(case, distance, time_s, [list(route) for route in routes])
    return {
        "case": case,
        "fleet_size": FLEET_SIZE_BY_CASE[case_name],
        "routes": routes,
        "dist_km": distance,
        "time_s": time_s,
        "stats": stats,
    }
