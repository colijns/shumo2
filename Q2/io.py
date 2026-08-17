"""Atomic JSON persistence and independent Q2 archive validation."""

import json
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from .balance_core import ScoredCandidate, score_candidate
from .domain import ProblemData

CHECKPOINT_SCHEMA = "q2-checkpoint-v1"
SOLUTION_SCHEMA = "q2-solution-v1"


class CheckpointError(ValueError):
    """Checkpoint content cannot safely resume this problem."""


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file only after complete temporary write."""
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
        _fsync_directory(target.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_checkpoint(path: str | Path, problem_contract_sha256: str) -> dict[str, Any]:
    """Load only compatible checkpoint JSON for exact frozen problem input."""
    if not _is_sha256(problem_contract_sha256):
        raise CheckpointError("checkpoint problem contract SHA-256 is malformed")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("checkpoint cannot be read") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise CheckpointError("checkpoint schema is unsupported")
    if not _is_sha256(payload.get("problem_contract_sha256")):
        raise CheckpointError("checkpoint problem contract SHA-256 is malformed")
    if payload.get("problem_contract_sha256") != problem_contract_sha256:
        raise CheckpointError("checkpoint problem contract hash mismatch")
    return payload


def validate_solution_archive(problem: ProblemData, archive: Mapping[str, Any]) -> ScoredCandidate:
    """Independently replay one Q2 archive without trusting stored metrics."""
    _validate_solution_header(problem, archive)
    routes = archive.get("task_routes")
    candidate = score_candidate(problem, routes)
    if candidate is None:
        raise ValueError("archive task routes violate Q2 constraints")
    _validate_stored_metrics(archive.get("metrics"), candidate)
    return candidate


def build_solution_archive(problem: ProblemData, candidate: ScoredCandidate) -> dict[str, Any]:
    """Build replayable strict-solution evidence from immutable candidate data."""
    if score_candidate(problem, candidate.routes) != candidate:
        raise ValueError("candidate must be fully replayed for this problem")
    metrics = candidate.metrics
    return {
        "schema_version": SOLUTION_SCHEMA,
        "case": problem.case_name,
        "fleet_size": problem.fleet_size,
        "parent_archive_sha256": problem.archive_sha256,
        "input": {
            "problem_contract_sha256": problem.problem_contract_sha256,
            "attachment_sha256": problem.attachment_sha256,
            "parent_archive_sha256": problem.archive_sha256,
        },
        "task_routes": [list(route) for route in candidate.routes],
        "metrics": {
            "route_flight_s": list(metrics.flight_s),
            "route_work_s": list(metrics.work_s),
            "Tmax_s": metrics.Tmax_s,
            "Tmin_s": metrics.Tmin_s,
            "delta_s": metrics.delta_s,
            "sum_T_s": metrics.sum_T_s,
            "mean_T_s": {"numerator": metrics.mean_T_s.numerator, "denominator": metrics.mean_T_s.denominator},
        },
    }


def write_result_workbook(path: str | Path, solutions: Mapping[str, tuple[ProblemData, ScoredCandidate]]) -> None:
    """Atomically publish strict point routes in result2-compatible sheets."""
    import pandas as pd

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".xlsx", dir=target.parent)
    os.close(descriptor)
    try:
        with pd.ExcelWriter(temporary_name, engine="openpyxl") as writer:
            for case_name, (problem, candidate) in solutions.items():
                if case_name != problem.case_name:
                    raise ValueError("solution map case key mismatch")
                _result_frame(problem, candidate).to_excel(writer, sheet_name=case_name, index=False)
        _fsync_file(Path(temporary_name))
        os.replace(temporary_name, target)
        _fsync_directory(target.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _result_frame(problem: ProblemData, candidate: ScoredCandidate):
    import pandas as pd

    point_by_task = {task.task_id: task.point_id for task in problem.tasks}
    point_routes = [tuple(point_by_task[task_id] for task_id in route) for route in candidate.routes]
    width = max(len(route) for route in point_routes)
    rows = [{"UAV ID": index, **{f"{column}th Inspection Point": route[column - 1] if column <= len(route) else None for column in range(1, width + 1)}} for index, route in enumerate(point_routes, 1)]
    return pd.DataFrame(rows)


def _validate_solution_header(problem: ProblemData, archive: Mapping[str, Any]) -> None:
    if not isinstance(archive, Mapping) or archive.get("schema_version") != SOLUTION_SCHEMA:
        raise ValueError("archive schema is unsupported")
    if archive.get("case") != problem.case_name or type(archive.get("fleet_size")) is not int:
        raise ValueError("archive case or fleet size mismatch")
    if archive["fleet_size"] != problem.fleet_size:
        raise ValueError("archive case or fleet size mismatch")
    if not _is_sha256(archive.get("parent_archive_sha256")):
        raise ValueError("archive parent SHA-256 is malformed")
    if archive["parent_archive_sha256"] != problem.archive_sha256:
        raise ValueError("archive parent hash mismatch")


def _validate_stored_metrics(stored: Any, candidate: ScoredCandidate) -> None:
    if not isinstance(stored, Mapping):
        raise ValueError("archive metrics must be an object")
    actual = candidate.metrics
    expected = {"Tmax_s": actual.Tmax_s, "Tmin_s": actual.Tmin_s, "delta_s": actual.delta_s, "sum_T_s": actual.sum_T_s}
    for field, value in expected.items():
        if type(stored.get(field)) is not int:
            raise ValueError(f"archive {field} must be an exact integer")
        if stored[field] != value:
            raise ValueError(f"archive {field} mismatch")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _fsync_file(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
