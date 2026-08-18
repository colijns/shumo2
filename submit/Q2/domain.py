

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping

SPEED_KMH = 55.0
UNIT_KM = 0.1
SERVICE_S = 300
CAP_S = 32_400
LEVEL_VISITS = MappingProxyType({"I": 3, "II": 2, "III": 1})


@dataclass(frozen=True, slots=True)
class Task:


    task_id: int
    point_id: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class RouteMetrics:


    task_sequence: tuple[int, ...]
    point_sequence: tuple[int, ...]
    flight_s: int
    distance_km: float
    work_s: int


@dataclass(frozen=True, slots=True)
class SolutionMetrics:


    routes: tuple[RouteMetrics, ...]
    work_s: tuple[int, ...]
    flight_s: tuple[int, ...]
    distance_km: tuple[float, ...]
    Tmax_s: int
    Tmin_s: int
    delta_s: int
    sum_T_s: int
    mean_T_s: Fraction
    std_T_s: float


@dataclass(frozen=True, slots=True)
class ProblemData:


    case_name: str
    fleet_size: int
    tasks: tuple[Task, ...]
    routes: tuple[tuple[int, ...], ...]
    distance_km: tuple[tuple[float, ...], ...]
    time_s: tuple[tuple[int, ...], ...]
    metrics: SolutionMetrics
    parent_archive: Mapping[str, Any]
    attachment_path: str
    archive_path: str
    attachment_sha256: str
    archive_sha256: str
    problem_contract_sha256: str
    q1_compatibility_checked: bool

    @property
    def point_sequences(self) -> tuple[tuple[int, ...], ...]:

        by_id = {task.task_id: task.point_id for task in self.tasks}
        return tuple(tuple(by_id[task_id] for task_id in route) for route in self.routes)


def freeze_json(value: Any) -> Any:

    if isinstance(value, dict):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value
