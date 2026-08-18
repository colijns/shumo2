

from dataclasses import dataclass, field
from typing import Any, Mapping

from Q2.domain import LEVEL_VISITS, SERVICE_S, SPEED_KMH, UNIT_KM

SAFETY_MARGIN_KM = 0.01
EPS_ARC_KM = 1e-6
EPS_VER_KM = 1e-9
START_T_S = 0
SOLUTION_SCHEMA = "q3-solution-v1"

FLEET_SIZE_BY_CASE = {"Case1": 4, "Case2": 2, "Case3": 5, "Case4": 4}

START_CLOCK_S = 8 * 3600


def hhmm_to_s(text: str) -> int:




    hours, minutes = (int(part) for part in str(text).split(":"))
    return hours * 3600 + minutes * 60


@dataclass(frozen=True, slots=True)
class NoFlyZone:


    zone_id: str
    cx_km: float
    cy_km: float
    radius_km: float
    start_s: int
    end_s: int

    def active(self, t_s: int) -> bool:
        return self.start_s <= t_s <= self.end_s

    def safe_radius(self) -> float:
        return self.radius_km + SAFETY_MARGIN_KM


@dataclass(frozen=True, slots=True)
class Base:


    task_id: int = 0
    point_id: int = 0
    x_km: float = 0.0
    y_km: float = 0.0


@dataclass(frozen=True, slots=True)
class Task:


    task_id: int
    point_id: int
    x_km: float
    y_km: float
    level: str


Node = Task | Base


@dataclass(frozen=True, slots=True)
class ProblemData:


    case: str
    tasks: tuple[Task, ...]
    zones: tuple[NoFlyZone, ...]
    fleet_size: int
    warnings: tuple[str, ...]
    input_sha256: str
    problem_contract_sha256: str
    parent_archive_sha256: str


@dataclass(frozen=True, slots=True)
class Config:


    fleet_size: dict[str, int] = field(default_factory=lambda: dict(FLEET_SIZE_BY_CASE))
    eta_km: float = SAFETY_MARGIN_KM
    seed: int = 42
    time_budget_s_per_case: int = 1800
    stall_rounds: int = 3
    conflict_focus: bool = False
    conflict_top_edges: int = 12
    conflict_full_fallback: bool = True
    nine_hour_cap_s: int | None = None
    lex_first_is_N: bool = False
    allow_empty_routes: bool = False


DEFAULT_CONFIG = Config()


@dataclass(frozen=True, slots=True)
class SegmentRecord:


    from_id: int
    to_id: int
    path_type: str
    depart_s: int
    arrive_s: int
    wait_s: int
    distance_km: float
    affected_zones: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UAVSchedule:


    uav_id: int
    task_route: tuple[int, ...]
    segments: tuple[SegmentRecord, ...]
    service_intervals: tuple[tuple[int, int], ...]
    S_k_s: int
    flight_s: int
    wait_s: int
    distance_km: float


@dataclass(frozen=True, slots=True)
class SolutionMetrics:


    S_max_s: int
    S_min_s: int
    delta_s: int
    sum_T_s: int
    total_wait_s: int
    total_distance_km: float

    def lex_key(self) -> tuple:
        return (
            self.S_max_s,
            self.delta_s,
            self.sum_T_s,
            self.total_wait_s,
            round(self.total_distance_km, 6),
        )


@dataclass(frozen=True, slots=True)
class Solution:


    problem: ProblemData
    schedules: tuple[UAVSchedule, ...]
    metrics: SolutionMetrics

    def freeze(self) -> dict[str, Any]:
        return {
            "schema_version": SOLUTION_SCHEMA,
            "case": self.problem.case,
            "fleet_size": self.problem.fleet_size,
            "input": {
                "attachment_sha256": self.problem.input_sha256,
                "problem_contract_sha256": self.problem.problem_contract_sha256,
                "parent_archive_sha256": self.problem.parent_archive_sha256,
            },
            "task_routes": [list(sched.task_route) for sched in self.schedules],
            "schedules": [
                {
                    "uav_id": sched.uav_id,
                    "segments": [
                        {
                            "from_id": seg.from_id,
                            "to_id": seg.to_id,
                            "path_type": seg.path_type,
                            "depart_s": seg.depart_s,
                            "arrive_s": seg.arrive_s,
                            "wait_s": seg.wait_s,
                            "distance_km": seg.distance_km,
                            "affected_zones": list(seg.affected_zones),
                        }
                        for seg in sched.segments
                    ],
                    "service_intervals": [list(iv) for iv in sched.service_intervals],
                    "S_k_s": sched.S_k_s,
                    "flight_s": sched.flight_s,
                    "wait_s": sched.wait_s,
                    "distance_km": sched.distance_km,
                }
                for sched in self.schedules
            ],
            "metrics": {
                "S_max_s": self.metrics.S_max_s,
                "S_min_s": self.metrics.S_min_s,
                "delta_s": self.metrics.delta_s,
                "sum_T_s": self.metrics.sum_T_s,
                "total_wait_s": self.metrics.total_wait_s,
                "total_distance_km": self.metrics.total_distance_km,
            },
        }


def freeze_json(value: Any) -> Any:

    if isinstance(value, dict):
        return {key: freeze_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value



__all__ = [
    "DEFAULT_CONFIG",
    "START_CLOCK_S",
    "EPS_ARC_KM",
    "EPS_VER_KM",
    "FLEET_SIZE_BY_CASE",
    "LEVEL_VISITS",
    "SAFETY_MARGIN_KM",
    "SERVICE_S",
    "SOLUTION_SCHEMA",
    "SPEED_KMH",
    "START_T_S",
    "UNIT_KM",
    "Base",
    "Config",
    "Node",
    "NoFlyZone",
    "ProblemData",
    "SegmentRecord",
    "Solution",
    "SolutionMetrics",
    "Task",
    "UAVSchedule",
    "freeze_json",
    "hhmm_to_s",
]
