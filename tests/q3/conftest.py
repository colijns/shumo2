"""Mini synthetic fixtures for Q3: 4 points + no-fly zones, hand-computable truths.

Layout (km): base at (0,0); tasks at (1,0), (0,1), (-1,0), (0,-1).
Z1 disk (0.5,0) r=0.2 blocks the direct base->T1 leg; Z2 disk (0,0.5) r=0.2
blocks the direct base->T2 leg. Safe radii are r + eta = 0.21.
"""

import hashlib
import math
from pathlib import Path

import pytest

from Q3.domain import (
    SAFETY_MARGIN_KM,
    NoFlyZone,
    ProblemData,
    Task,
)

ROOT = Path(__file__).resolve().parents[2]

SHA = hashlib.sha256(b"mini").hexdigest()


def task(task_id: int, x_km: float, y_km: float, level: str = "III") -> Task:
    return Task(task_id, 100 + task_id, x_km, y_km, level)


MINI_TASKS = (
    task(1, 1.0, 0.0),
    task(2, 0.0, 1.0),
    task(3, -1.0, 0.0),
    task(4, 0.0, -1.0),
)

Z1_ALL_DAY = NoFlyZone("Z1", 0.5, 0.0, 0.2, 0, 90_000)
Z2_ALL_DAY = NoFlyZone("Z2", 0.0, 0.5, 0.2, 0, 90_000)


def make_problem(tasks=MINI_TASKS, zones=(Z1_ALL_DAY, Z2_ALL_DAY), fleet_size=2) -> ProblemData:
    return ProblemData(
        case="Mini",
        tasks=tasks,
        zones=zones,
        fleet_size=fleet_size,
        warnings=(),
        input_sha256=SHA,
        problem_contract_sha256=SHA,
        parent_archive_sha256=SHA,
    )


@pytest.fixture
def mini_problem() -> ProblemData:
    """Both zones active all day: forces detour on legs to T1 and T2."""
    return make_problem()


@pytest.fixture
def mini_problem_windowed() -> ProblemData:
    """Z1 active [0, 300] only: waiting until 301 then direct is strictly best."""
    z1_window = NoFlyZone("Z1", 0.5, 0.0, 0.2, 0, 300)
    z2_window = NoFlyZone("Z2", 0.0, 0.5, 0.2, 0, 300)
    return make_problem(zones=(z1_window, z2_window))


@pytest.fixture
def mini_problem_service_conflict() -> ProblemData:
    """T1 sits inside Z4 (center (1,0) r=0.1, active [67, 5000]).

    Direct flight [0,66] is safe; the 300 s service window [66,366] overlaps the
    zone, so the UAV must delay departure from base until A' + 300 > 5000.
    """
    z4 = NoFlyZone("Z4", 1.0, 0.0, 0.1, 67, 5000)
    return make_problem(zones=(z4,))


@pytest.fixture
def base():
    from Q3.domain import Base

    return Base()


# ---------- hand-computable geometry truths ----------

DIRECT_BASE_TO_T1_S = math.ceil(1.0 / 55.0 * 3600)  # 66


def detour_distance_base_to_t1() -> float:
    """Exact tangent—arc—tangent length around Z1 (safe radius 0.21)."""
    r = 0.2 + SAFETY_MARGIN_KM
    center_distance = 0.5
    tangent_len = math.sqrt(center_distance * center_distance - r * r)
    half_angle = math.acos(r / center_distance)
    arc_angle = math.pi - 2 * half_angle
    return 2 * tangent_len + r * arc_angle


DETOUR_BASE_TO_T1_KM = detour_distance_base_to_t1()
DETOUR_BASE_TO_T1_S = math.ceil(DETOUR_BASE_TO_T1_KM / 55.0 * 3600)

WAIT_UNTIL_Z1_END_S = 301  # e_z - now + 1 with e_z=300, now=0
