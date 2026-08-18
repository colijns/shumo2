"""Short A/B test for the optional Q3 conflict-focused neighbourhood.

The published strict Q3 archive is used as the common incumbent. The two
searches receive the same wall-clock budget and seed. Checkpoints are written
to a temporary directory, so formal archives and repository outputs remain
untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ATTACH_XLSX = ROOT / "attachment" / "附件1.xlsx"
LEVEL_VISITS = {"I": 3, "II": 2, "III": 1}
SPEED_KMH = 55.0
UNIT_KM = 0.1
SERVICE_S = 300


def _load_case_without_solver(name: str) -> dict:
    """Q1 data helper without importing the optional OR-Tools solver."""
    frame = pd.read_excel(ATTACH_XLSX, sheet_name=name)
    points = []
    for row in frame.itertuples(index=False):
        points.append(
            {
                "pid": int(row.Point_ID),
                "x": float(row.X_Coordinate),
                "y": float(row.Y_Coordinate),
                "level": str(row.Inspection_Level).strip().upper(),
            }
        )
    tasks = []
    for point in points:
        for _ in range(LEVEL_VISITS[point["level"]]):
            tasks.append(
                {
                    "task_id": len(tasks) + 1,
                    "pid": point["pid"],
                    "x": point["x"],
                    "y": point["y"],
                }
            )
    return {"name": name, "points": points, "tasks": tasks}


def _build_matrices_without_solver(case: dict):
    coords = [(0.0, 0.0)] + [(task["x"], task["y"]) for task in case["tasks"]]
    array = np.asarray(coords) * UNIT_KM
    differences = array[:, None, :] - array[None, :, :]
    distances = np.sqrt((differences ** 2).sum(-1))
    times = np.ceil(distances / SPEED_KMH * 3600.0).astype(np.int64).tolist()
    return distances, times


def _evaluate_without_solver(case: dict, distances, times, routes):
    result = []
    for route in routes:
        path = [0] + list(route) + [0]
        flight = int(sum(times[a][b] for a, b in zip(path, path[1:])))
        distance = float(sum(distances[a][b] for a, b in zip(path, path[1:])))
        result.append(
            {
                "n_tasks": len(route),
                "fly_s": flight,
                "dist_km": round(distance, 6),
                "work_s": flight + SERVICE_S * len(route),
                "work_h": (flight + SERVICE_S * len(route)) / 3600.0,
                "task_seq": [int(task) for task in route],
                "point_seq": [int(case["tasks"][task - 1]["pid"]) for task in route],
            }
        )
    return result


# Q3 data loading reaches three helper functions in Q1.solve_q1. The formal
# solver dependencies are irrelevant to this replay test, so expose just those
# helpers and keep the test runnable in the lightweight bundled environment.
q1_stub = types.ModuleType("Q1.solve_q1")
q1_stub.load_case = _load_case_without_solver
q1_stub.build_matrices = _build_matrices_without_solver
q1_stub.evaluate = _evaluate_without_solver
sys.modules["Q1.solve_q1"] = q1_stub

from Q3.domain import Config
from Q3.io import load_archive, load_problem
from Q3.safe_path import eval_solution
from Q3.search import _conflict_focus_tasks, legality, local_search
from Q3.verify import verify_archive


PUBLISHED_Q3 = ROOT / "outputs" / "workbooks" / "q3" / "strict"


def metrics(solution):
    m = solution.metrics
    return {
        "S_max_s": m.S_max_s,
        "S_min_s": m.S_min_s,
        "delta_s": m.delta_s,
        "sum_T_s": m.sum_T_s,
        "total_wait_s": m.total_wait_s,
        "total_distance_km": round(m.total_distance_km, 6),
    }


def _run_search(problem, incumbent, config: Config):
    started = time.monotonic()
    solution = local_search(problem, config, [incumbent])
    return solution, round(time.monotonic() - started, 3)


def evaluate(case: str, budget_s: int, top_edges: int) -> dict:
    problem = load_problem(case, repository_root=ROOT)
    archive = load_archive(PUBLISHED_Q3 / f"{case}.json")
    routes = [tuple(map(int, route)) for route in archive["task_routes"]]
    assert legality(problem, routes), f"{case}: published routes are illegal"
    incumbent = eval_solution(problem, routes)
    assert incumbent is not None, f"{case}: published routes no longer replay"

    focused_tasks = _conflict_focus_tasks(problem, incumbent, top_edges)
    common = dict(
        time_budget_s_per_case=budget_s,
        stall_rounds=1,
        seed=42,
        conflict_top_edges=top_edges,
    )
    original_config = Config(**common, conflict_focus=False)
    focused_config = Config(
        **common,
        conflict_focus=True,
        conflict_full_fallback=False,
    )

    # local_search writes recoverable checkpoints relative to cwd. Redirect
    # them outside the repository so this experiment cannot alter formal data.
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="q3-conflict-smoke-") as temp_dir:
        os.chdir(temp_dir)
        try:
            original, original_elapsed = _run_search(
                problem, incumbent, original_config
            )
            focused, focused_elapsed = _run_search(
                problem, incumbent, focused_config
            )
        finally:
            os.chdir(previous_cwd)

    original_violations = verify_archive(problem, original.freeze())
    focused_violations = verify_archive(problem, focused.freeze())
    return {
        "case": case,
        "budget_s_each": budget_s,
        "top_edges": top_edges,
        "focused_task_ids_by_route": {
            str(route_index + 1): sorted(task_ids)
            for route_index, task_ids in focused_tasks.items()
        },
        "incumbent": metrics(incumbent),
        "original_full_scan": {
            "metrics": metrics(original),
            "elapsed_s": original_elapsed,
            "improved": original.metrics.lex_key() < incumbent.metrics.lex_key(),
            "verification_violations": original_violations,
        },
        "conflict_focused_scan": {
            "metrics": metrics(focused),
            "elapsed_s": focused_elapsed,
            "improved": focused.metrics.lex_key() < incumbent.metrics.lex_key(),
            "verification_violations": focused_violations,
        },
        "focused_vs_original": (
            "better"
            if focused.metrics.lex_key() < original.metrics.lex_key()
            else "equal"
            if focused.metrics.lex_key() == original.metrics.lex_key()
            else "worse"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["Case2", "Case3"])
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--top-edges", type=int, default=12)
    args = parser.parse_args()
    results = [
        evaluate(case, max(args.budget, 0), max(args.top_edges, 1))
        for case in args.cases
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
