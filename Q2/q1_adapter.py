"""Narrow, read-only boundary between the Q2 search and Q1 data contracts."""

from Q1.solve_q1 import build_matrices, evaluate, load_case

FLEET_SIZE_BY_CASE = {
    "Case1": 4,
    "Case2": 2,
    "Case3": 5,
    "Case4": 4,
}


def replay_archive(case_name: str, archive: dict) -> dict:
    """Replay a Q1 archive through the three approved Q1 helpers."""
    case = load_case(case_name)
    dist_km, time_s = build_matrices(case)
    immutable_routes = tuple(tuple(uav["task_seq"]) for uav in archive["uavs"])
    stats = evaluate(case, dist_km, time_s, [list(route) for route in immutable_routes])
    return {
        "case": case,
        "fleet_size": FLEET_SIZE_BY_CASE[case_name],
        "routes": immutable_routes,
        "dist_km": dist_km,
        "time_s": time_s,
        "stats": stats,
    }
