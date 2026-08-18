import hashlib
import time
from collections import Counter
from pathlib import Path
import numpy as np
from .domain import Base, Config, ProblemData, Solution, SolutionMetrics, UAVSchedule
from .io import (
    atomic_write_json,
    load_archive,
    load_hot_start_archive,
)
from .geometry import segment_circle_interval
from .safe_path import (
    earliest_safe_service_completion,
    earliest_safe_travel,
    eval_route,
    eval_solution,
    wait_safe,
)
LEVEL_PRIORITY = {"I": 0, "II": 1, "III": 2}
def derived_seed(seed: int, case: str, stage: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{case}:{stage}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
def _rng(seed: int, case: str, stage: str, index: int) -> np.random.RandomState:
    return np.random.RandomState(derived_seed(seed, case, stage, index))
def legality(problem: ProblemData, task_routes: list[tuple[int, ...]]) -> bool:
    if len(task_routes) != problem.fleet_size:
        return False
    seen: list[int] = []
    for route in task_routes:
        if not route:
            return False
        prev_point = None
        for task_id in route:
            if task_id < 1 or task_id > len(problem.tasks):
                return False
            point_id = problem.tasks[task_id - 1].point_id
            if point_id == prev_point:
                return False
            prev_point = point_id
            seen.append(task_id)
    if sorted(seen) != list(range(1, len(problem.tasks) + 1)):
        return False
    return True
def _greedy_nearest(problem: ProblemData, rng: np.random.RandomState, level_order: bool) -> list[tuple[int, ...]]:
    tasks = problem.tasks
    remaining: list[int] = list(range(1, len(tasks) + 1))
    if level_order:
        remaining.sort(key=lambda task_id: (LEVEL_PRIORITY[tasks[task_id - 1].level], task_id))
    else:
        remaining.sort()
    routes: list[list[int]] = [[] for _ in range(problem.fleet_size)]
    current = [Base() for _ in range(problem.fleet_size)]
    def closest(origin, candidates: list[int]) -> int | None:
        if not candidates:
            return None
        prev_point = origin.point_id
        feasible = [
            task_id for task_id in candidates
            if tasks[task_id - 1].point_id != prev_point
        ]
        if not feasible:
            return None
        origin_pos = (origin.x_km, origin.y_km)
        return min(
            feasible,
            key=lambda task_id: (
                (tasks[task_id - 1].x_km - origin_pos[0]) ** 2
                + (tasks[task_id - 1].y_km - origin_pos[1]) ** 2
            ),
        )
    while remaining:
        for index in range(problem.fleet_size):
            if not remaining:
                break
            pick = closest(current[index], remaining)
            if pick is None:
                continue
            routes[index].append(pick)
            remaining.remove(pick)
            current[index] = tasks[pick - 1]
    return [tuple(route) for route in routes]
def _greedy_random(problem: ProblemData, rng: np.random.RandomState) -> list[tuple[int, ...]]:
    tasks = problem.tasks
    remaining: list[int] = list(range(1, len(tasks) + 1))
    routes: list[list[int]] = [[] for _ in range(problem.fleet_size)]
    current = [Base() for _ in range(problem.fleet_size)]
    while remaining:
        for index in range(problem.fleet_size):
            if not remaining:
                break
            origin_pos = (current[index].x_km, current[index].y_km)
            ranked = sorted(
                remaining,
                key=lambda task_id: (
                    (tasks[task_id - 1].x_km - origin_pos[0]) ** 2
                    + (tasks[task_id - 1].y_km - origin_pos[1]) ** 2
                ),
            )
            pick = ranked[0]
            if len(ranked) > 1:
                pick = ranked[rng.randint(0, min(3, len(ranked)))]
            routes[index].append(pick)
            remaining.remove(pick)
            current[index] = tasks[pick - 1]
    return [tuple(route) for route in routes]
def _node_distance(a, b) -> float:
    return float(np.hypot(b.x_km - a.x_km, b.y_km - a.y_km))
def _min_feasible_wait(problem: ProblemData, prev, task, depart_s: int) -> int | None:
    wait = 0
    tried: set[int] = set()
    while True:
        if earliest_safe_service_completion(problem, prev, task, depart_s + wait) is not None:
            if wait == 0 or wait_safe(
                problem, (prev.x_km, prev.y_km), depart_s, depart_s + wait
            ):
                return wait
        events = sorted(
            {zone.end_s + 1 for zone in problem.zones if zone.end_s + 1 > depart_s + wait}
        )
        if not events:
            return None
        wait = events[0] - depart_s
        if wait in tried:
            return None
        tried.add(wait)
def _greedy_time_aware(problem: ProblemData) -> list[tuple[int, ...]] | None:
    tasks = problem.tasks
    fleet = problem.fleet_size
    remaining: set[int] = set(range(1, len(tasks) + 1))
    routes: list[list[int]] = [[] for _ in range(fleet)]
    prevs: list = [Base() for _ in range(fleet)]
    departs: list[int] = [0] * fleet
    history: list[list[tuple[int, object, int]]] = [[] for _ in range(fleet)]
    banned: set[tuple[int, int]] = set()
    rollbacks = 0
    max_rollbacks = fleet * 20
    def roll_back(index: int) -> bool:
        nonlocal rollbacks
        if not routes[index] or rollbacks >= max_rollbacks:
            return False
        task_id, prev0, depart0 = history[index].pop()
        routes[index].pop()
        remaining.add(task_id)
        prevs[index], departs[index] = prev0, depart0
        banned.add((index, task_id))
        rollbacks += 1
        return True
    while True:
        while remaining:
            progressed = False
            for index in range(fleet):
                if not remaining:
                    break
                prev = prevs[index]
                depart = departs[index]
                best_task, best_dist, best_record = None, None, None
                for task_id in sorted(remaining):
                    if (index, task_id) in banned:
                        continue
                    task = tasks[task_id - 1]
                    if task.point_id == prev.point_id:
                        continue
                    record = earliest_safe_service_completion(problem, prev, task, depart)
                    if record is None:
                        continue
                    dist = _node_distance(prev, task)
                    if best_task is None or dist < best_dist:
                        best_task, best_dist, best_record = task_id, dist, record
                if best_task is not None:
                    _, arrive, service_end = best_record
                    routes[index].append(best_task)
                    remaining.discard(best_task)
                    history[index].append((best_task, prev, depart))
                    prevs[index] = tasks[best_task - 1]
                    departs[index] = service_end
                    progressed = True
                    continue
                rescuable = False
                for task_id in sorted(remaining):
                    if (index, task_id) in banned:
                        continue
                    task = tasks[task_id - 1]
                    if task.point_id == prev.point_id:
                        continue
                    if _min_feasible_wait(problem, prev, task, depart) is not None:
                        rescuable = True
                        break
                if not rescuable:
                    roll_back(index)
            if progressed:
                continue
            best_advance: tuple[int, int] | None = None
            for index in range(fleet):
                for task_id in sorted(remaining):
                    if (index, task_id) in banned:
                        continue
                    task = tasks[task_id - 1]
                    if task.point_id == prevs[index].point_id:
                        continue
                    wait = _min_feasible_wait(problem, prevs[index], task, departs[index])
                    if wait is not None and wait > 0 and (
                        best_advance is None or wait < best_advance[0]
                    ):
                        best_advance = (wait, index)
            if best_advance is None:
                return None
            departs[best_advance[1]] += best_advance[0]
        failing = [
            index for index in range(fleet)
            if routes[index] and earliest_safe_travel(problem, prevs[index], Base(), departs[index]) is None
        ]
        if not failing:
            return [tuple(route) for route in routes]
        index = max(failing, key=lambda i: departs[i])
        if not roll_back(index):
            return None
def _q3_archive_routes(problem: ProblemData, repository_root: Path) -> list[tuple[int, ...]] | None:
    strict = repository_root / "outputs" / "workbooks" / "q3" / "strict" / f"{problem.case}.json"
    if not strict.is_file():
        return None
    archive = load_archive(strict)
    routes = archive.get("task_routes")
    if not isinstance(routes, list) or not routes:
        return None
    task_routes = [tuple(int(task_id) for task_id in uav) for uav in routes]
    if not legality(problem, task_routes):
        return None
    return task_routes
def _hot_start_routes(problem: ProblemData, repository_root: Path) -> tuple[list[tuple[int, ...]], dict] | None:
    strict = repository_root / "outputs" / "workbooks" / "q2" / "strict" / f"{problem.case}.json"
    if not strict.is_file():
        return None
    archive = load_hot_start_archive(problem, strict)
    routes = [tuple(int(task_id) for task_id in uav) for uav in archive["task_routes"]]
    if not legality(problem, routes):
        return None
    meta = {
        "source": "q2-strict",
        "archive_path": str(strict),
        "archive_sha256": _file_sha256(strict),
        "fallback": None,
    }
    return routes, meta
def _q1_baseline_routes(problem: ProblemData, repository_root: Path) -> list[tuple[int, ...]] | None:
    baseline = repository_root / "outputs" / "workbooks" / "baseline_20260816" / f"q1_solution_{problem.case}.json"
    if not baseline.is_file():
        return None
    archive = load_archive(baseline)
    if archive.get("case") != problem.case:
        return None
    uavs = archive.get("uavs")
    if not isinstance(uavs, list):
        return None
    routes = [tuple(int(task_id) for task_id in uav.get("task_seq", ())) for uav in uavs]
    if not legality(problem, routes):
        return None
    return routes
def _file_sha256(path: Path) -> str:
    import hashlib as _hashlib
    digest = _hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _perturbations(problem: ProblemData, base: list[tuple[int, ...]], count: int) -> list[list[tuple[int, ...]]]:
    results: list[list[tuple[int, ...]]] = []
    for index in range(count):
        rng = _rng(42, problem.case, "perturb", index)
        routes = [list(route) for route in base]
        moves_left = rng.randint(1, 4)
        for _ in range(moves_left):
            if rng.rand() < 0.5:
                _perturb_relocate(problem, routes, rng)
            else:
                _perturb_swap(problem, routes, rng)
        candidate = [tuple(route) for route in routes]
        if legality(problem, candidate):
            results.append(candidate)
    return results
def _perturb_relocate(problem: ProblemData, routes: list[list[int]], rng: np.random.RandomState) -> None:
    source = rng.randint(0, len(routes))
    if len(routes[source]) < 2:
        return
    position = rng.randint(0, len(routes[source]))
    task_id = routes[source].pop(position)
    target = rng.randint(0, len(routes))
    routes[target].insert(rng.randint(0, len(routes[target]) + 1), task_id)
def _perturb_swap(problem: ProblemData, routes: list[list[int]], rng: np.random.RandomState) -> None:
    if len(routes) < 2:
        return
    a, b = rng.choice(len(routes), size=2, replace=False)
    if not routes[a] or not routes[b]:
        return
    ia, ib = rng.randint(0, len(routes[a])), rng.randint(0, len(routes[b]))
    routes[a][ia], routes[b][ib] = routes[b][ib], routes[a][ia]
def build_initial_solutions(problem: ProblemData, rng_hashed, k: int, repository_root: Path) -> list[Solution]:
    pool: list[Solution] = []
    if repository_root is not None:
        sources: list[list[tuple[int, ...]]] = []
        q3 = _q3_archive_routes(problem, repository_root)
        if q3 is not None:
            sources.append(q3)
        hot = _hot_start_routes(problem, repository_root)
        if hot is not None:
            sources.append(hot[0])
        for routes in sources:
            solution = eval_solution(problem, routes)
            if solution is not None:
                pool.append(solution)
    if pool:
        pool.sort(key=lambda s: s.metrics.lex_key())
        base = [sched.task_route for sched in pool[0].schedules]
        pool.extend(_eval_perturbations(problem, base, k - 1))
    if not pool and repository_root is not None:
        baseline = _q1_baseline_routes(problem, repository_root)
        if baseline is not None:
            solution = eval_solution(problem, baseline)
            if solution is not None:
                pool.append(solution)
    for variant in (0, 1):
        routes = _greedy_nearest(
            problem, _rng(42, problem.case, "greedy", variant), level_order=bool(variant)
        )
        solution = eval_solution(problem, routes)
        if solution is not None:
            pool.append(solution)
    routes = _greedy_random(problem, _rng(42, problem.case, "greedy", 2))
    solution = eval_solution(problem, routes)
    if solution is not None:
        pool.append(solution)
    routes = _greedy_time_aware(problem)
    if routes is not None:
        solution = eval_solution(problem, routes)
        if solution is not None:
            pool.append(solution)
    if not pool:
        raise RuntimeError(f"{problem.case}: no feasible initial solution (fail closed)")
    pool.sort(key=lambda s: s.metrics.lex_key())
    return pool
def _eval_perturbations(problem: ProblemData, base: list[tuple[int, ...]], count: int) -> list[Solution]:
    results: list[Solution] = []
    for candidate in _perturbations(problem, base, max(count, 0)):
        solution = eval_solution(problem, candidate)
        if solution is not None:
            results.append(solution)
    return results
def _distance_km(problem: ProblemData, a: int, b: int) -> float:
    pa = (0.0, 0.0) if a == 0 else (problem.tasks[a - 1].x_km, problem.tasks[a - 1].y_km)
    pb = (0.0, 0.0) if b == 0 else (problem.tasks[b - 1].x_km, problem.tasks[b - 1].y_km)
    return float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
def _best_insert_position(problem: ProblemData, route: list[int], task_id: int) -> int:
    best_pos, best_gain = 0, None
    points = [0] + route + [0]
    for position in range(len(points) - 1):
        gain = (
            _distance_km(problem, points[position], task_id)
            + _distance_km(problem, task_id, points[position + 1])
            - _distance_km(problem, points[position], points[position + 1])
        )
        if best_gain is None or gain < best_gain:
            best_gain, best_pos = gain, position
    return best_pos
def _two_opt_routes(
    route: tuple[int, ...], focus_tasks: set[int] | None = None
) -> list[tuple[int, ...]]:
    if len(route) < 4:
        return []
    results = []
    for i in range(len(route) - 2):
        for j in range(i + 2, len(route)):
            if focus_tasks is not None and not ({route[i], route[j]} & focus_tasks):
                continue
            results.append(tuple(route[:i] + tuple(reversed(route[i:j + 1])) + route[j + 1:]))
    return results
def _node_position(problem: ProblemData, task_id: int) -> tuple[float, float]:
    if task_id == 0:
        return (0.0, 0.0)
    task = problem.tasks[task_id - 1]
    return (task.x_km, task.y_km)
def _boundary_gap_s(problem: ProblemData, segment) -> float:
    start = _node_position(problem, segment.from_id)
    end = _node_position(problem, segment.to_id)
    duration = max(segment.arrive_s - segment.depart_s, 0)
    best = float("inf")
    for zone in problem.zones:
        interval = segment_circle_interval(
            start,
            end,
            (zone.cx_km, zone.cy_km),
            zone.safe_radius(),
        )
        if interval is None or interval[1] - interval[0] <= 1e-6:
            continue
        enter_s = segment.depart_s + interval[0] * duration
        leave_s = segment.depart_s + interval[1] * duration
        best = min(
            best,
            abs(enter_s - zone.start_s),
            abs(enter_s - zone.end_s),
            abs(leave_s - zone.start_s),
            abs(leave_s - zone.end_s),
        )
    return best
def _conflict_focus_tasks(
    problem: ProblemData,
    solution: Solution,
    top_edges: int,
) -> dict[int, set[int]]:
    ranked: list[tuple[tuple, int, tuple[int, ...]]] = []
    for route_index, schedule in enumerate(solution.schedules):
        for segment in schedule.segments:
            direct_km = _distance_km(problem, segment.from_id, segment.to_id)
            detour_extra = max(segment.distance_km - direct_km, 0.0)
            boundary_gap = _boundary_gap_s(problem, segment)
            avoid = int(segment.path_type != "direct" or segment.wait_s > 0)
            endpoints = tuple(
                task_id for task_id in (segment.from_id, segment.to_id)
                if task_id != 0
            )
            if not endpoints:
                continue
            score = (
                schedule.S_k_s,
                avoid,
                segment.wait_s,
                round(detour_extra, 6),
                -boundary_gap,
            )
            ranked.append((score, route_index, endpoints))
    ranked.sort(key=lambda item: item[0], reverse=True)
    focused: dict[int, set[int]] = {}
    for _, route_index, endpoints in ranked[:max(int(top_edges), 1)]:
        focused.setdefault(route_index, set()).update(endpoints)
    return focused
def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline
def local_search(problem: ProblemData, config: Config, initial_pool: list[Solution]) -> Solution:
    best = initial_pool[0]
    checkpoint_dir = Path("outputs") / "workbooks" / "q3" / "checkpoints"
    stall = 0
    round_index = 0
    started = time.monotonic()
    deadline = started + max(float(config.time_budget_s_per_case), 0.0)
    operators = ("two_opt", "relocate", "swap", "block", "destroy_repair")
    while True:
        if _deadline_reached(deadline):
            break
        rng = _rng(config.seed, problem.case, "round", round_index)
        improved = False
        phases: list[dict[int, set[int]] | None] = [None]
        if config.conflict_focus:
            focus = _conflict_focus_tasks(problem, best, config.conflict_top_edges)
            phases = [focus]
            if config.conflict_full_fallback:
                phases.append(None)
        for focus_tasks in phases:
            phase_operators = operators if focus_tasks is None else operators[:-1]
            for operator in phase_operators:
                if _deadline_reached(deadline):
                    break
                candidate = _operator_candidate(
                    problem,
                    best,
                    operator,
                    rng,
                    focus_tasks=focus_tasks,
                    deadline=deadline,
                )
                if candidate is not None and candidate.metrics.lex_key() < best.metrics.lex_key():
                    best = candidate
                    improved = True
                    atomic_write_json(
                        checkpoint_dir / f"checkpoint_{problem.case}.json",
                        best.freeze(),
                    )
                    break
            if improved or _deadline_reached(deadline):
                break
        if _deadline_reached(deadline):
            break
        if improved:
            stall = 0
        else:
            stall += 1
            if stall >= config.stall_rounds:
                break
        round_index += 1
    return best
def _operator_candidate(
    problem: ProblemData,
    solution: Solution,
    operator: str,
    rng,
    focus_tasks: dict[int, set[int]] | None = None,
    deadline: float | None = None,
) -> Solution | None:
    if _deadline_reached(deadline):
        return None
    routes = [list(sched.task_route) for sched in solution.schedules]
    if operator == "two_opt":
        for route_index, route in enumerate(routes):
            route_focus = None if focus_tasks is None else focus_tasks.get(route_index, set())
            if focus_tasks is not None and not route_focus:
                continue
            for reversal in _two_opt_routes(tuple(route), route_focus):
                if _deadline_reached(deadline):
                    return None
                candidate_routes = [tuple(r) for r in routes]
                candidate_routes[route_index] = reversal
                if not legality(problem, candidate_routes):
                    continue
                candidate = eval_solution(problem, candidate_routes)
                if candidate is not None and candidate.metrics.lex_key() < solution.metrics.lex_key():
                    return candidate
    elif operator == "relocate":
        for source in range(len(routes)):
            source_focus = None if focus_tasks is None else focus_tasks.get(source, set())
            if focus_tasks is not None and not source_focus:
                continue
            for task_id in list(routes[source]):
                if source_focus is not None and task_id not in source_focus:
                    continue
                for target in range(len(routes)):
                    if _deadline_reached(deadline):
                        return None
                    if target == source:
                        continue
                    position = _best_insert_position(problem, routes[target], task_id)
                    candidate_routes = [tuple(r) for r in routes]
                    candidate_routes[source] = tuple(t for t in routes[source] if t != task_id)
                    inserted = list(routes[target])
                    inserted.insert(position, task_id)
                    candidate_routes[target] = tuple(inserted)
                    if not legality(problem, candidate_routes):
                        continue
                    candidate = eval_solution(problem, candidate_routes)
                    if candidate is not None and candidate.metrics.lex_key() < solution.metrics.lex_key():
                        return candidate
    elif operator == "swap":
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for task_a in list(routes[a]):
                    for task_b in list(routes[b]):
                        if _deadline_reached(deadline):
                            return None
                        if focus_tasks is not None and (
                            task_a not in focus_tasks.get(a, set())
                            and task_b not in focus_tasks.get(b, set())
                        ):
                            continue
                        candidate_routes = [tuple(r) for r in routes]
                        candidate_routes[a] = tuple(
                            task_b if task_id == task_a else task_id for task_id in routes[a]
                        )
                        candidate_routes[b] = tuple(
                            task_a if task_id == task_b else task_id for task_id in routes[b]
                        )
                        if not legality(problem, candidate_routes):
                            continue
                        candidate = eval_solution(problem, candidate_routes)
                        if candidate is not None and candidate.metrics.lex_key() < solution.metrics.lex_key():
                            return candidate
    elif operator == "block":
        for source in range(len(routes)):
            source_focus = None if focus_tasks is None else focus_tasks.get(source, set())
            if focus_tasks is not None and not source_focus:
                continue
            route = routes[source]
            for length in (2, 3, 4):
                for start in range(len(route) - length + 1):
                    block = tuple(route[start:start + length])
                    if source_focus is not None and not (set(block) & source_focus):
                        continue
                    for target in range(len(routes)):
                        if _deadline_reached(deadline):
                            return None
                        if target == source:
                            continue
                        position = _best_insert_position(problem, routes[target], block[0])
                        candidate_routes = [tuple(r) for r in routes]
                        candidate_routes[source] = tuple(
                            task_id for index, task_id in enumerate(route)
                            if index < start or index >= start + length
                        )
                        inserted = list(routes[target])
                        inserted[position:position] = block
                        candidate_routes[target] = tuple(inserted)
                        if not legality(problem, candidate_routes):
                            continue
                        candidate = eval_solution(problem, candidate_routes)
                        if candidate is not None and candidate.metrics.lex_key() < solution.metrics.lex_key():
                            return candidate
    elif operator == "destroy_repair":
        if focus_tasks is not None:
            return None
        candidate = _destroy_repair(problem, routes, rng)
        if candidate is not None and legality(problem, candidate):
            solution_candidate = eval_solution(problem, candidate)
            if solution_candidate is not None and solution_candidate.metrics.lex_key() < solution.metrics.lex_key():
                return solution_candidate
    return None
def _destroy_repair(problem: ProblemData, routes: list[list[int]], rng) -> list[tuple[int, ...]] | None:
    longest = max(range(len(routes)), key=lambda index: len(routes[index]))
    if len(routes[longest]) < 3:
        return None
    q = min(rng.randint(3, 6), len(routes[longest]))
    indices = sorted(rng.choice(len(routes[longest]), size=q, replace=False).tolist())
    removed = [routes[longest][index] for index in reversed(indices)]
    routes[longest] = [task_id for index, task_id in enumerate(routes[longest]) if index not in indices]
    for task_id in removed:
        best_target, best_pos, best_gain = None, None, None
        for target in range(len(routes)):
            position = _best_insert_position(problem, routes[target], task_id)
            points = [0] + routes[target] + [0]
            gain = (
                _distance_km(problem, points[position], task_id)
                + _distance_km(problem, task_id, points[position + 1])
                - _distance_km(problem, points[position], points[position + 1])
            )
            if best_gain is None or gain < best_gain:
                best_target, best_pos, best_gain = target, position, gain
        routes[best_target].insert(best_pos, task_id)
    return [tuple(route) for route in routes]
def solve_case(case: str, config: Config, repository_root: Path | None = None) -> Solution:
    from .io import load_problem
    problem = load_problem(case, repository_root=repository_root)
    pool = build_initial_solutions(problem, None, k=8, repository_root=repository_root)
    return local_search(problem, config, pool)
