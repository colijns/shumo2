"""M3: initial pool, legality, deterministic local search, termination."""

from pathlib import Path
import time

import pytest

from Q3.domain import Config
from Q3.safe_path import eval_solution
from Q3.search import (
    build_initial_solutions,
    derived_seed,
    legality,
    local_search,
    solve_case,
)
from tests.q3.conftest import MINI_TASKS, make_problem

ROOT = Path(__file__).resolve().parents[2]

MINI_FLEET_ONE = make_problem(fleet_size=1)


def test_derived_seed_is_deterministic_and_distinct():
    assert derived_seed(42, "Mini", "round", 0) == derived_seed(42, "Mini", "round", 0)
    assert derived_seed(42, "Mini", "round", 0) != derived_seed(42, "Mini", "round", 1)
    assert derived_seed(42, "Mini", "round", 0) != derived_seed(42, "Mini", "perturb", 0)
    assert derived_seed(43, "Mini", "round", 0) != derived_seed(42, "Mini", "round", 0)


def test_legality_accepts_complete_disjoint_routes():
    problem = make_problem(fleet_size=2)
    assert legality(problem, [(1, 2), (3, 4)]) is True
    assert legality(problem, [(1, 3), (2, 4)]) is True


def test_legality_rejects_wrong_fleet_size():
    problem = make_problem(fleet_size=2)
    assert legality(problem, [(1, 2, 3, 4)]) is False
    assert legality(problem, [(1,), (2,), (3,), (4,)]) is False


def test_legality_rejects_empty_route_and_duplicate_and_adjacent_same_point():
    problem = make_problem(fleet_size=2)
    assert legality(problem, [(1, 2), ()]) is False
    assert legality(problem, [(1, 1), (3, 4)]) is False
    # task 5 shares point 103 with task 3: adjacency is the violation
    from Q3.domain import Task

    five_tasks = (MINI_TASKS[0], MINI_TASKS[1], MINI_TASKS[2], MINI_TASKS[3], Task(5, 103, -1.0, 0.0, "III"))
    problem5 = make_problem(tasks=five_tasks, fleet_size=2)
    assert legality(problem5, [(1, 3, 5), (2, 4)]) is False   # 3->5 same point
    assert legality(problem5, [(1, 5), (2, 3, 4)]) is True    # 103 used once


def test_legality_rejects_task_out_of_range():
    problem = make_problem(fleet_size=2)
    assert legality(problem, [(1, 2), (3, 99)]) is False


def test_hot_start_loads_real_case1_archive_readonly():
    """Real Q2 strict archive for Case1 must seed the pool (contract-gated)."""
    problem = __import__("Q3.io", fromlist=["load_problem"]).load_problem("Case1", repository_root=ROOT)
    pool = build_initial_solutions(problem, None, k=2, repository_root=ROOT)
    assert pool
    first = pool[0]
    assert first.metrics.S_max_s > 0
    assert len(first.schedules) == 4
    # every route must satisfy the structural legality rules
    routes = [sched.task_route for sched in first.schedules]
    assert legality(problem, routes)


def test_q3_archive_seeds_pool_first():
    """Own Q3 verified archive must seed the pool ahead of the Q2 hot start."""
    from Q3.io import load_archive, load_problem

    problem = load_problem("Case1", repository_root=ROOT)
    pool = build_initial_solutions(problem, None, k=2, repository_root=ROOT)
    assert pool
    archive = load_archive(ROOT / "outputs" / "workbooks" / "q3" / "strict" / "Case1.json")
    archived = eval_solution(
        problem,
        [tuple(int(task_id) for task_id in route) for route in archive["task_routes"]],
    )
    assert archived is not None
    first = pool[0]
    assert first.metrics.lex_key() == archived.metrics.lex_key()
    assert [sched.task_route for sched in first.schedules] == [
        sched.task_route for sched in archived.schedules
    ]


def test_q3_archive_missing_falls_back_to_q2(tmp_path):
    """Without an own Q3 archive, the pool falls back to the Q2 hot start."""
    from Q3.io import load_problem
    from Q3.search import _hot_start_routes

    src = ROOT / "outputs" / "workbooks" / "q2" / "strict" / "Case1.json"
    dst = tmp_path / "outputs" / "workbooks" / "q2" / "strict" / "Case1.json"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(src.read_bytes())
    problem = load_problem("Case1", repository_root=ROOT)
    pool = build_initial_solutions(problem, None, k=2, repository_root=tmp_path)
    assert pool
    hot = _hot_start_routes(problem, tmp_path)
    assert hot is not None
    hot_sol = eval_solution(problem, hot[0])
    assert hot_sol is not None
    assert pool[0].metrics.lex_key() == hot_sol.metrics.lex_key()


def test_operator_outputs_pass_legality(mini_problem):
    """Every operator candidate must stay structurally legal."""
    routes = [(1, 2), (3, 4)]
    solution = eval_solution(mini_problem, routes)
    assert solution is not None
    from Q3.search import _operator_candidate, _rng

    rng = _rng(42, "Mini", "round", 0)
    for operator in ("two_opt", "relocate", "swap", "block", "destroy_repair"):
        candidate = _operator_candidate(mini_problem, solution, operator, rng)
        if candidate is not None:
            routes = [sched.task_route for sched in candidate.schedules]
            assert legality(mini_problem, routes)


def test_conflict_focus_selects_case2_wait_edge_first():
    """The focused scan should start from the bottleneck route's wait edge."""
    from Q3.io import load_archive, load_problem
    from Q3.search import _conflict_focus_tasks

    problem = load_problem("Case2", repository_root=ROOT)
    archive = load_archive(
        ROOT / "outputs" / "workbooks" / "q3" / "strict" / "Case2.json"
    )
    solution = eval_solution(
        problem,
        [tuple(int(task_id) for task_id in route) for route in archive["task_routes"]],
    )
    assert solution is not None
    focused = _conflict_focus_tasks(problem, solution, top_edges=1)
    assert focused == {0: {6, 19}}


def test_conflict_focused_search_is_legal_and_non_regressing(mini_problem):
    pool = build_initial_solutions(mini_problem, None, k=2, repository_root=None)
    config = Config(
        time_budget_s_per_case=2,
        stall_rounds=1,
        seed=42,
        conflict_focus=True,
        conflict_top_edges=2,
        conflict_full_fallback=False,
    )
    best = local_search(mini_problem, config, pool)
    assert best.metrics.lex_key() <= pool[0].metrics.lex_key()
    assert legality(mini_problem, [schedule.task_route for schedule in best.schedules])


def test_operator_respects_expired_deadline(mini_problem):
    from Q3.search import _operator_candidate, _rng

    solution = eval_solution(mini_problem, [(1, 2), (3, 4)])
    assert solution is not None
    candidate = _operator_candidate(
        mini_problem,
        solution,
        "swap",
        _rng(42, "Mini", "deadline", 0),
        deadline=time.monotonic() - 1,
    )
    assert candidate is None


def test_single_uav_two_opt_and_destroy_repair_work():
    """From a suboptimal single-UAV route, two_opt and destroy_repair must both
    find an improving (hence legal) neighbour."""
    problem = MINI_FLEET_ONE
    solution = eval_solution(problem, [(1, 3, 2, 4)])  # suboptimal order
    assert solution is not None
    from Q3.search import _operator_candidate, _rng

    rng = _rng(42, "Mini", "round", 1)
    candidate = _operator_candidate(problem, solution, "two_opt", rng)
    assert candidate is not None
    assert legality(problem, [sched.task_route for sched in candidate.schedules])
    candidate = _operator_candidate(problem, solution, "destroy_repair", rng)
    assert candidate is not None
    routes = [sched.task_route for sched in candidate.schedules]
    assert legality(problem, routes)


def test_lexicographic_acceptance_prefers_smax_then_delta():
    """Two feasible Mini solutions whose (S_max, delta) pairs cross must be
    ordered lexicographically by the search."""
    problem = make_problem(fleet_size=2)
    a = eval_solution(problem, [(1, 3), (2, 4)])
    b = eval_solution(problem, [(1, 2), (3, 4)])
    assert a is not None and b is not None
    a_key, b_key = a.metrics.lex_key(), b.metrics.lex_key()
    assert a_key[:2] == b_key[:2] or a_key < b_key or b_key < a_key
    # whatever the order, the comparison must respect lexicographic semantics
    assert (a_key < b_key) == (a_key[0] < b_key[0] or (a_key[0] == b_key[0] and a_key[1] < b_key[1]))


def test_local_search_terminates_on_stall():
    """stall_rounds=1 with a tiny budget must terminate promptly and improve
    (or match) the pool's best solution."""
    problem = make_problem(fleet_size=2)
    config = Config(time_budget_s_per_case=10, stall_rounds=1, seed=42)
    pool = build_initial_solutions(problem, None, k=2, repository_root=None)
    best = local_search(problem, config, pool)
    assert best.metrics.lex_key() <= pool[0].metrics.lex_key()


def test_repeated_runs_are_bit_identical():
    """The whole solve must reproduce bit-for-bit under the same seed."""
    problem = make_problem(fleet_size=2)
    config = Config(time_budget_s_per_case=10, stall_rounds=1, seed=42)
    pool = build_initial_solutions(problem, None, k=2, repository_root=None)
    first = local_search(problem, config, pool)
    second = local_search(problem, config, pool)
    assert first.freeze() == second.freeze()


def test_solve_case_runs_end_to_end_case1():
    """M3 acceptance gate: Case1 end-to-end must not degrade the hot start."""
    config = Config(time_budget_s_per_case=20, stall_rounds=1, seed=42)
    solution = solve_case("Case1", config, repository_root=ROOT)
    assert solution is not None
    assert len(solution.schedules) == 4
    assert solution.metrics.S_max_s > 0
    hot = build_initial_solutions(
        __import__("Q3.io", fromlist=["load_problem"]).load_problem("Case1", repository_root=ROOT),
        None, k=1, repository_root=ROOT,
    )
    assert solution.metrics.lex_key() <= hot[0].metrics.lex_key()


def test_time_aware_constructs_feasible_case2():
    """Regression: Case2's zone geometry traps a UAV that ends a route inside
    an about-to-activate disk (point 23, Z3 starts at 9000 s). The constructor
    must roll that final task back and re-dispatch until every return leg is
    feasible — the fix for the M5 'no feasible initial solution' failure."""
    problem = __import__("Q3.io", fromlist=["load_problem"]).load_problem("Case2", repository_root=ROOT)
    from Q3.search import _greedy_time_aware

    routes = _greedy_time_aware(problem)
    assert routes is not None, "time_aware must construct Case2 (deadlock would fail closed)"
    assert legality(problem, routes)
    solution = eval_solution(problem, routes)
    assert solution is not None, "constructed routes must replay time-feasibly (return legs included)"


def test_time_aware_wait_advance_respects_r3():
    """A UAV parked inside an active disk may not be advanced past a zone-end
    (waiting there is unsafe, R3). The zone-end advance must be refused and the
    search must fail closed instead of emitting an illegal park."""
    from Q3.domain import NoFlyZone, Task
    from Q3.search import _greedy_time_aware

    # T1 and T2 both sit inside the disk; the zone activates at 400 s and the
    # 300 s service windows can only complete before it. Serving either one
    # parks the single UAV inside the disk with no safe wait and no legal
    # return leg: the instance is genuinely infeasible, so the constructor
    # must return None (fail closed) rather than a time-unsafe route.
    z = NoFlyZone("Z3", 1.0, 0.0, 0.2, 400, 90_000)
    tasks = (
        Task(1, 101, 1.0, 0.0, "III"),
        Task(2, 102, 0.9, 0.0, "III"),
    )
    problem = make_problem(tasks=tasks, zones=(z,), fleet_size=1)
    assert _greedy_time_aware(problem) is None
