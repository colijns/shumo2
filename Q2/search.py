"""Deterministic local search for Q2 routes (strict and epsilon-constrained)."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, permutations
from random import Random

from .balance_core import (
    ScoredCandidate,
    deterministic_two_opt,
    relocate,
    route_work_s,
    score_candidate,
    swap,
)
from .domain import CAP_S, ProblemData
from .metrics import Routes, balanced_key


@dataclass(frozen=True, slots=True)
class StrictSearchResult:
    """Strict search state suitable for later checkpointing and reporting."""

    initial: ScoredCandidate
    incumbent: ScoredCandidate
    evaluations: int
    improvements: int


def derive_seed(base_seed: int, *namespace: object) -> int:
    """Derive platform-stable pseudo-random seed without Python hash()."""
    if type(base_seed) is not int:
        raise ValueError("base_seed must be an exact integer")
    components = (base_seed,) + namespace
    if any(type(part) not in {int, str} for part in components):
        raise ValueError("seed namespace components must be exact integers or strings")
    payload = b"q2-seed-v1\x00" + b"".join(_seed_component(part) for part in components)
    return int.from_bytes(sha256(payload).digest(), "big")


def _seed_component(value: int | str) -> bytes:
    kind, encoded = (b"i", str(value).encode("ascii")) if type(value) is int else (b"s", value.encode("utf-8"))
    return kind + len(encoded).to_bytes(4, "big") + encoded


def run_strict_search(problem: ProblemData, *, evaluation_limit: int, seed: int) -> StrictSearchResult:
    """Run deterministic first-improvement VND under the strict lexicographic key."""
    return _run_search(
        problem,
        key_of=lambda candidate: candidate.strict_key,
        bound_s=CAP_S,
        namespace="strict",
        evaluation_limit=evaluation_limit,
        seed=seed,
    )


def run_epsilon_search(
    problem: ProblemData,
    *,
    bound_s: int,
    evaluation_limit: int,
    seed: int,
    initial_routes: Routes | None = None,
) -> StrictSearchResult:
    """Run deterministic balance-first search under a fixed Tmax ceiling.

    ``initial_routes`` lets the epsilon stage inherit the verified strict-stage
    incumbent instead of restarting from the older Q1 parent solution.
    """
    if type(bound_s) is not int or not 1 <= bound_s <= CAP_S:
        raise ValueError("bound_s must be an exact integer in 1..CAP_S")
    return _run_search(
        problem,
        key_of=lambda candidate: balanced_key(
            candidate.metrics.Tmax_s, candidate.metrics.delta_s, candidate.metrics.sum_T_s, candidate.routes
        ),
        bound_s=bound_s,
        namespace="epsilon",
        evaluation_limit=evaluation_limit,
        seed=seed,
        initial_routes=initial_routes,
    )


def _run_search(
    problem: ProblemData,
    *,
    key_of,
    bound_s: int,
    namespace: str,
    evaluation_limit: int,
    seed: int,
    initial_routes: Routes | None = None,
) -> StrictSearchResult:
    """Shared first-improvement VND: accept only feasible bound-respecting key improvements."""
    if type(evaluation_limit) is not int or evaluation_limit < 0:
        raise ValueError("evaluation_limit must be a nonnegative exact integer")
    if type(seed) is not int:
        raise ValueError("seed must be an exact integer")
    if type(namespace) is not str:
        raise ValueError("namespace must be a string")
    randomizer = Random(derive_seed(seed, problem.case_name, namespace))
    initial = score_candidate(problem, problem.routes if initial_routes is None else initial_routes)
    if initial is None:
        raise ValueError("initial routes must be a feasible complete solution")
    if initial.metrics.Tmax_s > bound_s:
        raise ValueError("initial routes violate the Tmax bound")
    initial = score_candidate(problem, _normalize_all_routes(problem, initial.routes)) or initial
    incumbent = working = initial
    evaluations, improvements = 0, 0
    while evaluations < evaluation_limit:
        proposal, spent = _first_improvement(problem, working, key_of, bound_s, evaluation_limit - evaluations)
        evaluations += spent
        if proposal is not None:
            working = proposal
            if key_of(working) < key_of(incumbent):
                incumbent, improvements = working, improvements + 1
            continue
        perturbed, spent = _perturb(problem, working.routes, randomizer, bound_s, evaluation_limit - evaluations)
        evaluations += spent
        if perturbed is None:
            break
        working = perturbed
    return StrictSearchResult(initial, incumbent, evaluations, improvements)


def _normalize_all_routes(problem: ProblemData, routes: Routes) -> Routes:
    return tuple(deterministic_two_opt(route, problem.tasks, problem.time_s) for route in routes)


def _normalize_changed_routes(problem: ProblemData, before: Routes, after: Routes) -> Routes:
    """Run 2-opt only on routes touched by an inter-route move.

    Every search state is normalized before it becomes ``working``.  Therefore
    unchanged routes can be reused exactly, avoiding a full-fleet 2-opt replay
    for every relocate/swap candidate.
    """
    return tuple(
        route
        if route == before[index]
        else deterministic_two_opt(route, problem.tasks, problem.time_s)
        for index, route in enumerate(after)
    )


def _first_improvement(
    problem: ProblemData,
    incumbent: ScoredCandidate,
    key_of,
    bound_s: int,
    remaining: int,
) -> tuple[ScoredCandidate | None, int]:
    evaluations = 0
    for proposal in _ordered_move_candidates(problem, incumbent.routes):
        if evaluations >= remaining:
            break
        normalized = _normalize_changed_routes(problem, incumbent.routes, proposal)
        evaluations += 1
        improved = _accept_candidate(incumbent, problem, key_of, bound_s, normalized)
        if improved is not None:
            return improved, evaluations
    return None, evaluations


def _accept_candidate(
    incumbent: ScoredCandidate,
    problem: ProblemData,
    key_of,
    bound_s: int,
    routes: Routes,
) -> ScoredCandidate | None:
    """Return only a fully replayed, bound-respecting key improvement over incumbent."""
    candidate = score_candidate(problem, routes)
    if candidate is None or candidate.metrics.Tmax_s > bound_s:
        return None
    if key_of(candidate) >= key_of(incumbent):
        return None
    return candidate


def _perturb(
    problem: ProblemData, routes: Routes, randomizer: Random, bound_s: int, remaining: int
) -> tuple[ScoredCandidate | None, int]:
    if remaining == 0:
        return None, 0
    proposal = _reservoir_sample(_ordered_move_candidates(problem, routes), randomizer)
    if proposal is None:
        return None, 0
    normalized = _normalize_changed_routes(problem, routes, proposal)
    candidate = score_candidate(problem, normalized)
    if candidate is None or candidate.metrics.Tmax_s > bound_s:
        return None, 1
    return candidate, 1


def _reservoir_sample(proposals, randomizer: Random) -> Routes | None:
    """Select one generated move uniformly without materializing all routes."""
    selected = None
    for count, proposal in enumerate(proposals, 1):
        if randomizer.randrange(count) == 0:
            selected = proposal
    return selected


def _ordered_move_candidates(problem: ProblemData, routes: Routes):
    relocate_pairs, swap_pairs = _prioritized_route_pairs(problem, routes)
    for source_route, target_route in relocate_pairs:
        for source_index in range(len(routes[source_route])):
            for target_index in range(len(routes[target_route]) + 1):
                candidate = relocate(routes, source_route, source_index, target_route, target_index, problem.tasks)
                if candidate is not None:
                    yield candidate
    for left_route, right_route in swap_pairs:
        for left_index in range(len(routes[left_route])):
            for right_index in range(len(routes[right_route])):
                candidate = swap(routes, left_route, left_index, right_route, right_index, problem.tasks)
                if candidate is not None:
                    yield candidate


def _prioritized_route_pairs(problem: ProblemData, routes: Routes):
    """Order the full neighborhood by current workload imbalance."""
    workloads = tuple(route_work_s(route, problem.time_s) for route in routes)
    relocate_pairs = tuple(
        sorted(
            permutations(range(len(routes)), 2),
            key=lambda pair: (-workloads[pair[0]], workloads[pair[1]], pair),
        )
    )
    swap_pairs = tuple(
        sorted(
            combinations(range(len(routes)), 2),
            key=lambda pair: (-abs(workloads[pair[0]] - workloads[pair[1]]), pair),
        )
    )
    return relocate_pairs, swap_pairs
