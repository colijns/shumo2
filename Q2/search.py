"""Deterministic strict local search for Q2 routes."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, permutations
from random import Random

from .balance_core import (
    ScoredCandidate,
    accept_strict_improvement,
    deterministic_two_opt,
    relocate,
    score_candidate,
    swap,
)
from .domain import ProblemData
from .metrics import Routes


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
    """Run deterministic first-improvement VND under fixed evaluation budget."""
    if type(evaluation_limit) is not int or evaluation_limit < 0:
        raise ValueError("evaluation_limit must be a nonnegative exact integer")
    if type(seed) is not int:
        raise ValueError("seed must be an exact integer")
    randomizer = Random(derive_seed(seed, problem.case_name, "strict"))
    initial = score_candidate(problem, problem.routes)
    if initial is None:
        raise ValueError("problem initial routes must be feasible")
    initial = score_candidate(problem, _normalize_all_routes(problem, initial.routes)) or initial
    incumbent = working = initial
    evaluations, improvements = 0, 0
    while evaluations < evaluation_limit:
        proposal, spent = _first_improvement(problem, working, evaluation_limit - evaluations)
        evaluations += spent
        if proposal is not None:
            working = proposal
            if working.strict_key < incumbent.strict_key:
                incumbent, improvements = working, improvements + 1
            continue
        perturbed, spent = _perturb(problem, working.routes, randomizer, evaluation_limit - evaluations)
        evaluations += spent
        if perturbed is None:
            break
        working = perturbed
    return StrictSearchResult(initial, incumbent, evaluations, improvements)


def _normalize_all_routes(problem: ProblemData, routes: Routes) -> Routes:
    return tuple(deterministic_two_opt(route, problem.tasks, problem.time_s) for route in routes)


def _first_improvement(
    problem: ProblemData, incumbent: ScoredCandidate, remaining: int
) -> tuple[ScoredCandidate | None, int]:
    evaluations = 0
    for proposal in _ordered_move_candidates(problem, incumbent.routes):
        if evaluations >= remaining:
            break
        normalized = _normalize_all_routes(problem, proposal)
        evaluations += 1
        improved = accept_strict_improvement(incumbent, problem, normalized)
        if improved is not None:
            return improved, evaluations
    return None, evaluations



def _perturb(
    problem: ProblemData, routes: Routes, randomizer: Random, remaining: int
) -> tuple[ScoredCandidate | None, int]:
    if remaining == 0:
        return None, 0
    proposals = list(_ordered_move_candidates(problem, routes))
    if not proposals:
        return None, 0
    proposal = proposals[randomizer.randrange(len(proposals))]
    normalized = _normalize_all_routes(problem, proposal)
    candidate = score_candidate(problem, normalized)
    return candidate, 1


def _ordered_move_candidates(problem: ProblemData, routes: Routes):
    for source_route, target_route in permutations(range(len(routes)), 2):
        for source_index in range(len(routes[source_route])):
            for target_index in range(len(routes[target_route]) + 1):
                candidate = relocate(routes, source_route, source_index, target_route, target_index, problem.tasks)
                if candidate is not None:
                    yield candidate
    for left_route, right_route in combinations(range(len(routes)), 2):
        for left_index in range(len(routes[left_route])):
            for right_index in range(len(routes[right_route])):
                candidate = swap(routes, left_route, left_index, right_route, right_index, problem.tasks)
                if candidate is not None:
                    yield candidate
