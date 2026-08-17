"""Immutable epsilon-constrained dual-track selection state."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .balance_core import ScoredCandidate
from .metrics import balanced_key, epsilon_bound

EPSILON_VALUES = (Decimal("0"), Decimal("0.005"), Decimal("0.01"), Decimal("0.02"))


@dataclass(frozen=True, slots=True)
class EpsilonIncumbent:
    """Best balance-first candidate satisfying one epsilon bound."""

    epsilon: Decimal
    bound_s: int
    candidate: ScoredCandidate


@dataclass(frozen=True, slots=True)
class DualTrackState:
    """Strict incumbent, epsilon balance tracks, and deterministic frontier."""

    strict_incumbent: ScoredCandidate
    epsilon_incumbents: tuple[EpsilonIncumbent, ...]
    pareto_frontier: tuple[ScoredCandidate, ...]
    strict_epoch: int


def create_dual_track_state(
    initial: ScoredCandidate, *, epsilons: Iterable[object] = EPSILON_VALUES
) -> DualTrackState:
    """Create immutable dual tracks from one fully replayed strict candidate."""
    normalized = _normalize_epsilons(epsilons)
    return _build_state(initial, (initial,), 0, normalized)


def propagate_candidate(state: DualTrackState, candidate: ScoredCandidate) -> DualTrackState:
    """Propagate one fully replayed candidate through strict/epsilon state."""
    frontier = _reduce_frontier((*state.pareto_frontier, candidate))
    strict = min((*frontier, state.strict_incumbent), key=lambda item: item.strict_key)
    epoch = state.strict_epoch + int(strict.strict_key < state.strict_incumbent.strict_key)
    epsilons = tuple(item.epsilon for item in state.epsilon_incumbents)
    return _build_state(strict, frontier, epoch, epsilons)


def pareto_dominates(left: ScoredCandidate, right: ScoredCandidate) -> bool:
    """Compare numeric Q2 objectives without route-signature tie-breaking."""
    left_values = _numeric_objectives(left)
    right_values = _numeric_objectives(right)
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _build_state(
    strict: ScoredCandidate,
    frontier: Iterable[ScoredCandidate],
    epoch: int,
    epsilons: tuple[Decimal, ...],
) -> DualTrackState:
    reduced = _reduce_frontier((*frontier, strict))
    tracks = tuple(_select_epsilon(strict, reduced, epsilon) for epsilon in epsilons)
    return DualTrackState(strict, tracks, reduced, epoch)


def _select_epsilon(
    strict: ScoredCandidate, frontier: tuple[ScoredCandidate, ...], epsilon: Decimal
) -> EpsilonIncumbent:
    bound_s = epsilon_bound(strict.metrics.Tmax_s, epsilon)
    feasible = tuple(item for item in frontier if item.metrics.Tmax_s <= bound_s)
    candidate = min(feasible, key=_balanced_key)
    return EpsilonIncumbent(epsilon, bound_s, candidate)


def _balanced_key(candidate: ScoredCandidate):
    metrics = candidate.metrics
    return balanced_key(metrics.Tmax_s, metrics.delta_s, metrics.sum_T_s, candidate.routes)


def _numeric_objectives(candidate: ScoredCandidate) -> tuple[int, int, int]:
    metrics = candidate.metrics
    return metrics.Tmax_s, metrics.delta_s, metrics.sum_T_s


def _reduce_frontier(candidates: Iterable[ScoredCandidate]) -> tuple[ScoredCandidate, ...]:
    representatives: dict[tuple[int, int, int], ScoredCandidate] = {}
    for candidate in candidates:
        key = _numeric_objectives(candidate)
        current = representatives.get(key)
        if current is None or candidate.strict_key < current.strict_key:
            representatives[key] = candidate
    unique = tuple(representatives.values())
    survivors = tuple(
        candidate for candidate in unique if not any(pareto_dominates(other, candidate) for other in unique if other != candidate)
    )
    return tuple(sorted(survivors, key=lambda item: item.strict_key))


def _normalize_epsilons(epsilons: Iterable[object]) -> tuple[Decimal, ...]:
    values = tuple(_epsilon_value(value) for value in epsilons)
    if not values or len(set(values)) != len(values):
        raise ValueError("epsilons must be nonempty and unique")
    return tuple(sorted(values))


def _epsilon_value(value: object) -> Decimal:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("epsilon must be a finite nonnegative number") from exc
    if not decimal.is_finite() or decimal < 0:
        raise ValueError("epsilon must be a finite nonnegative number")
    return decimal
