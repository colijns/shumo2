"""Problem 2 workload-balancing domain foundation."""

from .domain import ProblemData, RouteMetrics, SolutionMetrics, Task
from .metrics import balanced_key, canonical_route_signature, epsilon_bound, strict_key
from .q1_adapter import FLEET_SIZE_BY_CASE, ParentArchiveError, load_problem, replay_archive

__all__ = [
    "FLEET_SIZE_BY_CASE",
    "ParentArchiveError",
    "ProblemData",
    "RouteMetrics",
    "SolutionMetrics",
    "Task",
    "balanced_key",
    "canonical_route_signature",
    "epsilon_bound",
    "load_problem",
    "replay_archive",
    "strict_key",
]
