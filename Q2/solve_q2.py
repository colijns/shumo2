"""Q2 strict search command-line entry point."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .epsilon_state import EPSILON_VALUES, create_dual_track_state
from .io import (
    atomic_write_json,
    build_solution_archive,
    validate_result_workbook,
    validate_solution_archive,
    write_result_workbook,
)
from .metrics import epsilon_bound
from .q1_adapter import FLEET_SIZE_BY_CASE, load_problem
from .search import run_strict_search

DEFAULT_SEED = 42
DEFAULT_EVALUATION_LIMIT = 2_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify Q2 strict routing results")
    parser.add_argument("--case", choices=tuple(FLEET_SIZE_BY_CASE))
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--attachment", type=Path)
    parser.add_argument("--parent-archive-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--evaluation-limit", type=int, default=DEFAULT_EVALUATION_LIMIT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    if options.case:
        raise SystemExit("partial formal publication is unsupported; run all four cases")
    if options.smoke and options.verify:
        raise SystemExit("--smoke and --verify are mutually exclusive")
    if options.verify:
        return _verify(options)
    return _run(options)


def _run(options: argparse.Namespace) -> int:
    root = _repository_root(options.repository_root)
    output_root = options.output_root or root / "outputs" / "workbooks"
    archive_dir = options.parent_archive_dir or root / "outputs" / "workbooks" / "baseline_20260816"
    attachment = options.attachment or root / "attachment" / "附件1.xlsx"
    solutions = {}
    archives = {}
    for case_name in FLEET_SIZE_BY_CASE:
        problem = load_problem(case_name, attachment_path=attachment, archive_path=archive_dir / f"q1_solution_{case_name}.json")
        result = run_strict_search(problem, evaluation_limit=options.evaluation_limit, seed=options.seed)
        archive = build_solution_archive(problem, result.incumbent)
        validate_solution_archive(problem, archive)
        solutions[case_name] = (problem, result.incumbent)
        archives[case_name] = archive
    if options.smoke:
        return 0
    strict_dir = output_root / "q2" / "strict"
    epsilon_archives = {epsilon: {} for epsilon in EPSILON_VALUES}
    pareto_archives = {}
    for case_name, archive in archives.items():
        problem, candidate = solutions[case_name]
        state = create_dual_track_state(candidate)
        pareto_archives[case_name] = [
            build_solution_archive(
                problem,
                observed,
                track="pareto_balanced",
                epsilon="observed-frontier",
            )
            for observed in state.pareto_frontier
        ]
        for incumbent in state.epsilon_incumbents:
            epsilon_archives[incumbent.epsilon][case_name] = build_solution_archive(
                problem,
                incumbent.candidate,
                track="epsilon_formal",
                epsilon=str(incumbent.epsilon),
            )
    for case_name, archive in archives.items():
        atomic_write_json(strict_dir / f"{case_name}.json", archive)
    for epsilon, case_archives in epsilon_archives.items():
        epsilon_dir = output_root / "q2" / "epsilon" / _epsilon_dir(epsilon)
        for case_name, archive in case_archives.items():
            atomic_write_json(epsilon_dir / f"{case_name}.json", archive)
    for case_name, frontier in pareto_archives.items():
        atomic_write_json(output_root / "q2" / "pareto" / f"{case_name}.json", {"frontier": frontier})
    write_result_workbook(output_root / "result2.xlsx", solutions)
    atomic_write_json(output_root / "q2" / "reports" / "run_manifest.json", _manifest(archives, epsilon_archives, pareto_archives))
    return 0


def _verify(options: argparse.Namespace) -> int:
    root = _repository_root(options.repository_root)
    output_root = options.output_root or root / "outputs" / "workbooks"
    manifest_path = output_root / "q2" / "reports" / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "q2-manifest-v1"
        or not isinstance(cases, dict)
        or set(cases) != set(FLEET_SIZE_BY_CASE)
    ):
        return 1
    archive_dir = options.parent_archive_dir or root / "outputs" / "workbooks" / "baseline_20260816"
    attachment = options.attachment or root / "attachment" / "附件1.xlsx"
    strict_dir = output_root / "q2" / "strict"
    if {path.stem for path in strict_dir.glob("*.json")} != set(FLEET_SIZE_BY_CASE):
        return 1
    try:
        solutions = {}
        for case_name in FLEET_SIZE_BY_CASE:
            problem = load_problem(case_name, attachment_path=attachment, archive_path=archive_dir / f"q1_solution_{case_name}.json")
            archive = json.loads((strict_dir / f"{case_name}.json").read_text(encoding="utf-8"))
            if archive != cases[case_name]:
                return 1
            solutions[case_name] = (problem, validate_solution_archive(problem, archive))
        validate_result_workbook(output_root / "result2.xlsx", solutions)
        if not _verify_epsilon_outputs(manifest, solutions, output_root / "q2" / "epsilon"):
            return 1
        if not _verify_pareto_outputs(manifest, solutions, output_root / "q2" / "pareto"):
            return 1
    except (OSError, ValueError, json.JSONDecodeError):
        return 1
    return 0


def _verify_epsilon_outputs(manifest: dict, solutions: dict, epsilon_root: Path) -> bool:
    expected_epsilons = {str(value) for value in EPSILON_VALUES}
    manifest_epsilons = manifest.get("epsilon")
    if not isinstance(manifest_epsilons, dict) or set(manifest_epsilons) != expected_epsilons:
        return False
    for epsilon in EPSILON_VALUES:
        case_archives = manifest_epsilons[str(epsilon)]
        if not isinstance(case_archives, dict) or set(case_archives) != set(FLEET_SIZE_BY_CASE):
            return False
        epsilon_dir = epsilon_root / _epsilon_dir(epsilon)
        if {path.stem for path in epsilon_dir.glob("*.json")} != set(FLEET_SIZE_BY_CASE):
            return False
        for case_name, (problem, _) in solutions.items():
            archive = json.loads((epsilon_dir / f"{case_name}.json").read_text(encoding="utf-8"))
            if archive != case_archives[case_name]:
                return False
            candidate = validate_solution_archive(problem, archive)
            if archive.get("epsilon") != str(epsilon) or archive.get("track") != "epsilon_formal":
                return False
            if candidate.metrics.Tmax_s > epsilon_bound(candidate.metrics.Tmax_s, epsilon):
                return False
    return True


def _verify_pareto_outputs(manifest: dict, solutions: dict, pareto_root: Path) -> bool:
    observed = manifest.get("pareto_observed")
    if not isinstance(observed, dict) or set(observed) != set(FLEET_SIZE_BY_CASE):
        return False
    if {path.stem for path in pareto_root.glob("*.json")} != set(FLEET_SIZE_BY_CASE):
        return False
    for case_name, (problem, _) in solutions.items():
        envelope = json.loads((pareto_root / f"{case_name}.json").read_text(encoding="utf-8"))
        if not isinstance(envelope, dict) or not isinstance(envelope.get("frontier"), list):
            return False
        archives = [validate_solution_archive(problem, item) for item in envelope["frontier"]]
        if envelope["frontier"] != observed[case_name]:
            return False
        if any(item.get("epsilon") != "observed-frontier" or item.get("track") != "pareto_balanced" for item in envelope["frontier"]):
            return False
        if any(archives[index].metrics.Tmax_s > 32400 for index in range(len(archives))):
            return False
    return True


def _manifest(
    archives: dict[str, dict], epsilon_archives: dict, pareto_archives: dict
) -> dict:
    return {
        "schema_version": "q2-manifest-v1",
        "publication_scope": "strict-plus-observed-epsilon-projection",
        "search_mode": "strict-local-search",
        "epsilon_selection": "strict-incumbent-projection",
        "cases": archives,
        "epsilon": {str(epsilon): case_archives for epsilon, case_archives in epsilon_archives.items()},
        "pareto_observed": pareto_archives,
    }


def _epsilon_dir(epsilon) -> str:
    return str(epsilon).replace(".", "_")


def _repository_root(explicit: Path | None) -> Path:
    return explicit.resolve() if explicit else Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main())
