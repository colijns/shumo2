"""Q2 strict search command-line entry point."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .io import (
    atomic_write_json,
    build_solution_archive,
    validate_result_workbook,
    validate_solution_archive,
    write_result_workbook,
)
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
    for case_name, archive in archives.items():
        atomic_write_json(strict_dir / f"{case_name}.json", archive)
    write_result_workbook(output_root / "result2.xlsx", solutions)
    atomic_write_json(output_root / "q2" / "reports" / "run_manifest.json", _manifest(archives))
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
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "q2-manifest-v1" or not isinstance(cases, dict) or set(cases) != set(FLEET_SIZE_BY_CASE):
        return 1
    archive_dir = options.parent_archive_dir or root / "outputs" / "workbooks" / "baseline_20260816"
    attachment = options.attachment or root / "attachment" / "附件1.xlsx"
    try:
        solutions = {}
        for case_name in FLEET_SIZE_BY_CASE:
            problem = load_problem(case_name, attachment_path=attachment, archive_path=archive_dir / f"q1_solution_{case_name}.json")
            archive_path = output_root / "q2" / "strict" / f"{case_name}.json"
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
            if archive != cases[case_name]:
                return 1
            candidate = validate_solution_archive(problem, archive)
            solutions[case_name] = (problem, candidate)
        validate_result_workbook(output_root / "result2.xlsx", solutions)
    except (OSError, ValueError, json.JSONDecodeError):
        return 1
    return 0


def _manifest(archives: dict[str, dict]) -> dict:
    return {"schema_version": "q2-manifest-v1", "publication_scope": "strict-only", "cases": archives}


def _repository_root(explicit: Path | None) -> Path:
    return explicit.resolve() if explicit else Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main())
