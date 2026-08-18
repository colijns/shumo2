

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import q3_conflict_focus_smoke as support

from Q3.io import (
    archive_sha256,
    atomic_write_json,
    load_archive,
    load_problem,
    write_result_workbook,
    write_timetable_csv,
)
from Q3.safe_path import eval_solution
from Q3.verify import verify_archive


ROOT = Path(__file__).resolve().parent
CASES = ("Case1", "Case2", "Case3", "Case4")


def _candidate_map(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"candidate must be CaseX=path: {item}")
        case, raw_path = item.split("=", 1)
        if case not in CASES:
            raise ValueError(f"unknown case: {case}")
        path = Path(raw_path)
        result[case] = path if path.is_absolute() else ROOT / path
    missing = set(CASES) - set(result)
    if missing:
        raise ValueError(f"missing candidates: {sorted(missing)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    candidates = _candidate_map(args.candidate)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    solutions = {}
    results = []
    for case in CASES:
        source_path = candidates[case]
        archive = load_archive(source_path)
        problem = load_problem(case, repository_root=ROOT)
        violations = verify_archive(problem, archive)
        if violations:
            raise RuntimeError(f"{case}: verification failed: {violations}")
        routes = [tuple(map(int, route)) for route in archive["task_routes"]]
        solution = eval_solution(problem, routes)
        if solution is None:
            raise RuntimeError(f"{case}: verified archive failed replay")
        solutions[case] = solution

        strict_path = output_dir / "strict" / f"{case}.json"
        atomic_write_json(strict_path, solution.freeze())
        timetable_path = output_dir / "timetables" / f"{case}_timetable.csv"
        write_timetable_csv(timetable_path, solution)
        results.append(
            {
                "case": case,
                "source_candidate": str(source_path.relative_to(ROOT)),
                "source_sha256": archive_sha256(source_path),
                "deliverable_archive": str(strict_path.relative_to(ROOT)),
                "timetable": str(timetable_path.relative_to(ROOT)),
                "metrics": support.metrics(solution),
                "verification_violations": violations,
            }
        )
        print(f"{case}: independently verified", flush=True)

    workbook_path = output_dir / "result3_conflict_focus.xlsx"
    write_result_workbook(workbook_path, solutions)
    summary = {
        "schema_version": "q3-conflict-focus-deliverable-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "branch": "q3-conflict-focus-run",
        "formal_archives_replaced": False,
        "workbook": str(workbook_path.relative_to(ROOT)),
        "results": results,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(f"deliverable={output_dir}")


if __name__ == "__main__":
    main()
