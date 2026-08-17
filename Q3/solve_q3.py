"""M4/M5 CLI: solve, verify and publish the Q3 four-piece deliverable.

Usage (from the repository root):
    python -m Q3.solve_q3 --all            # solve all four cases (30 min each)
    python -m Q3.solve_q3 --case Case1     # solve one case
    python -m Q3.solve_q3 --verify         # independent re-verification of everything
    python -m Q3.solve_q3 --all --verify   # solve, then verify the published pieces

Publication is fail-closed: a case whose own verify_archive reports a violation
is never written to the strict archive. `attachment/result3.xlsx` is never
touched; the real output goes to `outputs/workbooks/result3.xlsx`.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from .domain import Config, DEFAULT_CONFIG
from .io import (
    OUT_RELATIVE,
    Q3_OUT_RELATIVE,
    archive_sha256,
    atomic_write_json,
    load_archive,
    load_problem,
    write_archive_with_hot_start,
    write_result_workbook,
    write_timetable_csv,
)
from .search import _hot_start_routes, solve_case
from .verify import verify_all, verify_archive

ALL_CASES = ("Case1", "Case2", "Case3", "Case4")

MANIFEST_SCHEMA = "q3-manifest-v1"


def _strict_dir(root: Path) -> Path:
    return root / Q3_OUT_RELATIVE / "strict"


def _timetable_dir(root: Path) -> Path:
    return root / Q3_OUT_RELATIVE / "timetables"


def _report_dir(root: Path) -> Path:
    return root / Q3_OUT_RELATIVE / "reports"


def _result_workbook(root: Path) -> Path:
    return root / OUT_RELATIVE / "result3.xlsx"


def _manifest_path(root: Path) -> Path:
    return _report_dir(root) / "run_manifest.json"


def _repository_root(repository_root: str | Path | None) -> Path:
    return Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[1]


def _sha256_of(path: Path) -> str:
    return archive_sha256(path)


def _hot_start_block(problem, repository_root: Path) -> dict:
    """Warm-start provenance + the baseline metrics gating acceptance (PLAN 11.3)."""
    from .safe_path import eval_solution

    hot = _hot_start_routes(problem, repository_root)
    if hot is not None:
        routes, meta = hot
        solution = eval_solution(problem, routes)
        if solution is not None:
            return {**meta, "metrics": _metrics_dict(solution.metrics)}
        # Q2 routes ignore no-fly zones, so a hot start can be infeasible under
        # Q3 rules (e.g. Case2 crossing Z2 during 15:00-17:00); fall through to
        # the Q1 baseline and the greedy fallback instead of crashing.
    baseline = repository_root / "outputs" / "workbooks" / "baseline_20260816" / f"q1_solution_{problem.case}.json"
    if baseline.is_file():
        routes = _q1_routes(baseline)
        if routes is not None:
            solution = eval_solution(problem, routes)
            if solution is not None:
                return {
                    "source": "q1-baseline",
                    "archive_path": str(baseline),
                    "archive_sha256": _sha256_of(baseline),
                    "fallback": "q1-baseline",
                    "metrics": _metrics_dict(solution.metrics),
                }
    return {"source": "none", "archive_path": None, "archive_sha256": None,
            "fallback": "greedy", "metrics": None}


def _q1_routes(baseline: Path) -> list[tuple[int, ...]] | None:
    archive = load_archive(baseline)
    uavs = archive.get("uavs")
    if not isinstance(uavs, list):
        return None
    return [tuple(int(task_id) for task_id in uav.get("task_seq", ())) for uav in uavs]


def _metrics_dict(metrics) -> dict:
    return {
        "S_max_s": metrics.S_max_s,
        "S_min_s": metrics.S_min_s,
        "delta_s": metrics.delta_s,
        "sum_T_s": metrics.sum_T_s,
        "total_wait_s": metrics.total_wait_s,
        "total_distance_km": metrics.total_distance_km,
    }


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": MANIFEST_SCHEMA, "cases": {}, "config": None}
    return load_archive(path)


def _solve_and_publish(case: str, config: Config, repository_root: Path) -> dict:
    """Solve one case, verify it independently, and publish archive + timetable.
    Raises RuntimeError when verification fails (fail closed, nothing published)."""
    solution = solve_case(case, config, repository_root=repository_root)
    violations = verify_archive(solution.problem, solution.freeze())
    if violations:
        raise RuntimeError(f"{case}: independent verification failed:\n" + "\n".join(violations))

    hot_start = _hot_start_block(solution.problem, repository_root)
    config_snapshot = {
        "eta_km": config.eta_km,
        "seed": config.seed,
        "time_budget_s_per_case": config.time_budget_s_per_case,
        "nine_hour_cap_s": config.nine_hour_cap_s,
    }
    write_archive_with_hot_start(
        _strict_dir(repository_root) / f"{case}.json", solution,
        hot_start=hot_start, config=config_snapshot,
    )
    write_timetable_csv(_timetable_dir(repository_root) / f"{case}_timetable.csv", solution)
    return {
        "solution": solution,
        "hot_start": hot_start,
    }


def _publish_workbook(solutions: dict[str, object], repository_root: Path) -> None:
    if solutions:
        write_result_workbook(_result_workbook(repository_root), solutions)


def _update_manifest(case: str, entry: dict, repository_root: Path) -> None:
    manifest = _load_manifest(_manifest_path(repository_root))
    manifest.setdefault("cases", {})[case] = entry
    manifest["config"] = manifest.get("config") or {
        "eta_km": DEFAULT_CONFIG.eta_km,
        "seed": DEFAULT_CONFIG.seed,
        "time_budget_s": DEFAULT_CONFIG.time_budget_s_per_case,
        "nine_hour_cap_s": DEFAULT_CONFIG.nine_hour_cap_s,
    }
    atomic_write_json(_manifest_path(repository_root), manifest)


def _run_verify(config: Config, repository_root: Path) -> int:
    violations = verify_all(ALL_CASES, config, repository_root=repository_root)
    if violations:
        print(f"verify: {len(violations)} violation(s)", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("verify: all checks passed")
    return 0


def _reuse_verified(case: str, manifest: dict, root: Path) -> dict | None:
    """Reuse a previously published, verified archive when --all is rerun:
    skip the solve, re-verify with the current verifier, and rebuild the
    Solution for the workbook. None = must solve (missing/outdated/failed)."""
    from .safe_path import eval_solution

    entry = (manifest.get("cases") or {}).get(case)
    if not entry or not entry.get("verified"):
        return None
    archive_rel = entry.get("archive")
    if not archive_rel:
        return None
    archive_path = root / archive_rel
    if not archive_path.is_file():
        return None
    problem = load_problem(case, repository_root=root)
    archive = load_archive(archive_path)
    if verify_archive(problem, archive):
        return None  # verifier logic moved on; re-solve instead of reusing
    routes = [tuple(int(task_id) for task_id in route) for route in archive.get("task_routes", [])]
    solution = eval_solution(problem, routes)
    if solution is None:
        return None
    return {"solution": solution, "metrics": _metrics_dict(solution.metrics),
            "hot_start": (entry.get("hot_start_metrics") or None),
            "archive": archive_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Q3 solver + publisher")
    parser.add_argument("--case", choices=ALL_CASES)
    parser.add_argument("--all", action="store_true", help="solve all four cases")
    parser.add_argument("--verify", action="store_true", help="verify published outputs")
    parser.add_argument("--skip-solved", action="store_true",
                        help="reuse already-verified archives instead of re-solving")
    parser.add_argument("--time-budget", type=int, default=DEFAULT_CONFIG.time_budget_s_per_case,
                        help="seconds per case")
    parser.add_argument("--repository-root", default=None)
    args = parser.parse_args(argv)
    root = _repository_root(args.repository_root)
    config = Config(time_budget_s_per_case=args.time_budget)

    if args.verify:
        return _run_verify(config, root)

    cases = [args.case] if args.case else list(ALL_CASES)
    manifest = _load_manifest(_manifest_path(root))
    published: dict[str, object] = {}
    for case in cases:
        if args.skip_solved:
            reused = _reuse_verified(case, manifest, root)
            if reused is not None:
                published[case] = reused["solution"]
                metrics = reused["metrics"]
                print(f"{case}: reused verified archive "
                      f"(S_max={metrics['S_max_s']}s delta={metrics['delta_s']}s "
                      f"sum_T={metrics['sum_T_s']}s wait={metrics['total_wait_s']}s "
                      f"dist={metrics['total_distance_km']:.3f}km)")
                continue
        result = _solve_and_publish(case, config, root)
        solution = result["solution"]
        hot_start = result["hot_start"]
        published[case] = solution
        metrics = _metrics_dict(solution.metrics)
        baseline = hot_start.get("metrics") if hot_start.get("metrics") else metrics
        strict_path = _strict_dir(root) / f"{case}.json"
        entry = {
            "archive": str(strict_path.relative_to(root)),
            "archive_sha256": _sha256_of(strict_path),
            "timetable": str((_timetable_dir(root) / f"{case}_timetable.csv").relative_to(root)),
            "hot_start": {
                "source": hot_start.get("source"),
                "archive_path": hot_start.get("archive_path"),
                "archive_sha256": hot_start.get("archive_sha256"),
                "fallback": hot_start.get("fallback"),
            },
            "hot_start_metrics": baseline,
            "metrics": metrics,
            "verified": True,
        }
        _update_manifest(case, entry, root)
        print(f"{case}: S_max={metrics['S_max_s']}s delta={metrics['delta_s']}s "
              f"sum_T={metrics['sum_T_s']}s wait={metrics['total_wait_s']}s "
              f"dist={metrics['total_distance_km']:.3f}km")
        acceptance = tuple(metrics[k] for k in ("S_max_s", "delta_s", "sum_T_s", "total_wait_s"))
        base_tuple = tuple(baseline[k] for k in ("S_max_s", "delta_s", "sum_T_s", "total_wait_s"))
        if baseline and acceptance > base_tuple:
            raise RuntimeError(f"{case}: lexicographic regression vs hot-start baseline")

    if args.all or not args.case:
        _publish_workbook(published, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
