from __future__ import annotations
import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
import q3_conflict_focus_smoke as support
from Q3.domain import Config
from Q3.io import archive_sha256, atomic_write_json, load_archive, load_problem
from Q3.safe_path import eval_solution
from Q3.search import legality, local_search
from Q3.verify import verify_archive
ROOT = Path(__file__).resolve().parent
STRICT_DIR = ROOT / "outputs" / "workbooks" / "q3" / "strict"
RUNS_DIR = ROOT / "outputs" / "workbooks" / "q3" / "conflict_focus_runs"
def run_case(case: str, budget_s: int, top_edges: int, run_dir: Path) -> dict:
    problem = load_problem(case, repository_root=ROOT)
    strict_path = STRICT_DIR / f"{case}.json"
    archive = load_archive(strict_path)
    routes = [tuple(map(int, route)) for route in archive["task_routes"]]
    if not legality(problem, routes):
        raise RuntimeError(f"{case}: current strict archive is illegal")
    incumbent = eval_solution(problem, routes)
    if incumbent is None:
        raise RuntimeError(f"{case}: current strict archive cannot be replayed")
    config = Config(
        time_budget_s_per_case=budget_s,
        stall_rounds=3,
        seed=42,
        conflict_focus=True,
        conflict_top_edges=top_edges,
        conflict_full_fallback=True,
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"q3-{case}-focus-") as temp_dir:
        previous_cwd = Path.cwd()
        os.chdir(temp_dir)
        try:
            candidate = local_search(problem, config, [incumbent])
        finally:
            os.chdir(previous_cwd)
    elapsed_s = round(time.monotonic() - started, 3)
    frozen = candidate.freeze()
    violations = verify_archive(problem, frozen)
    if violations:
        raise RuntimeError(f"{case}: candidate verification failed: {violations}")
    if candidate.metrics.lex_key() > incumbent.metrics.lex_key():
        raise RuntimeError(f"{case}: continuation search regressed")
    candidate_path = run_dir / f"{case}_candidate.json"
    atomic_write_json(candidate_path, frozen)
    return {
        "case": case,
        "strict_archive": str(strict_path.relative_to(ROOT)),
        "strict_archive_sha256": archive_sha256(strict_path),
        "candidate_archive": str(candidate_path.relative_to(ROOT)),
        "budget_s": budget_s,
        "elapsed_s": elapsed_s,
        "seed": config.seed,
        "conflict_top_edges": top_edges,
        "conflict_full_fallback": True,
        "incumbent": support.metrics(incumbent),
        "candidate": support.metrics(candidate),
        "improved": candidate.metrics.lex_key() < incumbent.metrics.lex_key(),
        "verification_violations": violations,
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["Case2", "Case3"])
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--top-edges", type=int, default=12)
    args = parser.parse_args()
    case_tag = "-".join(args.cases)
    stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{case_tag}"
    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.json"
    progress = {
        "schema_version": "q3-conflict-focus-progress-v1",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "branch": "q3-conflict-focus-run",
        "status": "running",
        "planned_cases": args.cases,
        "budget_s_per_case": max(args.budget, 0),
        "conflict_top_edges": max(args.top_edges, 1),
        "results": [],
    }
    atomic_write_json(progress_path, progress)
    results = []
    for index, case in enumerate(args.cases, start=1):
        print(
            f"[{index}/{len(args.cases)}] {case}: start "
            f"(budget={max(args.budget, 0)}s)",
            flush=True,
        )
        try:
            result = run_case(
                case,
                max(args.budget, 0),
                max(args.top_edges, 1),
                run_dir,
            )
        except Exception as exc:
            progress["status"] = "failed"
            progress["failed_case"] = case
            progress["error"] = str(exc)
            progress["updated_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_write_json(progress_path, progress)
            raise
        results.append(result)
        progress["results"] = results
        progress["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(progress_path, progress)
        candidate = result["candidate"]
        print(
            f"[{index}/{len(args.cases)}] {case}: verified "
            f"S_max={candidate['S_max_s']}s delta={candidate['delta_s']}s "
            f"improved={result['improved']}",
            flush=True,
        )
    summary = {
        "schema_version": "q3-conflict-focus-run-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "branch": "q3-conflict-focus-run",
        "results": results,
    }
    summary_path = run_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    progress["status"] = "completed"
    progress["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(progress_path, progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
if __name__ == "__main__":
    main()
