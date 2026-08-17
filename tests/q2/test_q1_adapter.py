import ast
import importlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ATTACHMENT = ROOT.parents[2] / "attachment" / "附件1.xlsx"
WORKTREE_ATTACHMENT = ROOT / "attachment" / "附件1.xlsx"
ARCHIVE_DIR = ROOT / "outputs" / "workbooks" / "baseline_20260816"
EXPECTED_FLEET_SIZE = {"Case1": 4, "Case2": 2, "Case3": 5, "Case4": 4}
EXPECTED_FIRST_LEG_S = {"Case1": 5277, "Case2": 48, "Case3": 5725, "Case4": 2433}
ALLOWED_Q1_HELPERS = {"load_case", "build_matrices", "evaluate"}
FORBIDDEN_Q1_HELPERS = {
    "span_of",
    "better",
    "solve_n",
    "run_promote",
    "check_result1_xlsx",
    "write_result1",
    "write_summary",
    "write_reports",
}


@pytest.fixture(scope="module")
def adapter():
    return importlib.import_module("Q2.q1_adapter")


@contextmanager
def _available_attachment():
    if WORKTREE_ATTACHMENT.exists():
        yield
        return
    if not SOURCE_ATTACHMENT.exists():
        pytest.skip("附件1.xlsx is unavailable in both worktree and source checkout")
    WORKTREE_ATTACHMENT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ATTACHMENT, WORKTREE_ATTACHMENT)
    try:
        yield
    finally:
        WORKTREE_ATTACHMENT.unlink(missing_ok=True)
        try:
            WORKTREE_ATTACHMENT.parent.rmdir()
        except OSError:
            pass


@pytest.mark.parametrize("case_name, expected_n", EXPECTED_FLEET_SIZE.items())
def test_fleet_mapping_is_explicit(case_name, expected_n, adapter):
    """Catches an incorrect Q2 fleet size for any inherited Q1 case."""
    assert adapter.FLEET_SIZE_BY_CASE[case_name] == expected_n


@pytest.mark.parametrize("case_name", EXPECTED_FLEET_SIZE)
def test_real_archive_replays_through_q1_contract(case_name, adapter):
    """Catches task/point/time/stat drift at the Q1-to-Q2 boundary."""
    archive = json.loads(
        (ARCHIVE_DIR / f"q1_solution_{case_name}.json").read_text(encoding="utf-8")
    )
    with _available_attachment():
        replay = adapter.replay_archive(case_name, archive)

    assert replay["fleet_size"] == EXPECTED_FLEET_SIZE[case_name]
    assert replay["routes"] == tuple(tuple(uav["task_seq"]) for uav in archive["uavs"])
    expected_stats = [
        {key: value for key, value in uav.items() if key != "uav_id"}
        for uav in archive["uavs"]
    ]
    assert replay["stats"] == expected_stats
    assert replay["time_s"][0][0] == 0
    assert replay["time_s"][0][1] == EXPECTED_FIRST_LEG_S[case_name]
    assert all(isinstance(value, int) for row in replay["time_s"] for value in row)
    assert len(replay["time_s"]) == len(replay["case"]["tasks"]) + 1


def _q1_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"Q1", "Q1.solve_q1", "Q1.tight_search"}:
            yield node
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"Q1", "Q1.solve_q1", "Q1.tight_search"}:
                    yield node


def _referenced_attributes(path: Path, roots: set[str]):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in roots
    }


def test_q2_has_one_narrow_q1_import_boundary():
    """Catches Q2 bypassing the adapter or widening its Q1 dependency surface."""
    q2_files = sorted((ROOT / "Q2").rglob("*.py"))
    adapter_path = ROOT / "Q2" / "q1_adapter.py"

    for path in q2_files:
        imports = list(_q1_imports(path))
        if path != adapter_path:
            assert imports == [], f"{path.relative_to(ROOT)} bypasses Q2.q1_adapter"
            continue

        imported_names = {
            alias.name
            for node in imports
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert imported_names == ALLOWED_Q1_HELPERS
        assert _referenced_attributes(path, {"solve_q1", "tight_search"}) == set()


def test_q2_never_references_forbidden_q1_helpers():
    """Catches solver, comparison, or Q1 output/report logic leaking into Q2."""
    for path in (ROOT / "Q2").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        referenced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert referenced.isdisjoint(FORBIDDEN_Q1_HELPERS), (
            f"{path.relative_to(ROOT)} references "
            f"{sorted(referenced & FORBIDDEN_Q1_HELPERS)}"
        )
