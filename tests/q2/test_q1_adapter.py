import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ATTACHMENT = ROOT / "attachment" / "附件1.xlsx"
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


@pytest.fixture(scope="module")
def real_attachment_path():
    """Use the explicit parent checkout source without mutating any worktree."""
    if not SOURCE_ATTACHMENT.is_file():
        pytest.fail(f"required real attachment is unavailable: {SOURCE_ATTACHMENT}")
    return SOURCE_ATTACHMENT


@pytest.mark.parametrize("case_name, expected_n", EXPECTED_FLEET_SIZE.items())
def test_fleet_mapping_is_explicit(case_name, expected_n, adapter):
    """Catches an incorrect Q2 fleet size for any inherited Q1 case."""
    assert adapter.FLEET_SIZE_BY_CASE[case_name] == expected_n


@pytest.mark.parametrize("case_name", EXPECTED_FLEET_SIZE)
def test_real_archive_replays_through_q1_contract(case_name, adapter, real_attachment_path):
    """Catches task/point/time/stat drift at the Q1-to-Q2 boundary."""
    archive = json.loads(
        (ARCHIVE_DIR / f"q1_solution_{case_name}.json").read_text(encoding="utf-8")
    )
    replay = adapter.replay_archive(
        case_name,
        archive,
        attachment_path=real_attachment_path,
    )

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


def _is_q1_module(name: str | None) -> bool:
    return bool(name) and (name == "Q1" or name.startswith("Q1."))


def _q1_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _is_q1_module(node.module):
            yield node
        if isinstance(node, ast.Import) and any(_is_q1_module(alias.name) for alias in node.names):
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
        assert all(isinstance(node, ast.ImportFrom) for node in imports)
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
