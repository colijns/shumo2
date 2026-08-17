from pathlib import Path

import pytest

from Q2.solve_q2 import main


def test_smoke_runs_real_parent_cases_without_formal_output(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = root.parents[2]
    output = tmp_path / "workbooks"

    status = main([
        "--smoke", "--repository-root", str(source), "--output-root", str(output), "--evaluation-limit", "5",
    ])

    assert status == 0
    assert not (output / "result2.xlsx").exists()
    assert not (output / "q2" / "strict").exists()


def test_partial_formal_case_is_rejected():
    with pytest.raises(SystemExit, match="partial formal"):
        main(["--case", "Case1"])


def test_verify_rejects_incomplete_manifest(tmp_path):
    output = tmp_path / "workbooks"
    manifest = output / "q2" / "reports" / "run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"schema_version":"q2-manifest-v1","cases":{}}', encoding="utf-8")

    assert main(["--verify", "--output-root", str(output)]) == 1
