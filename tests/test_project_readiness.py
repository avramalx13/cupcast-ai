from __future__ import annotations

from scripts.check_project_readiness import check_project_readiness


def test_project_readiness_checker_passes_current_workspace() -> None:
    result = check_project_readiness()

    assert result["status"] == "PASS"
    assert result["errors"] == []
