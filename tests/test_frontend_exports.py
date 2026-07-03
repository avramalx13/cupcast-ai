from __future__ import annotations

import json

from scripts import export_frontend_reports as exporter
from scripts.export_frontend_reports import export_frontend_reports


def test_export_frontend_reports_writes_dataset_driven_team_file(tmp_path) -> None:
    teams_path = tmp_path / "teams_real.csv"
    teams_path.write_text(
        "team,confederation,initial_elo,fifa_rank,match_count,first_match_date,last_match_date\n"
        "Senegal,UNKNOWN,1500,,240,1961-12-31,2026-06-20\n"
        "Morocco,UNKNOWN,1500,,300,1957-10-19,2026-06-20\n"
        "Ghana,UNKNOWN,1500,,260,1950-05-28,2026-06-20\n"
        "Senegal,UNKNOWN,1500,,20,1961-12-31,1970-01-01\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    result = export_frontend_reports(
        teams_path=teams_path,
        output_dir=output_dir,
        matches_path=tmp_path / "missing_matches.csv",
        model_path=tmp_path / "missing_model.joblib",
    )

    payload = json.loads((output_dir / "teams_real.json").read_text(encoding="utf-8"))
    names = [team["name"] for team in payload["teams"]]
    assert payload["source"] == "real"
    assert names == sorted(set(names))
    assert "Senegal" in names
    assert "Ghana" in names
    assert "Morocco" in names
    assert next(team for team in payload["teams"] if team["name"] == "Senegal")["match_count"] == 240
    demo = json.loads((output_dir / "demo_matchups.json").read_text(encoding="utf-8"))
    assert demo["matchups"] == []
    assert any("Real prediction model not found" in warning for warning in result["warnings"])


def test_export_frontend_reports_empty_teams_file_fails_gracefully(tmp_path) -> None:
    teams_path = tmp_path / "teams_real.csv"
    teams_path.write_text("team,confederation\n", encoding="utf-8")

    result = export_frontend_reports(
        teams_path=teams_path,
        output_dir=tmp_path / "reports",
        matches_path=tmp_path / "missing_matches.csv",
        model_path=tmp_path / "missing_model.joblib",
    )

    payload = json.loads((tmp_path / "reports" / "teams_real.json").read_text(encoding="utf-8"))
    assert payload["teams"] == []
    assert any("empty" in warning.lower() for warning in result["warnings"])


def test_export_frontend_reports_copies_full_tournament_reports(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "full_tournament_simulation.json").write_text('{"available": true}', encoding="utf-8")
    (models_dir / "full_tournament_simulation.md").write_text("# Full Tournament\n", encoding="utf-8")
    teams_path = tmp_path / "teams_real.csv"
    teams_path.write_text("team,confederation,match_count\nFrance,UEFA,900\nBrazil,CONMEBOL,1000\n", encoding="utf-8")
    monkeypatch.setattr(exporter, "PROJECT_ROOT", project_root)

    result = exporter.export_frontend_reports(
        teams_path=teams_path,
        output_dir=tmp_path / "reports",
        matches_path=tmp_path / "missing_matches.csv",
        model_path=tmp_path / "missing_model.joblib",
    )

    copied = {path.split("\\")[-1].split("/")[-1] for path in result["reports"]}
    assert "full_tournament_simulation.json" in copied
    assert "full_tournament_simulation.md" in copied
