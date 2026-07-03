from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from cupcast.prediction_engine import real_data_pipeline


def test_real_data_pipeline_missing_file_fails_gracefully(tmp_path) -> None:
    missing = tmp_path / "results.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_data_pipeline.py",
            "--matches",
            str(missing),
            "--shootouts",
            str(tmp_path / "shootouts.csv"),
            "--years",
            "2014",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Real dataset not found." in completed.stderr
    assert "Traceback" not in completed.stderr


def test_real_data_pipeline_runs_on_tiny_real_format_dataset(tmp_path) -> None:
    matches = _write_tiny_real_results(tmp_path / "results.csv")
    shootouts = _write_tiny_shootouts(tmp_path / "shootouts.csv")
    models_dir = tmp_path / "models"

    result = real_data_pipeline.run_real_data_pipeline(
        matches_path=matches,
        shootouts_path=shootouts,
        years=[2014, 2018, 2022],
        models_dir=models_dir,
        teams_output_path=tmp_path / "teams_real.csv",
    )

    outputs = result["outputs"]
    assert (models_dir / "dataset_validation_report.json").exists()
    assert (models_dir / "rating_validation_report.json").exists()
    assert (models_dir / "model_leaderboard.json").exists()
    assert (models_dir / "comparison_results.json").exists()
    assert (models_dir / "world_cup_backtest_results.json").exists()
    assert (models_dir / "world_cup_error_analysis.json").exists()
    assert (models_dir / "feature_importance.json").exists()
    assert (models_dir / "feature_importance.md").exists()
    assert (models_dir / "ablation_study.json").exists()
    assert (models_dir / "ablation_study.md").exists()
    assert (models_dir / "real_data_results_summary.md").exists()
    assert json.loads((models_dir / "model_leaderboard.json").read_text(encoding="utf-8"))["models"]
    assert outputs["markdown_summary"].endswith("real_data_results_summary.md")
    assert result["ratings"]["valid"] is True


def test_real_data_pipeline_markdown_summary_is_created(tmp_path) -> None:
    matches = _write_tiny_real_results(tmp_path / "results.csv")
    models_dir = tmp_path / "models"

    real_data_pipeline.run_real_data_pipeline(
        matches_path=matches,
        shootouts_path=None,
        years=[2014, 2018, 2022],
        models_dir=models_dir,
        teams_output_path=tmp_path / "teams_real.csv",
    )

    summary = (models_dir / "real_data_results_summary.md").read_text(encoding="utf-8")
    assert "# CupCast AI Real Data Results" in summary
    assert "## Model Leaderboard" in summary
    assert "## World Cup Backtests" in summary
    assert "## Feature Importance" in summary
    assert "## Ablation Study" in summary
    assert "## Error Analysis" in summary
    assert "Real-data metrics are generated locally" in summary


def test_missing_optional_shootouts_only_warns(tmp_path) -> None:
    matches = _write_tiny_real_results(tmp_path / "results.csv")

    result = real_data_pipeline.run_real_data_pipeline(
        matches_path=matches,
        shootouts_path=tmp_path / "missing_shootouts.csv",
        years=[2014, 2018, 2022],
        models_dir=tmp_path / "models",
        teams_output_path=tmp_path / "teams_real.csv",
    )

    assert result["status"] == "ok"
    assert any("Shootouts file not found" in warning for warning in result["warnings"])
    assert any("ratings file missing" in warning for warning in result["warnings"])


def test_real_data_pipeline_excludes_unplayed_rows_before_validation(tmp_path) -> None:
    matches_path = tmp_path / "results.csv"
    rows = pd.read_csv(_write_tiny_real_results(matches_path))
    rows.loc[len(rows)] = [
        "2026-07-05",
        "Alpha",
        "Beta",
        None,
        None,
        "Friendly",
        "Test City",
        "Test Country",
        True,
    ]
    rows.to_csv(matches_path, index=False)

    result = real_data_pipeline.run_real_data_pipeline(
        matches_path=matches_path,
        shootouts_path=None,
        years=[2014, 2018, 2022],
        models_dir=tmp_path / "models",
        teams_output_path=tmp_path / "teams_real.csv",
    )

    excluded = pd.read_csv(tmp_path / "real_excluded_matches.csv")
    validation = json.loads((tmp_path / "models" / "dataset_validation_report.json").read_text(encoding="utf-8"))
    assert len(excluded) == 1
    assert excluded.loc[0, "exclusion_reason"] == "missing_score"
    assert validation["valid"] is True
    assert validation["invalid_score_count"] == 0
    assert any("Excluded 1 incomplete/unscored rows" in warning for warning in result["warnings"])


def test_model_failures_are_recorded_without_crashing_pipeline(tmp_path, monkeypatch) -> None:
    matches = _write_tiny_real_results(tmp_path / "results.csv")
    original_train = real_data_pipeline.train_prediction_model

    def flaky_train(*args, **kwargs):
        if kwargs.get("model_type") == "random_forest":
            raise RuntimeError("intentional random forest failure")
        return original_train(*args, **kwargs)

    monkeypatch.setattr(real_data_pipeline, "train_prediction_model", flaky_train)

    real_data_pipeline.run_real_data_pipeline(
        matches_path=matches,
        shootouts_path=None,
        years=[2014, 2018, 2022],
        models_dir=tmp_path / "models",
        teams_output_path=tmp_path / "teams_real.csv",
        model_names=["majority_baseline", "uniform_random_baseline", "random_forest"],
    )

    leaderboard = json.loads((tmp_path / "models" / "model_leaderboard.json").read_text(encoding="utf-8"))
    failed = [row for row in leaderboard["models"] if row["model_name"] == "random_forest"]
    assert failed
    assert failed[0]["status"] == "failed"
    assert "intentional random forest failure" in failed[0]["notes"]


def test_readme_does_not_hardcode_fake_real_data_metrics() -> None:
    readme = open("README.md", encoding="utf-8").read()

    assert "## Real Data Results" in readme
    assert "### Dataset Summary" in readme
    assert "### Model Leaderboard" in readme
    assert "### World Cup Backtests" in readme
    assert "This workspace does not currently contain a complete valid real-data report bundle" not in readme


def _write_tiny_real_results(path) -> str:
    rows = [
        ["2011-01-01", "Alpha", "Beta", 2, 0, "Friendly", "Paris", "France", False],
        ["2011-02-01", "Gamma", "Delta", 1, 1, "Friendly", "Berlin", "Germany", False],
        ["2011-03-01", "Beta", "Gamma", 0, 2, "Friendly", "Madrid", "Spain", False],
        ["2012-01-01", "Alpha", "Gamma", 3, 1, "Friendly", "Rome", "Italy", False],
        ["2012-02-01", "Beta", "Delta", 2, 2, "Friendly", "Lisbon", "Portugal", False],
        ["2012-03-01", "Delta", "Alpha", 1, 0, "Friendly", "London", "England", False],
        ["2013-01-01", "Alpha", "Beta", 1, 1, "Friendly", "Paris", "France", False],
        ["2013-02-01", "Gamma", "Delta", 2, 0, "Friendly", "Berlin", "Germany", False],
        ["2013-03-01", "Beta", "Delta", 0, 1, "Friendly", "Madrid", "Spain", False],
        ["2014-06-12", "Alpha", "Beta", 1, 0, "FIFA World Cup", "Sao Paulo", "Brazil", True],
        ["2014-06-13", "Gamma", "Delta", 1, 1, "FIFA World Cup", "Natal", "Brazil", True],
        ["2014-07-01", "Alpha", "Gamma", 0, 2, "FIFA World Cup", "Rio", "Brazil", True],
        ["2018-06-14", "Alpha", "Delta", 2, 0, "FIFA World Cup", "Moscow", "Russia", True],
        ["2018-06-15", "Beta", "Gamma", 1, 2, "FIFA World Cup", "Sochi", "Russia", True],
        ["2018-07-01", "Delta", "Gamma", 1, 1, "FIFA World Cup", "Kazan", "Russia", True],
        ["2022-11-20", "Alpha", "Gamma", 1, 1, "FIFA World Cup", "Al Khor", "Qatar", True],
        ["2022-11-21", "Beta", "Delta", 0, 1, "FIFA World Cup", "Doha", "Qatar", True],
        ["2022-12-09", "Gamma", "Delta", 2, 2, "FIFA World Cup", "Lusail", "Qatar", True],
    ]
    pd.DataFrame(
        rows,
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ],
    ).to_csv(path, index=False)
    return str(path)


def _write_tiny_shootouts(path) -> str:
    pd.DataFrame(
        [
            {
                "date": "2022-12-09",
                "home_team": "Gamma",
                "away_team": "Delta",
                "winner": "Delta",
                "first_shooter": "Gamma",
            }
        ]
    ).to_csv(path, index=False)
    return str(path)
