from __future__ import annotations

import json
import subprocess
import sys

from cupcast.prediction_engine.report_docs import (
    RealReportError,
    load_real_report_bundle,
    render_model_card,
    render_readme_real_results_section,
    write_portfolio_docs,
)


def test_readme_real_results_section_uses_report_metrics_not_placeholder(tmp_path) -> None:
    models_dir = _write_report_bundle(tmp_path / "models")
    bundle = load_real_report_bundle(models_dir)

    section = render_readme_real_results_section(bundle)

    assert "| Matches | 49300 |" in section
    assert "| Teams | 326 |" in section
    assert "Real-data metrics are generated locally after running the pipeline and are not hardcoded" not in section
    assert "majority baseline" in section
    assert "Elo-only logistic regression" in section
    assert "### Feature Importance" in section
    assert "### Ablation Study" in section
    assert "### Error Analysis" in section
    assert "This project does not use paid prediction APIs" in section


def test_model_card_can_be_generated_from_reports(tmp_path) -> None:
    models_dir = _write_report_bundle(tmp_path / "models")
    bundle = load_real_report_bundle(models_dir)

    card = render_model_card(bundle)

    assert "# CupCast AI Model Card" in card
    assert "This model produces probabilistic forecasts, not guarantees." in card
    assert "Best comparison model by log loss" in card


def test_missing_reports_fail_gracefully(tmp_path) -> None:
    try:
        load_real_report_bundle(tmp_path / "models")
    except RealReportError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected RealReportError")

    assert "Generated real-data reports are missing or incomplete" in message
    assert "run_real_data_pipeline.py" in message


def test_models_readme_lists_all_report_artifacts(tmp_path) -> None:
    models_dir = _write_report_bundle(tmp_path / "models")
    readme = tmp_path / "README.md"
    readme.write_text("# Test\n\n## Real Data Results\n\nplaceholder\n", encoding="utf-8")

    write_portfolio_docs(models_dir=models_dir, readme_path=readme)
    models_readme = (models_dir / "README.md").read_text(encoding="utf-8")

    for artifact in [
        "dataset_validation_report.json",
        "rating_validation_report.json",
        "model_leaderboard.json",
        "comparison_results.json",
        "world_cup_backtest_results.json",
        "world_cup_error_analysis.json",
        "feature_importance.json",
        "feature_importance.md",
        "ablation_study.json",
        "ablation_study.md",
        "real_data_results_summary.md",
        "MODEL_CARD.md",
    ]:
        assert artifact in models_readme


def test_plot_generation_handles_missing_optional_dependency(monkeypatch, tmp_path) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from scripts import generate_report_plots

    exit_code = generate_report_plots.main([])

    assert exit_code == 0


def test_generate_portfolio_docs_script_fails_gracefully_for_missing_reports(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_portfolio_docs.py",
            "--models-dir",
            str(tmp_path / "missing"),
            "--readme",
            str(tmp_path / "README.md"),
        ],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Generated real-data reports are missing or incomplete" in completed.stderr
    assert "Traceback" not in completed.stderr


def _write_report_bundle(models_dir):
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "dataset_validation_report.json").write_text(
        json.dumps(
            {
                "valid": True,
                "source_type": "international-results",
                "number_of_matches": 49300,
                "number_of_unique_teams": 326,
                "number_of_tournaments": 180,
                "world_cup_match_count": 964,
                "shootout_match_count": 52,
                "date_range": {"min": "1872-11-30", "max": "2026-06-30"},
            }
        ),
        encoding="utf-8",
    )
    (models_dir / "rating_validation_report.json").write_text(
        json.dumps(
            {
                "valid": True,
                "reports": [
                    {
                        "source": "elo",
                        "path": "data/real/elo_ratings.csv",
                        "exists": False,
                        "valid": True,
                        "warnings": ["elo ratings file missing; continuing without external elo features"],
                    },
                    {
                        "source": "fifa",
                        "path": "data/real/fifa_rankings.csv",
                        "exists": False,
                        "valid": True,
                        "warnings": ["fifa ratings file missing; continuing without external fifa features"],
                    },
                ],
                "warnings": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    models = [
        _model("majority_baseline", 0.44, 1.05, 0.63, 0.04, False, False),
        _model("uniform_random_baseline", 0.33, 1.10, 0.67, 0.10, False, False),
        _model("elo_logistic_regression", 0.51, 0.99, 0.60, 0.07, True, False),
        _model("recent_form_only", 0.47, 1.03, 0.62, 0.08, True, False),
        _model("logistic_regression", 0.52, 0.97, 0.59, 0.06, True, True),
        _model("random_forest", 0.50, 1.01, 0.61, 0.09, True, False),
        _model("gradient_boosting", 0.49, 1.02, 0.61, 0.10, True, False),
    ]
    (models_dir / "model_leaderboard.json").write_text(json.dumps({"models": models}), encoding="utf-8")
    (models_dir / "comparison_results.json").write_text(
        json.dumps({"dataset_source": "international-results", "models": models}),
        encoding="utf-8",
    )
    (models_dir / "world_cup_backtest_results.json").write_text(
        json.dumps(
            {
                "dataset_source": "international-results",
                "results": [
                    _backtest(2014, 32000, 64, "logistic_regression", 0.50, 1.02, 0.61, 0.08),
                    _backtest(2018, 38000, 64, "elo_logistic_regression", 0.48, 1.05, 0.63, 0.09),
                    _backtest(2022, 45000, 64, "logistic_regression", 0.53, 0.98, 0.59, 0.07),
                ],
            }
        ),
        encoding="utf-8",
    )
    (models_dir / "world_cup_error_analysis.json").write_text(
        json.dumps(
            {
                "errors": [
                    {
                        "year": 2022,
                        "match": "Alpha vs Beta",
                        "predicted_probabilities": {"A_WIN": 0.3, "DRAW": 0.2, "B_WIN": 0.5},
                        "actual_result": "A_WIN",
                        "predicted_result": "B_WIN",
                        "actual_probability": 0.3,
                        "error_type": "predicted_top_class_lost",
                        "notes": "example",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (models_dir / "feature_importance.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_name": "random_forest",
                        "status": "ok",
                        "features": [
                            {"feature": "elo_diff", "importance": 0.42},
                            {"feature": "recent_form_diff", "importance": 0.31},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (models_dir / "feature_importance.md").write_text("# Feature Importance\n", encoding="utf-8")
    (models_dir / "ablation_study.json").write_text(
        json.dumps(
            {
                "train_matches": 300,
                "test_matches": 100,
                "groups": [
                    {
                        "feature_group": "elo_only",
                        "model_name": "elo_logistic_regression",
                        "status": "ok",
                        "reason": None,
                        "accuracy": 0.51,
                        "log_loss": 1.01,
                        "brier_score": 0.62,
                        "ece": 0.08,
                    },
                    {
                        "feature_group": "all_features_plus_external_ratings",
                        "model_name": "logistic_regression",
                        "status": "unavailable",
                        "reason": "external rating CSVs not found",
                        "accuracy": None,
                        "log_loss": None,
                        "brier_score": None,
                        "ece": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (models_dir / "ablation_study.md").write_text("# Ablation Study\n", encoding="utf-8")
    (models_dir / "real_data_results_summary.md").write_text("# Summary\n", encoding="utf-8")
    return models_dir


def _model(name, accuracy, log_loss, brier, ece, beats_majority, beats_elo):
    return {
        "model_name": name,
        "dataset_source": "international-results",
        "status": "ok",
        "train_match_count": 45000,
        "test_match_count": 64,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier_score": brier,
        "ece": ece,
        "beats_majority_baseline": beats_majority,
        "beats_elo_baseline": beats_elo,
        "notes": "",
    }


def _backtest(year, train, test, best, accuracy, log_loss, brier, ece):
    return {
        "year": year,
        "status": "ok",
        "train_match_count": train,
        "test_match_count": test,
        "group_stage_match_count": 2,
        "knockout_match_count": max(0, test - 2),
        "best_overall_model": best,
        "best_group_stage_model": best,
        "best_knockout_model": best,
        "best_model": best,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier_score": brier,
        "ece": ece,
        "rps_if_available": None,
        "baseline_accuracy": 0.44,
        "elo_baseline_accuracy": 0.48,
        "ensemble_accuracy": 0.50,
        "calibration_ece": ece,
    }
