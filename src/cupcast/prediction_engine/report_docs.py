from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cupcast.shared.constants import PROJECT_ROOT


REQUIRED_REPORTS = {
    "validation": "dataset_validation_report.json",
    "ratings": "rating_validation_report.json",
    "leaderboard": "model_leaderboard.json",
    "comparison": "comparison_results.json",
    "backtests": "world_cup_backtest_results.json",
    "error_analysis": "world_cup_error_analysis.json",
    "feature_importance": "feature_importance.json",
    "feature_importance_md": "feature_importance.md",
    "ablation": "ablation_study.json",
    "ablation_md": "ablation_study.md",
    "summary": "real_data_results_summary.md",
}

RUN_PIPELINE_MESSAGE = (
    "Generated real-data reports are missing or incomplete.\n"
    "Run: python scripts/run_real_data_pipeline.py --matches data/real/results.csv "
    "--shootouts data/real/shootouts.csv --elo data/real/elo_ratings.csv "
    "--fifa data/real/fifa_rankings.csv --years 2014 2018 2022"
)


@dataclass(frozen=True)
class RealReportBundle:
    validation: dict[str, Any]
    ratings: dict[str, Any]
    leaderboard: dict[str, Any]
    comparison: dict[str, Any]
    backtests: dict[str, Any]
    error_analysis: dict[str, Any]
    feature_importance: dict[str, Any]
    ablation: dict[str, Any]
    feature_importance_markdown: str
    ablation_markdown: str
    summary_markdown: str


class RealReportError(RuntimeError):
    pass


def load_real_report_bundle(models_dir: str | Path = PROJECT_ROOT / "models") -> RealReportBundle:
    models_path = Path(models_dir)
    missing = [
        filename
        for filename in REQUIRED_REPORTS.values()
        if not (models_path / filename).exists()
    ]
    if missing:
        raise RealReportError(f"{RUN_PIPELINE_MESSAGE}\nMissing: {', '.join(missing)}")

    validation = _read_json(models_path / REQUIRED_REPORTS["validation"])
    ratings = _read_json(models_path / REQUIRED_REPORTS["ratings"])
    leaderboard = _read_json(models_path / REQUIRED_REPORTS["leaderboard"])
    comparison = _read_json(models_path / REQUIRED_REPORTS["comparison"])
    backtests = _read_json(models_path / REQUIRED_REPORTS["backtests"])
    error_analysis = _read_json(models_path / REQUIRED_REPORTS["error_analysis"])
    feature_importance = _read_json(models_path / REQUIRED_REPORTS["feature_importance"])
    ablation = _read_json(models_path / REQUIRED_REPORTS["ablation"])
    feature_importance_markdown = (models_path / REQUIRED_REPORTS["feature_importance_md"]).read_text(encoding="utf-8")
    ablation_markdown = (models_path / REQUIRED_REPORTS["ablation_md"]).read_text(encoding="utf-8")
    summary_markdown = (models_path / REQUIRED_REPORTS["summary"]).read_text(encoding="utf-8")

    errors = _validate_real_reports(validation, ratings, leaderboard, comparison, backtests, error_analysis, feature_importance, ablation)
    if errors:
        raise RealReportError(f"{RUN_PIPELINE_MESSAGE}\nProblems: {'; '.join(errors)}")

    return RealReportBundle(
        validation=validation,
        ratings=ratings,
        leaderboard=leaderboard,
        comparison=comparison,
        backtests=backtests,
        error_analysis=error_analysis,
        feature_importance=feature_importance,
        ablation=ablation,
        feature_importance_markdown=feature_importance_markdown,
        ablation_markdown=ablation_markdown,
        summary_markdown=summary_markdown,
    )


def write_portfolio_docs(
    models_dir: str | Path = PROJECT_ROOT / "models",
    readme_path: str | Path = PROJECT_ROOT / "README.md",
) -> dict[str, str]:
    bundle = load_real_report_bundle(models_dir)
    models_path = Path(models_dir)
    readme = Path(readme_path)

    model_card = models_path / "MODEL_CARD.md"
    model_card.write_text(render_model_card(bundle), encoding="utf-8")

    models_readme = models_path / "README.md"
    models_readme.write_text(render_models_readme(), encoding="utf-8")

    readme.write_text(
        replace_readme_real_results_section(
            readme.read_text(encoding="utf-8"),
            render_readme_real_results_section(bundle),
        ),
        encoding="utf-8",
    )
    return {
        "readme": str(readme),
        "model_card": str(model_card),
        "models_readme": str(models_readme),
    }


def render_readme_real_results_section(bundle: RealReportBundle) -> str:
    lines = [
        "## Real Data Results",
        "",
        "These metrics are generated from the local real-data reports in `models/`. They are not hardcoded claims.",
        "",
        "This project does not use paid prediction APIs. Forecasts are generated from historical match results, optional local rating CSVs, engineered features, model comparison, calibration, and Monte Carlo simulation.",
        "",
        "### Dataset Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Matches | {_value(bundle.validation.get('number_of_matches'))} |",
        f"| Teams | {_value(bundle.validation.get('number_of_unique_teams'))} |",
        f"| Date range | {_date_range(bundle.validation.get('date_range', {}))} |",
        f"| Tournaments | {_value(bundle.validation.get('number_of_tournaments'))} |",
        f"| World Cup matches | {_value(bundle.validation.get('world_cup_match_count'))} |",
        f"| Shootouts | {_value(bundle.validation.get('shootout_match_count'))} |",
        "",
        "### Model Leaderboard",
        "",
        "| Model | Accuracy | Log loss | Brier score | ECE | Beats majority | Beats Elo |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _ordered_leaderboard_rows(bundle.leaderboard.get("models", [])):
        lines.append(
            "| "
            + " | ".join(
                [
                    _model_label(str(row.get("model_name", ""))),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("log_loss")),
                    _fmt(row.get("brier_score")),
                    _fmt(row.get("ece")),
                    _yes_no(row.get("beats_majority_baseline")),
                    _yes_no(row.get("beats_elo_baseline")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### World Cup Backtests",
            "",
            "| Year | Train | Test | Group | Knockout | Best overall | Best group | Best knockout | Accuracy | Log loss | Brier | ECE |",
            "|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in bundle.backtests.get("results", []):
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('year')} | 0 | 0 | 0 | 0 | not available | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("year")),
                    str(row.get("train_match_count")),
                    str(row.get("test_match_count")),
                    str(row.get("group_stage_match_count", "n/a")),
                    str(row.get("knockout_match_count", "n/a")),
                    _model_label(str(row.get("best_overall_model") or row.get("best_model"))),
                    _model_label(str(row.get("best_group_stage_model") or "n/a")),
                    _model_label(str(row.get("best_knockout_model") or "n/a")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("log_loss")),
                    _fmt(row.get("brier_score")),
                    _fmt(row.get("ece", row.get("calibration_ece"))),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### Calibration",
            "",
            "Calibration is compared with log loss, Brier score, reliability bins, and expected calibration error. Calibrated variants are shown beside uncalibrated models, and the docs only claim improvement when metrics show it.",
            "",
            "### Feature Importance",
            "",
        ]
    )
    importance_lines = _feature_importance_lines(bundle.feature_importance)
    lines.extend(importance_lines or ["- Feature importance was unavailable for this run."])
    lines.extend(["", "### Ablation Study", ""])
    ablation_lines = _ablation_lines(bundle.ablation)
    lines.extend(ablation_lines or ["- Ablation results were unavailable for this run."])
    lines.extend(["", "### Error Analysis", ""])
    lines.extend(_error_analysis_lines(bundle.error_analysis))
    lines.extend(
        [
            "",
            "### Known Weaknesses",
            "",
            "- World Cup-only tests contain few matches, so year-level metrics are noisy.",
            "- Historical scores miss injuries, lineups, travel, tactical changes, weather, and market priors.",
            "- Optional Elo/FIFA snapshots improve current-strength features only when the user supplies valid local CSVs.",
            "- The model is a portfolio forecasting system, not betting-grade infrastructure.",
            "",
            "### Interview Talking Points",
            "",
            "- The LLM does not directly predict winners; structured models produce probabilities and the mini-LLM explains them.",
            "- Leakage is prevented by building each row from pre-match team state and rating snapshots dated strictly before the match.",
            "- Calibration matters because tournament simulation consumes probabilities, not just classes.",
            "- Baselines matter because Elo, majority, Poisson, and ablations show whether complexity is useful.",
            "- World Cup backtests are noisy because each tournament is a small test set.",
            "- The ensemble combines complementary model families while reporting weights and failures.",
            "- With paid data, the largest gains would likely come from lineups, injuries, player minutes, betting-market priors, and travel context.",
            "",
            "### What The Results Mean",
            "",
            *interpretation_lines(bundle),
            "",
        ]
    )
    return "\n".join(lines)


def render_model_card(bundle: RealReportBundle) -> str:
    best = _best_model(bundle)
    return "\n".join(
        [
            "# CupCast AI Model Card",
            "",
            "## Project Name",
            "",
            "CupCast AI",
            "",
            "## Model Purpose",
            "",
            "CupCast AI produces three-way probabilistic forecasts for international football matches: Team A win, draw, and Team B win. The structured model feeds tournament simulation and analyst explanations.",
            "",
            "This model produces probabilistic forecasts, not guarantees. It should not be used for betting or financial decisions.",
            "",
            "## Data Source",
            "",
            "User-supplied historical international football result CSVs placed in `data/real/`, normalized through the `international-results` adapter.",
            "",
            f"- Matches: {_value(bundle.validation.get('number_of_matches'))}",
            f"- Teams: {_value(bundle.validation.get('number_of_unique_teams'))}",
            f"- Date range: {_date_range(bundle.validation.get('date_range', {}))}",
            f"- Tournaments: {_value(bundle.validation.get('number_of_tournaments'))}",
            f"- World Cup matches: {_value(bundle.validation.get('world_cup_match_count'))}",
            "",
            "## Input Features",
            "",
            "- Pre-match Elo-style team ratings.",
            "- Elo difference.",
            "- Optional local Elo/FIFA snapshots looked up strictly before the match date.",
            "- Recent form from prior matches only.",
            "- Goals scored and conceded in the recent window.",
            "- Schedule/rest, tournament context, stage, neutral venue, and host-country features.",
            "- Elo trend, volatility, average, and recent maximum features from pre-match history.",
            "",
            "## Prediction Target",
            "",
            "Three-class match outcome: `A_WIN`, `DRAW`, or `B_WIN`, based on regulation/extra-time score. Penalty shootout winners are tracked separately and are not treated as regulation winners.",
            "",
            "## Training Setup",
            "",
            "The real-data pipeline compares majority baseline, uniform random baseline, Elo-only logistic regression, recent-form-only logistic regression, full feature logistic regression, Poisson goal model, calibrated variants, random forest, gradient boosting, and a probability ensemble. World Cup backtests train only on matches before each tournament start date.",
            "",
            "## Evaluation Metrics",
            "",
            "Accuracy, log loss, multiclass Brier score, expected calibration error, baseline comparisons, and World Cup-specific backtest metrics.",
            "",
            "## Calibration And Ensemble",
            "",
            "Calibrated variants are evaluated beside uncalibrated models. The ensemble reports member weights and component failures rather than hiding them.",
            "",
            "## Real-Data Results Summary",
            "",
            f"Best comparison model by log loss: `{_model_label(str(best.get('model_name', 'n/a')))}`.",
            f"Accuracy: {_fmt(best.get('accuracy'))}; log loss: {_fmt(best.get('log_loss'))}; Brier score: {_fmt(best.get('brier_score'))}; ECE: {_fmt(best.get('ece'))}.",
            "",
            "## Known Limitations",
            "",
            "- Historical results alone do not include injuries, player availability, tactical changes, travel, rest, lineups, or market expectations.",
            "- World Cup backtests are small samples and can be noisy.",
            "- Generated real teams use unknown confederation and blank FIFA rank until richer metadata is supplied.",
            "- The model is not a production-grade sports betting system.",
            "",
            "## Ethical/Product Limitations",
            "",
            "The project is for education and portfolio evaluation. It should not be used to make betting, financial, or high-stakes decisions.",
            "",
            "## Future Improvements",
            "",
            "- Add verified pre-match Elo/ranking snapshots.",
            "- Add squad strength, injuries, player minutes, and player form.",
            "- Add betting-market baseline comparisons.",
            "- Add richer calibration plots and reliability reporting.",
            "- Improve tournament simulation with scoreline and bracket-specific models.",
            "",
        ]
    )


def render_models_readme() -> str:
    rows = [
        ("dataset_validation_report.json", "Validation summary for real or synthetic CSV inputs. Generated by `scripts/validate_dataset.py` and the real-data pipeline."),
        ("rating_validation_report.json", "Validation summary for optional local Elo and FIFA snapshot CSVs."),
        ("model_leaderboard.json", "Compact model comparison table with accuracy, log loss, Brier score, ECE, and baseline flags. Generated by `scripts/compare_models.py` or `scripts/run_real_data_pipeline.py`."),
        ("comparison_results.json", "Detailed comparison split output, including per-model metrics and calibration bins. Generated by model comparison scripts."),
        ("world_cup_backtest_results.json", "Year-by-year World Cup backtests with pre-tournament training splits. Generated by `scripts/backtest_world_cups.py` or the real-data pipeline."),
        ("world_cup_error_analysis.json", "Big misses and underdog-call audit rows for requested World Cup years."),
        ("feature_importance.json", "Feature importance and coefficient magnitude report for supported models."),
        ("feature_importance.md", "Markdown rendering of feature importance results."),
        ("ablation_study.json", "Feature-group ablation study with unavailable groups marked explicitly."),
        ("ablation_study.md", "Markdown rendering of the ablation study."),
        ("real_data_results_summary.md", "Human-readable real-data evaluation summary generated by `scripts/run_real_data_pipeline.py`."),
        ("MODEL_CARD.md", "Portfolio-ready model card generated from the real-data reports by `scripts/generate_portfolio_docs.py`."),
    ]
    lines = [
        "# CupCast AI Generated Reports",
        "",
        "This directory contains local generated artifacts. Real-data metrics depend on the user's local dataset and are not hardcoded.",
        "",
        "| File | Purpose |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {purpose} |" for name, purpose in rows)
    lines.extend(
        [
            "",
            "To regenerate real-data reports:",
            "",
            "```bash",
            "python scripts/run_real_data_pipeline.py --matches data/real/results.csv --shootouts data/real/shootouts.csv --elo data/real/elo_ratings.csv --fifa data/real/fifa_rankings.csv --years 2014 2018 2022",
            "python scripts/generate_portfolio_docs.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def replace_readme_real_results_section(readme_text: str, new_section: str) -> str:
    marker = "## Real Data Results"
    start = readme_text.find(marker)
    if start < 0:
        return readme_text.rstrip() + "\n\n" + new_section.rstrip() + "\n"
    next_start = readme_text.find("\n## ", start + len(marker))
    if next_start < 0:
        return readme_text[:start].rstrip() + "\n\n" + new_section.rstrip() + "\n"
    return readme_text[:start].rstrip() + "\n\n" + new_section.rstrip() + "\n" + readme_text[next_start:]


def interpretation_lines(bundle: RealReportBundle) -> list[str]:
    leaderboard_rows = [row for row in bundle.leaderboard.get("models", []) if row.get("status", "ok") == "ok"]
    if not leaderboard_rows:
        return ["- No model completed successfully, so no performance claim should be made."]
    best = _best_model(bundle)
    full_feature = _find_model(bundle.leaderboard.get("models", []), "logistic_regression")
    elo = _find_model(bundle.leaderboard.get("models", []), "elo_logistic_regression")
    trees = [
        row
        for row in bundle.leaderboard.get("models", [])
        if row.get("model_name") in {"random_forest", "gradient_boosting"} and row.get("log_loss") is not None
    ]
    lines = [
        f"- Best model by comparison log loss: `{_model_label(str(best.get('model_name')))}`.",
    ]
    lines.append(
        "- The best model beats the majority baseline on log loss."
        if best.get("beats_majority_baseline")
        else "- The best model does not beat the majority baseline on log loss."
    )
    lines.append(
        "- The best model beats the Elo-only baseline on log loss."
        if best.get("beats_elo_baseline")
        else "- The best model does not beat the Elo-only baseline on log loss."
    )
    if full_feature and elo and full_feature.get("log_loss") is not None and elo.get("log_loss") is not None:
        if float(full_feature["log_loss"]) < float(elo["log_loss"]):
            lines.append("- The full feature logistic regression improves over Elo-only on the comparison split.")
        else:
            lines.append("- The full feature logistic regression does not improve over Elo-only on the comparison split.")
    if trees and elo and elo.get("log_loss") is not None:
        best_tree = min(trees, key=lambda row: float(row["log_loss"]))
        if float(best_tree["log_loss"]) < float(elo["log_loss"]):
            lines.append("- The best tree model outperforms Elo-only on the comparison split.")
        else:
            lines.append("- The tree models do not outperform Elo-only on the comparison split.")
    ece = best.get("ece")
    if ece is not None:
        lines.append(
            "- Calibration looks reasonable for this split."
            if float(ece) <= 0.08
            else "- Calibration is weak enough that probabilities should be interpreted cautiously."
        )
    successful_backtests = [row for row in bundle.backtests.get("results", []) if row.get("status") == "ok"]
    if successful_backtests:
        lines.append("- World Cup backtests are small samples, so year-to-year results should be treated as noisy.")
    lines.append("- Historical match results alone miss lineups, injuries, player form, tactics, and betting-market information.")
    return lines


def _feature_importance_lines(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for model in payload.get("models", [])[:4]:
        name = _model_label(str(model.get("model_name", "")))
        if model.get("status") != "ok":
            lines.append(f"- `{name}` unavailable: {model.get('error_message') or model.get('status')}")
            continue
        top = ", ".join(f"`{row.get('feature')}`" for row in model.get("features", [])[:5])
        lines.append(f"- `{name}` top features: {top or 'n/a'}.")
    return lines


def _ablation_lines(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("groups", [])
    if not rows:
        return []
    lines = ["| Feature group | Status | Log loss | Accuracy | Reason |", "|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("feature_group")),
                    str(row.get("status")),
                    _fmt(row.get("log_loss")),
                    _fmt(row.get("accuracy")),
                    str(row.get("reason") or ""),
                ]
            )
            + " |"
        )
    return lines


def _error_analysis_lines(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("errors", [])
    if not rows:
        return ["- No error-analysis rows were generated for this run."]
    misses = [row for row in rows if row.get("error_type") == "predicted_top_class_lost"]
    underdogs = [row for row in rows if row.get("error_type") == "correct_underdog_call"]
    biggest = sorted(
        [row for row in rows if row.get("actual_probability") is not None],
        key=lambda row: float(row["actual_probability"]),
    )[:3]
    lines = [
        f"- Predicted top-class misses: {len(misses)}.",
        f"- Correct underdog calls: {len(underdogs)}.",
    ]
    for row in biggest:
        lines.append(
            f"- Low-probability actual result: {row.get('year')} {row.get('match')} "
            f"actual `{row.get('actual_result')}` at {_fmt(row.get('actual_probability'))}."
        )
    return lines


def _validate_real_reports(
    validation: dict[str, Any],
    ratings: dict[str, Any],
    leaderboard: dict[str, Any],
    comparison: dict[str, Any],
    backtests: dict[str, Any],
    error_analysis: dict[str, Any],
    feature_importance: dict[str, Any],
    ablation: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if validation.get("valid") is not True:
        errors.append("dataset_validation_report.json is not valid")
    if validation.get("source_type") != "international-results":
        errors.append("dataset_validation_report.json is not from international-results")
    if not leaderboard.get("models"):
        errors.append("model_leaderboard.json has no models")
    if any(row.get("dataset_source") != "international-results" for row in leaderboard.get("models", [])):
        errors.append("model_leaderboard.json is not from international-results")
    if comparison.get("dataset_source") != "international-results":
        errors.append("comparison_results.json is not from international-results")
    if backtests.get("dataset_source") != "international-results":
        errors.append("world_cup_backtest_results.json is not from international-results")
    if not backtests.get("results"):
        errors.append("world_cup_backtest_results.json has no results")
    if ratings.get("valid") is not True:
        errors.append("rating_validation_report.json is not valid")
    if "errors" not in error_analysis:
        errors.append("world_cup_error_analysis.json missing errors")
    if "models" not in feature_importance:
        errors.append("feature_importance.json missing models")
    if "groups" not in ablation:
        errors.append("ablation_study.json missing groups")
    return errors


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        "majority_baseline": 0,
        "uniform_random_baseline": 1,
        "elo_logistic_regression": 2,
        "recent_form_only": 3,
        "logistic_regression": 4,
        "random_forest": 5,
        "gradient_boosting": 6,
        "poisson_goal_model": 7,
        "feature_logistic_regression_calibrated": 8,
        "random_forest_calibrated": 9,
        "gradient_boosting_calibrated": 10,
        "weighted_probability_ensemble": 11,
    }
    return sorted(rows, key=lambda row: order.get(str(row.get("model_name")), 99))


def _best_model(bundle: RealReportBundle) -> dict[str, Any]:
    rows = [
        row
        for row in bundle.leaderboard.get("models", [])
        if row.get("status", "ok") == "ok" and row.get("log_loss") is not None
    ]
    if not rows:
        return {}
    return min(rows, key=lambda row: float(row["log_loss"]))


def _find_model(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("model_name") == name:
            return row
    return None


def _date_range(value: dict[str, Any]) -> str:
    return f"{value.get('min') or 'n/a'} to {value.get('max') or 'n/a'}"


def _value(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _yes_no(value: object) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _model_label(name: str) -> str:
    return {
        "majority_baseline": "majority baseline",
        "uniform_random_baseline": "uniform random baseline",
        "elo_logistic_regression": "Elo-only logistic regression",
        "recent_form_only": "recent-form-only model",
        "logistic_regression": "full feature logistic regression",
        "random_forest": "random forest",
        "gradient_boosting": "gradient boosting",
        "poisson_goal_model": "Poisson goal model",
        "feature_logistic_regression_calibrated": "calibrated feature logistic regression",
        "random_forest_calibrated": "calibrated random forest",
        "gradient_boosting_calibrated": "calibrated gradient boosting",
        "weighted_probability_ensemble": "weighted probability ensemble",
    }.get(name, name)
