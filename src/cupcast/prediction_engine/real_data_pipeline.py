from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cupcast.shared.constants import PROJECT_ROOT

from .data_loader import load_dataset
from .data_sources import (
    build_teams_from_matches,
    normalize_international_results,
    validate_dataset_files,
)
from .evaluate import evaluate_model
from .features import build_feature_table
from .model import train_prediction_model
from .rating_sources import LocalEloRatingSource, LocalFifaRankingSource, validate_rating_sources
from .error_analysis import analyze_world_cup_errors
from .feature_importance import analyze_feature_importance
from .ablation import run_ablation_study


REAL_PIPELINE_DEFAULT_MODELS = [
    "majority_baseline",
    "uniform_random_baseline",
    "elo_logistic_regression",
    "recent_form_only",
    "logistic_regression",
    "poisson_goal_model",
    "feature_logistic_regression_calibrated",
    "weighted_probability_ensemble",
]

ERROR_ANALYSIS_MODEL_CANDIDATES = [
    "elo_logistic_regression",
    "logistic_regression",
    "poisson_goal_model",
]


@dataclass(frozen=True)
class RealDataPipelineOutputs:
    validation_report: Path
    rating_validation_report: Path
    completed_matches: Path
    excluded_matches: Path
    teams: Path
    comparison_results: Path
    model_leaderboard: Path
    world_cup_backtests: Path
    world_cup_error_analysis: Path
    feature_importance_json: Path
    feature_importance_md: Path
    ablation_study_json: Path
    ablation_study_md: Path
    markdown_summary: Path


def run_real_data_pipeline(
    matches_path: str | Path,
    shootouts_path: str | Path | None,
    years: list[int],
    models_dir: str | Path = PROJECT_ROOT / "models",
    teams_output_path: str | Path = PROJECT_ROOT / "data" / "processed" / "teams_real.csv",
    model_names: list[str] | None = None,
    recent_window: int = 5,
    elo_k: float = 28.0,
    elo_ratings_path: str | Path | None = None,
    fifa_rankings_path: str | Path | None = None,
) -> dict[str, Any]:
    matches_csv = Path(matches_path)
    if not matches_csv.exists():
        raise FileNotFoundError(_missing_real_data_message())
    if not years:
        raise ValueError("At least one World Cup year is required")

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    teams_path = Path(teams_output_path)
    outputs = RealDataPipelineOutputs(
        validation_report=models_path / "dataset_validation_report.json",
        rating_validation_report=models_path / "rating_validation_report.json",
        completed_matches=teams_path.parent / "real_completed_matches.csv",
        excluded_matches=teams_path.parent / "real_excluded_matches.csv",
        teams=teams_path,
        comparison_results=models_path / "comparison_results.json",
        model_leaderboard=models_path / "model_leaderboard.json",
        world_cup_backtests=models_path / "world_cup_backtest_results.json",
        world_cup_error_analysis=models_path / "world_cup_error_analysis.json",
        feature_importance_json=models_path / "feature_importance.json",
        feature_importance_md=models_path / "feature_importance.md",
        ablation_study_json=models_path / "ablation_study.json",
        ablation_study_md=models_path / "ablation_study.md",
        markdown_summary=models_path / "real_data_results_summary.md",
    )
    outputs.teams.parent.mkdir(parents=True, exist_ok=True)
    outputs.completed_matches.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    resolved_shootouts = _existing_optional_file(shootouts_path)
    if shootouts_path and resolved_shootouts is None:
        warnings.append(f"Shootouts file not found: {shootouts_path}; continuing without shootout metadata")
    elif shootouts_path is None:
        warnings.append("Shootouts file not provided; continuing without shootout metadata")
    resolved_elo_ratings = _existing_optional_file(elo_ratings_path)
    resolved_fifa_rankings = _existing_optional_file(fifa_rankings_path)
    if elo_ratings_path and resolved_elo_ratings is None:
        warnings.append(f"Elo ratings file not found: {elo_ratings_path}; continuing with internal Elo features")
    if fifa_rankings_path and resolved_fifa_rankings is None:
        warnings.append(f"FIFA rankings file not found: {fifa_rankings_path}; continuing without external FIFA features")
    rating_validation = validate_rating_sources(
        elo_path=elo_ratings_path,
        fifa_path=fifa_rankings_path,
        output_path=outputs.rating_validation_report,
    )
    warnings.extend(rating_validation.get("warnings", []))

    raw_matches = pd.read_csv(matches_csv)
    completed_matches, excluded_matches = _split_completed_matches(raw_matches)
    completed_matches.to_csv(outputs.completed_matches, index=False)
    excluded_matches.to_csv(outputs.excluded_matches, index=False)
    if not excluded_matches.empty:
        warnings.append(
            f"Excluded {len(excluded_matches)} incomplete/unscored rows before validation; "
            f"details written to {outputs.excluded_matches}"
        )

    validation = validate_dataset_files(
        matches_path=outputs.completed_matches,
        teams_path=None,
        shootouts_path=shootouts_path,
        goalscorers_path=_default_goalscorers_path(matches_csv),
        source_type="international-results",
    )
    _write_json(outputs.validation_report, validation.to_dict())
    warnings.extend(validation.warnings)
    if not validation.is_valid:
        raise ValueError("Real dataset validation failed:\n" + validation.format_text())

    raw_shootouts = pd.read_csv(resolved_shootouts) if resolved_shootouts else None
    normalized_matches = normalize_international_results(completed_matches, shootouts=raw_shootouts)
    generated_teams = build_teams_from_matches(normalized_matches)
    generated_teams.to_csv(outputs.teams, index=False)

    matches, teams = load_dataset(
        outputs.completed_matches,
        outputs.teams,
        source_type="international-results",
        shootouts_path=resolved_shootouts,
    )
    features = build_feature_table(
        matches,
        teams,
        recent_window=recent_window,
        elo_k=elo_k,
        external_elo_ratings=LocalEloRatingSource(resolved_elo_ratings).load() if resolved_elo_ratings else None,
        external_fifa_rankings=LocalFifaRankingSource(resolved_fifa_rankings).load() if resolved_fifa_rankings else None,
        feature_flags={
            "external_ratings": True,
            "schedule": True,
            "tournament_context": True,
            "elo_history": True,
        },
    )
    analysis_config_path = _write_pipeline_analysis_config(
        outputs=outputs,
        shootouts_path=resolved_shootouts,
        elo_ratings_path=resolved_elo_ratings,
        fifa_rankings_path=resolved_fifa_rankings,
        recent_window=recent_window,
        elo_k=elo_k,
    )

    comparison_year = max(int(year) for year in years)
    selected_models = model_names or REAL_PIPELINE_DEFAULT_MODELS
    comparison = _compare_models_for_world_cup_year(
        features=features,
        year=comparison_year,
        model_names=selected_models,
        recent_window=recent_window,
        elo_k=elo_k,
    )
    _write_json(outputs.comparison_results, comparison["comparison_results"])
    _write_json(outputs.model_leaderboard, {"models": comparison["leaderboard"]})

    backtests = _backtest_world_cup_years(
        features=features,
        years=[int(year) for year in years],
        model_names=selected_models,
        recent_window=recent_window,
        elo_k=elo_k,
    )
    _write_json(outputs.world_cup_backtests, backtests)
    error_analysis = analyze_world_cup_errors(
        features=features,
        years=[int(year) for year in years],
        model_names=_error_analysis_models(selected_models),
        output_path=outputs.world_cup_error_analysis,
    )
    try:
        feature_importance = analyze_feature_importance(
            analysis_config_path,
            output_json=outputs.feature_importance_json,
            output_md=outputs.feature_importance_md,
            max_rows=12000,
        )
    except Exception as exc:
        feature_importance = {"models": [], "error_message": str(exc)}
        _write_json(outputs.feature_importance_json, feature_importance)
        outputs.feature_importance_md.write_text(f"# Feature Importance\n\nUnavailable: {exc}\n", encoding="utf-8")
        warnings.append(f"Feature importance unavailable: {exc}")
    try:
        ablation = run_ablation_study(
            analysis_config_path,
            output_json=outputs.ablation_study_json,
            output_md=outputs.ablation_study_md,
            max_rows=12000,
        )
    except Exception as exc:
        ablation = {"groups": [], "error_message": str(exc)}
        _write_json(outputs.ablation_study_json, ablation)
        outputs.ablation_study_md.write_text(f"# Ablation Study\n\nUnavailable: {exc}\n", encoding="utf-8")
        warnings.append(f"Ablation study unavailable: {exc}")

    summary = _render_markdown_summary(
        validation=validation.to_dict(),
        leaderboard=comparison["leaderboard"],
        backtests=backtests,
        error_analysis=error_analysis,
        feature_importance=feature_importance,
        ablation=ablation,
        warnings=warnings,
    )
    outputs.markdown_summary.write_text(summary, encoding="utf-8")

    return {
        "status": "ok",
        "warnings": warnings,
        "outputs": {
            "validation_report": str(outputs.validation_report),
            "rating_validation_report": str(outputs.rating_validation_report),
            "completed_matches": str(outputs.completed_matches),
            "excluded_matches": str(outputs.excluded_matches),
            "teams": str(outputs.teams),
            "comparison_results": str(outputs.comparison_results),
            "model_leaderboard": str(outputs.model_leaderboard),
            "world_cup_backtests": str(outputs.world_cup_backtests),
            "world_cup_error_analysis": str(outputs.world_cup_error_analysis),
            "feature_importance_json": str(outputs.feature_importance_json),
            "feature_importance_md": str(outputs.feature_importance_md),
            "ablation_study_json": str(outputs.ablation_study_json),
            "ablation_study_md": str(outputs.ablation_study_md),
            "markdown_summary": str(outputs.markdown_summary),
        },
        "dataset": validation.to_dict(),
        "ratings": rating_validation,
        "comparison": comparison["comparison_results"],
        "world_cup_backtests": backtests,
        "world_cup_error_analysis": error_analysis,
        "feature_importance": feature_importance,
        "ablation": ablation,
    }


def _compare_models_for_world_cup_year(
    features: pd.DataFrame,
    year: int,
    model_names: list[str],
    recent_window: int,
    elo_k: float,
) -> dict[str, Any]:
    train_features, test_features, tournament_start, tournament_end = _world_cup_split(features, year)
    rows = _evaluate_models_safely(
        train_features=train_features,
        test_features=test_features,
        model_names=model_names,
        recent_window=recent_window,
        elo_k=elo_k,
        model_version_prefix=f"real-comparison-{year}",
    )
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        raise ValueError(f"All comparison models failed for World Cup {year}")
    successful_rows.sort(key=lambda row: float(row["log_loss"]))
    ordered_rows = successful_rows + [row for row in rows if row["status"] != "ok"]
    majority_log_loss = _metric_for(successful_rows, "majority_baseline", "log_loss")
    elo_log_loss = _metric_for(successful_rows, "elo_logistic_regression", "log_loss")
    leaderboard = [
        _leaderboard_row(
            row=row,
            train_count=len(train_features),
            test_count=len(test_features),
            majority_log_loss=majority_log_loss,
            elo_log_loss=elo_log_loss,
        )
        for row in ordered_rows
    ]
    return {
        "comparison_results": {
            "dataset_source": "international-results",
            "comparison_world_cup_year": int(year),
            "tournament_start": tournament_start.date().isoformat(),
            "tournament_end": tournament_end.date().isoformat(),
            "train_matches": int(len(train_features)),
            "test_matches": int(len(test_features)),
            "models": ordered_rows,
        },
        "leaderboard": leaderboard,
    }


def _split_completed_matches(raw_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = normalize_international_results(raw_matches)
    missing_score_mask = normalized["team_a_score"].isna() | normalized["team_b_score"].isna()
    completed = raw_matches.loc[~missing_score_mask].copy()
    excluded = raw_matches.loc[missing_score_mask].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = "missing_score"
    return completed, excluded


def _backtest_world_cup_years(
    features: pd.DataFrame,
    years: list[int],
    model_names: list[str],
    recent_window: int,
    elo_k: float,
) -> dict[str, Any]:
    results = []
    for year in years:
        try:
            train_features, test_features, tournament_start, tournament_end = _world_cup_split(features, int(year))
            model_rows = _evaluate_models_safely(
                train_features=train_features,
                test_features=test_features,
                model_names=model_names,
                recent_window=recent_window,
                elo_k=elo_k,
                model_version_prefix=f"real-world-cup-{year}",
                subset_features={
                    "group_stage": _subset_features(test_features, stage="group"),
                    "knockout": _subset_features(test_features, stage="knockout"),
                },
            )
            successful = [row for row in model_rows if row["status"] == "ok"]
            if not successful:
                raise ValueError(f"All models failed for World Cup {year}")
            successful.sort(key=lambda row: float(row["log_loss"]))
            best = successful[0]
            majority = _find_model_row(successful, "majority_baseline")
            elo = _find_model_row(successful, "elo_logistic_regression")
            ensemble = _find_model_row(successful, "weighted_probability_ensemble")
            best_group = _best_subset_model(successful, "group_stage_log_loss", lower_is_better=True)
            best_knockout = _best_subset_model(successful, "knockout_log_loss", lower_is_better=True)
            results.append(
                {
                    "year": int(year),
                    "status": "ok",
                    "tournament_start": tournament_start.date().isoformat(),
                    "tournament_end": tournament_end.date().isoformat(),
                    "train_match_count": int(len(train_features)),
                    "test_match_count": int(len(test_features)),
                    "group_stage_match_count": int(len(_subset_features(test_features, stage="group"))),
                    "knockout_match_count": int(len(_subset_features(test_features, stage="knockout"))),
                    "best_overall_model": best["model_name"],
                    "best_group_stage_model": best_group["model_name"] if best_group else None,
                    "best_knockout_model": best_knockout["model_name"] if best_knockout else None,
                    "best_group_stage_log_loss": best_group["group_stage_log_loss"] if best_group else None,
                    "best_knockout_log_loss": best_knockout["knockout_log_loss"] if best_knockout else None,
                    "best_model": best["model_name"],
                    "accuracy": best["accuracy"],
                    "log_loss": best["log_loss"],
                    "brier_score": best["brier_score"],
                    "ece": best["ece"],
                    "rps_if_available": None,
                    "calibration_ece": best["ece"],
                    "baseline_accuracy": majority["accuracy"] if majority else None,
                    "elo_baseline_accuracy": elo["accuracy"] if elo else None,
                    "ensemble_accuracy": ensemble["accuracy"] if ensemble else None,
                    "baseline_comparison": {
                        "majority_baseline_log_loss": majority["log_loss"] if majority else None,
                        "elo_baseline_log_loss": elo["log_loss"] if elo else None,
                        "best_beats_majority_baseline": (
                            majority is not None and float(best["log_loss"]) < float(majority["log_loss"])
                        ),
                        "best_beats_elo_baseline": (
                            elo is not None and float(best["log_loss"]) < float(elo["log_loss"])
                        ),
                    },
                    "models": successful + [row for row in model_rows if row["status"] != "ok"],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "year": int(year),
                    "status": "failed",
                    "error_message": str(exc),
                    "train_match_count": 0,
                    "test_match_count": 0,
                    "group_stage_match_count": 0,
                    "knockout_match_count": 0,
                    "best_overall_model": None,
                    "best_group_stage_model": None,
                    "best_knockout_model": None,
                    "best_group_stage_log_loss": None,
                    "best_knockout_log_loss": None,
                    "best_model": None,
                    "accuracy": None,
                    "log_loss": None,
                    "brier_score": None,
                    "ece": None,
                    "rps_if_available": None,
                    "calibration_ece": None,
                    "baseline_accuracy": None,
                    "elo_baseline_accuracy": None,
                    "ensemble_accuracy": None,
                    "baseline_comparison": {},
                    "models": [],
                }
            )
    if not any(row["status"] == "ok" for row in results):
        raise ValueError("No requested World Cup year could be backtested")
    return {
        "dataset_source": "international-results",
        "years": years,
        "results": results,
    }


def _evaluate_models_safely(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    model_names: list[str],
    recent_window: int,
    elo_k: float,
    model_version_prefix: str,
    subset_features: dict[str, pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        try:
            model = train_prediction_model(
                train_features,
                model_type=model_name,
                recent_window=recent_window,
                elo_k=elo_k,
                model_version=f"{model_version_prefix}-{model_name}",
                model_params=_real_pipeline_model_params(),
            )
            metrics = evaluate_model(model, test_features)
            subset_metrics = _subset_metrics(model, subset_features)
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_source": "international-results",
                    "status": "ok",
                    "accuracy": metrics["accuracy"],
                    "top_class_accuracy": metrics["top_class_accuracy"],
                    "log_loss": metrics["log_loss"],
                    "brier_score": metrics["brier_score"],
                    "ece": metrics["expected_calibration_error"],
                    "expected_calibration_error": metrics["expected_calibration_error"],
                    "number_of_matches": metrics["matches_tested"],
                    "calibration": metrics["calibration"],
                    "model_details": _model_details(model.estimator),
                    **subset_metrics,
                    "error_message": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_source": "international-results",
                    "status": "failed",
                    "accuracy": None,
                    "top_class_accuracy": None,
                    "log_loss": None,
                    "brier_score": None,
                    "ece": None,
                    "expected_calibration_error": None,
                    "number_of_matches": int(len(test_features)),
                    "calibration": [],
                    "model_details": {},
                    **_empty_subset_metrics(subset_features),
                    "error_message": str(exc),
                }
            )
    return rows


def _world_cup_split(features: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    tournament = features["tournament"].fillna("").astype(str).str.lower()
    world_cup_finals = tournament.str.contains("world cup", regex=False) & ~tournament.str.contains("qualif", regex=False)
    test_mask = world_cup_finals & (features["date"].dt.year == int(year))
    test_features = features.loc[test_mask].sort_values("date").reset_index(drop=True)
    if test_features.empty:
        raise ValueError(f"No World Cup matches found for {year}")
    tournament_start = pd.to_datetime(test_features["date"]).min()
    tournament_end = pd.to_datetime(test_features["date"]).max()
    train_features = features.loc[features["date"] < tournament_start].reset_index(drop=True)
    if train_features.empty:
        raise ValueError(f"No training matches found before World Cup {year} start date {tournament_start.date()}")
    return train_features, test_features, tournament_start, tournament_end


def _leaderboard_row(
    row: dict[str, Any],
    train_count: int,
    test_count: int,
    majority_log_loss: float | None,
    elo_log_loss: float | None,
) -> dict[str, Any]:
    if row["status"] != "ok":
        return {
            "model_name": row["model_name"],
            "dataset_source": "international-results",
            "status": "failed",
            "train_match_count": int(train_count),
            "test_match_count": int(test_count),
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "ece": None,
            "beats_majority_baseline": False,
            "beats_elo_baseline": False,
            "notes": f"model failed: {row['error_message']}",
            "model_details": row.get("model_details", {}),
        }
    log_loss = float(row["log_loss"])
    beats_majority = majority_log_loss is not None and log_loss < majority_log_loss
    beats_elo = elo_log_loss is not None and log_loss < elo_log_loss
    notes = []
    if majority_log_loss is not None and not beats_majority:
        notes.append("does not beat majority baseline on log loss")
    if elo_log_loss is not None and row["model_name"] != "elo_logistic_regression" and not beats_elo:
        notes.append("does not beat Elo baseline on log loss")
    return {
        "model_name": row["model_name"],
        "dataset_source": "international-results",
        "status": "ok",
        "train_match_count": int(train_count),
        "test_match_count": int(test_count),
        "accuracy": row["accuracy"],
        "log_loss": row["log_loss"],
        "brier_score": row["brier_score"],
        "ece": row["ece"],
        "beats_majority_baseline": beats_majority,
        "beats_elo_baseline": beats_elo,
        "notes": "; ".join(notes) if notes else "beats selected baselines on log loss for this split",
        "model_details": row.get("model_details", {}),
    }


def _write_pipeline_analysis_config(
    outputs: RealDataPipelineOutputs,
    shootouts_path: Path | None,
    elo_ratings_path: Path | None,
    fifa_rankings_path: Path | None,
    recent_window: int,
    elo_k: float,
) -> Path:
    config_path = outputs.markdown_summary.parent / "real_pipeline_analysis_config.json"
    payload: dict[str, Any] = {
        "data": {
            "mode": "real",
            "matches_path": str(outputs.completed_matches),
            "shootouts_path": str(shootouts_path) if shootouts_path else None,
            "teams_path": str(outputs.teams),
            "elo_ratings_path": str(elo_ratings_path) if elo_ratings_path else None,
            "fifa_rankings_path": str(fifa_rankings_path) if fifa_rankings_path else None,
        },
        "features": {
            "recent_window": int(recent_window),
            "elo_k": float(elo_k),
            "enable_groups": {
                "external_ratings": True,
                "schedule": True,
                "tournament_context": True,
                "elo_history": True,
            },
        },
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def _subset_features(features: pd.DataFrame, stage: str) -> pd.DataFrame:
    stage_values = features["stage"].fillna("").astype(str).str.lower()
    if stage == "group":
        return features.loc[stage_values.eq("group")].reset_index(drop=True)
    return features.loc[~stage_values.isin(["group", "unknown", "friendly", "qualifier"])].reset_index(drop=True)


def _subset_metrics(model: Any, subset_features: dict[str, pd.DataFrame] | None) -> dict[str, float | None]:
    if not subset_features:
        return {}
    metrics: dict[str, float | None] = {}
    for name, subset in subset_features.items():
        if subset.empty:
            metrics[f"{name}_accuracy"] = None
            metrics[f"{name}_log_loss"] = None
            metrics[f"{name}_brier_score"] = None
            metrics[f"{name}_ece"] = None
            continue
        subset_report = evaluate_model(model, subset)
        metrics[f"{name}_accuracy"] = float(subset_report["accuracy"])
        metrics[f"{name}_log_loss"] = float(subset_report["log_loss"])
        metrics[f"{name}_brier_score"] = float(subset_report["brier_score"])
        metrics[f"{name}_ece"] = float(subset_report["expected_calibration_error"])
    return metrics


def _empty_subset_metrics(subset_features: dict[str, pd.DataFrame] | None) -> dict[str, None]:
    if not subset_features:
        return {}
    return {
        metric_name: None
        for name in subset_features
        for metric_name in (
            f"{name}_accuracy",
            f"{name}_log_loss",
            f"{name}_brier_score",
            f"{name}_ece",
        )
    }


def _best_subset_model(
    rows: list[dict[str, Any]],
    metric_name: str,
    lower_is_better: bool = False,
) -> dict[str, Any] | None:
    available = [row for row in rows if row.get(metric_name) is not None]
    if not available:
        return None
    selector = min if lower_is_better else max
    return selector(available, key=lambda row: float(row[metric_name]))


def _model_details(estimator: Any) -> dict[str, Any]:
    details: dict[str, Any] = {}
    weights = getattr(estimator, "member_weights_", None)
    if weights:
        details["ensemble_weights"] = weights
    failures = getattr(estimator, "failures", None)
    if failures:
        details["ensemble_failures"] = failures
    method = getattr(estimator, "method", None)
    if method:
        details["calibration_method"] = method
    return details


def _real_pipeline_model_params() -> dict[str, Any]:
    return {
        "calibration_cv": 2,
        "n_estimators": 40,
        "max_depth": 2,
        "subsample": 0.85,
        "n_jobs": -1,
        "ensemble_candidates": [
            "elo_logistic_regression",
            "logistic_regression",
            "poisson_goal_model",
        ],
    }


def _error_analysis_models(selected_models: list[str]) -> list[str]:
    candidates = [model for model in selected_models if model in ERROR_ANALYSIS_MODEL_CANDIDATES]
    return candidates or ["elo_logistic_regression"]


def _render_markdown_summary(
    validation: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    backtests: dict[str, Any],
    error_analysis: dict[str, Any],
    feature_importance: dict[str, Any],
    ablation: dict[str, Any],
    warnings: list[str],
) -> str:
    lines = [
        "# CupCast AI Real Data Results",
        "",
        "This report was generated locally from user-supplied real historical match data.",
        "",
        "## Dataset Summary",
        "",
        f"- Match count: {validation.get('number_of_matches')}",
        f"- Date range: {validation.get('date_range', {}).get('min')} to {validation.get('date_range', {}).get('max')}",
        f"- Team count: {validation.get('number_of_unique_teams')}",
        f"- Tournament count: {validation.get('number_of_tournaments')}",
        f"- World Cup match count: {validation.get('world_cup_match_count')}",
        f"- Shootout count: {validation.get('shootout_match_count')}",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
        lines.append("")

    lines.extend(
        [
            "## Model Leaderboard",
            "",
            "| Model | Status | Accuracy | Log loss | Brier | ECE | Beats majority | Beats Elo | Notes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in leaderboard:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row.get("status", "ok")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("log_loss")),
                    _fmt(row.get("brier_score")),
                    _fmt(row.get("ece")),
                    _yes_no(row.get("beats_majority_baseline")),
                    _yes_no(row.get("beats_elo_baseline")),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## World Cup Backtests",
            "",
            "| Year | Status | Train matches | Test matches | Best model | Accuracy | Log loss | Brier | ECE |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in backtests.get("results", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["year"]),
                    str(row["status"]),
                    str(row.get("train_match_count", 0)),
                    str(row.get("test_match_count", 0)),
                    str(row.get("best_model") or "n/a"),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("log_loss")),
                    _fmt(row.get("brier_score")),
                    _fmt(row.get("calibration_ece")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "Calibration is evaluated with log loss, Brier score, calibration bins, and expected calibration error. Calibrated variants are reported alongside uncalibrated models; improvement is not assumed.",
            "",
            "## Feature Importance",
            "",
        ]
    )
    for model in feature_importance.get("models", [])[:3]:
        if model.get("status") != "ok":
            lines.append(f"- `{model.get('model_name')}` unavailable: {model.get('error_message')}")
            continue
        top_features = ", ".join(f"`{row['feature']}`" for row in model.get("features", [])[:5])
        lines.append(f"- `{model.get('model_name')}` top features: {top_features or 'n/a'}")
    lines.extend(["", "## Ablation Study", ""])
    for row in ablation.get("groups", []):
        metric = _fmt(row.get("log_loss")) if row.get("status") == "ok" else str(row.get("reason"))
        lines.append(f"- `{row.get('feature_group')}`: {row.get('status')} (log loss: {metric})")
    lines.extend(["", "## Error Analysis", ""])
    misses = [row for row in error_analysis.get("errors", []) if row.get("error_type") == "predicted_top_class_lost"]
    underdogs = [row for row in error_analysis.get("errors", []) if row.get("error_type") == "correct_underdog_call"]
    lines.append(f"- Predicted top-class misses recorded: {len(misses)}")
    lines.append(f"- Correct underdog calls recorded: {len(underdogs)}")
    lines.extend(
        [
            "",
            "## Honest Interpretation",
            "",
            *_interpretation_lines(leaderboard),
            "",
            "Historical match results are useful for learning broad team-strength and form signals, but they are not enough for elite forecasting. The model does not know injuries, player availability, travel, tactical matchups, betting-market information, squad age profiles, or current bookmaker priors. Betting odds and player-level data would improve realism because they capture recent public information and team news that raw historical scores miss.",
            "",
            "## Interview Talking Points",
            "",
            "- The LLM does not directly predict winners; structured models own the probabilities and the mini-LLM explains them.",
            "- Leakage is reduced by building every match row from team state and rating snapshots available strictly before that match date.",
            "- Calibration matters because tournament simulations consume probabilities, not just predicted classes.",
            "- Baselines matter because Elo-only, majority, random, Poisson, and ablations show whether complexity is earning its keep.",
            "- World Cup-only backtests are noisy because each tournament has a small sample of matches.",
            "- The ensemble is used to combine complementary signals while reporting member weights honestly.",
            "- Paid data would improve the model through lineups, injuries, player minutes, market priors, travel, and tactical information.",
            "",
            "These results should be read as an evaluation of the current CupCast feature pipeline on the supplied dataset, not as a claim of production-grade prediction quality.",
            "",
            "## README-Ready Snippet",
            "",
            "### Real Data Results",
            "",
            "To reproduce real-data evaluation:",
            "",
            "1. Download the international results dataset.",
            "2. Place `results.csv` in `data/real/`.",
            "3. Optionally place `shootouts.csv` in `data/real/`.",
            "4. Run `make real-data-pipeline`.",
            "5. Open `models/real_data_results_summary.md`.",
            "",
            "Real-data metrics are generated locally after running the pipeline and are not hardcoded in this repository.",
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation_lines(leaderboard: list[dict[str, Any]]) -> list[str]:
    successful = [row for row in leaderboard if row.get("status") == "ok"]
    if not successful:
        return ["- No model completed successfully, so no performance claim should be made."]
    best = min(successful, key=lambda row: float(row["log_loss"]))
    lines = [
        f"- Best model by log loss in the comparison split: `{best['model_name']}`.",
    ]
    if best.get("beats_majority_baseline"):
        lines.append("- The best model beats the majority baseline on log loss for this split.")
    else:
        lines.append("- The best model does not beat the majority baseline on log loss for this split.")
    if best.get("beats_elo_baseline"):
        lines.append("- The best model beats the Elo-only baseline on log loss for this split.")
    else:
        lines.append("- The best model does not beat the Elo-only baseline on log loss for this split.")
    failed = [row["model_name"] for row in leaderboard if row.get("status") == "failed"]
    if failed:
        lines.append(f"- Some models failed and were recorded rather than hidden: {', '.join(failed)}.")
    return lines


def _metric_for(rows: list[dict[str, Any]], model_name: str, metric_name: str) -> float | None:
    for row in rows:
        if row["model_name"] == model_name and row["status"] == "ok":
            return float(row[metric_name])
    return None


def _find_model_row(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any] | None:
    for row in rows:
        if row["model_name"] == model_name:
            return row
    return None


def _existing_optional_file(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def _default_goalscorers_path(matches_path: Path) -> Path | None:
    candidate = matches_path.parent / "goalscorers.csv"
    return candidate if candidate.exists() else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _yes_no(value: object) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _missing_real_data_message() -> str:
    return (
        "Real dataset not found.\n"
        "Place results.csv in data/real/results.csv.\n"
        "Optional: place shootouts.csv in data/real/shootouts.csv."
    )
