from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cupcast.shared.config import load_yaml, resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT

from .data_loader import load_dataset
from .evaluate import evaluate_model
from .features import build_feature_table
from .model import train_prediction_model
from .rating_sources import LocalEloRatingSource, LocalFifaRankingSource


DEFAULT_COMPARISON_MODELS = [
    "majority_baseline",
    "uniform_random_baseline",
    "elo_logistic_regression",
    "recent_form_only",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "poisson_goal_model",
    "feature_logistic_regression_calibrated",
    "random_forest_calibrated",
    "gradient_boosting_calibrated",
    "hist_gradient_boosting_calibrated",
    "weighted_probability_ensemble",
]


def compare_models(
    config_path: str | Path,
    train_before: int,
    test_tournament: str,
    output_path: str | Path = PROJECT_ROOT / "models" / "comparison_results.json",
    leaderboard_path: str | Path = PROJECT_ROOT / "models" / "model_leaderboard.json",
    model_names: list[str] | None = None,
    matches_path: str | Path | None = None,
    teams_path: str | Path | None = None,
    source_type: str | None = None,
    shootouts_path: str | Path | None = None,
    elo_ratings_path: str | Path | None = None,
    fifa_rankings_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_cfg = config.get("data", {})
    feature_cfg = config.get("features", {})
    model_cfg = config.get("model", {})
    calibration_cfg = config.get("calibration", {})
    ensemble_cfg = config.get("ensemble", {})

    resolved_matches_path = resolve_project_path(
        matches_path or data_cfg.get("matches_path", "data/raw/historical_matches.csv"),
        PROJECT_ROOT,
    )
    dataset_source = _infer_dataset_source(
        explicit_source=source_type,
        config_source=data_cfg.get("mode") or data_cfg.get("source"),
        matches_path=resolved_matches_path,
    )
    teams_value = teams_path if teams_path is not None else data_cfg.get("teams_path", "data/raw/teams.csv")
    resolved_teams_path = resolve_project_path(teams_value, PROJECT_ROOT) if teams_value else None
    resolved_shootouts_path = None
    shootouts_value = shootouts_path if shootouts_path is not None else data_cfg.get("shootouts_path")
    if shootouts_value:
        resolved_shootouts_path = resolve_project_path(shootouts_value, PROJECT_ROOT)
    resolved_elo_ratings_path = _optional_path(
        elo_ratings_path or data_cfg.get("elo_ratings_path") or data_cfg.get("elo_path")
    )
    resolved_fifa_rankings_path = _optional_path(
        fifa_rankings_path or data_cfg.get("fifa_rankings_path") or data_cfg.get("fifa_path")
    )
    recent_window = int(feature_cfg.get("recent_window", 5))
    elo_k = float(feature_cfg.get("elo_k", 28.0))

    matches, teams = load_dataset(
        resolved_matches_path,
        resolved_teams_path,
        source_type=dataset_source,
        shootouts_path=resolved_shootouts_path,
    )
    features = build_feature_table(
        matches,
        teams,
        recent_window=recent_window,
        elo_k=elo_k,
        external_elo_ratings=_load_elo_ratings(resolved_elo_ratings_path),
        external_fifa_rankings=_load_fifa_rankings(resolved_fifa_rankings_path),
        feature_flags=_feature_flags(feature_cfg),
    )
    test_tournament_mask = _test_tournament_mask(features, test_tournament)
    train_features = features.loc[
        (features["date"].dt.year < int(train_before)) & ~test_tournament_mask
    ].reset_index(drop=True)
    test_features = features.loc[test_tournament_mask].reset_index(drop=True)
    if train_features.empty:
        raise ValueError(f"No training matches found before {train_before}")
    if test_features.empty:
        raise ValueError(f"No matches found for tournament {test_tournament!r}")

    rows: list[dict[str, Any]] = []
    for model_name in model_names or DEFAULT_COMPARISON_MODELS:
        try:
            model = train_prediction_model(
                train_features,
                model_type=model_name,
                random_state=int(model_cfg.get("random_state", 42)),
                max_iter=int(model_cfg.get("max_iter", 1000)),
                recent_window=recent_window,
                elo_k=elo_k,
                model_version=f"comparison-{model_name}",
                model_params=_model_params(calibration_cfg, ensemble_cfg),
            )
            metrics = evaluate_model(model, test_features)
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_source": dataset_source,
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
                    "error_message": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_source": dataset_source,
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
                    "error_message": str(exc),
                }
            )
    successful_rows = [row for row in rows if row["status"] == "ok"]
    successful_rows.sort(key=lambda row: float(row["log_loss"]))
    rows = successful_rows + [row for row in rows if row["status"] != "ok"]
    majority_log_loss = _metric_for(successful_rows, "majority_baseline", "log_loss")
    elo_log_loss = _metric_for(successful_rows, "elo_logistic_regression", "log_loss")
    leaderboard = [
        {
            "model_name": row["model_name"],
            "dataset_source": dataset_source,
            "status": row.get("status", "ok"),
            "train_match_count": int(len(train_features)),
            "test_match_count": int(len(test_features)),
            "accuracy": row["accuracy"],
            "log_loss": row["log_loss"],
            "brier_score": row["brier_score"],
            "ece": row["ece"],
            "beats_majority_baseline": (
                row.get("status") == "ok" and majority_log_loss is not None and float(row["log_loss"]) < majority_log_loss
            ),
            "beats_elo_baseline": (
                row.get("status") == "ok" and elo_log_loss is not None and float(row["log_loss"]) < elo_log_loss
            ),
            "notes": _leaderboard_note(row, majority_log_loss, elo_log_loss),
            "model_details": row.get("model_details", {}),
        }
        for row in rows
    ]

    result = {
        "train_before": int(train_before),
        "test_tournament": test_tournament,
        "dataset_source": dataset_source,
        "train_matches": int(len(train_features)),
        "test_matches": int(len(test_features)),
        "models": rows,
        "leaderboard_path": str(leaderboard_path),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    leaderboard_output = Path(leaderboard_path)
    leaderboard_output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_output.write_text(json.dumps({"models": leaderboard}, indent=2), encoding="utf-8")
    return result


def _metric_for(rows: list[dict[str, Any]], model_name: str, metric_name: str) -> float | None:
    for row in rows:
        if str(row["model_name"]) == model_name:
            return float(row[metric_name])
    return None


def _optional_path(value: object) -> Path | None:
    if not value:
        return None
    path = resolve_project_path(value, PROJECT_ROOT)
    return path if path.exists() else None


def _load_elo_ratings(path: Path | None):
    if path is None:
        return None
    return LocalEloRatingSource(path).load()


def _load_fifa_rankings(path: Path | None):
    if path is None:
        return None
    return LocalFifaRankingSource(path).load()


def _feature_flags(feature_cfg: dict[str, Any]) -> dict[str, bool]:
    raw = feature_cfg.get("enable_groups", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _model_params(calibration_cfg: object, ensemble_cfg: object) -> dict[str, Any]:
    calibration = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    ensemble = ensemble_cfg if isinstance(ensemble_cfg, dict) else {}
    params: dict[str, Any] = {
        "calibration_method": calibration.get("method", "sigmoid"),
        "calibration_cv": calibration.get("cv", 3),
        "ensemble_mode": ensemble.get("mode", "validation_weighted"),
    }
    if isinstance(ensemble.get("candidates"), list):
        params["ensemble_candidates"] = ensemble["candidates"]
    return params


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


def _infer_dataset_source(
    explicit_source: str | None,
    config_source: object,
    matches_path: Path,
) -> str:
    if explicit_source:
        return str(explicit_source)
    path_text = str(matches_path).replace("\\", "/").lower()
    if "/data/real/" in path_text or matches_path.name.lower() == "results.csv":
        return "international-results"
    normalized_config_source = str(config_source or "csv")
    if normalized_config_source.strip().lower() in {"real", "international-results", "international_results"}:
        return "international-results"
    return normalized_config_source


def _test_tournament_mask(features: pd.DataFrame, test_tournament: str) -> pd.Series:
    normalized = test_tournament.strip().lower()
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match and "world cup" in normalized:
        tournament = features["tournament"].fillna("").astype(str).str.lower()
        is_world_cup_finals = tournament.str.contains("world cup", regex=False) & ~tournament.str.contains(
            "qualif",
            regex=False,
        )
        return is_world_cup_finals & features["date"].dt.year.eq(int(year_match.group(0)))
    return features["tournament"].fillna("").astype(str).str.lower().eq(normalized)


def _leaderboard_note(
    row: dict[str, Any],
    majority_log_loss: float | None,
    elo_log_loss: float | None,
) -> str:
    if row.get("status") != "ok":
        return f"model failed: {row.get('error_message')}"
    model_name = str(row["model_name"])
    log_loss = float(row["log_loss"])
    notes = []
    if majority_log_loss is not None and log_loss >= majority_log_loss:
        notes.append("does not beat majority baseline on log loss")
    if elo_log_loss is not None and model_name != "elo_logistic_regression" and log_loss >= elo_log_loss:
        notes.append("does not beat Elo baseline on log loss")
    return "; ".join(notes) if notes else "beats selected baselines on log loss for this split"
