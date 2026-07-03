from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cupcast.shared.constants import PROJECT_ROOT

from .compare import DEFAULT_COMPARISON_MODELS
from .data_loader import load_dataset
from .evaluate import evaluate_model
from .features import build_feature_table
from .model import train_prediction_model


def backtest_world_cups(
    matches_path: str | Path,
    years: list[int],
    teams_path: str | Path | None = None,
    shootouts_path: str | Path | None = None,
    output_path: str | Path = PROJECT_ROOT / "models" / "world_cup_backtest_results.json",
    recent_window: int = 5,
    elo_k: float = 28.0,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    matches, teams = load_dataset(
        matches_path,
        teams_path,
        source_type="international-results",
        shootouts_path=shootouts_path,
    )
    features = build_feature_table(matches, teams, recent_window=recent_window, elo_k=elo_k)
    results = []
    for year in years:
        results.append(
            backtest_single_world_cup_year(
                features=features,
                year=int(year),
                recent_window=recent_window,
                elo_k=elo_k,
                model_names=model_names or DEFAULT_COMPARISON_MODELS,
            )
        )
    payload = {
        "dataset_source": "international-results",
        "years": [int(year) for year in years],
        "results": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def backtest_single_world_cup_year(
    features: pd.DataFrame,
    year: int,
    recent_window: int = 5,
    elo_k: float = 28.0,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    tournament_mask = _world_cup_mask(features, year)
    test_features = features.loc[tournament_mask].sort_values("date").reset_index(drop=True)
    if test_features.empty:
        raise ValueError(f"No World Cup matches found for {year}")
    tournament_start = pd.to_datetime(test_features["date"]).min()
    tournament_end = pd.to_datetime(test_features["date"]).max()
    train_features = features.loc[features["date"] < tournament_start].reset_index(drop=True)
    if train_features.empty:
        raise ValueError(f"No training matches found before World Cup {year} start date {tournament_start.date()}")

    model_rows: list[dict[str, Any]] = []
    for model_name in model_names or DEFAULT_COMPARISON_MODELS:
        model = train_prediction_model(
            train_features,
            model_type=model_name,
            recent_window=recent_window,
            elo_k=elo_k,
            model_version=f"world-cup-{year}-{model_name}",
        )
        metrics = evaluate_model(model, test_features)
        model_rows.append(
            {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["expected_calibration_error"],
                **_subset_metrics(model, test_features, stage="group", prefix="group_stage"),
                **_subset_metrics(model, test_features, stage="knockout", prefix="knockout"),
                "matches_tested": metrics["matches_tested"],
            }
        )
    model_rows.sort(key=lambda row: float(row["log_loss"]))
    best = model_rows[0]
    majority = _find_model_row(model_rows, "majority_baseline")
    elo = _find_model_row(model_rows, "elo_logistic_regression")
    ensemble = _find_model_row(model_rows, "weighted_probability_ensemble")
    group_stage_count = _subset_count(test_features, stage="group")
    knockout_count = _subset_count(test_features, stage="knockout")
    best_group = _best_subset_model(model_rows, "group_stage_log_loss", lower_is_better=True)
    best_knockout = _best_subset_model(model_rows, "knockout_log_loss", lower_is_better=True)
    return {
        "year": int(year),
        "tournament_start": tournament_start.date().isoformat(),
        "tournament_end": tournament_end.date().isoformat(),
        "train_match_count": int(len(train_features)),
        "test_match_count": int(len(test_features)),
        "group_stage_match_count": group_stage_count,
        "knockout_match_count": knockout_count,
        "best_overall_model": best["model_name"],
        "best_group_stage_model": best_group["model_name"] if best_group else None,
        "best_knockout_model": best_knockout["model_name"] if best_knockout else None,
        "best_group_stage_log_loss": best_group["group_stage_log_loss"] if best_group else None,
        "best_knockout_log_loss": best_knockout["knockout_log_loss"] if best_knockout else None,
        "accuracy": best["accuracy"],
        "log_loss": best["log_loss"],
        "brier_score": best["brier_score"],
        "ece": best["calibration_ece"],
        "ranked_probability_score_if_available": None,
        "rps_if_available": None,
        "group_stage_accuracy": best["group_stage_accuracy"],
        "knockout_accuracy": best["knockout_accuracy"],
        "calibration_ece": best["calibration_ece"],
        "best_model": best["model_name"],
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
        "models": model_rows,
    }


def _world_cup_mask(features: pd.DataFrame, year: int) -> pd.Series:
    tournament = features["tournament"].fillna("").astype(str).str.lower()
    world_cup_finals = tournament.str.contains("world cup", regex=False) & ~tournament.str.contains("qualif", regex=False)
    return world_cup_finals & (features["date"].dt.year == int(year))


def _subset_accuracy(model, features: pd.DataFrame, stage: str) -> float | None:
    metrics = _subset_metrics(model, features, stage=stage, prefix=stage)
    return metrics[f"{stage}_accuracy"]


def _subset_metrics(model, features: pd.DataFrame, stage: str, prefix: str) -> dict[str, float | None]:
    stage_values = features["stage"].fillna("").astype(str).str.lower()
    if stage == "group":
        subset = features.loc[stage_values.eq("group")]
    else:
        subset = features.loc[~stage_values.isin(["group", "unknown", "friendly", "qualifier"])]
    if subset.empty:
        return {
            f"{prefix}_accuracy": None,
            f"{prefix}_log_loss": None,
            f"{prefix}_brier_score": None,
            f"{prefix}_ece": None,
        }
    report = evaluate_model(model, subset)
    return {
        f"{prefix}_accuracy": float(report["accuracy"]),
        f"{prefix}_log_loss": float(report["log_loss"]),
        f"{prefix}_brier_score": float(report["brier_score"]),
        f"{prefix}_ece": float(report["expected_calibration_error"]),
    }


def _subset_count(features: pd.DataFrame, stage: str) -> int:
    stage_values = features["stage"].fillna("").astype(str).str.lower()
    if stage == "group":
        return int(stage_values.eq("group").sum())
    return int((~stage_values.isin(["group", "unknown", "friendly", "qualifier"])).sum())


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


def _find_model_row(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any] | None:
    for row in rows:
        if row["model_name"] == model_name:
            return row
    return None
