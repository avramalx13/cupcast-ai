from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cupcast.shared.constants import OUTCOME_LABELS, PROJECT_ROOT

from .model import PredictionModel, train_prediction_model


def analyze_world_cup_errors(
    features: pd.DataFrame,
    years: list[int],
    model_names: list[str],
    output_path: str | Path = PROJECT_ROOT / "models" / "world_cup_error_analysis.json",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for year in years:
        try:
            train_features, test_features = _world_cup_split(features, int(year))
            best_model, best_name = _best_model_for_split(train_features, test_features, model_names)
            rows.extend(_error_rows_for_year(best_model, best_name, test_features, int(year)))
        except Exception as exc:
            rows.append(
                {
                    "year": int(year),
                    "match": None,
                    "predicted_probabilities": {},
                    "actual_result": None,
                    "error_type": "analysis_failed",
                    "notes": str(exc),
                }
            )
    payload = {"errors": rows}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _best_model_for_split(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    model_names: list[str],
) -> tuple[PredictionModel, str]:
    from .evaluate import evaluate_model

    best_model: PredictionModel | None = None
    best_name = ""
    best_loss = float("inf")
    for model_name in model_names:
        try:
            model = train_prediction_model(train_features, model_type=model_name)
            loss = float(evaluate_model(model, test_features)["log_loss"])
        except Exception:
            continue
        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_name = model_name
    if best_model is None:
        raise ValueError("No model could be trained for error analysis")
    return best_model, best_name


def _error_rows_for_year(
    model: PredictionModel,
    model_name: str,
    test_features: pd.DataFrame,
    year: int,
) -> list[dict[str, Any]]:
    rows = []
    for _, match in test_features.iterrows():
        probabilities = model.predict_features(pd.DataFrame([match]))
        predicted = max(probabilities, key=probabilities.get)
        actual = str(match["label"])
        actual_probability = float(probabilities.get(actual, 0.0))
        top_probability = float(probabilities[predicted])
        if predicted != actual:
            error_type = "predicted_top_class_lost"
            notes = f"{model_name} assigned {top_probability:.3f} to {predicted}, actual was {actual}"
        elif top_probability < 0.4:
            error_type = "correct_underdog_call"
            notes = f"{model_name} correctly picked {actual} with only {top_probability:.3f} confidence"
        else:
            error_type = "correct_favorite_call"
            notes = f"{model_name} correctly picked {actual}"
        rows.append(
            {
                "year": year,
                "match": f"{match['team_a']} vs {match['team_b']}",
                "date": pd.to_datetime(match["date"]).date().isoformat(),
                "model_name": model_name,
                "predicted_probabilities": {label: float(probabilities[label]) for label in OUTCOME_LABELS},
                "actual_result": actual,
                "predicted_result": predicted,
                "actual_probability": actual_probability,
                "error_type": error_type,
                "notes": notes,
            }
        )
    rows.sort(key=lambda row: (row["error_type"] != "predicted_top_class_lost", row["actual_probability"]))
    return rows[:25]


def _world_cup_split(features: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tournament = features["tournament"].fillna("").astype(str).str.lower()
    world_cup_finals = tournament.str.contains("world cup", regex=False) & ~tournament.str.contains("qualif", regex=False)
    test_mask = world_cup_finals & (features["date"].dt.year == int(year))
    test_features = features.loc[test_mask].sort_values("date").reset_index(drop=True)
    if test_features.empty:
        raise ValueError(f"No World Cup matches found for {year}")
    start_date = pd.to_datetime(test_features["date"]).min()
    train_features = features.loc[features["date"] < start_date].reset_index(drop=True)
    if train_features.empty:
        raise ValueError(f"No training matches found before World Cup {year}")
    return train_features, test_features
