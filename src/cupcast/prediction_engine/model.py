from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from cupcast.shared.constants import OUTCOME_LABELS

from .features import FEATURE_COLUMNS, TeamStates, features_for_match
from .model_registry import create_model, feature_columns_for_model


ARTIFACT_VERSION = 1


@dataclass
class PredictionResult:
    team_a: str
    team_b: str
    team_a_win_probability: float
    draw_probability: float
    team_b_win_probability: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "team_a_win_probability": self.team_a_win_probability,
            "draw_probability": self.draw_probability,
            "team_b_win_probability": self.team_b_win_probability,
        }


@dataclass
class PredictionModel:
    estimator: Any
    feature_columns: list[str]
    classes: list[str]
    recent_window: int = 5
    elo_k: float = 28.0
    model_version: str = "local"

    def predict_features(self, features: pd.DataFrame) -> dict[str, float]:
        probabilities = self.estimator.predict_proba(features[self.feature_columns])[0]
        by_class = {label: 0.0 for label in OUTCOME_LABELS}
        for label, probability in zip(self.estimator.classes_, probabilities, strict=False):
            by_class[str(label)] = float(probability)
        total = sum(by_class.values())
        if total <= 0:
            return {"A_WIN": 1 / 3, "DRAW": 1 / 3, "B_WIN": 1 / 3}
        return {label: value / total for label, value in by_class.items()}

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        teams: pd.DataFrame,
        matches: pd.DataFrame | None = None,
        neutral: bool = True,
        stage: str = "group",
        current_states: TeamStates | None = None,
    ) -> PredictionResult:
        features = features_for_match(
            team_a=team_a,
            team_b=team_b,
            teams=teams,
            matches=matches,
            neutral=neutral,
            stage=stage,
            recent_window=self.recent_window,
            elo_k=self.elo_k,
            current_states=current_states,
        )
        probabilities = self.predict_features(features)
        return PredictionResult(
            team_a=team_a,
            team_b=team_b,
            team_a_win_probability=probabilities["A_WIN"],
            draw_probability=probabilities["DRAW"],
            team_b_win_probability=probabilities["B_WIN"],
        )


def build_estimator(
    model_type: str = "logistic_regression",
    random_state: int = 42,
    max_iter: int = 1000,
    model_params: dict[str, Any] | None = None,
) -> Any:
    config = {
        "random_state": random_state,
        "max_iter": max_iter,
    }
    if model_params:
        config.update(model_params)
    return create_model(
        model_type,
        config,
    )


def train_prediction_model(
    feature_table: pd.DataFrame,
    model_type: str = "logistic_regression",
    random_state: int = 42,
    max_iter: int = 1000,
    recent_window: int = 5,
    elo_k: float = 28.0,
    model_version: str = "local",
    model_params: dict[str, Any] | None = None,
) -> PredictionModel:
    feature_columns = feature_columns_for_model(model_type)
    missing = [column for column in feature_columns + ["label"] if column not in feature_table]
    if missing:
        raise ValueError(f"Feature table missing columns: {missing}")
    estimator = build_estimator(
        model_type=model_type,
        random_state=random_state,
        max_iter=max_iter,
        model_params=model_params,
    )
    estimator.fit(feature_table[feature_columns], feature_table["label"])
    return PredictionModel(
        estimator=estimator,
        feature_columns=feature_columns,
        classes=list(OUTCOME_LABELS),
        recent_window=recent_window,
        elo_k=elo_k,
        model_version=model_version,
    )


def save_model(model: PredictionModel, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "estimator": model.estimator,
        "feature_columns": model.feature_columns,
        "classes": model.classes,
        "recent_window": model.recent_window,
        "elo_k": model.elo_k,
        "model_version": model.model_version,
        "created_at": datetime.now(UTC).isoformat(),
    }
    joblib.dump(artifact, model_path)


def load_model(path: str | Path) -> PredictionModel:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Prediction model not found: {model_path}")
    loaded = joblib.load(model_path)
    if isinstance(loaded, PredictionModel):
        return loaded
    if not isinstance(loaded, dict):
        raise TypeError(f"File is not a PredictionModel bundle: {model_path}")
    if loaded.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported prediction model artifact version: {loaded.get('artifact_version')}"
        )
    feature_columns = [str(column) for column in loaded.get("feature_columns", [])]
    classes = [str(label) for label in loaded.get("classes", [])]
    unknown_columns = sorted(set(feature_columns) - set(FEATURE_COLUMNS))
    if unknown_columns:
        raise ValueError(f"Prediction model artifact contains unknown feature columns: {unknown_columns}")
    if classes != list(OUTCOME_LABELS):
        raise ValueError("Prediction model artifact classes do not match current code")
    return PredictionModel(
        estimator=loaded["estimator"],
        feature_columns=feature_columns,
        classes=classes,
        recent_window=int(loaded.get("recent_window", 5)),
        elo_k=float(loaded.get("elo_k", 28.0)),
        model_version=str(loaded.get("model_version", "local")),
    )
