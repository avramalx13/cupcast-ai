from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cupcast.shared.constants import OUTCOME_LABELS

from .features import FEATURE_COLUMNS


ELO_ONLY_COLUMNS = ["elo_a", "elo_b", "elo_diff"]
RECENT_FORM_ONLY_COLUMNS = [
    "team_a_recent_form",
    "team_b_recent_form",
    "recent_form_diff",
    "team_a_goals_scored_last_5",
    "team_b_goals_scored_last_5",
    "team_a_goals_conceded_last_5",
    "team_b_goals_conceded_last_5",
]
RANKING_ONLY_COLUMNS = [
    "fifa_rank_a",
    "fifa_rank_b",
    "rank_diff",
    "fifa_rank_external_a",
    "fifa_rank_external_b",
    "fifa_rank_external_diff",
    "fifa_points_a",
    "fifa_points_b",
    "fifa_points_diff",
]
SCHEDULE_ONLY_COLUMNS = [
    "days_since_last_match_a",
    "days_since_last_match_b",
    "rest_days_diff",
    "matches_last_30_days_a",
    "matches_last_30_days_b",
    "matches_last_90_days_a",
    "matches_last_90_days_b",
]
GOALS_ONLY_COLUMNS = [
    "team_a_goals_scored_last_5",
    "team_b_goals_scored_last_5",
    "team_a_goals_conceded_last_5",
    "team_b_goals_conceded_last_5",
    "weighted_goal_diff_last_5_a",
    "weighted_goal_diff_last_5_b",
]
ELO_PLUS_FORM_COLUMNS = list(dict.fromkeys(ELO_ONLY_COLUMNS + RECENT_FORM_ONLY_COLUMNS))


class MatchOutcomeModel(ABC):
    classes_: np.ndarray

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MatchOutcomeModel":
        ...

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "MatchOutcomeModel":
        loaded = joblib.load(path)
        if not isinstance(loaded, MatchOutcomeModel):
            raise TypeError(f"File does not contain a MatchOutcomeModel: {path}")
        return loaded


class MajorityBaseline(MatchOutcomeModel):
    def __init__(self) -> None:
        self.classes_ = np.array(OUTCOME_LABELS)
        self.probabilities = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MajorityBaseline":
        counts = y.astype(str).value_counts()
        raw = np.array([float(counts.get(label, 0.0)) for label in OUTCOME_LABELS], dtype=float)
        if raw.sum() == 0:
            self.probabilities = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        else:
            self.probabilities = raw / raw.sum()
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n_rows = len(x)
        return np.tile(self.probabilities, (n_rows, 1))


class UniformRandomBaseline(MatchOutcomeModel):
    def __init__(self) -> None:
        self.classes_ = np.array(OUTCOME_LABELS)
        self.probabilities = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "UniformRandomBaseline":
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(self.probabilities, (len(x), 1))


def create_model(model_name: str, config: dict[str, Any] | None = None) -> Any:
    config = config or {}
    name = model_name.strip().lower()
    random_state = int(config.get("random_state", 42))
    max_iter = int(config.get("max_iter", 1000))

    if name in {"majority", "majority_baseline"}:
        return MajorityBaseline()
    if name in {"uniform", "uniform_random", "uniform_random_baseline"}:
        return UniformRandomBaseline()
    if name in {"elo_logistic_regression", "elo_only_logistic_regression"}:
        return _logistic_regression(random_state=random_state, max_iter=max_iter)
    if name in {"recent_form_only", "recent_form_logistic_regression"}:
        return _logistic_regression(random_state=random_state, max_iter=max_iter)
    if name in {
        "ranking_only",
        "ranking_only_logistic_regression",
        "schedule_only",
        "schedule_only_logistic_regression",
        "goals_only",
        "goals_only_logistic_regression",
        "elo_plus_form",
        "elo_plus_form_logistic_regression",
    }:
        return _logistic_regression(random_state=random_state, max_iter=max_iter)
    if name in {"logistic_regression", "feature_logistic_regression"}:
        return _logistic_regression(random_state=random_state, max_iter=max_iter)
    if name in {
        "feature_logistic_regression_calibrated",
        "logistic_regression_calibrated",
        "random_forest_calibrated",
        "gradient_boosting_calibrated",
    }:
        from .calibration import CalibratedMatchModel

        base_name = name.replace("_calibrated", "")
        if base_name == "feature_logistic_regression":
            base_name = "logistic_regression"
        return CalibratedMatchModel(
            create_model(base_name, config),
            method=str(config.get("calibration_method", "sigmoid")),
            cv=int(config.get("calibration_cv", 3)),
        )
    if name == "poisson_goal_model":
        from .poisson_model import PoissonGoalModel

        return PoissonGoalModel(max_goals=int(config.get("max_goals", 8)))
    if name in {"weighted_probability_ensemble", "probability_ensemble"}:
        from .ensemble import ProbabilityEnsemble

        candidates = config.get("ensemble_candidates")
        return ProbabilityEnsemble(
            candidates=list(candidates) if isinstance(candidates, list) else None,
            mode=str(config.get("ensemble_mode", "validation_weighted")),
            random_state=random_state,
            max_iter=max_iter,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 60)),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            random_state=random_state,
            n_jobs=int(config.get("n_jobs", -1)),
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(config.get("n_estimators", 60)),
            max_depth=int(config.get("max_depth", 2)),
            subsample=float(config.get("subsample", 0.85)),
            random_state=random_state,
        )
    raise ValueError(
        "model_name must be one of: majority_baseline, uniform_random_baseline, "
        "elo_logistic_regression, recent_form_only, logistic_regression, "
        "random_forest, gradient_boosting, poisson_goal_model, "
        "feature_logistic_regression_calibrated, random_forest_calibrated, "
        "gradient_boosting_calibrated, weighted_probability_ensemble"
    )


def feature_columns_for_model(model_name: str) -> list[str]:
    name = model_name.strip().lower()
    if name in {"majority", "majority_baseline"}:
        return []
    if name in {"uniform", "uniform_random", "uniform_random_baseline"}:
        return []
    if name in {"elo_logistic_regression", "elo_only_logistic_regression"}:
        return list(ELO_ONLY_COLUMNS)
    if name in {"recent_form_only", "recent_form_logistic_regression"}:
        return list(RECENT_FORM_ONLY_COLUMNS)
    if name in {"ranking_only", "ranking_only_logistic_regression"}:
        return list(RANKING_ONLY_COLUMNS)
    if name in {"schedule_only", "schedule_only_logistic_regression"}:
        return list(SCHEDULE_ONLY_COLUMNS)
    if name in {"goals_only", "goals_only_logistic_regression"}:
        return list(GOALS_ONLY_COLUMNS)
    if name in {"elo_plus_form", "elo_plus_form_logistic_regression"}:
        return list(ELO_PLUS_FORM_COLUMNS)
    return list(FEATURE_COLUMNS)


def _logistic_regression(random_state: int, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
