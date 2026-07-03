from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cupcast.shared.constants import OUTCOME_LABELS

from .model_registry import MatchOutcomeModel


DEFAULT_ENSEMBLE_CANDIDATES = [
    "elo_logistic_regression",
    "logistic_regression",
    "poisson_goal_model",
    "gradient_boosting",
]


@dataclass
class EnsembleMember:
    name: str
    model: Any
    feature_columns: list[str]
    weight: float
    validation_log_loss: float | None = None


class ProbabilityEnsemble(MatchOutcomeModel):
    def __init__(
        self,
        candidates: list[str] | None = None,
        mode: str = "validation_weighted",
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:
        self.candidates = candidates or list(DEFAULT_ENSEMBLE_CANDIDATES)
        self.mode = mode
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.classes_ = np.array(OUTCOME_LABELS)
        self.members: list[EnsembleMember] = []
        self.failures: list[dict[str, str]] = []

    @property
    def member_weights_(self) -> dict[str, float]:
        return {member.name: member.weight for member in self.members}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ProbabilityEnsemble":
        from .model_registry import create_model, feature_columns_for_model

        y_text = y.astype(str).reset_index(drop=True)
        x_reset = x.reset_index(drop=True)
        split_idx = max(1, int(len(x_reset) * 0.8))
        if split_idx >= len(x_reset):
            split_idx = len(x_reset)
        train_x = x_reset.iloc[:split_idx]
        train_y = y_text.iloc[:split_idx]
        val_x = x_reset.iloc[split_idx:]
        val_y = y_text.iloc[split_idx:]

        provisional: list[EnsembleMember] = []
        for name in self.candidates:
            try:
                columns = feature_columns_for_model(name)
                model = create_model(name, {"random_state": self.random_state, "max_iter": self.max_iter})
                fit_x = train_x[columns] if not val_x.empty else x_reset[columns]
                fit_y = train_y if not val_x.empty else y_text
                model.fit(fit_x, fit_y)
                validation_loss = None
                if not val_x.empty:
                    probabilities = _ordered_probabilities(model, val_x[columns])
                    validation_loss = _ordered_log_loss(val_y, probabilities)
                model = create_model(name, {"random_state": self.random_state, "max_iter": self.max_iter})
                model.fit(x_reset[columns], y_text)
                provisional.append(
                    EnsembleMember(
                        name=name,
                        model=model,
                        feature_columns=columns,
                        weight=1.0,
                        validation_log_loss=validation_loss,
                    )
                )
            except Exception as exc:
                self.failures.append({"model_name": name, "error_message": str(exc)})
        if not provisional:
            raise ValueError("No ensemble member model could be fitted")
        self.members = _normalize_member_weights(provisional, mode=self.mode)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if not self.members:
            raise ValueError("ProbabilityEnsemble must be fitted before prediction")
        total = np.zeros((len(x), len(OUTCOME_LABELS)), dtype=float)
        for member in self.members:
            total += member.weight * _ordered_probabilities(member.model, x[member.feature_columns])
        row_sums = total.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return total / row_sums


def _ordered_probabilities(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    ordered = np.zeros((len(x), len(OUTCOME_LABELS)), dtype=float)
    for source_idx, label in enumerate(model.classes_):
        if str(label) in OUTCOME_LABELS:
            ordered[:, OUTCOME_LABELS.index(str(label))] = raw[:, source_idx]
    row_sums = ordered.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return ordered / row_sums


def _normalize_member_weights(members: list[EnsembleMember], mode: str) -> list[EnsembleMember]:
    if mode == "simple_average":
        raw_weights = [1.0 for _member in members]
    else:
        raw_weights = [
            1.0 / max(0.001, float(member.validation_log_loss))
            if member.validation_log_loss is not None
            else 1.0
            for member in members
        ]
    total = sum(raw_weights)
    if total <= 0:
        raw_weights = [1.0 for _member in members]
        total = sum(raw_weights)
    for member, raw_weight in zip(members, raw_weights, strict=False):
        member.weight = float(raw_weight / total)
    return members


def _ordered_log_loss(y_true: pd.Series, probabilities: np.ndarray) -> float:
    label_to_idx = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}
    clipped = np.clip(probabilities, 1e-15, 1.0)
    losses = [
        -np.log(clipped[row_idx, label_to_idx[str(label)]])
        for row_idx, label in enumerate(y_true.astype(str).to_numpy())
    ]
    return float(np.mean(losses))
