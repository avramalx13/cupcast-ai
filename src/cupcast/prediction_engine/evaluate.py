from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from cupcast.shared.constants import OUTCOME_LABELS

from .model import PredictionModel


def evaluate_model(model: PredictionModel, feature_table: pd.DataFrame) -> dict[str, object]:
    if feature_table.empty:
        raise ValueError("Cannot evaluate on an empty feature table")
    y_true = feature_table["label"].astype(str).to_numpy()
    probabilities = _predict_probability_matrix(model, feature_table)
    y_pred = np.array(OUTCOME_LABELS)[np.argmax(probabilities, axis=1)]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top_class_accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(_ordered_log_loss(y_true, probabilities)),
        "brier_score": float(_multiclass_brier_score(y_true, probabilities)),
        "matches_tested": int(len(feature_table)),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(OUTCOME_LABELS),
        ).tolist(),
        "calibration": calibration_summary(y_true, y_pred, probabilities),
        "expected_calibration_error": expected_calibration_error(y_true, y_pred, probabilities),
    }


def calibration_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    confidence = probabilities.max(axis=1)
    correct = y_true == y_pred
    bins = [(0.0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]
    rows: list[dict[str, float | int]] = []
    for lower, upper in bins:
        mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            rows.append(
                {
                    "lower": lower,
                    "upper": min(upper, 1.0),
                    "count": 0,
                    "average_confidence": 0.0,
                    "accuracy": 0.0,
                }
            )
            continue
        rows.append(
            {
                "lower": lower,
                "upper": min(upper, 1.0),
                "count": int(mask.sum()),
                "average_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    rows = calibration_summary(y_true, y_pred, probabilities)
    total = max(1, int(sum(int(row["count"]) for row in rows)))
    error = 0.0
    for row in rows:
        count = int(row["count"])
        if count == 0:
            continue
        error += (count / total) * abs(float(row["average_confidence"]) - float(row["accuracy"]))
    return float(error)


def _predict_probability_matrix(model: PredictionModel, feature_table: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in feature_table.iterrows():
        probabilities = model.predict_features(pd.DataFrame([row]))
        rows.append([probabilities[label] for label in OUTCOME_LABELS])
    return np.asarray(rows, dtype=float)


def _multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = _one_hot(y_true, probabilities)
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _ordered_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    label_to_idx = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}
    clipped = np.clip(probabilities, 1e-15, 1.0)
    losses = [-np.log(clipped[row_idx, label_to_idx[str(label)]]) for row_idx, label in enumerate(y_true)]
    return float(np.mean(losses))


def _one_hot(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    one_hot = np.zeros_like(probabilities)
    label_to_idx = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}
    for row_idx, label in enumerate(y_true):
        one_hot[row_idx, label_to_idx[str(label)]] = 1.0
    return one_hot
