from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from cupcast.shared.constants import OUTCOME_LABELS

from .model_registry import MatchOutcomeModel


class CalibratedMatchModel(MatchOutcomeModel):
    def __init__(self, base_estimator: Any, method: str = "sigmoid", cv: int = 3) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.cv = int(cv)
        self.classes_ = np.array(OUTCOME_LABELS)
        self.estimator: Any | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CalibratedMatchModel":
        y_text = y.astype(str)
        min_class_count = int(y_text.value_counts().min()) if not y_text.empty else 0
        cv = min(self.cv, min_class_count)
        if cv < 2:
            self.base_estimator.fit(x, y_text)
            self.estimator = self.base_estimator
        else:
            try:
                self.estimator = CalibratedClassifierCV(
                    estimator=self.base_estimator,
                    method=self.method,
                    cv=cv,
                )
            except TypeError:  # pragma: no cover - older sklearn compatibility
                self.estimator = CalibratedClassifierCV(
                    base_estimator=self.base_estimator,
                    method=self.method,
                    cv=cv,
                )
            self.estimator.fit(x, y_text)
        self.classes_ = np.array(OUTCOME_LABELS)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("CalibratedMatchModel must be fitted before prediction")
        raw = np.asarray(self.estimator.predict_proba(x), dtype=float)
        by_label = np.zeros((len(x), len(OUTCOME_LABELS)), dtype=float)
        for source_idx, label in enumerate(self.estimator.classes_):
            if str(label) in OUTCOME_LABELS:
                by_label[:, OUTCOME_LABELS.index(str(label))] = raw[:, source_idx]
        row_sums = by_label.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return by_label / row_sums
