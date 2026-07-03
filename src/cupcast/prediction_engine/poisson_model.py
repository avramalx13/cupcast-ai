from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cupcast.shared.constants import OUTCOME_LABELS

from .model_registry import MatchOutcomeModel


class PoissonGoalModel(MatchOutcomeModel):
    def __init__(self, max_goals: int = 8) -> None:
        self.max_goals = int(max_goals)
        self.classes_ = np.array(OUTCOME_LABELS)
        self.base_goals = 1.25
        self.attack_scale = 0.22
        self.elo_scale = 0.00045
        self.home_advantage = 0.10
        self.importance_scale = 0.025

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "PoissonGoalModel":
        goal_columns = [
            column
            for column in ["team_a_goals_scored_last_5", "team_b_goals_scored_last_5"]
            if column in x
        ]
        if goal_columns:
            self.base_goals = float(np.clip(x[goal_columns].mean().mean(), 0.75, 2.2))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in x.iterrows():
            lambda_a, lambda_b = self._expected_goals(row)
            rows.append(_scoreline_probabilities(lambda_a, lambda_b, self.max_goals))
        return np.asarray(rows, dtype=float)

    def _expected_goals(self, row: pd.Series) -> tuple[float, float]:
        gf_a = float(row.get("team_a_goals_scored_last_5", 1.0))
        gf_b = float(row.get("team_b_goals_scored_last_5", 1.0))
        ga_a = float(row.get("team_a_goals_conceded_last_5", 1.0))
        ga_b = float(row.get("team_b_goals_conceded_last_5", 1.0))
        elo_diff = float(row.get("elo_external_diff", row.get("elo_diff", 0.0)))
        home_flag = float(row.get("home_advantage_flag", 1 - int(row.get("neutral", 1))))
        importance = float(row.get("tournament_importance", 1.0))

        attack_a = (gf_a - ga_b) * self.attack_scale
        attack_b = (gf_b - ga_a) * self.attack_scale
        context = (importance - 1.0) * self.importance_scale
        lambda_a = self.base_goals + attack_a + (elo_diff * self.elo_scale) + (home_flag * self.home_advantage) + context
        lambda_b = self.base_goals + attack_b - (elo_diff * self.elo_scale) - (home_flag * self.home_advantage * 0.35) + context
        return float(np.clip(lambda_a, 0.15, 4.5)), float(np.clip(lambda_b, 0.15, 4.5))


def _scoreline_probabilities(lambda_a: float, lambda_b: float, max_goals: int) -> list[float]:
    a_win = 0.0
    draw = 0.0
    b_win = 0.0
    a_probs = [_poisson_pmf(goals, lambda_a) for goals in range(max_goals + 1)]
    b_probs = [_poisson_pmf(goals, lambda_b) for goals in range(max_goals + 1)]
    for goals_a, probability_a in enumerate(a_probs):
        for goals_b, probability_b in enumerate(b_probs):
            probability = probability_a * probability_b
            if goals_a > goals_b:
                a_win += probability
            elif goals_a == goals_b:
                draw += probability
            else:
                b_win += probability
    total = a_win + draw + b_win
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [a_win / total, draw / total, b_win / total]


def _poisson_pmf(k: int, lam: float) -> float:
    return float(math.exp(-lam) * (lam**k) / math.factorial(k))
