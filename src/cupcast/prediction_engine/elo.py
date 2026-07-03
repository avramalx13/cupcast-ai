from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected Elo score for team A against team B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    result_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Update two Elo ratings after a match.

    result_a is 1.0 for a team A win, 0.5 for a draw, and 0.0 for a loss.
    """
    if result_a not in {0.0, 0.5, 1.0}:
        raise ValueError("result_a must be 1.0, 0.5, or 0.0")
    if k <= 0:
        raise ValueError("k must be positive")
    expected_a = expected_score(rating_a, rating_b)
    change = k * (result_a - expected_a)
    return rating_a + change, rating_b - change


def match_result_for_team_a(score_a: int, score_b: int) -> float:
    if score_a > score_b:
        return 1.0
    if score_a == score_b:
        return 0.5
    return 0.0


def label_from_score(score_a: int, score_b: int) -> str:
    if score_a > score_b:
        return "A_WIN"
    if score_a == score_b:
        return "DRAW"
    return "B_WIN"
