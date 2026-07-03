from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchPredictionContext:
    team_a: str
    team_b: str
    team_a_win_probability: float
    draw_probability: float
    team_b_win_probability: float
    elo_a: float | None = None
    elo_b: float | None = None


@dataclass(frozen=True)
class TournamentOddsContext:
    top_teams: list[dict[str, float | str]]


@dataclass(frozen=True)
class ProbabilityChangeContext:
    event: str
    probability_changes: dict[str, dict[str, float]]
