from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cupcast.prediction_engine.elo import match_result_for_team_a, update_elo

from .bracket import TournamentBracket
from .monte_carlo import MatchPredictor, simulate_tournament


@dataclass
class ResultUpdate:
    event: str
    eliminated_team: str | None
    advanced_team: str
    probability_changes: dict[str, dict[str, float]]
    elo_update: dict[str, float]


def apply_match_result(
    bracket: TournamentBracket,
    prediction_model: MatchPredictor,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None,
    team_a: str,
    team_b: str,
    score_a: int,
    score_b: int,
    penalty_winner: str | None = None,
    simulations: int = 1000,
    seed: int = 42,
) -> ResultUpdate:
    winner = _winner(team_a, team_b, score_a, score_b, penalty_winner)
    loser = team_b if winner == team_a else team_a
    updated_bracket = bracket.with_completed_match(team_a=team_a, team_b=team_b, winner=winner)

    before = simulate_tournament(
        bracket=bracket,
        prediction_model=prediction_model,
        teams=teams,
        matches=matches,
        n_simulations=simulations,
        seed=seed,
        simulation_version="before-update",
    )
    after = simulate_tournament(
        bracket=updated_bracket,
        prediction_model=prediction_model,
        teams=teams,
        matches=matches,
        n_simulations=simulations,
        seed=seed,
        simulation_version="after-update",
    )

    before_title = {str(row["team"]): float(row["win_tournament_probability"]) for row in before.team_probabilities}
    after_title = {str(row["team"]): float(row["win_tournament_probability"]) for row in after.team_probabilities}
    always_report = {winner, loser}
    probability_changes = {
        team: {"before": before_title.get(team, 0.0), "after": after_title.get(team, 0.0)}
        for team in sorted(set(before_title) | set(after_title))
        if before_title.get(team, 0.0) != after_title.get(team, 0.0) or team in always_report
    }

    # Expose the Elo update calculation for the live-update pipeline. The persisted
    # state is intentionally left to a caller/database layer.
    rating_a_before = _rating_for_team(teams, team_a)
    rating_b_before = _rating_for_team(teams, team_b)
    rating_a_after, rating_b_after = update_elo(
        rating_a_before,
        rating_b_before,
        match_result_for_team_a(score_a, score_b),
    )

    suffix = " on penalties" if score_a == score_b and penalty_winner else ""
    return ResultUpdate(
        event=f"{winner} eliminated {loser}{suffix}",
        eliminated_team=loser,
        advanced_team=winner,
        probability_changes=probability_changes,
        elo_update={
            f"{team_a}_before": rating_a_before,
            f"{team_a}_after": rating_a_after,
            f"{team_b}_before": rating_b_before,
            f"{team_b}_after": rating_b_after,
        },
    )


def _winner(
    team_a: str,
    team_b: str,
    score_a: int,
    score_b: int,
    penalty_winner: str | None,
) -> str:
    if score_a > score_b:
        return team_a
    if score_b > score_a:
        return team_b
    if penalty_winner in {team_a, team_b}:
        return str(penalty_winner)
    raise ValueError("A drawn knockout match requires penalty_winner")


def _rating_for_team(teams: pd.DataFrame, team: str) -> float:
    row = teams.loc[teams["team"] == team]
    if row.empty:
        raise ValueError(f"Team not found in teams.csv: {team}")
    return float(row["initial_elo"].iloc[0])
