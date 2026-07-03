from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from cupcast.prediction_engine.features import TeamStates, build_current_states
from cupcast.prediction_engine.model import PredictionModel
from cupcast.prediction_engine.model import PredictionResult

from .bracket import KnockoutMatch, TournamentBracket


class MatchPredictor(Protocol):
    model_version: str

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
        ...


@dataclass
class SimulationResult:
    team_probabilities: list[dict[str, float | str]]
    n_simulations: int
    model_version: str
    simulation_version: str


def simulate_match(
    team_a: str,
    team_b: str,
    prediction_model: MatchPredictor,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None,
    rng: random.Random,
    stage: str = "round_of_16",
    knockout: bool = True,
) -> str:
    result = prediction_model.predict_match(
        team_a=team_a,
        team_b=team_b,
        teams=teams,
        matches=matches,
        neutral=True,
        stage=stage,
    )
    if knockout:
        probability_a = result.team_a_win_probability + result.draw_probability * 0.5
        probability_b = result.team_b_win_probability + result.draw_probability * 0.5
        total = probability_a + probability_b
        probability_a = probability_a / total if total else 0.5
        return team_a if rng.random() < probability_a else team_b

    roll = rng.random()
    if roll < result.team_a_win_probability:
        return team_a
    if roll < result.team_a_win_probability + result.draw_probability:
        return "DRAW"
    return team_b


def simulate_tournament(
    bracket: TournamentBracket,
    prediction_model: MatchPredictor,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None,
    n_simulations: int = 10000,
    seed: int | None = 42,
    simulation_version: str = "local",
) -> SimulationResult:
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    rng = random.Random(seed)
    prediction_cache = _MatchPredictionCache(
        prediction_model=prediction_model,
        teams=teams,
        matches=matches,
    )
    counts = {
        team: {
            "reach_round_of_16_probability": 0,
            "reach_quarterfinal_probability": 0,
            "reach_semifinal_probability": 0,
            "reach_final_probability": 0,
            "win_tournament_probability": 0,
        }
        for team in bracket.teams
    }

    for eliminated in bracket.eliminated:
        if eliminated in counts:
            counts[eliminated] = {key: 0 for key in counts[eliminated]}

    for _ in range(n_simulations):
        pending_matches: list[KnockoutMatch] = []
        quarterfinalists: list[str] = []
        for match in bracket.round_of_16:
            completed_winner = bracket.completed_round_of_16.get(_match_key(match.team_a, match.team_b))
            team_a_out = match.team_a in bracket.eliminated
            team_b_out = match.team_b in bracket.eliminated
            if not team_a_out:
                counts[match.team_a]["reach_round_of_16_probability"] += 1
            if not team_b_out:
                counts[match.team_b]["reach_round_of_16_probability"] += 1
            if completed_winner:
                quarterfinalists.append(completed_winner)
                continue
            if team_a_out and team_b_out:
                continue
            if team_a_out:
                quarterfinalists.append(match.team_b)
                continue
            if team_b_out:
                quarterfinalists.append(match.team_a)
                continue
            pending_matches.append(match)

        quarterfinalists.extend(_simulate_round(pending_matches, prediction_cache, rng))
        for team in quarterfinalists:
            counts[team]["reach_quarterfinal_probability"] += 1

        semifinalists = _simulate_round(_pair_round(quarterfinalists, "quarterfinal"), prediction_cache, rng)
        for team in semifinalists:
            counts[team]["reach_semifinal_probability"] += 1

        finalists = _simulate_round(_pair_round(semifinalists, "semifinal"), prediction_cache, rng)
        for team in finalists:
            counts[team]["reach_final_probability"] += 1

        champion = _simulate_round(_pair_round(finalists, "final"), prediction_cache, rng)[0]
        counts[champion]["win_tournament_probability"] += 1

    rows = []
    for team in bracket.teams:
        row: dict[str, float | str] = {"team": team}
        for key, value in counts[team].items():
            row[key] = float(value / n_simulations)
        rows.append(row)

    rows.sort(key=lambda item: float(item["win_tournament_probability"]), reverse=True)
    return SimulationResult(
        team_probabilities=rows,
        n_simulations=n_simulations,
        model_version=prediction_model.model_version,
        simulation_version=simulation_version,
    )


def _simulate_round(
    round_matches: list[KnockoutMatch],
    prediction_cache: "_MatchPredictionCache",
    rng: random.Random,
) -> list[str]:
    winners = []
    for match in round_matches:
        winners.append(
            simulate_match(
                team_a=match.team_a,
                team_b=match.team_b,
                prediction_model=prediction_cache,
                teams=prediction_cache.teams,
                matches=prediction_cache.matches,
                rng=rng,
                stage=match.stage,
                knockout=True,
            )
        )
    return winners


def _pair_round(teams: list[str], stage: str) -> list[KnockoutMatch]:
    if len(teams) < 2:
        return []
    if len(teams) % 2 != 0:
        raise ValueError(f"Cannot pair odd number of teams: {teams}")
    return [
        KnockoutMatch(team_a=teams[idx], team_b=teams[idx + 1], stage=stage)
        for idx in range(0, len(teams), 2)
    ]


class _MatchPredictionCache:
    def __init__(
        self,
        prediction_model: MatchPredictor,
        teams: pd.DataFrame,
        matches: pd.DataFrame | None,
    ) -> None:
        self.prediction_model = prediction_model
        self.teams = teams
        self.matches = matches
        self.model_version = prediction_model.model_version
        self.current_states = self._build_current_states()
        self._cache: dict[tuple[str, str, str], PredictionResult] = {}

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
        cache_key = (team_a, team_b, stage)
        if cache_key not in self._cache:
            if isinstance(self.prediction_model, PredictionModel):
                self._cache[cache_key] = self.prediction_model.predict_match(
                    team_a=team_a,
                    team_b=team_b,
                    teams=self.teams,
                    matches=self.matches,
                    neutral=neutral,
                    stage=stage,
                    current_states=self.current_states,
                )
            else:
                self._cache[cache_key] = self.prediction_model.predict_match(
                    team_a=team_a,
                    team_b=team_b,
                    teams=teams,
                    matches=matches,
                    neutral=neutral,
                    stage=stage,
                )
        return self._cache[cache_key]

    def _build_current_states(self) -> TeamStates | None:
        if self.matches is None or not isinstance(self.prediction_model, PredictionModel):
            return None
        return build_current_states(
            matches=self.matches,
            teams=self.teams,
            recent_window=self.prediction_model.recent_window,
            elo_k=self.prediction_model.elo_k,
        )


def _match_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted((team_a, team_b)))
