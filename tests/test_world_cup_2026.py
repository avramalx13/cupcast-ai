from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cupcast.prediction_engine.model import PredictionResult
from cupcast.simulator.world_cup_2026 import (
    build_round_of_32,
    rank_group_table,
    select_best_third_place_teams,
    simulate_group_stage,
    simulate_world_cup_2026,
)


@dataclass
class FixedPredictor:
    model_version: str = "fixed"

    def predict_match(self, team_a, team_b, teams, matches=None, neutral=True, stage="group", current_states=None):
        return PredictionResult(
            team_a=team_a,
            team_b=team_b,
            team_a_win_probability=0.45,
            draw_probability=0.25,
            team_b_win_probability=0.30,
        )


def test_rank_group_table_uses_points_goal_difference_and_goals_scored() -> None:
    table = rank_group_table(
        [
            {"team_a": "A", "team_b": "B", "team_a_score": 2, "team_b_score": 0},
            {"team_a": "C", "team_b": "D", "team_a_score": 1, "team_b_score": 1},
            {"team_a": "A", "team_b": "C", "team_a_score": 1, "team_b_score": 0},
            {"team_a": "B", "team_b": "D", "team_a_score": 3, "team_b_score": 0},
        ]
    )

    assert table.iloc[0]["team"] == "A"
    assert table.iloc[0]["points"] == 6


def test_world_cup_2026_group_stage_and_round_of_32_shape() -> None:
    groups = _groups()
    teams = _teams(groups)
    results = simulate_group_stage(groups, FixedPredictor(), teams, seed=1)
    group_tables = {
        group_name: rank_group_table([row for row in results if row["group"] == group_name])
        for group_name in groups
    }
    best_third = select_best_third_place_teams(group_tables)
    round_of_32 = build_round_of_32(group_tables, best_third)

    assert len(results) == 72
    assert len(best_third) == 8
    assert len(round_of_32) == 16


def test_world_cup_2026_simulation_returns_valid_probabilities() -> None:
    groups = _groups()
    teams = _teams(groups)

    result = simulate_world_cup_2026(groups, FixedPredictor(), teams, n_simulations=3, seed=2)

    assert len(result.team_probabilities) == 48
    title_total = sum(float(row["win_tournament_probability"]) for row in result.team_probabilities)
    third_place_total = sum(float(row["win_third_place_probability"]) for row in result.team_probabilities)
    assert abs(title_total - 1.0) < 1e-9
    assert abs(third_place_total - 1.0) < 1e-9


def _groups() -> dict[str, list[str]]:
    teams = [f"Team {idx:02d}" for idx in range(1, 49)]
    return {
        chr(ord("A") + group_idx): teams[group_idx * 4 : group_idx * 4 + 4]
        for group_idx in range(12)
    }


def _teams(groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for rank, team in enumerate([team for group in groups.values() for team in group], start=1):
        rows.append(
            {
                "team": team,
                "confederation": "TEST",
                "initial_elo": 1600 - rank,
                "fifa_rank": rank,
            }
        )
    return pd.DataFrame(rows)
