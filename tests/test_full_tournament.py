from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cupcast.prediction_engine.model import PredictionResult
from cupcast.simulator.full_tournament import (
    TournamentConfigError,
    TournamentPredictionService,
    build_round_of_32_from_group_results,
    load_tournament_config,
    rank_group_table,
    select_best_third_place_teams,
    simulate_full_tournament,
    simulate_group_stage,
    validate_tournament_groups,
)


@dataclass
class FixedPredictor:
    model_version: str = "fixed"

    def predict_match(self, team_a, team_b, teams, matches=None, neutral=True, stage="group", current_states=None):
        return PredictionResult(
            team_a=team_a,
            team_b=team_b,
            team_a_win_probability=0.45,
            draw_probability=0.20,
            team_b_win_probability=0.35,
        )


def test_tournament_groups_yaml_validates_against_team_registry(tmp_path) -> None:
    path = tmp_path / "groups.yaml"
    groups = _groups()
    groups["A"][3] = "Czechia"
    groups["E"][1] = "Curacao"
    path.write_text(_yaml(groups), encoding="utf-8")
    teams = _teams(groups)
    teams.loc[teams["team"] == "Czechia", "team"] = "Czech Republic"
    teams.loc[teams["team"] == "Curacao", "team"] = "Curaçao"

    config = load_tournament_config(path, teams=teams)

    assert config.groups["A"][3] == "Czech Republic"
    assert config.groups["E"][1] == "Curaçao"
    assert config.aliases == {"Czechia": "Czech Republic", "Curacao": "Curaçao"}


def test_duplicate_team_detection() -> None:
    groups = _groups()
    groups["B"][0] = groups["A"][0]

    try:
        validate_tournament_groups(groups)
    except TournamentConfigError as exc:
        assert "Duplicate teams" in str(exc)
    else:
        raise AssertionError("Expected duplicate team config to fail")


def test_missing_team_detection() -> None:
    groups = _groups()
    teams = _teams(groups)
    available = set(teams.loc[teams["team"] != "Team 48", "team"])

    try:
        validate_tournament_groups(groups, available_teams=available)
    except TournamentConfigError as exc:
        assert "Team 48" in str(exc)
    else:
        raise AssertionError("Expected missing team config to fail")


def test_group_stage_creates_six_matches_per_group() -> None:
    groups = _groups()
    service = TournamentPredictionService(FixedPredictor(), teams=_teams(groups))

    result = simulate_group_stage(groups, service, random_state=1)

    assert len(result.matches) == 72
    assert all(len([match for match in result.matches if match["group"] == group]) == 6 for group in groups)


def test_group_ranking_uses_points_goal_difference_goals_scored_then_seeded_tiebreaker() -> None:
    table = rank_group_table(
        [
            {"team_a": "A", "team_b": "B", "team_a_score": 2, "team_b_score": 0},
            {"team_a": "C", "team_b": "D", "team_a_score": 1, "team_b_score": 1},
            {"team_a": "A", "team_b": "C", "team_a_score": 1, "team_b_score": 0},
            {"team_a": "B", "team_b": "D", "team_a_score": 3, "team_b_score": 0},
            {"team_a": "A", "team_b": "D", "team_a_score": 0, "team_b_score": 0},
            {"team_a": "B", "team_b": "C", "team_a_score": 1, "team_b_score": 1},
        ],
        tie_seed=7,
    )

    assert table[0]["team"] == "A"
    assert table[0]["points"] == 7
    assert table[1]["team"] == "B"
    assert table[1]["goal_difference"] == 1


def test_best_third_place_selection_returns_eight_teams_and_round_of_32_shape() -> None:
    groups = _groups()
    service = TournamentPredictionService(FixedPredictor(), teams=_teams(groups))
    result = simulate_group_stage(groups, service, random_state=3)

    best_third = select_best_third_place_teams(result.group_tables)
    bracket = build_round_of_32_from_group_results(result.group_tables, best_third)

    assert len(best_third) == 8
    assert len(bracket["round_of_32"]) == 16
    assert all(match["team_a_group"] != match["team_b_group"] for match in bracket["round_of_32"])


def test_full_tournament_simulation_probabilities_are_valid() -> None:
    groups = _groups()
    service = TournamentPredictionService(FixedPredictor(), teams=_teams(groups))

    summary = simulate_full_tournament(groups, service, n_simulations=20, seed=4)

    assert len(summary["team_probabilities"]) == 48
    for row in summary["team_probabilities"]:
        for key, value in row.items():
            if key.endswith("_probability"):
                assert 0.0 <= float(value) <= 1.0
    champion_total = sum(float(row["title_probability"]) for row in summary["team_probabilities"])
    assert abs(champion_total - 1.0) < 1e-9


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


def _yaml(groups: dict[str, list[str]]) -> str:
    lines = ['tournament: "World Cup 2026"', 'format: "world_cup_2026"', "groups:"]
    for group, teams in groups.items():
        lines.append(f"  {group}:")
        lines.extend(f"    - {team}" for team in teams)
    return "\n".join(lines) + "\n"
