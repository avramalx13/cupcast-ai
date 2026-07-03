from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from .bracket import KnockoutMatch
from .monte_carlo import MatchPredictor, SimulationResult, simulate_match


def simulate_group_stage(
    groups: dict[str, list[str]],
    prediction_model: MatchPredictor,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    seed: int | None = 42,
) -> list[dict[str, object]]:
    _validate_groups(groups)
    rng = random.Random(seed)
    results: list[dict[str, object]] = []
    for group_name, group_teams in groups.items():
        for idx, team_a in enumerate(group_teams):
            for team_b in group_teams[idx + 1 :]:
                prediction = prediction_model.predict_match(
                    team_a=team_a,
                    team_b=team_b,
                    teams=teams,
                    matches=matches,
                    neutral=True,
                    stage="group",
                )
                score_a, score_b = _sample_scoreline(
                    prediction.team_a_win_probability,
                    prediction.draw_probability,
                    prediction.team_b_win_probability,
                    rng,
                )
                results.append(
                    {
                        "group": group_name,
                        "team_a": team_a,
                        "team_b": team_b,
                        "team_a_score": score_a,
                        "team_b_score": score_b,
                    }
                )
    return results


def rank_group_table(results: list[dict[str, object]]) -> pd.DataFrame:
    table: dict[str, dict[str, int | str]] = {}
    for result in results:
        team_a = str(result["team_a"])
        team_b = str(result["team_b"])
        score_a = int(result["team_a_score"])
        score_b = int(result["team_b_score"])
        for team in [team_a, team_b]:
            table.setdefault(
                team,
                {"team": team, "played": 0, "points": 0, "goals_for": 0, "goals_against": 0},
            )
        table[team_a]["played"] = int(table[team_a]["played"]) + 1
        table[team_b]["played"] = int(table[team_b]["played"]) + 1
        table[team_a]["goals_for"] = int(table[team_a]["goals_for"]) + score_a
        table[team_a]["goals_against"] = int(table[team_a]["goals_against"]) + score_b
        table[team_b]["goals_for"] = int(table[team_b]["goals_for"]) + score_b
        table[team_b]["goals_against"] = int(table[team_b]["goals_against"]) + score_a
        points_a, points_b = _points(score_a, score_b)
        table[team_a]["points"] = int(table[team_a]["points"]) + points_a
        table[team_b]["points"] = int(table[team_b]["points"]) + points_b

    frame = pd.DataFrame(table.values())
    frame["goal_difference"] = frame["goals_for"] - frame["goals_against"]
    return frame.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def select_best_third_place_teams(group_tables: dict[str, pd.DataFrame]) -> list[str]:
    third_place_rows = []
    for group_name, table in group_tables.items():
        if len(table) < 3:
            raise ValueError(f"Group {group_name} does not have a third-place team")
        row = table.iloc[2].copy()
        row["group"] = group_name
        third_place_rows.append(row)
    third_frame = pd.DataFrame(third_place_rows)
    third_frame = third_frame.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    )
    return [str(team) for team in third_frame.head(8)["team"].tolist()]


def build_round_of_32(
    group_tables: dict[str, pd.DataFrame],
    best_third_place_teams: list[str],
) -> list[KnockoutMatch]:
    qualified: list[str] = []
    for group_name in sorted(group_tables):
        table = group_tables[group_name]
        qualified.extend([str(table.iloc[0]["team"]), str(table.iloc[1]["team"])])
    qualified.extend(best_third_place_teams)
    if len(qualified) != 32:
        raise ValueError(f"Round of 32 requires 32 teams, got {len(qualified)}")
    return [
        KnockoutMatch(qualified[idx], qualified[-idx - 1], "round_of_32")
        for idx in range(16)
    ]


def simulate_world_cup_2026(
    groups: dict[str, list[str]],
    prediction_model: MatchPredictor,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    n_simulations: int = 10000,
    seed: int | None = 42,
) -> SimulationResult:
    _validate_groups(groups)
    rng = random.Random(seed)
    all_teams = [team for group in groups.values() for team in group]
    counts = {
        team: {
            "reach_round_of_32_probability": 0,
            "reach_round_of_16_probability": 0,
            "reach_quarterfinal_probability": 0,
            "reach_semifinal_probability": 0,
            "reach_final_probability": 0,
            "win_third_place_probability": 0,
            "win_tournament_probability": 0,
        }
        for team in all_teams
    }

    for sim_idx in range(n_simulations):
        group_results = simulate_group_stage(
            groups,
            prediction_model,
            teams,
            matches,
            seed=rng.randint(0, 2**31 - 1),
        )
        group_tables = {
            group_name: rank_group_table([row for row in group_results if row["group"] == group_name])
            for group_name in groups
        }
        best_third = select_best_third_place_teams(group_tables)
        round_of_32 = build_round_of_32(group_tables, best_third)
        round_of_32_teams = [team for match in round_of_32 for team in [match.team_a, match.team_b]]
        for team in round_of_32_teams:
            counts[team]["reach_round_of_32_probability"] += 1

        round_of_16 = _simulate_knockout_round(round_of_32, prediction_model, teams, matches, rng)
        for team in round_of_16:
            counts[team]["reach_round_of_16_probability"] += 1
        quarterfinalists = _simulate_knockout_round(_pair_round(round_of_16, "round_of_16"), prediction_model, teams, matches, rng)
        for team in quarterfinalists:
            counts[team]["reach_quarterfinal_probability"] += 1
        semifinalists = _simulate_knockout_round(_pair_round(quarterfinalists, "quarterfinal"), prediction_model, teams, matches, rng)
        for team in semifinalists:
            counts[team]["reach_semifinal_probability"] += 1
        finalists, semifinal_losers = _simulate_knockout_round_with_losers(
            _pair_round(semifinalists, "semifinal"),
            prediction_model,
            teams,
            matches,
            rng,
        )
        for team in finalists:
            counts[team]["reach_final_probability"] += 1
        third_place_winner = _simulate_knockout_round(
            _pair_round(semifinal_losers, "third_place"),
            prediction_model,
            teams,
            matches,
            rng,
        )[0]
        counts[third_place_winner]["win_third_place_probability"] += 1
        champion = _simulate_knockout_round(_pair_round(finalists, "final"), prediction_model, teams, matches, rng)[0]
        counts[champion]["win_tournament_probability"] += 1

    rows: list[dict[str, float | str]] = []
    for team in all_teams:
        row: dict[str, float | str] = {"team": team}
        for key, value in counts[team].items():
            row[key] = float(value / n_simulations)
        rows.append(row)
    rows.sort(key=lambda row: float(row["win_tournament_probability"]), reverse=True)
    return SimulationResult(
        team_probabilities=rows,
        n_simulations=n_simulations,
        model_version=prediction_model.model_version,
        simulation_version="world-cup-2026-skeleton",
    )


def _sample_scoreline(
    team_a_win: float,
    draw: float,
    team_b_win: float,
    rng: random.Random,
) -> tuple[int, int]:
    roll = rng.random()
    if roll < team_a_win:
        return rng.choice([(1, 0), (2, 0), (2, 1), (3, 1)])
    if roll < team_a_win + draw:
        return rng.choice([(0, 0), (1, 1), (2, 2)])
    return rng.choice([(0, 1), (0, 2), (1, 2), (1, 3)])


def _simulate_knockout_round(
    round_matches: list[KnockoutMatch],
    prediction_model: MatchPredictor,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None,
    rng: random.Random,
) -> list[str]:
    return [
        simulate_match(
            match.team_a,
            match.team_b,
            prediction_model,
            teams,
            matches,
            rng,
            stage=match.stage,
            knockout=True,
        )
        for match in round_matches
    ]


def _simulate_knockout_round_with_losers(
    round_matches: list[KnockoutMatch],
    prediction_model: MatchPredictor,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    winners: list[str] = []
    losers: list[str] = []
    for match in round_matches:
        winner = simulate_match(
            match.team_a,
            match.team_b,
            prediction_model,
            teams,
            matches,
            rng,
            stage=match.stage,
            knockout=True,
        )
        winners.append(winner)
        losers.append(match.team_b if winner == match.team_a else match.team_a)
    return winners, losers


def _pair_round(teams: list[str], stage: str) -> list[KnockoutMatch]:
    return [KnockoutMatch(teams[idx], teams[idx + 1], stage) for idx in range(0, len(teams), 2)]


def _points(score_a: int, score_b: int) -> tuple[int, int]:
    if score_a > score_b:
        return 3, 0
    if score_b > score_a:
        return 0, 3
    return 1, 1


def _validate_groups(groups: dict[str, list[str]]) -> None:
    if len(groups) != 12:
        raise ValueError(f"World Cup 2026 skeleton requires 12 groups, got {len(groups)}")
    all_teams = [team for group in groups.values() for team in group]
    if any(len(group) != 4 for group in groups.values()):
        raise ValueError("World Cup 2026 skeleton requires 4 teams per group")
    duplicates = [team for team, count in _counts(all_teams).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate teams in World Cup groups: {duplicates}")


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts
