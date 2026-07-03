from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cupcast.prediction_engine.features import TeamStates, build_current_states
from cupcast.prediction_engine.model import PredictionModel, PredictionResult
from cupcast.shared.config import load_yaml

from .monte_carlo import MatchPredictor


SIMULATION_VERSION = "full-world-cup-2026-v1"
PLACEMENT_NOTE = (
    "Round of 32 placement uses a deterministic approximation. It separates same-group teams "
    "where possible, but it is not the official FIFA third-place placement matrix."
)
PLACEMENT_TODO = "Replace deterministic approximation with official FIFA third-place placement matrix when finalized/needed."

GROUP_MATCH_INDEXES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SCORELINES = {
    "A_WIN": [(1, 0), (2, 0), (2, 1), (3, 1)],
    "DRAW": [(0, 0), (1, 1), (2, 2)],
    "B_WIN": [(0, 1), (0, 2), (1, 2), (1, 3)],
}
TEAM_NAME_ALIASES = {
    "Czechia": "Czech Republic",
    "Curacao": "Cura\u00e7ao",
    "Cote d'Ivoire": "Ivory Coast",
    "United States of America": "United States",
}


class TournamentConfigError(ValueError):
    """Raised when a tournament config cannot be simulated safely."""


@dataclass(frozen=True)
class TournamentConfig:
    tournament: str
    format: str
    groups: dict[str, list[str]]
    aliases: dict[str, str]


@dataclass
class TournamentPredictionService:
    prediction_model: MatchPredictor
    teams: pd.DataFrame
    matches: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.model_version = self.prediction_model.model_version
        self.current_states = self._build_current_states()
        self._cache: dict[tuple[str, str, str], PredictionResult] = {}

    def predict_match(self, team_a: str, team_b: str, stage: str = "group") -> PredictionResult:
        cache_key = (team_a, team_b, stage)
        if cache_key not in self._cache:
            if isinstance(self.prediction_model, PredictionModel):
                self._cache[cache_key] = self.prediction_model.predict_match(
                    team_a=team_a,
                    team_b=team_b,
                    teams=self.teams,
                    matches=self.matches,
                    neutral=True,
                    stage=stage,
                    current_states=self.current_states,
                )
            else:
                self._cache[cache_key] = self.prediction_model.predict_match(
                    team_a=team_a,
                    team_b=team_b,
                    teams=self.teams,
                    matches=self.matches,
                    neutral=True,
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


@dataclass(frozen=True)
class GroupStageResult:
    matches: list[dict[str, Any]]
    group_tables: dict[str, list[dict[str, Any]]]
    best_third_place_teams: list[str]
    third_place_eliminated: list[str]
    qualifiers_by_group: dict[str, list[str]]


def load_tournament_config(path: str | Path, teams: pd.DataFrame | None = None) -> TournamentConfig:
    config_path = Path(path)
    raw = load_yaml(config_path)
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, dict):
        raise TournamentConfigError(f"Tournament config requires a groups mapping: {config_path}")

    available = _available_team_names(teams)
    aliases: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for group_name, group_teams in groups_raw.items():
        if not isinstance(group_teams, list):
            raise TournamentConfigError(f"Group {group_name} must be a list of teams")
        resolved: list[str] = []
        for team in group_teams:
            original = str(team).strip()
            canonical = canonical_team_name(original, available)
            if canonical != original:
                aliases[original] = canonical
            resolved.append(canonical)
        groups[str(group_name)] = resolved

    validate_tournament_groups(groups, available_teams=available)
    return TournamentConfig(
        tournament=str(raw.get("tournament", "World Cup 2026")),
        format=str(raw.get("format", "world_cup_2026")),
        groups=groups,
        aliases=aliases,
    )


def canonical_team_name(team: str, available_teams: set[str] | None = None) -> str:
    if available_teams and team in available_teams:
        return team
    normalized_available = {_normalize_team_name(value): value for value in available_teams or set()}
    normalized_match = normalized_available.get(_normalize_team_name(team))
    if normalized_match:
        return normalized_match
    alias = TEAM_NAME_ALIASES.get(team)
    if alias and (not available_teams or alias in available_teams):
        return alias
    return alias or team


def validate_tournament_groups(groups: dict[str, list[str]], available_teams: set[str] | None = None) -> None:
    if len(groups) != 12:
        raise TournamentConfigError(f"World Cup 2026 requires exactly 12 groups, got {len(groups)}")
    bad_sizes = {group: len(teams) for group, teams in groups.items() if len(teams) != 4}
    if bad_sizes:
        raise TournamentConfigError(f"World Cup 2026 requires exactly 4 teams per group: {bad_sizes}")
    all_teams = [team for teams in groups.values() for team in teams]
    duplicates = sorted(team for team in set(all_teams) if all_teams.count(team) > 1)
    if duplicates:
        raise TournamentConfigError(f"Duplicate teams in tournament config: {duplicates}")
    if available_teams is not None:
        missing = sorted(team for team in all_teams if team not in available_teams)
        if missing:
            raise TournamentConfigError(
                "Tournament teams missing from prediction dataset/team registry: "
                + ", ".join(missing)
            )


def simulate_group_stage(
    groups: dict[str, list[str]],
    prediction_service: TournamentPredictionService,
    random_state: random.Random | int | None = None,
) -> GroupStageResult:
    validate_tournament_groups(groups)
    rng = _rng(random_state)
    matches: list[dict[str, Any]] = []
    group_tables: dict[str, list[dict[str, Any]]] = {}
    for group_name in sorted(groups):
        group_teams = groups[group_name]
        group_matches = []
        for team_a_idx, team_b_idx in GROUP_MATCH_INDEXES:
            match = simulate_group_match(
                group_teams[team_a_idx],
                group_teams[team_b_idx],
                prediction_service,
                rng,
            )
            match["group"] = group_name
            group_matches.append(match)
            matches.append(match)
        group_tables[group_name] = rank_group_table(group_matches, tie_seed=_stable_int(group_name))

    best_third = select_best_third_place_teams(group_tables)
    qualifiers_by_group: dict[str, list[str]] = {}
    third_place_eliminated = []
    for group_name, table in group_tables.items():
        qualifiers = [table[0]["team"], table[1]["team"]]
        third_team = table[2]["team"]
        if third_team in best_third:
            qualifiers.append(third_team)
        else:
            third_place_eliminated.append(third_team)
        qualifiers_by_group[group_name] = qualifiers
    return GroupStageResult(
        matches=matches,
        group_tables=group_tables,
        best_third_place_teams=best_third,
        third_place_eliminated=third_place_eliminated,
        qualifiers_by_group=qualifiers_by_group,
    )


def simulate_group_match(
    team_a: str,
    team_b: str,
    prediction_service: TournamentPredictionService,
    random_state: random.Random | int | None = None,
) -> dict[str, Any]:
    rng = _rng(random_state)
    prediction = prediction_service.predict_match(team_a, team_b, stage="group")
    roll = rng.random()
    if roll < prediction.team_a_win_probability:
        outcome = "A_WIN"
    elif roll < prediction.team_a_win_probability + prediction.draw_probability:
        outcome = "DRAW"
    else:
        outcome = "B_WIN"
    score_a, score_b = rng.choice(SCORELINES[outcome])
    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_score": score_a,
        "team_b_score": score_b,
        "outcome": outcome,
        "team_a_win_probability": prediction.team_a_win_probability,
        "draw_probability": prediction.draw_probability,
        "team_b_win_probability": prediction.team_b_win_probability,
    }


def rank_group_table(matches: list[dict[str, Any]], tie_seed: int = 0) -> list[dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for result in matches:
        team_a = str(result["team_a"])
        team_b = str(result["team_b"])
        score_a = int(result["team_a_score"])
        score_b = int(result["team_b_score"])
        for team in [team_a, team_b]:
            table.setdefault(
                team,
                {
                    "team": team,
                    "played": 0,
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goal_difference": 0,
                },
            )
        table[team_a]["played"] += 1
        table[team_b]["played"] += 1
        table[team_a]["goals_for"] += score_a
        table[team_a]["goals_against"] += score_b
        table[team_b]["goals_for"] += score_b
        table[team_b]["goals_against"] += score_a
        points_a, points_b = _points(score_a, score_b)
        table[team_a]["points"] += points_a
        table[team_b]["points"] += points_b

    rows = []
    for row in table.values():
        row = dict(row)
        row["goal_difference"] = int(row["goals_for"]) - int(row["goals_against"])
        row["tie_breaker"] = _stable_tie_breaker(str(row["team"]), tie_seed)
        rows.append(row)
    rows.sort(key=_table_sort_key)
    for position, row in enumerate(rows, start=1):
        row["position"] = position
    return rows


def select_best_third_place_teams(group_tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    third_rows = []
    for group_name, table in group_tables.items():
        if len(table) < 3:
            raise TournamentConfigError(f"Group {group_name} does not have a third-place team")
        row = dict(table[2])
        row["group"] = group_name
        third_rows.append(row)
    third_rows.sort(key=_table_sort_key)
    return [str(row["team"]) for row in third_rows[:8]]


def build_round_of_32_from_group_results(
    group_tables: dict[str, list[dict[str, Any]]],
    best_third_place_teams: list[str],
) -> dict[str, Any]:
    winner_rows = []
    runner_rows = []
    third_rows = []
    for group_name in sorted(group_tables):
        table = group_tables[group_name]
        if len(table) != 4:
            raise TournamentConfigError(f"Group {group_name} must have exactly four ranked teams")
        winner_rows.append(_with_group(table[0], group_name))
        runner_rows.append(_with_group(table[1], group_name))
        third = _with_group(table[2], group_name)
        if third["team"] in best_third_place_teams:
            third_rows.append(third)

    runner_rows.sort(key=_table_sort_key)
    third_rows.sort(key=_table_sort_key)
    seeded = winner_rows + runner_rows[:4]
    unseeded = third_rows + runner_rows[4:]
    if len(seeded) != 16 or len(unseeded) != 16:
        raise TournamentConfigError(
            f"Round of 32 requires 16 seeded and 16 unseeded teams, got {len(seeded)} and {len(unseeded)}"
        )

    opponent_pool = list(reversed(unseeded))
    fixtures = []
    for seeded_row in seeded:
        opponent_idx = _find_opponent_index(opponent_pool, str(seeded_row["group"]))
        opponent = opponent_pool.pop(opponent_idx)
        fixtures.append(
            {
                "team_a": seeded_row["team"],
                "team_b": opponent["team"],
                "stage": "round_of_32",
                "team_a_group": seeded_row["group"],
                "team_b_group": opponent["group"],
            }
        )
    _separate_same_group_pairings(fixtures)
    return {
        "round_of_32": fixtures,
        "placement_note": PLACEMENT_NOTE,
        "todo": PLACEMENT_TODO,
    }


def simulate_knockout_round(
    matches: list[dict[str, Any]],
    prediction_service: TournamentPredictionService,
    random_state: random.Random | int | None = None,
) -> dict[str, Any]:
    rng = _rng(random_state)
    stage = str(matches[0].get("stage", "knockout")) if matches else "knockout"
    results = []
    winners = []
    for fixture in matches:
        team_a = str(fixture["team_a"])
        team_b = str(fixture["team_b"])
        prediction = prediction_service.predict_match(team_a, team_b, stage=stage)
        probability_a = prediction.team_a_win_probability + prediction.draw_probability * 0.5
        probability_b = prediction.team_b_win_probability + prediction.draw_probability * 0.5
        total = probability_a + probability_b
        probability_a = probability_a / total if total else 0.5
        probability_b = probability_b / total if total else 0.5
        winner = team_a if rng.random() < probability_a else team_b
        loser = team_b if winner == team_a else team_a
        winners.append(winner)
        results.append(
            {
                "stage": stage,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_advancement_probability": probability_a,
                "team_b_advancement_probability": probability_b,
                "winner": winner,
                "loser": loser,
            }
        )
    return {"stage": stage, "matches": results, "winners": winners}


def simulate_knockout_bracket(
    round_of_32: dict[str, Any] | list[dict[str, Any]],
    prediction_service: TournamentPredictionService,
    random_state: random.Random | int | None = None,
) -> dict[str, Any]:
    rng = _rng(random_state)
    current_matches = (
        list(round_of_32["round_of_32"])
        if isinstance(round_of_32, dict)
        else list(round_of_32)
    )
    rounds: dict[str, dict[str, Any]] = {}
    round_sequence = [
        ("round_of_32", "round_of_16"),
        ("round_of_16", "quarterfinal"),
        ("quarterfinal", "semifinal"),
        ("semifinal", "final"),
        ("final", None),
    ]
    for stage, next_stage in round_sequence:
        current_matches = [{**match, "stage": stage} for match in current_matches]
        round_result = simulate_knockout_round(current_matches, prediction_service, rng)
        rounds[stage] = round_result
        if next_stage is None:
            break
        current_matches = _pair_teams(round_result["winners"], next_stage)
    champion = rounds["final"]["winners"][0]
    return {"rounds": rounds, "champion": champion}


def simulate_full_tournament(
    groups: TournamentConfig | dict[str, list[str]],
    prediction_service: TournamentPredictionService,
    n_simulations: int = 10000,
    seed: int | None = 42,
) -> dict[str, Any]:
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if isinstance(groups, TournamentConfig):
        tournament = groups.tournament
        tournament_format = groups.format
        group_map = groups.groups
        aliases = groups.aliases
    else:
        tournament = "World Cup 2026"
        tournament_format = "world_cup_2026"
        group_map = groups
        aliases = {}
    validate_tournament_groups(group_map, available_teams=_available_team_names(prediction_service.teams))

    rng = random.Random(seed)
    team_to_group = {team: group for group, teams in group_map.items() for team in teams}
    all_teams = [team for group in sorted(group_map) for team in group_map[group]]
    counts = {
        team: {
            "win_group_count": 0,
            "finish_second_count": 0,
            "finish_third_count": 0,
            "qualify_from_group_count": 0,
            "best_third_place_qualifier_count": 0,
            "reach_round_of_32_count": 0,
            "reach_round_of_16_count": 0,
            "reach_quarterfinal_count": 0,
            "reach_semifinal_count": 0,
            "reach_final_count": 0,
            "win_tournament_count": 0,
        }
        for team in all_teams
    }

    for _ in range(n_simulations):
        group_result = simulate_group_stage(group_map, prediction_service, rng)
        for group_name, table in group_result.group_tables.items():
            for row in table:
                team = str(row["team"])
                position = int(row["position"])
                if position == 1:
                    counts[team]["win_group_count"] += 1
                elif position == 2:
                    counts[team]["finish_second_count"] += 1
                elif position == 3:
                    counts[team]["finish_third_count"] += 1
                if position <= 2 or team in group_result.best_third_place_teams:
                    counts[team]["qualify_from_group_count"] += 1
            for team in group_result.best_third_place_teams:
                if team_to_group[team] == group_name:
                    counts[team]["best_third_place_qualifier_count"] += 1

        bracket = build_round_of_32_from_group_results(
            group_result.group_tables,
            group_result.best_third_place_teams,
        )
        round_of_32_teams = _teams_from_fixtures(bracket["round_of_32"])
        for team in round_of_32_teams:
            counts[team]["reach_round_of_32_count"] += 1
        knockout = simulate_knockout_bracket(bracket, prediction_service, rng)
        for team in knockout["rounds"]["round_of_32"]["winners"]:
            counts[team]["reach_round_of_16_count"] += 1
        for team in knockout["rounds"]["round_of_16"]["winners"]:
            counts[team]["reach_quarterfinal_count"] += 1
        for team in knockout["rounds"]["quarterfinal"]["winners"]:
            counts[team]["reach_semifinal_count"] += 1
        for team in knockout["rounds"]["semifinal"]["winners"]:
            counts[team]["reach_final_count"] += 1
        counts[str(knockout["champion"])]["win_tournament_count"] += 1

    team_rows = [_team_summary_row(team, team_to_group[team], counts[team], n_simulations) for team in all_teams]
    team_rows.sort(key=lambda row: float(row["title_probability"]), reverse=True)
    group_predictions = _group_predictions(group_map, team_rows)
    top_champions = [
        {"team": row["team"], "title_probability": row["title_probability"]}
        for row in team_rows[:10]
    ]
    summary = {
        "available": True,
        "tournament": tournament,
        "format": tournament_format,
        "simulations": n_simulations,
        "seed": seed,
        "model_version": prediction_service.model_version,
        "simulation_version": SIMULATION_VERSION,
        "placement_note": PLACEMENT_NOTE,
        "todo": PLACEMENT_TODO,
        "aliases": aliases,
        "team_probabilities": team_rows,
        "group_predictions": group_predictions,
        "most_likely_group_winners": _most_likely_group_winners(group_predictions),
        "most_likely_qualifiers_by_group": _most_likely_qualifiers_by_group(group_predictions),
        "top_champions": top_champions,
        "dark_horses": _dark_horses(team_rows, all_teams),
        "volatile_teams": _volatile_teams(team_rows),
    }
    return summary


def write_full_tournament_reports(summary: dict[str, Any], out_json: str | Path, out_md: str | Path) -> None:
    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_full_tournament_markdown(summary), encoding="utf-8")


def render_full_tournament_markdown(summary: dict[str, Any]) -> str:
    top_champions = summary.get("top_champions", [])
    group_predictions = summary.get("group_predictions", {})
    rows = [
        "# Full Tournament Simulation",
        "",
        f"Tournament: {summary.get('tournament', 'World Cup 2026')}",
        f"Simulations: {summary.get('simulations', 'n/a')}",
        f"Model version: {summary.get('model_version', 'unknown')}",
        "",
        "## Top Title Probabilities",
        "",
        "| Team | Title Probability |",
        "| --- | ---: |",
    ]
    for item in top_champions:
        rows.append(f"| {item['team']} | {_pct(float(item['title_probability']))} |")
    rows.extend(["", "## Group Winners", "", "| Group | Most Likely Winner | Win Group |", "| --- | --- | ---: |"])
    for group_name in sorted(group_predictions):
        leader = group_predictions[group_name][0]
        rows.append(f"| {group_name} | {leader['team']} | {_pct(float(leader['win_group_probability']))} |")
    rows.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Group stage is simulated from scratch using match-level win/draw/loss probabilities.",
            "- Top two teams from each group qualify, plus the eight strongest third-place teams.",
            f"- {summary.get('placement_note', PLACEMENT_NOTE)}",
            "- Predictions are probabilistic estimates, not guarantees.",
        ]
    )
    return "\n".join(rows) + "\n"


def _team_summary_row(team: str, group: str, counts: dict[str, int], n_simulations: int) -> dict[str, Any]:
    row: dict[str, Any] = {"team": team, "group": group}
    probability_names = {
        "win_group_count": "win_group_probability",
        "finish_second_count": "finish_second_probability",
        "finish_third_count": "finish_third_probability",
        "qualify_from_group_count": "qualify_probability",
        "best_third_place_qualifier_count": "best_third_place_probability",
        "reach_round_of_32_count": "round_of_32_probability",
        "reach_round_of_16_count": "round_of_16_probability",
        "reach_quarterfinal_count": "quarterfinal_probability",
        "reach_semifinal_count": "semifinal_probability",
        "reach_final_count": "final_probability",
        "win_tournament_count": "title_probability",
    }
    for count_name, probability_name in probability_names.items():
        row[count_name] = counts[count_name]
        row[probability_name] = counts[count_name] / n_simulations
    row["win_tournament_probability"] = row["title_probability"]
    row["reach_final_probability"] = row["final_probability"]
    return row


def _group_predictions(groups: dict[str, list[str]], rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_team = {str(row["team"]): row for row in rows}
    output = {}
    for group_name in sorted(groups):
        group_rows = [by_team[team] for team in groups[group_name]]
        group_rows.sort(key=lambda row: float(row["win_group_probability"]), reverse=True)
        output[group_name] = group_rows
    return output


def _most_likely_group_winners(group_predictions: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        group_name: {
            "team": rows[0]["team"],
            "win_group_probability": rows[0]["win_group_probability"],
        }
        for group_name, rows in group_predictions.items()
        if rows
    }


def _most_likely_qualifiers_by_group(group_predictions: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for group_name, rows in group_predictions.items():
        qualifiers = sorted(rows, key=lambda row: float(row["qualify_probability"]), reverse=True)
        output[group_name] = [
            {"team": row["team"], "qualify_probability": row["qualify_probability"]}
            for row in qualifiers[:3]
        ]
    return output


def _dark_horses(rows: list[dict[str, Any]], seed_order: list[str], threshold: float = 0.015) -> list[dict[str, Any]]:
    seed_rank = {team: idx + 1 for idx, team in enumerate(seed_order)}
    candidates = [
        {
            "team": row["team"],
            "group": row["group"],
            "seed_rank": seed_rank[str(row["team"])],
            "title_probability": row["title_probability"],
        }
        for row in rows
        if seed_rank[str(row["team"])] > 16 and float(row["title_probability"]) >= threshold
    ]
    candidates.sort(key=lambda row: float(row["title_probability"]), reverse=True)
    return candidates[:8]


def _volatile_teams(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        qualify_probability = float(row["qualify_probability"])
        variance = qualify_probability * (1.0 - qualify_probability)
        candidates.append(
            {
                "team": row["team"],
                "group": row["group"],
                "qualify_probability": qualify_probability,
                "qualification_variance": variance,
            }
        )
    candidates.sort(key=lambda row: float(row["qualification_variance"]), reverse=True)
    return candidates[:8]


def _pair_teams(teams: list[str], stage: str) -> list[dict[str, str]]:
    if len(teams) % 2 != 0:
        raise TournamentConfigError(f"Cannot build {stage} with odd number of teams: {len(teams)}")
    return [
        {"team_a": teams[idx], "team_b": teams[idx + 1], "stage": stage}
        for idx in range(0, len(teams), 2)
    ]


def _teams_from_fixtures(fixtures: list[dict[str, Any]]) -> list[str]:
    teams = []
    for fixture in fixtures:
        teams.extend([str(fixture["team_a"]), str(fixture["team_b"])])
    return teams


def _with_group(row: dict[str, Any], group_name: str) -> dict[str, Any]:
    copied = dict(row)
    copied["group"] = group_name
    return copied


def _find_opponent_index(rows: list[dict[str, Any]], group_name: str) -> int:
    for idx, row in enumerate(rows):
        if str(row.get("group")) != group_name:
            return idx
    return 0


def _separate_same_group_pairings(fixtures: list[dict[str, Any]]) -> None:
    for idx, fixture in enumerate(fixtures):
        if fixture["team_a_group"] != fixture["team_b_group"]:
            continue
        for swap_idx, candidate in enumerate(fixtures):
            if idx == swap_idx:
                continue
            if (
                fixture["team_a_group"] != candidate["team_b_group"]
                and candidate["team_a_group"] != fixture["team_b_group"]
            ):
                fixture["team_b"], candidate["team_b"] = candidate["team_b"], fixture["team_b"]
                fixture["team_b_group"], candidate["team_b_group"] = candidate["team_b_group"], fixture["team_b_group"]
                break


def _table_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -int(row["points"]),
        -int(row["goal_difference"]),
        -int(row["goals_for"]),
        int(row["tie_breaker"]),
    )


def _points(score_a: int, score_b: int) -> tuple[int, int]:
    if score_a > score_b:
        return 3, 0
    if score_b > score_a:
        return 0, 3
    return 1, 1


def _stable_tie_breaker(team: str, seed: int) -> int:
    return _stable_int(f"{seed}:{team}")


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def _rng(random_state: random.Random | int | None) -> random.Random:
    if isinstance(random_state, random.Random):
        return random_state
    return random.Random(random_state)


def _available_team_names(teams: pd.DataFrame | None) -> set[str] | None:
    if teams is None:
        return None
    if teams.empty or "team" not in teams:
        return set()
    return {str(team).strip() for team in teams["team"].dropna() if str(team).strip()}


def _normalize_team_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return "".join(char for char in ascii_text.lower() if char.isalnum())


def _pct(value: float) -> str:
    return f"{value:.1%}"
