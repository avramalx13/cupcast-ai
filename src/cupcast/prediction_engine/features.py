from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from pandas.testing import assert_series_equal

from .data_sources import normalize_player_events
from .elo import label_from_score, match_result_for_team_a, update_elo


FEATURE_COLUMNS = [
    "elo_a",
    "elo_b",
    "elo_diff",
    "fifa_rank_a",
    "fifa_rank_b",
    "rank_diff",
    "team_a_recent_form",
    "team_b_recent_form",
    "recent_form_diff",
    "team_a_goals_scored_last_5",
    "team_b_goals_scored_last_5",
    "team_a_goals_conceded_last_5",
    "team_b_goals_conceded_last_5",
    "team_a_goal_diff_last_5",
    "team_b_goal_diff_last_5",
    "goal_diff_form_diff",
    "team_a_competitive_form_last_5",
    "team_b_competitive_form_last_5",
    "competitive_form_diff",
    "team_a_competitive_goal_diff_last_5",
    "team_b_competitive_goal_diff_last_5",
    "competitive_goal_diff_diff",
    "neutral",
    "stage_encoded",
    "elo_external_a",
    "elo_external_b",
    "elo_external_diff",
    "fifa_rank_external_a",
    "fifa_rank_external_b",
    "fifa_rank_external_diff",
    "fifa_points_a",
    "fifa_points_b",
    "fifa_points_diff",
    "days_since_last_match_a",
    "days_since_last_match_b",
    "rest_days_diff",
    "matches_last_30_days_a",
    "matches_last_30_days_b",
    "matches_last_90_days_a",
    "matches_last_90_days_b",
    "weighted_form_last_5_a",
    "weighted_form_last_5_b",
    "weighted_form_diff",
    "weighted_goal_diff_last_5_a",
    "weighted_goal_diff_last_5_b",
    "goal_events_last_90_days_a",
    "goal_events_last_90_days_b",
    "goal_events_last_90_days_diff",
    "injury_events_last_30_days_a",
    "injury_events_last_30_days_b",
    "injury_events_last_30_days_diff",
    "opponent_adjusted_form_a",
    "opponent_adjusted_form_b",
    "opponent_adjusted_form_diff",
    "tournament_importance",
    "is_friendly",
    "is_qualifier",
    "is_world_cup",
    "is_continental_tournament",
    "is_knockout",
    "home_advantage_flag",
    "host_country_advantage_flag",
    "same_confederation_match",
    "elo_trend_5_matches_a",
    "elo_trend_5_matches_b",
    "elo_trend_diff",
    "elo_volatility_10_matches_a",
    "elo_volatility_10_matches_b",
    "elo_volatility_diff",
    "avg_elo_last_5_a",
    "avg_elo_last_5_b",
    "avg_elo_diff",
    "max_elo_last_10_a",
    "max_elo_last_10_b",
]

STAGE_ENCODING = {
    "friendly": 0,
    "qualifier": 1,
    "group": 2,
    "round_of_16": 3,
    "quarterfinal": 4,
    "semifinal": 5,
    "third_place": 6,
    "final": 7,
}

UNKNOWN_FIFA_RANK_SENTINEL = 999


@dataclass
class TeamState:
    rating: float
    fifa_rank: int
    confederation: str
    recent_points: deque[float]
    goals_for: deque[int]
    goals_against: deque[int]
    goal_diffs: deque[int]
    competitive_points: deque[float]
    competitive_goal_diffs: deque[int]
    match_dates: deque[pd.Timestamp]
    opponent_adjusted_points: deque[float]
    elo_history: deque[float]
    last_match_date: pd.Timestamp | None = None


TeamStates = dict[str, TeamState]
RatingLookup = dict[str, pd.DataFrame]
RatingFrameOrLookup = pd.DataFrame | RatingLookup | None


def build_feature_table(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    recent_window: int = 5,
    elo_k: float = 28.0,
    external_elo_ratings: pd.DataFrame | None = None,
    external_fifa_rankings: pd.DataFrame | None = None,
    player_events: pd.DataFrame | None = None,
    feature_flags: dict[str, bool] | None = None,
) -> pd.DataFrame:
    if recent_window <= 0:
        raise ValueError("recent_window must be positive")

    states = initialize_team_states(teams, recent_window=recent_window)
    external_elo_lookup = _prepare_rating_lookup(external_elo_ratings)
    external_fifa_lookup = _prepare_rating_lookup(external_fifa_rankings)
    player_event_lookup = _prepare_player_event_lookup(player_events)
    rows: list[dict[str, object]] = []

    for match in matches.sort_values("date").itertuples(index=False):
        team_a = str(match.team_a)
        team_b = str(match.team_b)
        _ensure_team(states, team_a, teams, recent_window)
        _ensure_team(states, team_b, teams, recent_window)

        features = features_from_state(
            states=states,
            team_a=team_a,
            team_b=team_b,
            neutral=int(match.neutral),
            stage=str(match.stage),
            match_date=pd.to_datetime(match.date),
            tournament=str(match.tournament),
            country=str(getattr(match, "country", "") or ""),
            external_elo_ratings=external_elo_lookup,
            external_fifa_rankings=external_fifa_lookup,
            player_event_lookup=player_event_lookup,
            feature_flags=feature_flags,
        )
        features.update(
            {
                "date": match.date,
                "team_a": team_a,
                "team_b": team_b,
                "tournament": str(match.tournament),
                "stage": str(match.stage),
                "label": label_from_score(int(match.team_a_score), int(match.team_b_score)),
            }
        )
        rows.append(features)
        _apply_result(
            states=states,
            team_a=team_a,
            team_b=team_b,
            score_a=int(match.team_a_score),
            score_b=int(match.team_b_score),
            match_date=pd.to_datetime(match.date),
            tournament=str(match.tournament),
            stage=str(match.stage),
            elo_k=elo_k,
        )

    return pd.DataFrame(rows)


def assert_no_future_leakage(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    recent_window: int = 5,
    elo_k: float = 28.0,
    max_checks: int | None = None,
) -> None:
    sorted_matches = matches.sort_values("date").reset_index(drop=True)
    full_features = build_feature_table(
        sorted_matches,
        teams,
        recent_window=recent_window,
        elo_k=elo_k,
    ).reset_index(drop=True)
    check_count = len(sorted_matches) if max_checks is None else min(len(sorted_matches), max_checks)
    for idx in range(check_count):
        prefix_matches = sorted_matches.iloc[: idx + 1].copy()
        prefix_features = build_feature_table(
            prefix_matches,
            teams,
            recent_window=recent_window,
            elo_k=elo_k,
        ).reset_index(drop=True)
        assert_series_equal(
            full_features.loc[idx, FEATURE_COLUMNS],
            prefix_features.loc[idx, FEATURE_COLUMNS],
            check_names=False,
            rtol=1e-10,
            atol=1e-10,
        )


def initialize_team_states(teams: pd.DataFrame, recent_window: int = 5) -> TeamStates:
    return {
        str(row.team): TeamState(
            rating=float(row.initial_elo),
            fifa_rank=_rank_or_unknown(row.fifa_rank),
            confederation=str(getattr(row, "confederation", "UNKNOWN") or "UNKNOWN"),
            recent_points=deque(maxlen=recent_window),
            goals_for=deque(maxlen=recent_window),
            goals_against=deque(maxlen=recent_window),
            goal_diffs=deque(maxlen=recent_window),
            competitive_points=deque(maxlen=recent_window),
            competitive_goal_diffs=deque(maxlen=recent_window),
            match_dates=deque(maxlen=20),
            opponent_adjusted_points=deque(maxlen=recent_window),
            elo_history=deque([float(row.initial_elo)], maxlen=20),
        )
        for row in teams.itertuples(index=False)
    }


def build_current_states(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    recent_window: int = 5,
    elo_k: float = 28.0,
) -> TeamStates:
    states = initialize_team_states(teams, recent_window=recent_window)
    for match in matches.sort_values("date").itertuples(index=False):
        team_a = str(match.team_a)
        team_b = str(match.team_b)
        _ensure_team(states, team_a, teams, recent_window)
        _ensure_team(states, team_b, teams, recent_window)
        _apply_result(
            states=states,
            team_a=team_a,
            team_b=team_b,
            score_a=int(match.team_a_score),
            score_b=int(match.team_b_score),
            match_date=pd.to_datetime(match.date),
            tournament=str(match.tournament),
            stage=str(match.stage),
            elo_k=elo_k,
        )
    return states


def features_for_match(
    team_a: str,
    team_b: str,
    teams: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    neutral: bool = True,
    stage: str = "group",
    recent_window: int = 5,
    elo_k: float = 28.0,
    current_states: TeamStates | None = None,
    external_elo_ratings: pd.DataFrame | None = None,
    external_fifa_rankings: pd.DataFrame | None = None,
    player_events: pd.DataFrame | None = None,
    feature_flags: dict[str, bool] | None = None,
) -> pd.DataFrame:
    if current_states is not None:
        states = current_states
    elif matches is not None:
        states = build_current_states(matches, teams, recent_window=recent_window, elo_k=elo_k)
    else:
        states = initialize_team_states(teams, recent_window=recent_window)
    _ensure_team(states, team_a, teams, recent_window)
    _ensure_team(states, team_b, teams, recent_window)
    external_elo_lookup = _prepare_rating_lookup(external_elo_ratings)
    external_fifa_lookup = _prepare_rating_lookup(external_fifa_rankings)
    player_event_lookup = _prepare_player_event_lookup(player_events)
    return pd.DataFrame(
        [
            features_from_state(
                states=states,
                team_a=team_a,
                team_b=team_b,
                neutral=int(neutral),
                stage=stage,
                match_date=pd.Timestamp.today(),
                tournament=stage,
                country="",
                external_elo_ratings=external_elo_lookup,
                external_fifa_rankings=external_fifa_lookup,
                player_event_lookup=player_event_lookup,
                feature_flags=feature_flags,
            )
        ]
    )


def features_from_state(
    states: TeamStates,
    team_a: str,
    team_b: str,
    neutral: int,
    stage: str,
    match_date: pd.Timestamp,
    tournament: str = "",
    country: str = "",
    external_elo_ratings: RatingFrameOrLookup = None,
    external_fifa_rankings: RatingFrameOrLookup = None,
    player_event_lookup: dict[str, pd.DataFrame] | None = None,
    feature_flags: dict[str, bool] | None = None,
) -> dict[str, float | int]:
    feature_flags = feature_flags or {}
    state_a = states[team_a]
    state_b = states[team_b]
    form_a = _recent_form(state_a.recent_points)
    form_b = _recent_form(state_b.recent_points)
    gf_a = _average(state_a.goals_for)
    gf_b = _average(state_b.goals_for)
    ga_a = _average(state_a.goals_against)
    ga_b = _average(state_b.goals_against)
    gd_a = _average_float(state_a.goal_diffs, default=0.0)
    gd_b = _average_float(state_b.goal_diffs, default=0.0)
    competitive_form_a = _recent_form(state_a.competitive_points)
    competitive_form_b = _recent_form(state_b.competitive_points)
    competitive_gd_a = _average_float(state_a.competitive_goal_diffs, default=0.0)
    competitive_gd_b = _average_float(state_b.competitive_goal_diffs, default=0.0)
    stage_key = stage.strip().lower().replace(" ", "_")
    match_ts = pd.to_datetime(match_date)
    elo_external_a = _external_value_or_default(
        _latest_rating_value_before(
            team_a,
            match_ts,
            external_elo_ratings if feature_flags.get("external_ratings", True) else None,
            value_column="elo",
        ),
        state_a.rating,
    )
    elo_external_b = _external_value_or_default(
        _latest_rating_value_before(
            team_b,
            match_ts,
            external_elo_ratings if feature_flags.get("external_ratings", True) else None,
            value_column="elo",
        ),
        state_b.rating,
    )
    fifa_row_a = _latest_rating_row_before(team_a, match_ts, external_fifa_rankings if feature_flags.get("external_ratings", True) else None)
    fifa_row_b = _latest_rating_row_before(team_b, match_ts, external_fifa_rankings if feature_flags.get("external_ratings", True) else None)
    fifa_rank_a = _numeric_or_default(fifa_row_a.get("rank") if fifa_row_a else None, state_a.fifa_rank)
    fifa_rank_b = _numeric_or_default(fifa_row_b.get("rank") if fifa_row_b else None, state_b.fifa_rank)
    fifa_points_a = _numeric_or_default(fifa_row_a.get("points") if fifa_row_a else None, 0.0)
    fifa_points_b = _numeric_or_default(fifa_row_b.get("points") if fifa_row_b else None, 0.0)
    days_a = _days_since_last_match(state_a, match_ts)
    days_b = _days_since_last_match(state_b, match_ts)
    weighted_form_a = _weighted_form(state_a.recent_points)
    weighted_form_b = _weighted_form(state_b.recent_points)
    weighted_gd_a = _weighted_average(state_a.goal_diffs, default=0.0)
    weighted_gd_b = _weighted_average(state_b.goal_diffs, default=0.0)
    goal_events_last_90_days_a = _count_player_events_before(
        team_a,
        match_ts,
        player_event_lookup,
        "goal",
        days=90,
    )
    goal_events_last_90_days_b = _count_player_events_before(
        team_b,
        match_ts,
        player_event_lookup,
        "goal",
        days=90,
    )
    injury_events_last_30_days_a = _count_player_events_before(
        team_a,
        match_ts,
        player_event_lookup,
        "injury",
        days=30,
    )
    injury_events_last_30_days_b = _count_player_events_before(
        team_b,
        match_ts,
        player_event_lookup,
        "injury",
        days=30,
    )
    opponent_form_a = _recent_form(state_a.opponent_adjusted_points)
    opponent_form_b = _recent_form(state_b.opponent_adjusted_points)
    importance = tournament_importance(tournament, stage)
    host_advantage = _host_country_advantage(team_a, team_b, country)
    same_confed = int(
        state_a.confederation
        and state_b.confederation
        and state_a.confederation != "UNKNOWN"
        and state_a.confederation == state_b.confederation
    )
    elo_trend_a = _elo_trend(state_a.elo_history, window=5)
    elo_trend_b = _elo_trend(state_b.elo_history, window=5)
    elo_vol_a = _elo_volatility(state_a.elo_history, window=10)
    elo_vol_b = _elo_volatility(state_b.elo_history, window=10)
    avg_elo_a = _recent_elo_average(state_a.elo_history, window=5, default=state_a.rating)
    avg_elo_b = _recent_elo_average(state_b.elo_history, window=5, default=state_b.rating)

    features = {
        "elo_a": state_a.rating,
        "elo_b": state_b.rating,
        "elo_diff": state_a.rating - state_b.rating,
        "fifa_rank_a": state_a.fifa_rank,
        "fifa_rank_b": state_b.fifa_rank,
        "rank_diff": state_b.fifa_rank - state_a.fifa_rank,
        "team_a_recent_form": form_a,
        "team_b_recent_form": form_b,
        "recent_form_diff": form_a - form_b,
        "team_a_goals_scored_last_5": gf_a,
        "team_b_goals_scored_last_5": gf_b,
        "team_a_goals_conceded_last_5": ga_a,
        "team_b_goals_conceded_last_5": ga_b,
        "team_a_goal_diff_last_5": gd_a,
        "team_b_goal_diff_last_5": gd_b,
        "goal_diff_form_diff": gd_a - gd_b,
        "team_a_competitive_form_last_5": competitive_form_a,
        "team_b_competitive_form_last_5": competitive_form_b,
        "competitive_form_diff": competitive_form_a - competitive_form_b,
        "team_a_competitive_goal_diff_last_5": competitive_gd_a,
        "team_b_competitive_goal_diff_last_5": competitive_gd_b,
        "competitive_goal_diff_diff": competitive_gd_a - competitive_gd_b,
        "neutral": int(neutral),
        "stage_encoded": STAGE_ENCODING.get(stage_key, 0),
        "elo_external_a": elo_external_a,
        "elo_external_b": elo_external_b,
        "elo_external_diff": elo_external_a - elo_external_b,
        "fifa_rank_external_a": fifa_rank_a,
        "fifa_rank_external_b": fifa_rank_b,
        "fifa_rank_external_diff": fifa_rank_b - fifa_rank_a,
        "fifa_points_a": fifa_points_a,
        "fifa_points_b": fifa_points_b,
        "fifa_points_diff": fifa_points_a - fifa_points_b,
        "days_since_last_match_a": days_a,
        "days_since_last_match_b": days_b,
        "rest_days_diff": days_a - days_b,
        "matches_last_30_days_a": _matches_within_days(state_a.match_dates, match_ts, days=30),
        "matches_last_30_days_b": _matches_within_days(state_b.match_dates, match_ts, days=30),
        "matches_last_90_days_a": _matches_within_days(state_a.match_dates, match_ts, days=90),
        "matches_last_90_days_b": _matches_within_days(state_b.match_dates, match_ts, days=90),
        "weighted_form_last_5_a": weighted_form_a,
        "weighted_form_last_5_b": weighted_form_b,
        "weighted_form_diff": weighted_form_a - weighted_form_b,
        "weighted_goal_diff_last_5_a": weighted_gd_a,
        "weighted_goal_diff_last_5_b": weighted_gd_b,
        "goal_events_last_90_days_a": goal_events_last_90_days_a,
        "goal_events_last_90_days_b": goal_events_last_90_days_b,
        "goal_events_last_90_days_diff": goal_events_last_90_days_a - goal_events_last_90_days_b,
        "injury_events_last_30_days_a": injury_events_last_30_days_a,
        "injury_events_last_30_days_b": injury_events_last_30_days_b,
        "injury_events_last_30_days_diff": injury_events_last_30_days_a - injury_events_last_30_days_b,
        "opponent_adjusted_form_a": opponent_form_a,
        "opponent_adjusted_form_b": opponent_form_b,
        "opponent_adjusted_form_diff": opponent_form_a - opponent_form_b,
        "tournament_importance": importance,
        "is_friendly": int(_is_friendly(tournament, stage)),
        "is_qualifier": int(_is_qualifier(tournament, stage)),
        "is_world_cup": int(_is_world_cup(tournament)),
        "is_continental_tournament": int(_is_continental_tournament(tournament)),
        "is_knockout": int(_is_knockout_stage(stage)),
        "home_advantage_flag": int(not int(neutral)),
        "host_country_advantage_flag": host_advantage,
        "same_confederation_match": same_confed,
        "elo_trend_5_matches_a": elo_trend_a,
        "elo_trend_5_matches_b": elo_trend_b,
        "elo_trend_diff": elo_trend_a - elo_trend_b,
        "elo_volatility_10_matches_a": elo_vol_a,
        "elo_volatility_10_matches_b": elo_vol_b,
        "elo_volatility_diff": elo_vol_a - elo_vol_b,
        "avg_elo_last_5_a": avg_elo_a,
        "avg_elo_last_5_b": avg_elo_b,
        "avg_elo_diff": avg_elo_a - avg_elo_b,
        "max_elo_last_10_a": _recent_elo_max(state_a.elo_history, window=10, default=state_a.rating),
        "max_elo_last_10_b": _recent_elo_max(state_b.elo_history, window=10, default=state_b.rating),
    }
    _apply_feature_flags(features, feature_flags, state_a=state_a, state_b=state_b)
    return features


def _apply_result(
    states: TeamStates,
    team_a: str,
    team_b: str,
    score_a: int,
    score_b: int,
    match_date: pd.Timestamp,
    tournament: str,
    stage: str,
    elo_k: float,
) -> None:
    state_a = states[team_a]
    state_b = states[team_b]
    result_a = match_result_for_team_a(score_a, score_b)
    pre_rating_a = state_a.rating
    pre_rating_b = state_b.rating
    state_a.rating, state_b.rating = update_elo(
        state_a.rating,
        state_b.rating,
        result_a=result_a,
        k=elo_k,
    )

    points_a, points_b = _points(score_a, score_b)
    state_a.recent_points.append(points_a)
    state_b.recent_points.append(points_b)
    state_a.goals_for.append(score_a)
    state_a.goals_against.append(score_b)
    state_a.goal_diffs.append(score_a - score_b)
    state_b.goals_for.append(score_b)
    state_b.goals_against.append(score_a)
    state_b.goal_diffs.append(score_b - score_a)
    if not _is_friendly(tournament, stage):
        state_a.competitive_points.append(points_a)
        state_b.competitive_points.append(points_b)
        state_a.competitive_goal_diffs.append(score_a - score_b)
        state_b.competitive_goal_diffs.append(score_b - score_a)
    state_a.match_dates.append(match_date)
    state_b.match_dates.append(match_date)
    state_a.last_match_date = match_date
    state_b.last_match_date = match_date
    state_a.opponent_adjusted_points.append(points_a * max(0.5, pre_rating_b / 1500.0))
    state_b.opponent_adjusted_points.append(points_b * max(0.5, pre_rating_a / 1500.0))
    state_a.elo_history.append(state_a.rating)
    state_b.elo_history.append(state_b.rating)


def _ensure_team(
    states: TeamStates,
    team: str,
    teams: pd.DataFrame,
    recent_window: int,
) -> None:
    if team in states:
        return
    row = teams.loc[teams["team"] == team]
    if row.empty:
        raise ValueError(f"Team not found in teams.csv: {team}")
    record = row.iloc[0]
    states[team] = TeamState(
        rating=float(record["initial_elo"]),
        fifa_rank=_rank_or_unknown(record["fifa_rank"]),
        confederation=str(record.get("confederation", "UNKNOWN") or "UNKNOWN"),
        recent_points=deque(maxlen=recent_window),
        goals_for=deque(maxlen=recent_window),
        goals_against=deque(maxlen=recent_window),
        goal_diffs=deque(maxlen=recent_window),
        competitive_points=deque(maxlen=recent_window),
        competitive_goal_diffs=deque(maxlen=recent_window),
        match_dates=deque(maxlen=20),
        opponent_adjusted_points=deque(maxlen=recent_window),
        elo_history=deque([float(record["initial_elo"])], maxlen=20),
    )


def _points(score_a: int, score_b: int) -> tuple[float, float]:
    if score_a > score_b:
        return 3.0, 0.0
    if score_a < score_b:
        return 0.0, 3.0
    return 1.0, 1.0


def _recent_form(points: Iterable[float]) -> float:
    values = list(points)
    if not values:
        return 0.5
    return sum(values) / (3.0 * len(values))


def _average(values: Iterable[int]) -> float:
    items = list(values)
    if not items:
        return 1.0
    return float(sum(items) / len(items))


def _average_float(values: Iterable[float], default: float) -> float:
    items = [float(value) for value in values]
    if not items:
        return float(default)
    return float(sum(items) / len(items))


def _weighted_form(points: Iterable[float]) -> float:
    return _weighted_average(points, default=1.5) / 3.0


def _weighted_average(values: Iterable[float], default: float) -> float:
    items = [float(value) for value in values]
    if not items:
        return float(default)
    weights = list(range(1, len(items) + 1))
    return float(sum(value * weight for value, weight in zip(items, weights, strict=False)) / sum(weights))


def _days_since_last_match(state: TeamState, match_date: pd.Timestamp) -> float:
    if state.last_match_date is None or pd.isna(match_date):
        return 999.0
    return float(max(0, (match_date - state.last_match_date).days))


def _matches_within_days(match_dates: Iterable[pd.Timestamp], match_date: pd.Timestamp, days: int) -> int:
    if pd.isna(match_date):
        return 0
    return int(sum(0 < (match_date - date).days <= days for date in match_dates if not pd.isna(date)))


def tournament_importance(tournament: str, stage: str = "") -> int:
    text = f"{tournament} {stage}".lower()
    if "world cup" in text and "qualif" not in text:
        return 5
    if _is_continental_tournament(tournament) and "qualif" not in text:
        return 4
    if "world cup" in text and "qualif" in text:
        return 3
    if "qualif" in text:
        return 2
    return 1


def _is_friendly(tournament: str, stage: str = "") -> bool:
    return "friendly" in f"{tournament} {stage}".lower()


def _is_qualifier(tournament: str, stage: str = "") -> bool:
    return "qualif" in f"{tournament} {stage}".lower()


def _is_world_cup(tournament: str) -> bool:
    text = tournament.lower()
    return "world cup" in text and "qualif" not in text


def _is_continental_tournament(tournament: str) -> bool:
    text = tournament.lower()
    continental_markers = [
        "euro",
        "copa america",
        "africa cup",
        "asian cup",
        "gold cup",
        "nations cup",
        "concacaf",
        "ofc",
    ]
    return any(marker in text for marker in continental_markers)


def _is_knockout_stage(stage: str) -> bool:
    stage_key = stage.strip().lower().replace(" ", "_")
    return stage_key in {"round_of_16", "quarterfinal", "semifinal", "third_place", "final"}


def _host_country_advantage(team_a: str, team_b: str, country: str) -> int:
    host = country.strip().casefold()
    if not host:
        return 0
    if host == team_a.strip().casefold():
        return 1
    if host == team_b.strip().casefold():
        return -1
    return 0


def _elo_trend(values: Iterable[float], window: int) -> float:
    items = list(values)[-window:]
    if len(items) < 2:
        return 0.0
    return float(items[-1] - items[0])


def _elo_volatility(values: Iterable[float], window: int) -> float:
    items = list(values)[-window:]
    if len(items) < 2:
        return 0.0
    diffs = [items[idx] - items[idx - 1] for idx in range(1, len(items))]
    mean = sum(diffs) / len(diffs)
    return float((sum((value - mean) ** 2 for value in diffs) / len(diffs)) ** 0.5)


def _recent_elo_average(values: Iterable[float], window: int, default: float) -> float:
    items = list(values)[-window:]
    if not items:
        return float(default)
    return float(sum(items) / len(items))


def _recent_elo_max(values: Iterable[float], window: int, default: float) -> float:
    items = list(values)[-window:]
    if not items:
        return float(default)
    return float(max(items))


def _apply_feature_flags(
    features: dict[str, float | int],
    feature_flags: dict[str, bool],
    state_a: TeamState,
    state_b: TeamState,
) -> None:
    if not feature_flags.get("external_ratings", True):
        features.update(
            {
                "elo_external_a": float(state_a.rating),
                "elo_external_b": float(state_b.rating),
                "elo_external_diff": float(state_a.rating - state_b.rating),
                "fifa_rank_external_a": float(state_a.fifa_rank),
                "fifa_rank_external_b": float(state_b.fifa_rank),
                "fifa_rank_external_diff": float(state_b.fifa_rank - state_a.fifa_rank),
                "fifa_points_a": 0.0,
                "fifa_points_b": 0.0,
                "fifa_points_diff": 0.0,
            }
        )
    if not feature_flags.get("schedule", True):
        features.update(
            {
                "days_since_last_match_a": 999.0,
                "days_since_last_match_b": 999.0,
                "rest_days_diff": 0.0,
                "matches_last_30_days_a": 0,
                "matches_last_30_days_b": 0,
                "matches_last_90_days_a": 0,
                "matches_last_90_days_b": 0,
            }
        )
    if not feature_flags.get("tournament_context", True):
        features.update(
            {
                "tournament_importance": 1,
                "is_friendly": 0,
                "is_qualifier": 0,
                "is_world_cup": 0,
                "is_continental_tournament": 0,
                "is_knockout": 0,
                "home_advantage_flag": 0,
                "host_country_advantage_flag": 0,
                "same_confederation_match": 0,
            }
        )
    if not feature_flags.get("elo_history", True):
        features.update(
            {
                "elo_trend_5_matches_a": 0.0,
                "elo_trend_5_matches_b": 0.0,
                "elo_trend_diff": 0.0,
                "elo_volatility_10_matches_a": 0.0,
                "elo_volatility_10_matches_b": 0.0,
                "elo_volatility_diff": 0.0,
                "avg_elo_last_5_a": float(state_a.rating),
                "avg_elo_last_5_b": float(state_b.rating),
                "avg_elo_diff": float(state_a.rating - state_b.rating),
                "max_elo_last_10_a": float(state_a.rating),
                "max_elo_last_10_b": float(state_b.rating),
            }
        )


def _prepare_player_event_lookup(player_events: pd.DataFrame | None) -> dict[str, pd.DataFrame] | None:
    if player_events is None or player_events.empty:
        return None

    frame = player_events.copy()
    if "event_category" not in frame.columns:
        frame = normalize_player_events(frame)
    if not all(column in frame.columns for column in ("date", "team", "event_category")):
        return None

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["team"] = frame["team"].astype("string").str.strip().str.casefold()
    frame["event_category"] = frame["event_category"].astype("string").str.strip().str.lower()
    frame = frame.dropna(subset=["date", "team"])

    lookup: dict[str, pd.DataFrame] = {}
    for team, rows in frame.groupby("team"):
        if not team:
            continue
        lookup[str(team)] = rows.sort_values("date").reset_index(drop=True)
    return lookup


def _count_player_events_before(
    team: str,
    match_date: pd.Timestamp,
    player_event_lookup: dict[str, pd.DataFrame] | None,
    event_category: str,
    days: int,
) -> int:
    if player_event_lookup is None:
        return 0
    if pd.isna(match_date):
        return 0

    team_key = str(team).strip().casefold()
    events = player_event_lookup.get(team_key)
    if events is None or events.empty:
        return 0

    match_ts = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(match_ts):
        return 0

    cutoff = match_ts - pd.Timedelta(days=days)
    mask = (
        (events["date"] < match_ts)
        & (events["date"] >= cutoff)
        & (events["event_category"].astype("string").str.lower() == str(event_category).strip().lower())
    )
    return int(mask.sum())


def _external_value_or_default(value: float | None, default: float) -> float:
    return float(default if value is None else value)


def _prepare_rating_lookup(ratings: RatingFrameOrLookup) -> RatingLookup | None:
    if ratings is None:
        return None
    if isinstance(ratings, dict):
        return ratings
    if ratings.empty or "date" not in ratings or "team" not in ratings:
        return None
    cached = ratings.attrs.get("_cupcast_rating_lookup")
    if isinstance(cached, dict):
        return cached
    frame = ratings.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "team"])
    lookup: RatingLookup = {}
    for team, rows in frame.groupby(frame["team"].astype(str).str.strip().str.casefold()):
        if not team:
            continue
        lookup[str(team)] = rows.sort_values("date").reset_index(drop=True)
    ratings.attrs["_cupcast_rating_lookup"] = lookup
    return lookup


def _latest_rating_value_before(
    team: str,
    match_date: pd.Timestamp,
    ratings: RatingFrameOrLookup,
    value_column: str,
) -> float | None:
    row = _latest_rating_row_before(team, match_date, ratings)
    if not row:
        return None
    value = pd.to_numeric(row.get(value_column), errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def _latest_rating_row_before(
    team: str,
    match_date: pd.Timestamp,
    ratings: RatingFrameOrLookup,
) -> dict[str, object] | None:
    lookup = _prepare_rating_lookup(ratings)
    if not lookup:
        return None
    rows = lookup.get(str(team).strip().casefold())
    if rows is None or rows.empty:
        return None
    match_ts = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(match_ts):
        return None
    dates = pd.to_datetime(rows["date"], errors="coerce")
    row_index = int(dates.searchsorted(match_ts, side="left")) - 1
    if row_index < 0:
        return None
    return rows.iloc[row_index].to_dict()


def _numeric_or_default(value: object, default: float) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _rank_or_unknown(value: object) -> int:
    if pd.isna(value):
        return UNKNOWN_FIFA_RANK_SENTINEL
    return int(value)
