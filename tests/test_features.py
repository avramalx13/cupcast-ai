import pandas as pd
import pytest

from cupcast.prediction_engine.features import FEATURE_COLUMNS, assert_no_future_leakage, build_feature_table


def test_feature_engineering_creates_required_columns() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 2,
                "team_b_score": 1,
                "tournament": "Test Cup",
                "neutral": 1,
                "stage": "group",
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {"team": "France", "confederation": "UEFA", "initial_elo": 2040, "fifa_rank": 2},
            {"team": "Brazil", "confederation": "CONMEBOL", "initial_elo": 2050, "fifa_rank": 5},
        ]
    )

    features = build_feature_table(matches, teams)

    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert features.loc[0, "label"] == "A_WIN"


def test_feature_engineering_uses_pre_match_elo_and_no_future_matches() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 2,
                "team_b_score": 0,
                "tournament": "Test Cup",
                "neutral": 1,
                "stage": "group",
            },
            {
                "date": pd.Timestamp("2020-02-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 0,
                "team_b_score": 3,
                "tournament": "Test Cup",
                "neutral": 1,
                "stage": "group",
            },
            {
                "date": pd.Timestamp("2020-03-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 1,
                "team_b_score": 1,
                "tournament": "Test Cup",
                "neutral": 1,
                "stage": "group",
            },
        ]
    )
    teams = pd.DataFrame(
        [
            {"team": "France", "confederation": "UEFA", "initial_elo": 1500, "fifa_rank": 2},
            {"team": "Brazil", "confederation": "CONMEBOL", "initial_elo": 1500, "fifa_rank": 5},
        ]
    )

    features = build_feature_table(matches, teams, elo_k=20)

    assert features.loc[0, "elo_a"] == pytest.approx(1500)
    assert features.loc[0, "elo_b"] == pytest.approx(1500)
    assert features.loc[1, "elo_a"] > 1500
    assert features.loc[1, "elo_b"] < 1500
    assert_no_future_leakage(matches, teams, elo_k=20)


def test_feature_group_flags_neutralize_disabled_groups() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 2,
                "team_b_score": 0,
                "tournament": "Friendly",
                "neutral": 0,
                "stage": "friendly",
                "country": "France",
            },
            {
                "date": pd.Timestamp("2020-02-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 1,
                "team_b_score": 1,
                "tournament": "FIFA World Cup",
                "neutral": 0,
                "stage": "group",
                "country": "France",
            },
        ]
    )
    teams = pd.DataFrame(
        [
            {"team": "France", "confederation": "UEFA", "initial_elo": 1500, "fifa_rank": 2},
            {"team": "Brazil", "confederation": "CONMEBOL", "initial_elo": 1500, "fifa_rank": 5},
        ]
    )

    features = build_feature_table(
        matches,
        teams,
        feature_flags={"schedule": False, "tournament_context": False, "elo_history": False},
    )

    assert features.loc[1, "days_since_last_match_a"] == pytest.approx(999)
    assert features.loc[1, "matches_last_90_days_a"] == 0
    assert features.loc[1, "tournament_importance"] == 1
    assert features.loc[1, "home_advantage_flag"] == 0
    assert features.loc[1, "elo_trend_5_matches_a"] == pytest.approx(0)


def test_competitive_form_ignores_friendlies_until_competitive_match() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 5,
                "team_b_score": 0,
                "tournament": "Friendly",
                "neutral": 1,
                "stage": "friendly",
            },
            {
                "date": pd.Timestamp("2020-02-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 1,
                "team_b_score": 1,
                "tournament": "FIFA World Cup",
                "neutral": 1,
                "stage": "group",
            },
            {
                "date": pd.Timestamp("2020-03-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 0,
                "team_b_score": 1,
                "tournament": "FIFA World Cup qualification",
                "neutral": 1,
                "stage": "qualifier",
            },
        ]
    )
    teams = pd.DataFrame(
        [
            {"team": "France", "confederation": "UEFA", "initial_elo": 1500, "fifa_rank": 2},
            {"team": "Brazil", "confederation": "CONMEBOL", "initial_elo": 1500, "fifa_rank": 5},
        ]
    )

    features = build_feature_table(matches, teams)

    assert features.loc[1, "team_a_recent_form"] == pytest.approx(1.0)
    assert features.loc[1, "team_a_competitive_form_last_5"] == pytest.approx(0.5)
    assert features.loc[2, "team_a_competitive_form_last_5"] == pytest.approx(1.0 / 3.0)
    assert features.loc[2, "team_a_competitive_goal_diff_last_5"] == pytest.approx(0.0)
