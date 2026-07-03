from __future__ import annotations

import pandas as pd
import pytest

from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.rating_sources import (
    LocalEloRatingSource,
    get_latest_rating_before,
    normalize_elo_ratings,
    normalize_fifa_rankings,
    validate_rating_sources,
)


def test_rating_aliases_normalize_and_latest_prior_rating_is_selected(tmp_path) -> None:
    elo_path = tmp_path / "elo.csv"
    pd.DataFrame(
        [
            {"date": "2020-01-01", "country": "France", "rating": 1900},
            {"date": "2021-01-01", "country": "France", "rating": 1950},
            {"date": "2022-01-01", "country": "France", "rating": 2010},
        ]
    ).to_csv(elo_path, index=False)

    ratings = LocalEloRatingSource(elo_path).load()

    assert list(ratings.columns) == ["date", "team", "elo", "rank", "points", "source"]
    assert get_latest_rating_before("France", "2021-06-01", ratings) == pytest.approx(1950)
    assert get_latest_rating_before("France", "2020-01-01", ratings) is None
    assert get_latest_rating_before("Brazil", "2021-06-01", ratings) is None


def test_future_rating_is_not_used_for_prior_match() -> None:
    ratings = normalize_elo_ratings(
        pd.DataFrame(
            [
                {"date": "2022-01-01", "nation": "France", "elo_rating": 2042},
                {"date": "2022-01-01", "nation": "Brazil", "elo_rating": 2056},
            ]
        )
    )
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-06-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 1,
                "team_b_score": 0,
                "tournament": "Friendly",
                "neutral": 1,
                "stage": "friendly",
            },
            {
                "date": pd.Timestamp("2022-06-01"),
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 0,
                "team_b_score": 0,
                "tournament": "Friendly",
                "neutral": 1,
                "stage": "friendly",
            },
        ]
    )
    teams = pd.DataFrame(
        [
            {"team": "France", "confederation": "UEFA", "initial_elo": 1500, "fifa_rank": 2},
            {"team": "Brazil", "confederation": "CONMEBOL", "initial_elo": 1500, "fifa_rank": 5},
        ]
    )

    features = build_feature_table(matches, teams, external_elo_ratings=ratings)

    assert features.loc[0, "elo_external_a"] == pytest.approx(1500)
    assert features.loc[1, "elo_external_a"] == pytest.approx(2042)
    assert features.loc[1, "elo_external_b"] == pytest.approx(2056)


def test_fifa_aliases_and_missing_files_warn_without_fake_data(tmp_path) -> None:
    fifa = normalize_fifa_rankings(
        pd.DataFrame(
            [
                {"date": "2022-01-01", "country": "France", "ranking": 3, "total_points": 1789.5},
            ]
        )
    )
    report_path = tmp_path / "ratings_report.json"

    payload = validate_rating_sources(
        elo_path=tmp_path / "missing_elo.csv",
        fifa_path=tmp_path / "missing_fifa.csv",
        output_path=report_path,
    )

    assert fifa.loc[0, "team"] == "France"
    assert fifa.loc[0, "rank"] == pytest.approx(3)
    assert payload["valid"] is True
    assert report_path.exists()
    assert len(payload["warnings"]) == 2
