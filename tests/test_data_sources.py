from __future__ import annotations

import pandas as pd

from cupcast.prediction_engine.data_sources import (
    CsvDataSource,
    InternationalResultsCsvDataSource,
    build_teams_from_matches,
    normalize_matches,
    validate_dataset_files,
)


def test_csv_data_source_normalizes_common_public_dataset_aliases(tmp_path) -> None:
    matches_path = tmp_path / "matches.csv"
    teams_path = tmp_path / "teams.csv"
    pd.DataFrame(
        [
            {
                "date": "2022-11-20",
                "home_team": "Qatar",
                "away_team": "Ecuador",
                "home_score": 0,
                "away_score": 2,
                "tournament": "World Cup 2022",
                "neutral": True,
                "country": "Qatar",
                "city": "Al Khor",
            }
        ]
    ).to_csv(matches_path, index=False)
    pd.DataFrame(
        [
            {"team": "Qatar", "confederation": "AFC", "initial_elo": 1660, "fifa_rank": 50},
            {"team": "Ecuador", "confederation": "CONMEBOL", "initial_elo": 1780, "fifa_rank": 44},
        ]
    ).to_csv(teams_path, index=False)

    source = CsvDataSource(matches_path=matches_path, teams_path=teams_path)
    matches = source.load_matches()

    assert matches.loc[0, "team_a"] == "Qatar"
    assert matches.loc[0, "team_b"] == "Ecuador"
    assert matches.loc[0, "team_a_score"] == 0
    assert matches.loc[0, "team_b_score"] == 2


def test_dataset_validation_reports_duplicates_invalid_scores_and_unknown_teams(tmp_path) -> None:
    matches_path = tmp_path / "matches.csv"
    teams_path = tmp_path / "teams.csv"
    pd.DataFrame(
        [
            {
                "date": "2022-01-01",
                "team_a": "France",
                "team_b": "Atlantis",
                "team_a_score": 2,
                "team_b_score": -1,
                "tournament": "Test",
                "neutral": 1,
            },
            {
                "date": "2022-01-01",
                "team_a": "France",
                "team_b": "Atlantis",
                "team_a_score": 2,
                "team_b_score": -1,
                "tournament": "Test",
                "neutral": 1,
            },
        ]
    ).to_csv(matches_path, index=False)
    pd.DataFrame(
        [
            {"team": "France", "confederation": "UEFA", "initial_elo": 2042, "fifa_rank": 2},
        ]
    ).to_csv(teams_path, index=False)

    report = validate_dataset_files(matches_path=matches_path, teams_path=teams_path)

    assert not report.is_valid
    assert report.duplicate_rows == 1
    assert report.invalid_scores == 2
    assert report.unknown_teams == ["Atlantis"]


def test_normalize_matches_adds_optional_defaults() -> None:
    frame = normalize_matches(
        pd.DataFrame(
            [
                {
                    "date": "2022-01-01",
                    "home_team": "France",
                    "away_team": "Brazil",
                    "home_score": 1,
                    "away_score": 1,
                }
            ]
        )
    )

    assert frame.loc[0, "tournament"] == "Unknown"
    assert frame.loc[0, "stage"] == "unknown"
    assert frame.loc[0, "neutral"] == 1


def test_international_results_source_normalizes_and_merges_shootouts(tmp_path) -> None:
    matches_path = tmp_path / "results.csv"
    shootouts_path = tmp_path / "shootouts.csv"
    pd.DataFrame(
        [
            {
                "date": "2022-12-09",
                "home_team": "Netherlands",
                "away_team": "Argentina",
                "home_score": 2,
                "away_score": 2,
                "tournament": "FIFA World Cup",
                "city": "Lusail",
                "country": "Qatar",
                "neutral": "TRUE",
            }
        ]
    ).to_csv(matches_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2022-12-09",
                "home_team": "Netherlands",
                "away_team": "Argentina",
                "winner": "Argentina",
                "first_shooter": "Netherlands",
            }
        ]
    ).to_csv(shootouts_path, index=False)

    source = InternationalResultsCsvDataSource(matches_path=matches_path, shootouts_path=shootouts_path)
    matches = source.load_matches()

    assert matches.loc[0, "team_a"] == "Netherlands"
    assert matches.loc[0, "team_b"] == "Argentina"
    assert matches.loc[0, "team_a_score"] == 2
    assert matches.loc[0, "team_b_score"] == 2
    assert matches.loc[0, "source"] == "real_csv"
    assert matches.loc[0, "went_to_penalties"] == 1
    assert matches.loc[0, "penalty_winner"] == "Argentina"


def test_validation_report_schema_for_international_results(tmp_path) -> None:
    matches_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "date": "2018-06-14",
                "home_team": "Russia",
                "away_team": "Saudi Arabia",
                "home_score": 5,
                "away_score": 0,
                "tournament": "FIFA World Cup",
                "neutral": False,
            }
        ]
    ).to_csv(matches_path, index=False)

    report = validate_dataset_files(matches_path, source_type="international-results")
    payload = report.to_dict()

    assert report.is_valid
    assert payload["source_type"] == "international-results"
    assert payload["number_of_matches"] == 1
    assert payload["world_cup_match_count"] == 1
    assert "missing_values_by_column" in payload
    assert "teams_with_few_matches" in payload


def test_world_cup_match_count_excludes_qualification_rows(tmp_path) -> None:
    matches_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "date": "2022-11-20",
                "home_team": "Qatar",
                "away_team": "Ecuador",
                "home_score": 0,
                "away_score": 2,
                "tournament": "FIFA World Cup",
                "neutral": True,
            },
            {
                "date": "2021-09-01",
                "home_team": "France",
                "away_team": "Bosnia and Herzegovina",
                "home_score": 1,
                "away_score": 1,
                "tournament": "FIFA World Cup qualification",
                "neutral": False,
            },
        ]
    ).to_csv(matches_path, index=False)

    report = validate_dataset_files(matches_path, source_type="international-results")

    assert report.world_cup_match_count == 1


def test_build_teams_from_matches_keeps_unknown_metadata_blank() -> None:
    matches = normalize_matches(
        pd.DataFrame(
            [
                {
                    "date": "2020-01-01",
                    "home_team": "France",
                    "away_team": "Brazil",
                    "home_score": 1,
                    "away_score": 0,
                }
            ]
        )
    )

    teams = build_teams_from_matches(matches)

    assert set(teams["team"]) == {"Brazil", "France"}
    assert set(teams["confederation"]) == {"UNKNOWN"}
    assert set(teams["initial_elo"]) == {1500}
    assert teams["fifa_rank"].isna().all()
