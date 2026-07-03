import pandas as pd

from cupcast.prediction_engine.data_loader import load_matches, load_teams
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.model import load_model, save_model, train_prediction_model
from cupcast.prediction_engine.rating_sources import normalize_elo_ratings
from scripts.create_sample_data import main as create_sample_data


def test_model_prediction_probabilities_sum_to_one() -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    features = build_feature_table(matches, teams)
    model = train_prediction_model(features)

    result = model.predict_match("France", "Brazil", teams=teams, matches=matches)
    total = (
        result.team_a_win_probability
        + result.draw_probability
        + result.team_b_win_probability
    )

    assert abs(total - 1.0) < 1e-9


def test_prediction_model_artifact_roundtrip(tmp_path) -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    model = train_prediction_model(build_feature_table(matches, teams), model_version="roundtrip-test")
    model_path = tmp_path / "prediction_model.joblib"

    save_model(model, model_path)
    loaded = load_model(model_path)
    result = loaded.predict_match("France", "Brazil", teams=teams, matches=matches)

    assert loaded.model_version == "roundtrip-test"
    assert result.team_a == "France"
    assert abs(
        result.team_a_win_probability
        + result.draw_probability
        + result.team_b_win_probability
        - 1.0
    ) < 1e-9


def test_prediction_model_artifact_preserves_external_rating_snapshots(tmp_path) -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    external_elo = normalize_elo_ratings(
        pd.DataFrame(
            [
                {"date": "2020-01-01", "team": "France", "elo": 2010},
                {"date": "2020-01-01", "team": "Brazil", "elo": 1990},
            ]
        )
    )
    feature_flags = {"external_ratings": True}
    model = train_prediction_model(
        build_feature_table(matches, teams, external_elo_ratings=external_elo, feature_flags=feature_flags),
        model_version="ratings-roundtrip-test",
        external_elo_ratings=external_elo,
        feature_flags=feature_flags,
    )
    model_path = tmp_path / "prediction_model.joblib"

    save_model(model, model_path)
    loaded = load_model(model_path)

    assert loaded.external_elo_ratings is not None
    assert loaded.feature_flags == feature_flags
    assert loaded.external_elo_ratings["elo"].tolist() == [1990, 2010]
