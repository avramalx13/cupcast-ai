from __future__ import annotations

import json

import numpy as np

from cupcast.prediction_engine.compare import DEFAULT_COMPARISON_MODELS, compare_models
from cupcast.prediction_engine.compare import _test_tournament_mask
from cupcast.prediction_engine.data_loader import load_matches, load_teams
from cupcast.prediction_engine.evaluate import calibration_summary, expected_calibration_error
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.model import train_prediction_model
from scripts.create_sample_data import main as create_sample_data


def test_compare_models_returns_all_expected_metrics(tmp_path) -> None:
    create_sample_data()
    output = tmp_path / "comparison_results.json"
    leaderboard = tmp_path / "model_leaderboard.json"

    result = compare_models(
        config_path="configs/prediction_model.yaml",
        train_before=2022,
        test_tournament="World Cup 2022",
        output_path=output,
        leaderboard_path=leaderboard,
    )

    names = {row["model_name"] for row in result["models"]}
    assert names == set(DEFAULT_COMPARISON_MODELS)
    for row in result["models"]:
        assert "accuracy" in row
        assert "log_loss" in row
        assert "brier_score" in row
        assert "top_class_accuracy" in row
        assert "expected_calibration_error" in row
    assert json.loads(output.read_text(encoding="utf-8"))["models"]
    leaderboard_payload = json.loads(leaderboard.read_text(encoding="utf-8"))
    assert leaderboard_payload["models"]
    first_row = leaderboard_payload["models"][0]
    assert "beats_majority_baseline" in first_row
    assert "beats_elo_baseline" in first_row
    assert "notes" in first_row


def test_baseline_model_probabilities_sum_to_one() -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    features = build_feature_table(matches, teams)
    model = train_prediction_model(features, model_type="majority_baseline")

    probabilities = model.predict_features(features.head(1))

    assert abs(sum(probabilities.values()) - 1.0) < 1e-9


def test_calibration_bins_and_ece_are_valid() -> None:
    y_true = np.array(["A_WIN", "DRAW", "B_WIN"])
    y_pred = np.array(["A_WIN", "A_WIN", "B_WIN"])
    probabilities = np.array(
        [
            [0.6, 0.2, 0.2],
            [0.55, 0.25, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )

    bins = calibration_summary(y_true, y_pred, probabilities)
    ece = expected_calibration_error(y_true, y_pred, probabilities)

    assert all(0 <= row["average_confidence"] <= 1 for row in bins)
    assert 0 <= ece <= 1


def test_world_cup_year_tournament_mask_matches_final_tournament_only() -> None:
    import pandas as pd

    features = pd.DataFrame(
        {
            "tournament": ["FIFA World Cup", "FIFA World Cup qualification", "Friendly", "FIFA World Cup"],
            "date": pd.to_datetime(["2022-11-20", "2022-03-01", "2022-01-01", "2018-06-14"]),
        }
    )

    mask = _test_tournament_mask(features, "World Cup 2022")

    assert mask.tolist() == [True, False, False, False]
