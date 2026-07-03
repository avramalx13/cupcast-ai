from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cupcast.prediction_engine.ablation import run_ablation_study
from cupcast.prediction_engine.calibration import CalibratedMatchModel
from cupcast.prediction_engine.data_loader import load_matches, load_teams
from cupcast.prediction_engine.ensemble import ProbabilityEnsemble
from cupcast.prediction_engine.error_analysis import analyze_world_cup_errors
from cupcast.prediction_engine.feature_importance import analyze_feature_importance
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.model_registry import create_model
from cupcast.prediction_engine.poisson_model import PoissonGoalModel
from scripts.create_sample_data import main as create_sample_data


def test_poisson_model_probabilities_are_valid_and_reflect_attack_strength() -> None:
    model = PoissonGoalModel()
    rows = pd.DataFrame(
        [
            _poisson_row(gf_a=2.4, gf_b=0.8, ga_a=0.8, ga_b=1.8, elo_diff=180),
            _poisson_row(gf_a=0.8, gf_b=2.4, ga_a=1.8, ga_b=0.8, elo_diff=-180),
        ]
    )
    probabilities = model.predict_proba(rows)

    assert probabilities.shape == (2, 3)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > probabilities[1, 0]


def test_calibrated_model_probability_shape_and_sum() -> None:
    features = _sample_features()
    x = features[["elo_a", "elo_b", "elo_diff"]]
    y = features["label"]
    model = CalibratedMatchModel(create_model("elo_logistic_regression"), method="sigmoid", cv=2)

    model.fit(x, y)
    probabilities = model.predict_proba(x.head(3))

    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_ensemble_probabilities_sum_and_failed_member_is_reported() -> None:
    features = _sample_features()
    model = ProbabilityEnsemble(
        candidates=["majority_baseline", "poisson_goal_model", "not_a_real_model"],
        mode="simple_average",
    )

    model.fit(features, features["label"])
    probabilities = model.predict_proba(features.head(4))

    assert probabilities.shape == (4, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert any(failure["model_name"] == "not_a_real_model" for failure in model.failures)
    assert sum(model.member_weights_.values()) == pytest.approx(1.0)


def test_world_cup_error_analysis_schema(tmp_path) -> None:
    features = _sample_features()
    output = tmp_path / "errors.json"

    payload = analyze_world_cup_errors(
        features=features,
        years=[2022],
        model_names=["majority_baseline"],
        output_path=output,
    )

    assert output.exists()
    assert "errors" in payload
    assert payload["errors"]
    first = payload["errors"][0]
    for field in ["year", "match", "predicted_probabilities", "actual_result", "error_type", "notes"]:
        assert field in first


def test_feature_importance_report_schema_and_unsupported_model(tmp_path) -> None:
    output_json = tmp_path / "feature_importance.json"
    output_md = tmp_path / "feature_importance.md"

    payload = analyze_feature_importance(
        "configs/prediction_model.yaml",
        output_json=output_json,
        output_md=output_md,
        model_names=["logistic_regression", "majority_baseline"],
    )

    assert output_json.exists()
    assert output_md.exists()
    assert payload["models"]
    statuses = {row["model_name"]: row["status"] for row in payload["models"]}
    assert statuses["logistic_regression"] == "ok"
    assert statuses["majority_baseline"] == "unsupported"


def test_ablation_report_schema_marks_missing_external_ratings(tmp_path) -> None:
    output_json = tmp_path / "ablation.json"
    output_md = tmp_path / "ablation.md"

    payload = run_ablation_study(
        "configs/prediction_model.yaml",
        output_json=output_json,
        output_md=output_md,
    )

    assert output_json.exists()
    assert output_md.exists()
    assert payload["groups"]
    external = [row for row in payload["groups"] if row["feature_group"] == "all_features_plus_external_ratings"]
    assert external
    assert external[0]["status"] in {"unavailable", "ok"}
    for row in payload["groups"]:
        if row["status"] == "ok":
            assert isinstance(row["log_loss"], float)
        else:
            assert row["log_loss"] is None
            assert row["reason"]
    assert json.loads(output_json.read_text(encoding="utf-8"))["groups"]


def _sample_features() -> pd.DataFrame:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    return build_feature_table(matches, teams)


def _poisson_row(gf_a: float, gf_b: float, ga_a: float, ga_b: float, elo_diff: float) -> dict[str, float]:
    return {
        "team_a_goals_scored_last_5": gf_a,
        "team_b_goals_scored_last_5": gf_b,
        "team_a_goals_conceded_last_5": ga_a,
        "team_b_goals_conceded_last_5": ga_b,
        "elo_external_diff": elo_diff,
        "home_advantage_flag": 0,
        "tournament_importance": 1,
    }
