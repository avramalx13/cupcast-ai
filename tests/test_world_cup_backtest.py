from __future__ import annotations

import json

import pandas as pd

from cupcast.prediction_engine.world_cup_backtest import backtest_world_cups


def test_world_cup_backtest_split_and_output_schema(tmp_path) -> None:
    matches_path = tmp_path / "results.csv"
    output_path = tmp_path / "world_cup_backtest_results.json"
    pd.DataFrame(
        [
            ["2012-01-01", "A", "B", 2, 0, "Friendly", "City", "Country", False],
            ["2012-02-01", "C", "D", 0, 0, "Friendly", "City", "Country", False],
            ["2013-01-01", "A", "C", 0, 1, "Friendly", "City", "Country", False],
            ["2013-02-01", "B", "D", 2, 2, "Friendly", "City", "Country", False],
            ["2013-03-01", "A", "D", 3, 1, "Friendly", "City", "Country", False],
            ["2013-04-01", "B", "C", 1, 2, "Friendly", "City", "Country", False],
            ["2014-06-12", "A", "B", 1, 0, "FIFA World Cup", "Sao Paulo", "Brazil", True],
            ["2014-06-13", "C", "D", 1, 1, "FIFA World Cup", "Natal", "Brazil", True],
            ["2014-07-01", "A", "C", 0, 2, "FIFA World Cup", "Rio", "Brazil", True],
        ],
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ],
    ).to_csv(matches_path, index=False)

    result = backtest_world_cups(
        matches_path=matches_path,
        years=[2014],
        output_path=output_path,
        model_names=["majority_baseline", "uniform_random_baseline", "elo_logistic_regression"],
    )

    year_result = result["results"][0]
    assert year_result["year"] == 2014
    assert year_result["train_match_count"] == 6
    assert year_result["test_match_count"] == 3
    assert "best_model" in year_result
    assert "baseline_comparison" in year_result
    assert json.loads(output_path.read_text(encoding="utf-8"))["results"][0]["year"] == 2014
