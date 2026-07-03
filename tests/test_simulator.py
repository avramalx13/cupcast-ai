from dataclasses import dataclass

from cupcast.prediction_engine.data_loader import load_matches, load_teams
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.model import PredictionResult, train_prediction_model
from cupcast.simulator.bracket import default_bracket
from cupcast.simulator.monte_carlo import simulate_match, simulate_tournament
from cupcast.simulator.update_after_result import apply_match_result
from scripts.create_sample_data import main as create_sample_data


def test_simulated_match_returns_valid_team() -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    model = train_prediction_model(build_feature_table(matches, teams))

    winner = simulate_match("France", "Brazil", model, teams, matches, __import__("random").Random(1))

    assert winner in {"France", "Brazil"}


def test_simulation_probabilities_are_between_zero_and_one() -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    model = train_prediction_model(build_feature_table(matches, teams))

    result = simulate_tournament(default_bracket(), model, teams, matches, n_simulations=50, seed=1)

    assert result.team_probabilities
    for row in result.team_probabilities:
        assert 0.0 <= float(row["win_tournament_probability"]) <= 1.0


def test_completed_match_eliminates_loser_and_advances_winner() -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    model = train_prediction_model(build_feature_table(matches, teams))

    update = apply_match_result(
        bracket=default_bracket(),
        prediction_model=model,
        teams=teams,
        matches=matches,
        team_a="Germany",
        team_b="Paraguay",
        score_a=1,
        score_b=1,
        penalty_winner="Paraguay",
        simulations=40,
    )

    assert update.eliminated_team == "Germany"
    assert update.advanced_team == "Paraguay"
    assert update.probability_changes["Germany"]["after"] == 0.0
    assert update.elo_update["Germany_after"] != update.elo_update["Germany_before"]


def test_completed_match_must_exist_in_bracket() -> None:
    create_sample_data()
    matches = load_matches("data/raw/historical_matches.csv")
    teams = load_teams("data/raw/teams.csv")
    model = train_prediction_model(build_feature_table(matches, teams))

    try:
        apply_match_result(
            bracket=default_bracket(),
            prediction_model=model,
            teams=teams,
            matches=matches,
            team_a="France",
            team_b="Paraguay",
            score_a=1,
            score_b=0,
            simulations=10,
        )
    except ValueError as exc:
        assert "not paired" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for a non-bracket match")


@dataclass
class CountingPredictor:
    model_version: str = "counting"
    calls: int = 0

    def predict_match(self, team_a, team_b, teams, matches=None, neutral=True, stage="group", current_states=None):
        self.calls += 1
        return PredictionResult(
            team_a=team_a,
            team_b=team_b,
            team_a_win_probability=0.45,
            draw_probability=0.10,
            team_b_win_probability=0.45,
        )


def test_simulation_caches_repeated_match_predictions() -> None:
    create_sample_data()
    teams = load_teams("data/raw/teams.csv")
    predictor = CountingPredictor()

    simulate_tournament(default_bracket(), predictor, teams, matches=None, n_simulations=100, seed=4)

    assert 0 < predictor.calls < 200
