from cupcast.prediction_engine.elo import expected_score, update_elo


def test_expected_score_is_valid_probability() -> None:
    score = expected_score(2000, 1900)

    assert 0.0 < score < 1.0


def test_elo_update_moves_winner_up() -> None:
    new_a, new_b = update_elo(1900, 2000, result_a=1.0, k=32)

    assert new_a > 1900
    assert new_b < 2000
