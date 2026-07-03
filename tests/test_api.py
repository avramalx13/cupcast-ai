from fastapi.testclient import TestClient

from scripts.create_sample_data import main as create_sample_data


def test_health_returns_ok() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_data_status_reports_configured_synthetic_mode(monkeypatch) -> None:
    create_sample_data()
    from cupcast.api import services
    from cupcast.api.main import app

    monkeypatch.setenv("CUPCAST_API_CONFIG", "configs/prediction_model.yaml")
    services.get_api_config.cache_clear()
    services.get_data_config.cache_clear()
    services.get_model_config.cache_clear()
    client = TestClient(app)
    response = client.get("/data/status")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "synthetic"
    assert body["matches_loaded"] > 0
    assert body["last_validation_valid"] is True


def test_api_defaults_to_real_config_when_real_artifacts_exist(tmp_path, monkeypatch) -> None:
    from cupcast.api import services

    real_config = tmp_path / "prediction_model_real.yaml"
    real_matches = tmp_path / "real_completed_matches.csv"
    real_teams = tmp_path / "teams_real.csv"
    real_model = tmp_path / "prediction_model_real.joblib"

    for path in (real_config, real_matches, real_teams, real_model):
        path.write_text("present", encoding="utf-8")

    monkeypatch.delenv("CUPCAST_API_CONFIG", raising=False)
    monkeypatch.setattr(services, "REAL_API_CONFIG_PATH", real_config)
    monkeypatch.setattr(services, "REAL_MATCHES_PATH", real_matches)
    monkeypatch.setattr(services, "REAL_TEAMS_PATH", real_teams)
    monkeypatch.setattr(services, "REAL_MODEL_PATH", real_model)

    assert services.api_config_path() == real_config


def test_teams_endpoint_returns_generated_real_teams_when_present(tmp_path, monkeypatch) -> None:
    from cupcast.api import services

    teams_path = tmp_path / "teams_real.csv"
    teams_path.write_text(
        "team,confederation,initial_elo,fifa_rank,match_count,first_match_date,last_match_date\n"
        "Senegal,UNKNOWN,1500,,240,1961-12-31,2026-06-20\n"
        "Morocco,UNKNOWN,1500,,300,1957-10-19,2026-06-20\n"
        "Senegal,UNKNOWN,1500,,100,1961-12-31,2020-01-01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "REAL_TEAMS_PATH", teams_path)
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.get("/teams")

    assert response.status_code == 200
    body = response.json()
    names = [team["name"] for team in body["teams"]]
    assert body["source"] == "real"
    assert names.count("Senegal") == 1
    assert "Morocco" in names
    senegal = next(team for team in body["teams"] if team["name"] == "Senegal")
    assert senegal["match_count"] == 240


def test_cors_allows_local_frontend_preflight() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.options(
        "/backtesting/run",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_predict_match_returns_probabilities() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.post("/predict/match", json={"team_a": "France", "team_b": "Brazil"})

    assert response.status_code == 200
    body = response.json()
    total = (
        body["team_a_win_probability"]
        + body["draw_probability"]
        + body["team_b_win_probability"]
    )
    assert abs(total - 1.0) < 1e-9
    assert body["explanation"]


def test_predict_match_returns_400_for_unknown_team() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.post("/predict/match", json={"team_a": "Atlantis", "team_b": "Brazil"})

    assert response.status_code == 400


def test_full_tournament_latest_returns_unavailable_when_report_missing(tmp_path, monkeypatch) -> None:
    from cupcast.api import services
    from cupcast.api.main import app

    monkeypatch.setattr(services, "FULL_TOURNAMENT_JSON_PATH", tmp_path / "missing_full_tournament.json")
    monkeypatch.setattr(services, "FULL_TOURNAMENT_MD_PATH", tmp_path / "missing_full_tournament.md")
    client = TestClient(app)
    response = client.get("/simulation/full-tournament/latest")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "Run full tournament simulation first" in response.json()["message"]


def test_analyst_explain_returns_text() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.post(
        "/analyst/explain",
        json={
            "kind": "match_prediction",
            "team_a": "France",
            "team_b": "Brazil",
            "team_a_win_probability": 0.42,
            "draw_probability": 0.24,
            "team_b_win_probability": 0.34,
        },
    )

    assert response.status_code == 200
    assert response.json()["explanation"]
    assert "France" in response.json()["explanation"]
    assert "Brazil" in response.json()["explanation"]


def test_analyst_explain_changes_with_selected_teams() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    first = client.post(
        "/analyst/explain",
        json={
            "kind": "match_prediction",
            "team_a": "France",
            "team_b": "Brazil",
            "team_a_win_probability": 0.42,
            "draw_probability": 0.24,
            "team_b_win_probability": 0.34,
        },
    )
    second = client.post(
        "/analyst/explain",
        json={
            "kind": "match_prediction",
            "team_a": "Senegal",
            "team_b": "Morocco",
            "team_a_win_probability": 0.31,
            "draw_probability": 0.29,
            "team_b_win_probability": 0.40,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert "Senegal" in second.json()["explanation"]
    assert "Morocco" in second.json()["explanation"]
    assert first.json()["explanation"] != second.json()["explanation"]


def test_update_result_returns_400_for_non_bracket_match() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.post(
        "/matches/update-result",
        json={
            "team_a": "France",
            "team_b": "Paraguay",
            "score_a": 1,
            "score_b": 0,
            "simulations": 10,
        },
    )

    assert response.status_code == 400
