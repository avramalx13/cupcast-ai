from __future__ import annotations

from cupcast.shared.config import load_yaml


def test_prediction_config_exposes_data_mode_paths() -> None:
    config = load_yaml("configs/prediction_model.yaml")

    assert config["data"]["mode"] in {"synthetic", "real"}
    assert "matches_path" in config["data"]
    assert "teams_path" in config["data"]
