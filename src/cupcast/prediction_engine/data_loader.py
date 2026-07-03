from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_sources import CsvDataSource, InternationalResultsCsvDataSource


def load_matches(path: str | Path) -> pd.DataFrame:
    data_source = CsvDataSource(matches_path=path, teams_path=_default_teams_path(path))
    return data_source.load_matches()


def load_teams(path: str | Path) -> pd.DataFrame:
    data_source = CsvDataSource(matches_path=_default_matches_path(path), teams_path=path)
    return data_source.load_teams()


def load_dataset(
    matches_path: str | Path,
    teams_path: str | Path | None,
    source_type: str = "csv",
    shootouts_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source_type.strip().lower() in {"international-results", "international_results", "real"}:
        data_source = InternationalResultsCsvDataSource(
            matches_path=matches_path,
            teams_path=teams_path,
            shootouts_path=shootouts_path,
        )
        return data_source.load_matches(), data_source.load_teams()
    if teams_path is None:
        raise FileNotFoundError("Teams CSV is required unless source_type is international-results")
    data_source = CsvDataSource(matches_path=matches_path, teams_path=teams_path)
    return data_source.load_matches(), data_source.load_teams()


def _default_teams_path(match_path: str | Path) -> Path:
    return Path(match_path).parent / "teams.csv"


def _default_matches_path(team_path: str | Path) -> Path:
    return Path(team_path).parent / "historical_matches.csv"
