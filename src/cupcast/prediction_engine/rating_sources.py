from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


RATING_OUTPUT_COLUMNS = ["date", "team", "elo", "rank", "points", "source"]

ELO_ALIASES = {
    "date": ["date"],
    "team": ["team", "country", "nation"],
    "elo": ["elo", "rating", "elo_rating"],
    "rank": ["rank"],
}

FIFA_ALIASES = {
    "date": ["date"],
    "team": ["team", "country", "nation"],
    "rank": ["rank", "ranking"],
    "points": ["points", "total_points"],
}


@dataclass
class RatingValidationReport:
    source: str
    path: str
    exists: bool
    rows: int = 0
    date_min: str | None = None
    date_max: str | None = None
    teams: int = 0
    duplicate_team_date_rows: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "path": self.path,
            "exists": self.exists,
            "valid": self.valid,
            "rows": self.rows,
            "date_range": {"min": self.date_min, "max": self.date_max},
            "teams": self.teams,
            "duplicate_team_date_rows": self.duplicate_team_date_rows,
            "missing_values": self.missing_values,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class RatingSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def validate(self) -> RatingValidationReport:
        ...


class LocalEloRatingSource(RatingSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return _load_rating_file(self.path, ELO_ALIASES, source="elo", required_numeric=("elo",))

    def validate(self) -> RatingValidationReport:
        return validate_rating_file(self.path, ELO_ALIASES, source="elo", required_numeric=("elo",))


class LocalFifaRankingSource(RatingSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return _load_rating_file(self.path, FIFA_ALIASES, source="fifa", required_numeric=("rank",))

    def validate(self) -> RatingValidationReport:
        return validate_rating_file(self.path, FIFA_ALIASES, source="fifa", required_numeric=("rank",))


def validate_rating_sources(
    elo_path: str | Path | None,
    fifa_path: str | Path | None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    reports = []
    if elo_path:
        reports.append(LocalEloRatingSource(elo_path).validate())
    if fifa_path:
        reports.append(LocalFifaRankingSource(fifa_path).validate())
    if not reports:
        reports.append(_missing_report("elo", ""))
        reports.append(_missing_report("fifa", ""))
    payload = {
        "valid": all(report.valid for report in reports),
        "reports": [report.to_dict() for report in reports],
        "warnings": [warning for report in reports for warning in report.warnings],
        "errors": [error for report in reports for error in report.errors],
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_rating_file(
    path: str | Path,
    aliases: dict[str, list[str]],
    source: str,
    required_numeric: tuple[str, ...],
) -> RatingValidationReport:
    rating_path = Path(path)
    if not rating_path.exists():
        return _missing_report(source, str(rating_path))
    try:
        frame = _load_rating_file(rating_path, aliases, source=source, required_numeric=required_numeric)
    except ValueError as exc:
        report = RatingValidationReport(source=source, path=str(rating_path), exists=True)
        report.errors.append(str(exc))
        return report
    report = RatingValidationReport(
        source=source,
        path=str(rating_path),
        exists=True,
        rows=int(len(frame)),
        teams=int(frame["team"].dropna().astype(str).nunique()),
        duplicate_team_date_rows=int(frame.duplicated(subset=["team", "date"]).sum()),
        missing_values={
            column: int(frame[column].isna().sum())
            for column in ["date", "team", "elo", "rank", "points"]
            if column in frame and int(frame[column].isna().sum()) > 0
        },
    )
    valid_dates = frame["date"].dropna()
    if not valid_dates.empty:
        report.date_min = valid_dates.min().date().isoformat()
        report.date_max = valid_dates.max().date().isoformat()
    if report.duplicate_team_date_rows:
        report.warnings.append(f"Duplicate team/date rows: {report.duplicate_team_date_rows}")
    return report


def get_latest_rating_before(team: str, match_date: object, ratings_df: pd.DataFrame) -> float | None:
    if ratings_df is None or ratings_df.empty:
        return None
    frame = ratings_df.copy()
    if "date" not in frame or "team" not in frame:
        return None
    value_column = "elo" if "elo" in frame and frame["elo"].notna().any() else "rank"
    if value_column not in frame:
        return None
    match_ts = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(match_ts):
        return None
    team_rows = frame.loc[
        frame["team"].astype(str).str.casefold().eq(str(team).strip().casefold())
        & (pd.to_datetime(frame["date"], errors="coerce") < match_ts)
    ].sort_values("date")
    if team_rows.empty:
        return None
    value = pd.to_numeric(team_rows.iloc[-1][value_column], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def normalize_elo_ratings(raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_rating_frame(raw, ELO_ALIASES, source="elo", required_numeric=("elo",))


def normalize_fifa_rankings(raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_rating_frame(raw, FIFA_ALIASES, source="fifa", required_numeric=("rank",))


def _load_rating_file(
    path: Path,
    aliases: dict[str, list[str]],
    source: str,
    required_numeric: tuple[str, ...],
) -> pd.DataFrame:
    return _normalize_rating_frame(pd.read_csv(path), aliases, source=source, required_numeric=required_numeric)


def _normalize_rating_frame(
    raw: pd.DataFrame,
    aliases: dict[str, list[str]],
    source: str,
    required_numeric: tuple[str, ...],
) -> pd.DataFrame:
    frame = _rename_aliases(raw, aliases)
    required = {"date", "team", *required_numeric}
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"{source} ratings missing required columns after alias normalization: {missing}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["team"] = frame["team"].astype("string").str.strip()
    for column in ["elo", "rank", "points"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["source"] = source
    invalid_dates = int(frame["date"].isna().sum())
    empty_teams = int(frame["team"].fillna("").eq("").sum())
    numeric_errors = {
        column: int(frame[column].isna().sum())
        for column in required_numeric
        if int(frame[column].isna().sum()) > 0
    }
    errors = []
    if invalid_dates:
        errors.append(f"Invalid dates: {invalid_dates}")
    if empty_teams:
        errors.append(f"Empty team names: {empty_teams}")
    if numeric_errors:
        errors.append(f"Missing/non-numeric required values: {numeric_errors}")
    if errors:
        raise ValueError("; ".join(errors))
    return frame[RATING_OUTPUT_COLUMNS].sort_values(["team", "date"]).reset_index(drop=True)


def _missing_report(source: str, path: str) -> RatingValidationReport:
    report = RatingValidationReport(source=source, path=path, exists=False)
    report.warnings.append(f"{source} ratings file missing; continuing without external {source} features")
    return report


def _rename_aliases(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    normalized_columns = {_normalize_column(column): column for column in frame.columns}
    rename_map: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            original = normalized_columns.get(_normalize_column(candidate))
            if original is not None:
                rename_map[original] = canonical
                break
    return frame.rename(columns=rename_map)


def _normalize_column(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")
