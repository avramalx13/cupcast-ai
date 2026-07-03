from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cupcast.shared.constants import DEFAULT_MATCHES_PATH, DEFAULT_TEAMS_PATH


MATCH_REQUIRED_COLUMNS = {"date", "team_a", "team_b", "team_a_score", "team_b_score"}
MATCH_OUTPUT_COLUMNS = [
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
    "tournament",
    "neutral",
    "stage",
    "country",
    "city",
    "source",
    "went_to_penalties",
    "penalty_winner",
]
TEAM_REQUIRED_COLUMNS = {"team", "initial_elo", "fifa_rank"}
TEAM_OUTPUT_COLUMNS = [
    "team",
    "confederation",
    "initial_elo",
    "fifa_rank",
    "match_count",
    "first_match_date",
    "last_match_date",
]

MATCH_ALIASES = {
    "team_a": ["team_a", "home_team", "home", "team1", "home_name"],
    "team_b": ["team_b", "away_team", "away", "team2", "away_name"],
    "team_a_score": ["team_a_score", "home_score", "home_goals", "score_a", "home_ft_score"],
    "team_b_score": ["team_b_score", "away_score", "away_goals", "score_b", "away_ft_score"],
    "date": ["date", "match_date", "utc_date"],
    "tournament": ["tournament", "competition", "league", "event"],
    "neutral": ["neutral", "neutral_venue", "is_neutral"],
    "stage": ["stage", "round", "phase"],
    "country": ["country", "host_country"],
    "city": ["city", "venue_city"],
    "source": ["source", "data_source"],
    "went_to_penalties": ["went_to_penalties", "penalties", "shootout"],
    "penalty_winner": ["penalty_winner", "shootout_winner"],
}

TEAM_ALIASES = {
    "team": ["team", "name", "team_name", "country", "nation"],
    "confederation": ["confederation", "confed", "region"],
    "initial_elo": ["initial_elo", "elo", "rating", "elo_rating"],
    "fifa_rank": ["fifa_rank", "rank", "fifa_ranking"],
}


@dataclass
class DatasetValidationReport:
    matches_count: int = 0
    teams_count: int = 0
    source_type: str = "csv"
    date_min: str | None = None
    date_max: str | None = None
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    invalid_scores: int = 0
    invalid_dates: int = 0
    empty_team_names: int = 0
    unknown_teams: list[str] = field(default_factory=list)
    number_of_tournaments: int = 0
    world_cup_match_count: int = 0
    friendly_match_count: int = 0
    shootout_match_count: int = 0
    teams_with_few_matches: list[str] = field(default_factory=list)
    suspicious_team_names: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "source_type": self.source_type,
            "number_of_matches": self.matches_count,
            "number_of_unique_teams": self.teams_count,
            "number_of_tournaments": self.number_of_tournaments,
            "world_cup_match_count": self.world_cup_match_count,
            "friendly_match_count": self.friendly_match_count,
            "missing_values_by_column": self.missing_values,
            "duplicate_match_count": self.duplicate_rows,
            "invalid_score_count": self.invalid_scores,
            "unknown_team_count": len(self.unknown_teams),
            "shootout_match_count": self.shootout_match_count,
            "teams_with_few_matches": self.teams_with_few_matches,
            "suspicious_team_names": self.suspicious_team_names,
            "matches": self.matches_count,
            "teams": self.teams_count,
            "date_range": {"min": self.date_min, "max": self.date_max},
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "invalid_scores": self.invalid_scores,
            "invalid_dates": self.invalid_dates,
            "empty_team_names": self.empty_team_names,
            "unknown_teams": self.unknown_teams,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def format_text(self) -> str:
        lines = [
            f"valid={str(self.is_valid).lower()}",
            f"source_type={self.source_type}",
            f"number_of_matches={self.matches_count}",
            f"matches={self.matches_count}",
            f"date_range={self.date_min or 'n/a'}..{self.date_max or 'n/a'}",
            f"number_of_unique_teams={self.teams_count}",
            f"teams={self.teams_count}",
            f"number_of_tournaments={self.number_of_tournaments}",
            f"world_cup_match_count={self.world_cup_match_count}",
            f"friendly_match_count={self.friendly_match_count}",
            f"shootout_match_count={self.shootout_match_count}",
            f"duplicate_match_count={self.duplicate_rows}",
            f"invalid_score_count={self.invalid_scores}",
            f"unknown_team_count={len(self.unknown_teams)}",
            f"missing_values={self.missing_values}",
            f"duplicate_rows={self.duplicate_rows}",
            f"invalid_scores={self.invalid_scores}",
            f"invalid_dates={self.invalid_dates}",
            f"empty_team_names={self.empty_team_names}",
            f"unknown_teams={self.unknown_teams}",
            f"teams_with_few_matches={self.teams_with_few_matches}",
            f"suspicious_team_names={self.suspicious_team_names}",
        ]
        if self.warnings:
            lines.append(f"warnings={self.warnings}")
        if self.errors:
            lines.append(f"errors={self.errors}")
        return "\n".join(lines)


class MatchDataSource(ABC):
    @abstractmethod
    def load_matches(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def load_teams(self) -> pd.DataFrame:
        ...


class CsvDataSource(MatchDataSource):
    def __init__(self, matches_path: str | Path, teams_path: str | Path) -> None:
        self.matches_path = Path(matches_path)
        self.teams_path = Path(teams_path)

    def load_matches(self) -> pd.DataFrame:
        matches, _report = load_and_validate_dataset(self.matches_path, self.teams_path)
        return matches

    def load_teams(self) -> pd.DataFrame:
        _matches, _report, teams = load_and_validate_dataset(
            self.matches_path,
            self.teams_path,
            include_teams=True,
        )
        return teams


class SyntheticDataSource(CsvDataSource):
    def __init__(
        self,
        matches_path: str | Path = DEFAULT_MATCHES_PATH,
        teams_path: str | Path = DEFAULT_TEAMS_PATH,
    ) -> None:
        super().__init__(matches_path=matches_path, teams_path=teams_path)


class InternationalResultsCsvDataSource(MatchDataSource):
    def __init__(
        self,
        matches_path: str | Path,
        teams_path: str | Path | None = None,
        shootouts_path: str | Path | None = None,
    ) -> None:
        self.matches_path = Path(matches_path)
        self.teams_path = Path(teams_path) if teams_path is not None else None
        self.shootouts_path = Path(shootouts_path) if shootouts_path is not None else None

    def load_matches(self) -> pd.DataFrame:
        if not self.matches_path.exists():
            raise FileNotFoundError(f"International results CSV not found: {self.matches_path}")
        raw_matches = pd.read_csv(self.matches_path)
        shootouts = None
        if self.shootouts_path is not None and self.shootouts_path.exists():
            shootouts = pd.read_csv(self.shootouts_path)
        matches = normalize_international_results(raw_matches, shootouts=shootouts)
        teams = self.load_teams()
        report = validate_normalized_dataset(matches, teams, source_type="international-results")
        if not report.is_valid:
            raise ValueError("Dataset validation failed:\n" + report.format_text())
        return matches

    def load_teams(self) -> pd.DataFrame:
        if self.teams_path is not None and self.teams_path.exists():
            return normalize_teams(pd.read_csv(self.teams_path), allow_missing_rank=True)
        return build_teams_from_matches(self._load_matches_without_validation())

    def _load_matches_without_validation(self) -> pd.DataFrame:
        raw_matches = pd.read_csv(self.matches_path)
        shootouts = None
        if self.shootouts_path is not None and self.shootouts_path.exists():
            shootouts = pd.read_csv(self.shootouts_path)
        return normalize_international_results(raw_matches, shootouts=shootouts)


def load_and_validate_dataset(
    matches_path: str | Path,
    teams_path: str | Path,
    include_teams: bool = False,
) -> tuple[pd.DataFrame, DatasetValidationReport] | tuple[pd.DataFrame, DatasetValidationReport, pd.DataFrame]:
    matches_csv = Path(matches_path)
    teams_csv = Path(teams_path)
    if not matches_csv.exists():
        raise FileNotFoundError(f"Historical matches CSV not found: {matches_csv}")
    if not teams_csv.exists():
        raise FileNotFoundError(f"Teams CSV not found: {teams_csv}")

    raw_matches = pd.read_csv(matches_csv)
    raw_teams = pd.read_csv(teams_csv)
    matches = normalize_matches(raw_matches)
    teams = normalize_teams(raw_teams)
    report = validate_normalized_dataset(matches, teams)
    if not report.is_valid:
        raise ValueError("Dataset validation failed:\n" + report.format_text())
    if include_teams:
        return matches, report, teams
    return matches, report


def validate_dataset_files(
    matches_path: str | Path,
    teams_path: str | Path | None = None,
    shootouts_path: str | Path | None = None,
    goalscorers_path: str | Path | None = None,
    source_type: str = "csv",
) -> DatasetValidationReport:
    source_name = source_type.strip().lower()
    matches_csv = Path(matches_path)
    if not matches_csv.exists():
        report = DatasetValidationReport(source_type=source_type)
        report.errors.append(f"Matches CSV not found: {matches_csv}")
        return report

    raw_matches = pd.read_csv(matches_csv)
    if source_name in {"international-results", "international_results", "real"}:
        shootouts = None
        if shootouts_path:
            shootouts_csv = Path(shootouts_path)
            if shootouts_csv.exists():
                shootouts = pd.read_csv(shootouts_csv)
            else:
                shootouts = None
        matches = normalize_international_results(raw_matches, shootouts=shootouts)
        teams = _load_or_build_teams(matches, teams_path=teams_path, allow_missing_rank=True)
        report = validate_normalized_dataset(matches, teams, source_type="international-results")
        if not teams_path or not Path(teams_path).exists():
            report.warnings.append("Teams file missing; generated team metadata from matches with UNKNOWN confederation and blank fifa_rank")
        if not shootouts_path or not Path(shootouts_path).exists():
            report.warnings.append("Shootouts file missing; penalty winners are unavailable")
        if goalscorers_path and not Path(goalscorers_path).exists():
            report.warnings.append("Goalscorers file missing; goalscorer-derived features are unavailable")
        elif not goalscorers_path:
            report.warnings.append("Goalscorers file missing; goalscorer-derived features are unavailable")
        return report

    if teams_path is None:
        report = DatasetValidationReport(source_type=source_type)
        report.errors.append("Teams CSV is required for synthetic/csv validation")
        return report
    raw_teams = pd.read_csv(teams_path)
    matches = normalize_matches(raw_matches)
    teams = normalize_teams(raw_teams)
    return validate_normalized_dataset(matches, teams, source_type=source_type)


def normalize_matches(raw_matches: pd.DataFrame) -> pd.DataFrame:
    frame = _rename_aliases(raw_matches, MATCH_ALIASES)
    missing_required = missing_required_match_columns(raw_matches)
    for column in MATCH_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if "tournament" not in frame.columns:
        frame["tournament"] = "Unknown"
    if "neutral" not in frame.columns:
        frame["neutral"] = 1
    if "stage" not in frame.columns:
        frame["stage"] = "unknown"
    if "country" not in frame.columns:
        frame["country"] = ""
    if "city" not in frame.columns:
        frame["city"] = ""
    if "source" not in frame.columns:
        frame["source"] = "csv"
    if "went_to_penalties" not in frame.columns:
        frame["went_to_penalties"] = 0
    if "penalty_winner" not in frame.columns:
        frame["penalty_winner"] = ""

    frame = frame.copy()
    frame.attrs["missing_required_columns"] = missing_required
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["team_a"] = frame["team_a"].astype("string").str.strip()
    frame["team_b"] = frame["team_b"].astype("string").str.strip()
    frame["team_a_score"] = pd.to_numeric(frame["team_a_score"], errors="coerce")
    frame["team_b_score"] = pd.to_numeric(frame["team_b_score"], errors="coerce")
    frame["neutral"] = frame["neutral"].map(_parse_boolish)
    frame["went_to_penalties"] = frame["went_to_penalties"].map(_parse_boolish)
    for column in ["tournament", "stage", "country", "city", "source", "penalty_winner"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame[MATCH_OUTPUT_COLUMNS]


def normalize_international_results(
    raw_matches: pd.DataFrame,
    shootouts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = normalize_matches(raw_matches)
    frame["source"] = "real_csv"
    frame["stage"] = frame.apply(_infer_stage, axis=1)
    if shootouts is not None:
        frame = merge_shootouts(frame, shootouts)
    return frame


def merge_shootouts(matches: pd.DataFrame, raw_shootouts: pd.DataFrame) -> pd.DataFrame:
    if raw_shootouts.empty:
        return matches
    shootouts = _rename_aliases(
        raw_shootouts,
        {
            "date": ["date", "match_date"],
            "team_a": ["team_a", "home_team", "home"],
            "team_b": ["team_b", "away_team", "away"],
            "winner": ["winner", "penalty_winner", "shootout_winner"],
            "first_shooter": ["first_shooter"],
        },
    ).copy()
    required = {"date", "team_a", "team_b", "winner"}
    missing = sorted(required - set(shootouts.columns))
    if missing:
        merged = matches.copy()
        merged.attrs["shootout_errors"] = [f"Shootouts file missing required columns: {missing}"]
        return merged

    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
    for column in ["team_a", "team_b", "winner"]:
        shootouts[column] = shootouts[column].astype("string").str.strip()
    shootouts = shootouts.dropna(subset=["date", "team_a", "team_b", "winner"])
    shootouts = shootouts[["date", "team_a", "team_b", "winner"]].drop_duplicates()

    merged = matches.merge(
        shootouts,
        how="left",
        on=["date", "team_a", "team_b"],
    )
    winner = merged["winner"].fillna("").astype(str).str.strip()
    merged["went_to_penalties"] = winner.ne("").astype(int)
    merged["penalty_winner"] = winner
    return merged.drop(columns=["winner"])[MATCH_OUTPUT_COLUMNS]


def normalize_teams(raw_teams: pd.DataFrame, allow_missing_rank: bool = False) -> pd.DataFrame:
    frame = _rename_aliases(raw_teams, TEAM_ALIASES)
    for column in TEAM_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if "confederation" not in frame.columns:
        frame["confederation"] = "UNKNOWN"
    for column in ["match_count", "first_match_date", "last_match_date"]:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame.copy()
    frame["team"] = frame["team"].astype("string").str.strip()
    frame["confederation"] = frame["confederation"].fillna("UNKNOWN").astype(str).str.strip()
    frame["initial_elo"] = pd.to_numeric(frame["initial_elo"], errors="coerce")
    frame["fifa_rank"] = pd.to_numeric(frame["fifa_rank"], errors="coerce")
    if allow_missing_rank:
        frame["initial_elo"] = frame["initial_elo"].fillna(1500)
    frame["match_count"] = pd.to_numeric(frame["match_count"], errors="coerce")
    for column in ["first_match_date", "last_match_date"]:
        frame[column] = frame[column].fillna("").astype(str)
    return frame[TEAM_OUTPUT_COLUMNS]


def build_teams_from_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=TEAM_OUTPUT_COLUMNS)
    records: list[dict[str, object]] = []
    for team in sorted(set(matches["team_a"].dropna().astype(str)) | set(matches["team_b"].dropna().astype(str))):
        if not team.strip():
            continue
        team_matches = matches.loc[(matches["team_a"] == team) | (matches["team_b"] == team)]
        dates = pd.to_datetime(team_matches["date"], errors="coerce").dropna()
        records.append(
            {
                "team": team,
                "confederation": "UNKNOWN",
                "initial_elo": 1500,
                "fifa_rank": pd.NA,
                "match_count": int(len(team_matches)),
                "first_match_date": dates.min().date().isoformat() if not dates.empty else "",
                "last_match_date": dates.max().date().isoformat() if not dates.empty else "",
            }
        )
    return pd.DataFrame(records, columns=TEAM_OUTPUT_COLUMNS)


def validate_normalized_dataset(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    source_type: str = "csv",
) -> DatasetValidationReport:
    report = DatasetValidationReport(
        matches_count=int(len(matches)),
        teams_count=int(len(teams)),
        source_type=source_type,
    )
    if matches.empty:
        report.errors.append("Dataset is empty")
        return report
    missing_required = list(matches.attrs.get("missing_required_columns", []))
    if missing_required:
        report.errors.append(f"Required match columns could not be normalized: {missing_required}")
    shootout_errors = list(matches.attrs.get("shootout_errors", []))
    report.errors.extend(shootout_errors)
    missing_columns = [column for column in MATCH_REQUIRED_COLUMNS if column not in matches.columns]
    if missing_columns:
        report.errors.append(f"Missing required match columns after normalization: {missing_columns}")
    missing_team_columns = [column for column in TEAM_REQUIRED_COLUMNS if column not in teams.columns]
    if missing_team_columns:
        report.errors.append(f"Missing required team columns after normalization: {missing_team_columns}")

    report.missing_values = {
        column: int(matches[column].isna().sum())
        for column in matches.columns
        if int(matches[column].isna().sum()) > 0
    }
    report.invalid_dates = int(matches["date"].isna().sum()) if "date" in matches else 0
    report.invalid_scores = int(
        matches["team_a_score"].isna().sum()
        + matches["team_b_score"].isna().sum()
        + (matches["team_a_score"] < 0).fillna(False).sum()
        + (matches["team_b_score"] < 0).fillna(False).sum()
    )
    report.empty_team_names = int(
        matches["team_a"].fillna("").eq("").sum()
        + matches["team_b"].fillna("").eq("").sum()
        + teams["team"].fillna("").eq("").sum()
    )
    duplicate_subset = ["date", "team_a", "team_b", "team_a_score", "team_b_score", "tournament"]
    report.duplicate_rows = int(matches.duplicated(subset=duplicate_subset).sum())
    report.number_of_tournaments = int(matches["tournament"].replace("", pd.NA).dropna().nunique())
    tournament_text = matches["tournament"].fillna("").astype(str).str.lower()
    stage_text = matches["stage"].fillna("").astype(str).str.lower()
    world_cup_finals_mask = (
        tournament_text.str.contains("world cup", regex=False)
        & ~tournament_text.str.contains("qualif", regex=False)
    )
    report.world_cup_match_count = int(world_cup_finals_mask.sum())
    friendly_mask = tournament_text.str.contains("friendly", regex=False) | stage_text.eq("friendly")
    report.friendly_match_count = int(friendly_mask.sum())
    report.shootout_match_count = int(matches.get("went_to_penalties", pd.Series(dtype=int)).fillna(0).astype(int).sum())

    if report.invalid_dates:
        report.errors.append(f"Invalid dates: {report.invalid_dates}")
    if report.invalid_scores:
        report.errors.append(f"Invalid scores: {report.invalid_scores}")
    if report.empty_team_names:
        report.errors.append(f"Empty team names: {report.empty_team_names}")
    if report.duplicate_rows:
        report.warnings.append(f"Duplicate matches detected: {report.duplicate_rows}")

    team_names = {str(team) for team in teams["team"].dropna() if str(team).strip()}
    match_teams = {
        *{str(team) for team in matches["team_a"].dropna() if str(team).strip()},
        *{str(team) for team in matches["team_b"].dropna() if str(team).strip()},
    }
    report.unknown_teams = sorted(match_teams - team_names)
    if report.unknown_teams:
        report.errors.append(f"Unknown teams in matches: {report.unknown_teams[:20]}")

    if teams["team"].duplicated().any():
        duplicated = sorted(teams.loc[teams["team"].duplicated(), "team"].astype(str).unique())
        report.errors.append(f"Duplicate teams: {duplicated[:20]}")

    if teams["initial_elo"].isna().any():
        report.errors.append("Team initial_elo contains missing or non-numeric values")
    if teams["fifa_rank"].isna().any() and source_type != "international-results":
        report.errors.append("Team fifa_rank contains missing or non-numeric values")
    elif teams["fifa_rank"].isna().any():
        report.warnings.append("Team fifa_rank contains missing values; rank features should be replaced with real rankings before making claims")

    match_counts = (
        pd.concat([matches["team_a"], matches["team_b"]])
        .dropna()
        .astype(str)
        .value_counts()
    )
    report.teams_with_few_matches = sorted(match_counts.loc[match_counts <= 2].index.tolist())
    if report.teams_with_few_matches:
        report.warnings.append(f"Teams with very few matches: {report.teams_with_few_matches[:20]}")
    report.suspicious_team_names = sorted(
        team
        for team in match_counts.index.tolist()
        if _is_suspicious_team_name(team)
    )
    if report.suspicious_team_names:
        report.warnings.append(f"Suspicious team names: {report.suspicious_team_names[:20]}")

    valid_dates = matches["date"].dropna()
    if not valid_dates.empty:
        report.date_min = valid_dates.min().date().isoformat()
        report.date_max = valid_dates.max().date().isoformat()
    return report


def missing_required_match_columns(raw_matches: pd.DataFrame) -> list[str]:
    normalized_columns = {_normalize_column(column) for column in raw_matches.columns}
    missing = []
    for canonical in MATCH_REQUIRED_COLUMNS:
        candidates = MATCH_ALIASES.get(canonical, [canonical])
        if not any(_normalize_column(candidate) in normalized_columns for candidate in candidates):
            missing.append(canonical)
    return sorted(missing)


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


def _parse_boolish(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "neutral"}:
        return 1
    if text in {"0", "false", "no", "n", "home"}:
        return 0
    return 0


def _load_or_build_teams(
    matches: pd.DataFrame,
    teams_path: str | Path | None,
    allow_missing_rank: bool,
) -> pd.DataFrame:
    if teams_path is not None and Path(teams_path).exists():
        return normalize_teams(pd.read_csv(teams_path), allow_missing_rank=allow_missing_rank)
    return build_teams_from_matches(matches)


def _infer_stage(row: pd.Series) -> str:
    stage = str(row.get("stage", "") or "").strip().lower().replace(" ", "_").replace("-", "_")
    tournament = str(row.get("tournament", "") or "").strip().lower()
    if stage and stage != "unknown":
        if stage in {"round_of_16", "quarterfinal", "semifinal", "third_place", "final", "group", "friendly", "qualifier"}:
            return stage
        if "round of 16" in stage or "last 16" in stage:
            return "round_of_16"
        if "quarter" in stage:
            return "quarterfinal"
        if "semi" in stage:
            return "semifinal"
        if "third" in stage:
            return "third_place"
        if "final" in stage:
            return "final"
        if "group" in stage:
            return "group"
    if "friendly" in tournament:
        return "friendly"
    if "qualif" in tournament:
        return "qualifier"
    return "unknown"


def _is_suspicious_team_name(team: str) -> bool:
    stripped = team.strip()
    if len(stripped) <= 1:
        return True
    if stripped.lower() in {"nan", "none", "unknown", "tbd"}:
        return True
    if any(char in stripped for char in ["?", "\ufffd"]):
        return True
    return False
