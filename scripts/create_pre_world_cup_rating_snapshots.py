from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cupcast.prediction_engine.data_sources import normalize_matches
from cupcast.prediction_engine.elo import match_result_for_team_a, update_elo


# Static pre-tournament FIFA ranks for qualified teams only.
# Sources:
# - 2014 FIFA World Cup page: last pre-tournament FIFA ranking, FIFA ranking dated 2014-06-05.
# - 2018 FIFA World Cup page: qualified team rankings in June 2018, before the 2018-06-14 opener.
# - 2022 FIFA World Cup page: final FIFA ranking positions before the tournament, dated 2022-10-06.
PRE_WORLD_CUP_FIFA_RANKS: dict[str, dict[str, int]] = {
    "2014-06-05": {
        "Algeria": 22,
        "Argentina": 5,
        "Australia": 62,
        "Belgium": 11,
        "Bosnia and Herzegovina": 21,
        "Brazil": 3,
        "Cameroon": 56,
        "Chile": 14,
        "Colombia": 8,
        "Costa Rica": 28,
        "Croatia": 18,
        "Ecuador": 26,
        "England": 10,
        "France": 17,
        "Germany": 2,
        "Ghana": 37,
        "Greece": 12,
        "Honduras": 33,
        "Iran": 43,
        "Italy": 9,
        "Ivory Coast": 23,
        "Japan": 46,
        "Mexico": 20,
        "Netherlands": 15,
        "Nigeria": 44,
        "Portugal": 4,
        "Russia": 19,
        "South Korea": 57,
        "Spain": 1,
        "Switzerland": 6,
        "United States": 13,
        "Uruguay": 7,
    },
    "2018-06-07": {
        "Argentina": 5,
        "Australia": 36,
        "Belgium": 3,
        "Brazil": 2,
        "Colombia": 16,
        "Costa Rica": 23,
        "Croatia": 20,
        "Denmark": 12,
        "Egypt": 45,
        "England": 12,
        "France": 7,
        "Germany": 1,
        "Iceland": 22,
        "Iran": 37,
        "Japan": 61,
        "Mexico": 15,
        "Morocco": 41,
        "Nigeria": 48,
        "Panama": 55,
        "Peru": 11,
        "Poland": 8,
        "Portugal": 4,
        "Russia": 70,
        "Saudi Arabia": 67,
        "Senegal": 27,
        "Serbia": 34,
        "South Korea": 57,
        "Spain": 10,
        "Sweden": 24,
        "Switzerland": 6,
        "Tunisia": 21,
        "Uruguay": 14,
    },
    "2022-10-06": {
        "Argentina": 3,
        "Australia": 38,
        "Belgium": 2,
        "Brazil": 1,
        "Cameroon": 43,
        "Canada": 41,
        "Costa Rica": 31,
        "Croatia": 12,
        "Denmark": 10,
        "Ecuador": 44,
        "England": 5,
        "France": 4,
        "Germany": 11,
        "Ghana": 61,
        "Iran": 20,
        "Japan": 24,
        "Mexico": 13,
        "Morocco": 22,
        "Netherlands": 8,
        "Poland": 26,
        "Portugal": 9,
        "Qatar": 50,
        "Saudi Arabia": 51,
        "Senegal": 18,
        "Serbia": 21,
        "South Korea": 28,
        "Spain": 7,
        "Switzerland": 15,
        "Tunisia": 30,
        "United States": 16,
        "Uruguay": 14,
        "Wales": 19,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create offline pre-World-Cup Elo and FIFA ranking snapshots for backtests."
    )
    parser.add_argument("--matches", default="data/processed/real_completed_matches.csv")
    parser.add_argument("--elo-output", default="data/real/elo_ratings.csv")
    parser.add_argument("--fifa-output", default="data/real/fifa_rankings.csv")
    parser.add_argument("--base-elo", type=float, default=1500.0)
    parser.add_argument("--elo-k", type=float, default=28.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches = load_completed_matches(args.matches)
    snapshot_dates = sorted(PRE_WORLD_CUP_FIFA_RANKS)
    elo = build_elo_snapshots(
        matches,
        snapshot_dates=snapshot_dates,
        base_elo=args.base_elo,
        elo_k=args.elo_k,
    )
    fifa = build_fifa_rank_snapshots(PRE_WORLD_CUP_FIFA_RANKS)
    write_csv(elo, args.elo_output)
    write_csv(fifa, args.fifa_output)
    print(f"matches={len(matches)}")
    print(f"elo_rows={len(elo)} output={args.elo_output}")
    print(f"fifa_rows={len(fifa)} output={args.fifa_output}")
    return 0


def load_completed_matches(path: str | Path) -> pd.DataFrame:
    matches_path = Path(path)
    if not matches_path.exists():
        raise FileNotFoundError(f"Matches file not found: {matches_path}")
    raw = pd.read_csv(matches_path)
    matches = normalize_matches(raw)
    matches = matches.dropna(subset=["date", "team_a", "team_b", "team_a_score", "team_b_score"])
    matches = matches.sort_values("date").reset_index(drop=True)
    return matches


def build_elo_snapshots(
    matches: pd.DataFrame,
    snapshot_dates: Iterable[str],
    base_elo: float,
    elo_k: float,
) -> pd.DataFrame:
    dates = [pd.Timestamp(date) for date in snapshot_dates]
    rows: list[dict[str, object]] = []
    ratings: dict[str, float] = {}
    match_index = 0
    ordered = matches.sort_values("date").reset_index(drop=True)

    for snapshot_date in dates:
        while match_index < len(ordered) and pd.Timestamp(ordered.loc[match_index, "date"]) < snapshot_date:
            match = ordered.loc[match_index]
            team_a = str(match["team_a"]).strip()
            team_b = str(match["team_b"]).strip()
            if not team_a or not team_b:
                match_index += 1
                continue
            ratings.setdefault(team_a, float(base_elo))
            ratings.setdefault(team_b, float(base_elo))
            result_a = match_result_for_team_a(int(match["team_a_score"]), int(match["team_b_score"]))
            ratings[team_a], ratings[team_b] = update_elo(
                ratings[team_a],
                ratings[team_b],
                result_a=result_a,
                k=elo_k,
            )
            match_index += 1

        for team, elo in sorted(ratings.items()):
            rows.append(
                {
                    "date": snapshot_date.date().isoformat(),
                    "team": team,
                    "elo": round(float(elo), 3),
                    "source": f"local_results_pre_wc_k{elo_k:g}",
                }
            )
    return pd.DataFrame(rows, columns=["date", "team", "elo", "source"])


def build_fifa_rank_snapshots(ranks_by_date: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = [
        {
            "date": date,
            "team": team,
            "rank": rank,
            "source": "static_pre_world_cup_fifa_rank",
        }
        for date, ranks in ranks_by_date.items()
        for team, rank in sorted(ranks.items())
    ]
    return pd.DataFrame(rows, columns=["date", "team", "rank", "source"])


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
