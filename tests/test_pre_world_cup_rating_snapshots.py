from __future__ import annotations

import pandas as pd

from scripts.create_pre_world_cup_rating_snapshots import (
    PRE_WORLD_CUP_FIFA_RANKS,
    build_elo_snapshots,
    build_fifa_rank_snapshots,
)


def test_pre_world_cup_fifa_ranks_include_expected_reference_rows() -> None:
    ranks = build_fifa_rank_snapshots(PRE_WORLD_CUP_FIFA_RANKS)

    france_2022 = ranks.loc[(ranks["date"] == "2022-10-06") & (ranks["team"] == "France")]
    germany_2014 = ranks.loc[(ranks["date"] == "2014-06-05") & (ranks["team"] == "Germany")]

    assert int(france_2022.iloc[0]["rank"]) == 4
    assert int(germany_2014.iloc[0]["rank"]) == 2
    assert len(ranks) == 96


def test_elo_snapshots_only_use_matches_before_snapshot_date() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": "2014-06-01",
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 1,
                "team_b_score": 0,
            },
            {
                "date": "2014-06-05",
                "team_a": "France",
                "team_b": "Brazil",
                "team_a_score": 0,
                "team_b_score": 5,
            },
        ]
    )

    ratings = build_elo_snapshots(
        matches,
        snapshot_dates=["2014-06-05"],
        base_elo=1500.0,
        elo_k=28.0,
    )

    france = float(ratings.loc[ratings["team"] == "France"].iloc[0]["elo"])
    brazil = float(ratings.loc[ratings["team"] == "Brazil"].iloc[0]["elo"])
    assert france > brazil
