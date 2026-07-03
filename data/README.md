# CupCast AI Data

The repository includes deterministic synthetic sample data so the full pipeline can run without paid APIs or external downloads.

Files:

- `raw/teams.csv`: team metadata, confederation, initial Elo, and FIFA-rank-style signal.
- `raw/historical_matches.csv`: 180 synthetic historical fixtures with scores, stage, tournament, and neutral-venue flag.
- `raw/football_text.txt`: 81 training examples for the custom mini-LLM analyst.
- `real/`: user-supplied real international football CSVs. The repo only tracks `.gitkeep` here.
- `processed/`: generated corpora, probability snapshots, and local evaluation outputs.

Regenerate the sample data:

```bash
python scripts/create_sample_data.py
```

The sample data is intentionally realistic enough to validate architecture, feature engineering, testing, and demos. For production-grade predictions, replace it with verified match history, current Elo ratings, player availability, squad form, and tournament bracket data.

## Real Data CSVs

Use `scripts/validate_dataset.py` before training on any external CSV.

For the bundled synthetic data:

```bash
python scripts/validate_dataset.py --matches data/raw/historical_matches.csv --teams data/raw/teams.csv
```

For real international results:

```bash
python scripts/validate_dataset.py \
  --matches data/real/results.csv \
  --shootouts data/real/shootouts.csv \
  --source international-results
```

Expected real files:

- `real/results.csv`: required.
- `real/shootouts.csv`: optional but recommended for knockout advancement history.
- `real/goalscorers.csv`: optional and currently reserved for later feature work.

Accepted match aliases include `home_team`/`away_team`, `home_score`/`away_score`, `competition`, `round`, `neutral_venue`, `country`, and `city`. After normalization the pipeline requires `date`, `team_a`, `team_b`, `team_a_score`, and `team_b_score`.

The teams file must provide team names plus numeric Elo and ranking signals. Accepted aliases include `name`, `team_name`, `elo`, `rating`, `rank`, and `fifa_ranking`.

If no teams file exists for real data, generate one from match history:

```bash
python scripts/build_teams_from_matches.py \
  --matches data/real/results.csv \
  --out data/processed/teams_real.csv
```

Generated teams use `UNKNOWN` confederation, `initial_elo=1500`, and blank `fifa_rank`. FIFA rankings and confederations are not fabricated.
