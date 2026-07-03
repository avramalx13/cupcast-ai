# CupCast AI

A World Cup prediction engine with a custom mini-LLM analyst.

CupCast AI is an end-to-end portfolio project that separates prediction from explanation:

```text
Data Source
  -> Validation
  -> Feature Engineering
  -> Model Training
  -> Backtesting + Calibration
  -> Match Prediction
  -> Tournament Simulation
  -> Probability Snapshots
  -> Analyst Explanation
  -> FastAPI
  -> Next.js Dashboard
```

The LLM does not randomly choose a World Cup winner. The structured model produces probabilities, the simulator turns those probabilities into tournament odds, and the analyst layer explains the output.

This project is not claiming perfect football prediction. It demonstrates an end-to-end ML system for probabilistic forecasting, simulation, evaluation, and explanation.

This project does not use paid prediction APIs. Forecasts are generated from historical match results, optional local rating CSVs, engineered features, model comparison, calibration, and Monte Carlo simulation.

## Architecture

```text
data/raw/*.csv
  -> cupcast.prediction_engine
     -> Elo, recent form, rank, goals, stage features
     -> baseline/model comparison and calibration
     -> scikit-learn 3-class match models
  -> cupcast.simulator
     -> simple knockout or World Cup 2026 skeleton
     -> repeated Monte Carlo runs
     -> probability snapshots
  -> cupcast.analyst
     -> deterministic templates
     -> optional custom mini-LLM generation
  -> cupcast.api
     -> FastAPI endpoints for predictions, simulation, updates, and analyst text
  -> frontend
     -> Next.js dashboard scaffold
```

## Implemented

- Custom decoder-only Transformer in PyTorch.
- Word and byte-level tokenizer support with special tokens.
- Mini-LLM training, checkpointing, resume, and generation.
- Structured football prediction engine with Elo and recent-form features.
- Real-data-ready CSV normalization and validation.
- International results CSV adapter for common national-team datasets.
- Shootout merge support that preserves regulation draws and records penalty winners separately.
- Automatic team metadata generation from real match files without fake rankings or confederations.
- Majority, uniform random, Elo-only logistic regression, recent-form-only, feature logistic regression, random forest, and gradient boosting comparison.
- Accuracy, log loss, Brier score, confusion matrix, calibration bins, and expected calibration error.
- Real World Cup backtesting command with strict pre-tournament training splits.
- Monte Carlo simple knockout simulator with title and round probabilities.
- World Cup 2026 knockout skeleton with 12 groups, Round of 32, Round of 16, quarterfinals, semifinals, third-place match placeholder, and final.
- Full pre-tournament World Cup simulator from group stage through champion probabilities.
- File-backed async-style backtesting jobs for the API.
- Mini-LLM evaluation report with loss, perplexity, repetition, unknown-token, and format-compliance checks.
- Probability snapshot writer.
- Live result update flow for knockout results and penalty winners.
- Template analyst with optional mini-LLM fallback.
- FastAPI backend.
- Next.js/Tailwind frontend scaffold.
- Synthetic sample data and pytest coverage.

## Why The LLM Does Not Predict Winners

This project contains a custom mini language model trained from scratch for football explanation generation. It is not intended to compete with commercial LLMs. The prediction quality comes from the structured match prediction engine and Monte Carlo simulator. The mini-LLM is used as an analyst layer to explain model outputs.

## Install

From `D:\WorldCupSIM_AI\cupcast-ai`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
python scripts/create_sample_data.py
```

During development in this workspace you can also reuse the existing venv:

```bash
..\cupcast-mini-llm\.venv\Scripts\python -m pip install -e . --no-build-isolation
```

## Mini-LLM

Train the mini-LLM:

```bash
python scripts/train_mini_llm.py --config configs/mini_llm_small.yaml --data data/raw/football_text.txt --checkpoint-dir checkpoints/mini_llm
```

For a quick CPU smoke run:

```bash
python scripts/train_mini_llm.py --config configs/mini_llm_tiny.yaml --data data/raw/football_text.txt --checkpoint-dir checkpoints/mini_llm_tiny --max-steps 50
```

Generate analysis:

```bash
python -m cupcast.mini_llm.generate --checkpoint checkpoints/mini_llm/best.pt --prompt "### Context:\nTeam A: France\nTeam B: Brazil\n"
```

The mini-LLM architecture is:

```text
token IDs
-> token embeddings + learned position embeddings
-> decoder block x N
-> final LayerNorm
-> tied LM head
-> next-token loss
```

Each decoder block uses pre-norm causal self-attention, a feed-forward network, residual connections, dropout, and a causal mask.

Evaluate a checkpoint:

```bash
python scripts/evaluate_mini_llm.py --checkpoint checkpoints/mini_llm/best.pt --eval-data data/raw/football_text_eval.txt
```

If the checkpoint is missing, the script writes `models/mini_llm_eval_report.json` with `status=missing_checkpoint` instead of failing with an unclear traceback.

The mini-LLM is an educational domain-specific language model. It is evaluated for loss, perplexity, repetition, and format compliance, but prediction correctness comes from the structured prediction engine.

## Synthetic vs Real Data Mode

The repo ships with deterministic synthetic data so commands run immediately. Synthetic data is only for architecture demos.

The API and frontend expose the active data mode:

```yaml
data:
  mode: synthetic  # synthetic | real
  matches_path: data/raw/historical_matches.csv
  shootouts_path:
  teams_path: data/raw/teams.csv
```

For real data, set `mode: real` and point the paths at `data/real/results.csv`, `data/real/shootouts.csv`, and `data/processed/teams_real.csv`.

Validate the bundled synthetic data:

```bash
python scripts/validate_dataset.py --matches data/raw/historical_matches.csv --teams data/raw/teams.csv
```

Required match fields after normalization:

```text
date
team_a / home_team
team_b / away_team
team_a_score / home_score
team_b_score / away_score
```

Accepted optional aliases include:

```text
tournament / competition / league
neutral / neutral_venue
stage / round
country / host_country
city / venue_city
```

Required team fields:

```text
team / name / team_name
initial_elo / elo / rating
fifa_rank / rank
```

Validation reports match count, date range, team count, missing values, duplicate rows, invalid scores, empty team names, and unknown teams. Bad data is not silently ignored.

## Using Real Historical Football Data

Manually place real international football CSVs here:

```text
data/real/results.csv
data/real/shootouts.csv
data/real/goalscorers.csv
```

The expected `results.csv` shape follows common international-results datasets:

```text
date
home_team
away_team
home_score
away_score
tournament
city
country
neutral
```

Validate real data:

```bash
python scripts/validate_dataset.py ^
  --matches data/real/results.csv ^
  --shootouts data/real/shootouts.csv ^
  --source international-results
```

This writes `models/dataset_validation_report.json` with:

```text
valid
source_type
number_of_matches
date_range
number_of_unique_teams
number_of_tournaments
world_cup_match_count
friendly_match_count
missing_values_by_column
duplicate_match_count
invalid_score_count
unknown_team_count
shootout_match_count
teams_with_few_matches
suspicious_team_names
```

Build teams from matches if no real teams file exists:

```bash
python scripts/build_teams_from_matches.py ^
  --matches data/real/results.csv ^
  --out data/processed/teams_real.csv
```

Generated teams use:

```text
confederation = UNKNOWN
initial_elo = 1500
fifa_rank = blank
```

The pipeline uses an internal unknown-rank sentinel for model features, but the generated CSV does not pretend to know FIFA rankings.

Compare models on real data:

```bash
python scripts/compare_models.py ^
  --config configs/prediction_model.yaml ^
  --matches data/real/results.csv ^
  --teams data/processed/teams_real.csv ^
  --shootouts data/real/shootouts.csv ^
  --source international-results ^
  --train-before 2022 ^
  --test-tournament "FIFA World Cup"
```

This writes `models/comparison_results.json` and `models/model_leaderboard.json`. The leaderboard depends entirely on the user's dataset and should not be copied as a universal claim.

Backtest real World Cups:

```bash
python scripts/backtest_world_cups.py ^
  --matches data/real/results.csv ^
  --shootouts data/real/shootouts.csv ^
  --years 2014 2018 2022
```

This writes `models/world_cup_backtest_results.json`. For each year, training uses only matches before that World Cup's first match date, and testing uses only World Cup matches from that year.

Run the full real-data evaluation and portfolio report pipeline:

```bash
python scripts/run_real_data_pipeline.py ^
  --matches data/real/results.csv ^
  --shootouts data/real/shootouts.csv ^
  --elo data/real/elo_ratings.csv ^
  --fifa data/real/fifa_rankings.csv ^
  --years 2014 2018 2022

python scripts/generate_portfolio_docs.py
python scripts/export_frontend_reports.py
```

Forecasting quality depends heavily on data quality. The synthetic dataset is only for architecture demos. Real claims should only be made after running validation, model comparison, calibration, and World Cup backtests on a real historical dataset.

## Real Data Results

These metrics are generated from the local real-data reports in `models/`. They are not hardcoded claims.

This project does not use paid prediction APIs. Forecasts are generated from historical match results, optional local rating CSVs, engineered features, model comparison, calibration, and Monte Carlo simulation.

### Dataset Summary

| Metric | Value |
|---|---:|
| Matches | 49484 |
| Teams | 336 |
| Date range | 1872-11-30 to 2026-06-30 |
| Tournaments | 200 |
| World Cup matches | 1103 |
| Shootouts | 679 |

### Model Leaderboard

| Model | Accuracy | Log loss | Brier score | ECE | Beats majority | Beats Elo |
|---|---:|---:|---:|---:|---:|---:|
| majority baseline | 0.4375 | 1.0743 | 0.6512 | 0.0540 | no | no |
| uniform random baseline | 0.4375 | 1.0986 | 0.6667 | 0.1042 | no | no |
| Elo-only logistic regression | 0.4844 | 1.0453 | 0.6039 | 0.0830 | yes | no |
| recent-form-only model | 0.4688 | 1.0928 | 0.6622 | 0.0835 | no | no |
| full feature logistic regression | 0.5156 | 1.0496 | 0.6144 | 0.1069 | yes | no |
| Poisson goal model | 0.4375 | 1.0724 | 0.6484 | 0.1118 | yes | no |
| calibrated feature logistic regression | 0.5156 | 1.0500 | 0.6151 | 0.0737 | yes | no |
| weighted probability ensemble | 0.4688 | 1.0202 | 0.6060 | 0.1177 | yes | yes |

### World Cup Backtests

| Year | Train | Test | Group | Knockout | Best overall | Best group | Best knockout | Accuracy | Log loss | Brier | ECE |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 2014 | 37861 | 64 | 0 | 0 | calibrated feature logistic regression | n/a | n/a | 0.5938 | 0.9487 | 0.5605 | 0.0722 |
| 2018 | 41639 | 64 | 0 | 0 | full feature logistic regression | n/a | n/a | 0.5625 | 0.9314 | 0.5503 | 0.0510 |
| 2022 | 45700 | 64 | 0 | 0 | weighted probability ensemble | n/a | n/a | 0.4688 | 1.0202 | 0.6060 | 0.1177 |

### Calibration

Calibration is compared with log loss, Brier score, reliability bins, and expected calibration error. Calibrated variants are shown beside uncalibrated models, and the docs only claim improvement when metrics show it.

### Feature Importance

- `random forest` top features: `elo_external_diff`, `avg_elo_diff`, `elo_diff`, `elo_b`, `elo_a`.
- `gradient boosting` top features: `elo_diff`, `elo_external_diff`, `avg_elo_diff`, `home_advantage_flag`, `team_b_goals_conceded_last_5`.
- `full feature logistic regression` top features: `elo_diff`, `elo_external_diff`, `avg_elo_diff`, `elo_a`, `elo_external_a`.

### Ablation Study

| Feature group | Status | Log loss | Accuracy | Reason |
|---|---:|---:|---:|---|
| elo_only | ok | 0.8680 | 0.6023 |  |
| form_only | ok | 0.9704 | 0.5380 |  |
| ranking_only | ok | 1.0543 | 0.4727 |  |
| schedule_only | ok | 1.0464 | 0.4740 |  |
| goals_only | ok | 0.9680 | 0.5390 |  |
| elo_plus_form | ok | 0.8610 | 0.6090 |  |
| all_features | ok | 0.8601 | 0.6090 |  |
| all_features_plus_external_ratings | unavailable | n/a | n/a | external rating CSVs not found |

### Error Analysis

- Predicted top-class misses: 75.
- Correct underdog calls: 0.
- Low-probability actual result: 2022 Argentina vs Saudi Arabia actual `B_WIN` at 0.0321.
- Low-probability actual result: 2022 Cameroon vs Brazil actual `A_WIN` at 0.0448.
- Low-probability actual result: 2022 South Korea vs Ghana actual `B_WIN` at 0.0807.

### Known Weaknesses

- World Cup-only tests contain few matches, so year-level metrics are noisy.
- Historical scores miss injuries, lineups, travel, tactical changes, weather, and market priors.
- Optional Elo/FIFA snapshots improve current-strength features only when the user supplies valid local CSVs.
- The model is a portfolio forecasting system, not betting-grade infrastructure.

### Interview Talking Points

- The LLM does not directly predict winners; structured models produce probabilities and the mini-LLM explains them.
- Leakage is prevented by building each row from pre-match team state and rating snapshots dated strictly before the match.
- Calibration matters because tournament simulation consumes probabilities, not just classes.
- Baselines matter because Elo, majority, Poisson, and ablations show whether complexity is useful.
- World Cup backtests are noisy because each tournament is a small test set.
- The ensemble combines complementary model families while reporting weights and failures.
- With paid data, the largest gains would likely come from lineups, injuries, player minutes, betting-market priors, and travel context.

### What The Results Mean

- Best model by comparison log loss: `weighted probability ensemble`.
- The best model beats the majority baseline on log loss.
- The best model beats the Elo-only baseline on log loss.
- The full feature logistic regression does not improve over Elo-only on the comparison split.
- Calibration is weak enough that probabilities should be interpreted cautiously.
- World Cup backtests are small samples, so year-to-year results should be treated as noisy.
- Historical match results alone miss lineups, injuries, player form, tactics, and betting-market information.

## Prediction Model

Train the structured model:

```bash
python scripts/train_prediction_model.py --config configs/prediction_model.yaml
```

Predict a match:

```bash
python -m cupcast.prediction_engine.predict --team-a France --team-b Brazil
```

Backtest:

```bash
python scripts/run_backtest.py --train-before 2022 --test-tournament "World Cup 2022"
```

Compare models and save `models/comparison_results.json`:

```bash
python scripts/compare_models.py --config configs/prediction_model.yaml --train-before 2022 --test-tournament "World Cup 2022"
```

Feature groups:

- Elo before match.
- FIFA-rank-style team strength.
- Recent form from the last 5 matches.
- Goals scored and conceded in the last 5 matches.
- Neutral venue flag.
- Stage encoding.

## Model Comparison and Calibration

`compare_models.py` evaluates:

- Majority baseline.
- Uniform random baseline.
- Elo-only logistic regression.
- Recent-form-only logistic regression.
- Feature-based logistic regression.
- Random forest.
- Gradient boosting.

The report includes accuracy, top-class accuracy, log loss, Brier score, calibration bins, expected calibration error, match counts, and a model leaderboard. This makes the baseline explicit and avoids pretending one model is best without comparison.

## Simulator

Run a tournament simulation:

```bash
python scripts/run_simulation.py --config configs/simulation.yaml --simulations 10000
```

For knockout matches, draw probability is converted into advancement probability:

```text
team_a_advances = team_a_win + draw * 0.5
team_b_advances = team_b_win + draw * 0.5
```

Snapshots are written to `data/processed/probability_snapshots.csv`.

## World Cup 2026 Format Support

The default command still uses `simple_16_team_knockout` because the bundled demo teams file has 20 teams. A `world_cup_2026` skeleton is implemented for 48-team inputs:

```text
12 groups of 4
top 2 from each group
8 best third-place teams
Round of 32
Round of 16
Quarterfinal
Semifinal
Third-place match
Final
```

Group ranking uses points, goal difference, goals scored, then deterministic team-name fallback. Scoreline generation is intentionally simple: it samples W/D/L from model probabilities, then maps the outcome to plausible fixed scorelines. This is documented as a limitation, not a final football simulation model.

## Full Tournament Simulation

Run the 48-team World Cup simulation from the start of the group stage:

```bash
python scripts/simulate_full_tournament.py --groups data/tournaments/world_cup_2026_groups.yaml --simulations 1000 --seed 42
```

The full tournament simulator:

- simulates all six matches in every group from the existing trained match predictor,
- ranks groups by points, goal difference, goals scored, then deterministic seeded tie-breaker,
- qualifies the top two teams from each group plus the eight strongest third-place teams,
- builds an approximate Round of 32 bracket,
- simulates Round of 32, Round of 16, quarterfinals, semifinals, final, and champion,
- writes `models/full_tournament_simulation.json` and `models/full_tournament_simulation.md`.

The report includes group winner probabilities, qualification probabilities, best third-place qualification probabilities, round progression probabilities, top title contenders, dark horses, volatile teams, and an analyst explanation. No new ML model is added; the simulator consumes probabilities from the trained prediction model.

Honesty note: the 2026 third-place knockout placement is approximated deterministically unless the official placement matrix is implemented. This affects exact path probabilities but still provides useful group qualification and tournament-strength estimates.

## Live Update

Apply a knockout result:

```bash
python scripts/update_after_match.py --team-a Germany --team-b Paraguay --score-a 1 --score-b 1 --penalty-winner Paraguay
```

The update flow:

```text
result
-> validate winner
-> update knockout state
-> expose Elo update calculation
-> rerun simulation before and after
-> return probability changes
-> analyst can explain the movement
```

## API

Start the backend:

```bash
uvicorn cupcast.api.main:app --reload
```

On Windows from the repo root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_api_dev.ps1
```

Keep this terminal open while using the dashboard. If the frontend shows `API request failed: 503`, check the backend with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_api_dev.ps1
```

Endpoints:

- `GET /health`
- `GET /data/status`
- `POST /predict/match`
- `GET /teams`
- `GET /simulation/latest`
- `POST /simulation/run`
- `GET /simulation/full-tournament/latest`
- `POST /simulation/full-tournament/run`
- `POST /matches/update-result`
- `POST /analyst/explain`
- `POST /backtesting/run`
- `GET /backtesting/jobs/{job_id}`
- `GET /backtesting/summary`

Example:

```bash
curl -X POST http://localhost:8000/predict/match ^
  -H "Content-Type: application/json" ^
  -d "{\"team_a\":\"France\",\"team_b\":\"Brazil\"}"
```

## Frontend

The frontend is scaffolded in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

On Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File frontend/start_frontend_dev.ps1
```

This starts the local dashboard at `https://127.0.0.1:3001`. The frontend calls the backend through its same-origin proxy at `/api/cupcast` by default. Keep FastAPI running on `http://127.0.0.1:8000`, or set `CUPCAST_API_INTERNAL_BASE` before starting Next.js if the backend runs elsewhere. Only set `NEXT_PUBLIC_CUPCAST_API_BASE` if you intentionally want the browser to bypass the proxy and call a public API URL directly.

Export static frontend reports and dataset-driven teams:

```bash
python scripts/export_frontend_reports.py
```

This writes `frontend/public/reports/teams_real.json` and report artifacts for static portfolio mode. Static mode can show saved report results, but exact live matchup predictions require API mode unless the matchup exists in `demo_matchups.json`.

It also copies `full_tournament_simulation.json` and `full_tournament_simulation.md` when they exist. The frontend full tournament page reads the cached API report first and falls back to `frontend/public/reports/full_tournament_simulation.json` in static mode.

The frontend uses original CupCast AI artwork in `frontend/public/brand/`, including the mark-only app logo and mascot assets. The full logo art is kept as optional source artwork, while the visible app header uses `cupcast-header-mark-art.png` and the homepage showcase uses the mascot PNGs/SVG fallbacks. Do not commit FIFA logos, official mascot artwork, screenshots, scraped images, or any asset unless you have explicit rights to use it. The app displays an unofficial-project disclaimer and is not affiliated with FIFA.

Pages:

- `/`: dashboard home.
- `/predictions`: match predictor.
- `/simulator`: tournament odds and probability timeline.
- `/full-tournament`: full 48-team group-to-final simulator.
- `/analyst`: controlled analyst prompts.
- `/backtesting`: metrics page scaffold.
- `/backtesting`: triggers a cached backtest job, polls status, and displays model comparison results.

## Tests

```bash
pytest
```

Current coverage includes tokenizer roundtrip, dataset chunking, model shapes, causal mask, generation smoke test, dataset normalization/validation, international-results normalization, shootout merge behavior, generated teams, no-future-leakage feature checks, Elo behavior, feature engineering, prediction probability sums, model comparison, leaderboard schema, calibration, backtest jobs, World Cup backtest split correctness, World Cup 2026 skeleton behavior, full tournament group validation and simulation accounting, mini-LLM evaluation helpers, simulator bounds, and API responses.

## Limitations

- Included match and team data is synthetic by default, not production-grade.
- Real data is supported through validated CSV ingestion, but no live data provider or auto-download is bundled.
- The default model is a baseline, not a calibrated elite forecasting model.
- The default bracket is a simple 16-team knockout bracket for local simulation.
- World Cup 2026 support includes a full 48-team pre-tournament simulation, but group-stage score simulation is intentionally simple.
- The 2026 third-place knockout placement is approximated deterministically until an official matrix is implemented.
- Player-level form and goalscorer features are not yet connected in this monorepo version.
- The frontend is functional but still lacks production-grade charts and state persistence.
- Mini-LLM output quality depends heavily on corpus size and CPU training time.

## Production Roadmap

- Replace synthetic data with verified historical international data and current Elo snapshots.
- Add current national Elo feeds and squad/player form features.
- Improve the World Cup 2026 group-stage score model.
- Add reliability plots and calibration visualization.
- Persist bracket state and snapshots in a database.
- Move backtesting/simulation jobs to a real queue for deployment.
- Expand frontend charts and error states.
- Train the mini-LLM on larger structured analyst corpora.
