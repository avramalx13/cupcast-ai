# CupCast AI Real Data Results

This report was generated locally from user-supplied real historical match data.

## Dataset Summary

- Match count: 49484
- Date range: 1872-11-30 to 2026-06-30
- Team count: 336
- Tournament count: 200
- World Cup match count: 1103
- Shootout count: 679

## Warnings

- Duplicate matches detected: 1
- Elo ratings file not found: data/real/elo_ratings.csv; continuing with internal Elo features
- Excluded 12 incomplete/unscored rows before validation; details written to data\processed\real_excluded_matches.csv
- FIFA rankings file not found: data/real/fifa_rankings.csv; continuing without external FIFA features
- Team fifa_rank contains missing values; rank features should be replaced with real rankings before making claims
- Teams file missing; generated team metadata from matches with UNKNOWN confederation and blank fifa_rank
- Teams with very few matches: ['Asturias', 'Aymara', 'Central Spain', 'Cilento', 'Délvidék', 'Elba Island', 'Madrid', 'Mapuche', 'Marshall Islands', 'Maule Sur', 'Niue', 'Palau', 'Ryūkyū', 'Saugeais', 'Seborga', 'Surrey', 'Ticino', 'West Papua', 'Yoruba Nation']
- elo ratings file missing; continuing without external elo features
- fifa ratings file missing; continuing without external fifa features

## Model Leaderboard

| Model | Status | Accuracy | Log loss | Brier | ECE | Beats majority | Beats Elo | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| weighted_probability_ensemble | ok | 0.4844 | 1.0190 | 0.6053 | 0.1047 | yes | yes | beats selected baselines on log loss for this split |
| elo_logistic_regression | ok | 0.4844 | 1.0453 | 0.6039 | 0.0830 | yes | no | beats selected baselines on log loss for this split |
| logistic_regression | ok | 0.5156 | 1.0458 | 0.6121 | 0.0884 | yes | no | does not beat Elo baseline on log loss |
| feature_logistic_regression_calibrated | ok | 0.5156 | 1.0465 | 0.6131 | 0.0497 | yes | no | does not beat Elo baseline on log loss |
| poisson_goal_model | ok | 0.4375 | 1.0724 | 0.6484 | 0.1118 | yes | no | does not beat Elo baseline on log loss |
| majority_baseline | ok | 0.4375 | 1.0743 | 0.6512 | 0.0540 | no | no | does not beat majority baseline on log loss; does not beat Elo baseline on log loss |
| recent_form_only | ok | 0.4219 | 1.0886 | 0.6593 | 0.0646 | no | no | does not beat majority baseline on log loss; does not beat Elo baseline on log loss |
| uniform_random_baseline | ok | 0.4375 | 1.0986 | 0.6667 | 0.1042 | no | no | does not beat majority baseline on log loss; does not beat Elo baseline on log loss |

## World Cup Backtests

| Year | Status | Train matches | Test matches | Best model | Accuracy | Log loss | Brier | ECE |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2014 | ok | 37861 | 64 | feature_logistic_regression_calibrated | 0.6094 | 0.9451 | 0.5581 | 0.0640 |
| 2018 | ok | 41639 | 64 | feature_logistic_regression_calibrated | 0.5625 | 0.9326 | 0.5520 | 0.0554 |
| 2022 | ok | 45700 | 64 | weighted_probability_ensemble | 0.4844 | 1.0190 | 0.6053 | 0.1047 |

## Calibration

Calibration is evaluated with log loss, Brier score, calibration bins, and expected calibration error. Calibrated variants are reported alongside uncalibrated models; improvement is not assumed.

## Feature Importance

- `random_forest` top features: `elo_diff`, `elo_external_diff`, `avg_elo_diff`, `max_elo_last_10_b`, `elo_a`
- `gradient_boosting` top features: `elo_diff`, `elo_external_diff`, `avg_elo_diff`, `weighted_goal_diff_last_5_b`, `team_a_goals_conceded_last_5`
- `logistic_regression` top features: `avg_elo_diff`, `elo_diff`, `elo_external_diff`, `avg_elo_last_5_b`, `elo_b`

## Ablation Study

- `elo_only`: ok (log loss: 0.8680)
- `form_only`: ok (log loss: 0.9771)
- `ranking_only`: ok (log loss: 1.0543)
- `schedule_only`: ok (log loss: 1.0464)
- `goals_only`: ok (log loss: 0.9745)
- `elo_plus_form`: ok (log loss: 0.8610)
- `all_features`: ok (log loss: 0.8599)
- `all_features_plus_external_ratings`: unavailable (log loss: external rating CSVs not found)

## Error Analysis

- Predicted top-class misses recorded: 75
- Correct underdog calls recorded: 0

## Honest Interpretation

- Best model by log loss in the comparison split: `weighted_probability_ensemble`.
- The best model beats the majority baseline on log loss for this split.
- The best model beats the Elo-only baseline on log loss for this split.

Historical match results are useful for learning broad team-strength and form signals, but they are not enough for elite forecasting. The model does not know injuries, player availability, travel, tactical matchups, betting-market information, squad age profiles, or current bookmaker priors. Betting odds and player-level data would improve realism because they capture recent public information and team news that raw historical scores miss.

## Interview Talking Points

- The LLM does not directly predict winners; structured models own the probabilities and the mini-LLM explains them.
- Leakage is reduced by building every match row from team state and rating snapshots available strictly before that match date.
- Calibration matters because tournament simulations consume probabilities, not just predicted classes.
- Baselines matter because Elo-only, majority, random, Poisson, and ablations show whether complexity is earning its keep.
- World Cup-only backtests are noisy because each tournament has a small sample of matches.
- The ensemble is used to combine complementary signals while reporting member weights honestly.
- Paid data would improve the model through lineups, injuries, player minutes, market priors, travel, and tactical information.

These results should be read as an evaluation of the current CupCast feature pipeline on the supplied dataset, not as a claim of production-grade prediction quality.

## README-Ready Snippet

### Real Data Results

To reproduce real-data evaluation:

1. Download the international results dataset.
2. Place `results.csv` in `data/real/`.
3. Optionally place `shootouts.csv` in `data/real/`.
4. Run `make real-data-pipeline`.
5. Open `models/real_data_results_summary.md`.

Real-data metrics are generated locally after running the pipeline and are not hardcoded in this repository.
