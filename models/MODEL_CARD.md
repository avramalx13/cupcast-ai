# CupCast AI Model Card

## Project Name

CupCast AI

## Model Purpose

CupCast AI produces three-way probabilistic forecasts for international football matches: Team A win, draw, and Team B win. The structured model feeds tournament simulation and analyst explanations.

This model produces probabilistic forecasts, not guarantees. It should not be used for betting or financial decisions.

## Data Source

User-supplied historical international football result CSVs placed in `data/real/`, normalized through the `international-results` adapter.

- Matches: 49484
- Teams: 336
- Date range: 1872-11-30 to 2026-06-30
- Tournaments: 200
- World Cup matches: 1103

## Input Features

- Pre-match Elo-style team ratings.
- Elo difference.
- Optional local Elo/FIFA snapshots looked up strictly before the match date.
- Recent form from prior matches only.
- Goals scored and conceded in the recent window.
- Schedule/rest, tournament context, stage, neutral venue, and host-country features.
- Elo trend, volatility, average, and recent maximum features from pre-match history.

## Prediction Target

Three-class match outcome: `A_WIN`, `DRAW`, or `B_WIN`, based on regulation/extra-time score. Penalty shootout winners are tracked separately and are not treated as regulation winners.

## Training Setup

The real-data pipeline compares majority baseline, uniform random baseline, Elo-only logistic regression, recent-form-only logistic regression, full feature logistic regression, Poisson goal model, calibrated variants, random forest, gradient boosting, and a probability ensemble. World Cup backtests train only on matches before each tournament start date.

## Evaluation Metrics

Accuracy, log loss, multiclass Brier score, expected calibration error, baseline comparisons, and World Cup-specific backtest metrics.

## Calibration And Ensemble

Calibrated variants are evaluated beside uncalibrated models. The ensemble reports member weights and component failures rather than hiding them.

## Real-Data Results Summary

Best comparison model by log loss: `weighted probability ensemble`.
Accuracy: 0.4688; log loss: 1.0202; Brier score: 0.6060; ECE: 0.1177.

## Known Limitations

- Historical results alone do not include injuries, player availability, tactical changes, travel, rest, lineups, or market expectations.
- World Cup backtests are small samples and can be noisy.
- Generated real teams use unknown confederation and blank FIFA rank until richer metadata is supplied.
- The model is not a production-grade sports betting system.

## Ethical/Product Limitations

The project is for education and portfolio evaluation. It should not be used to make betting, financial, or high-stakes decisions.

## Future Improvements

- Add verified pre-match Elo/ranking snapshots.
- Add squad strength, injuries, player minutes, and player form.
- Add betting-market baseline comparisons.
- Add richer calibration plots and reliability reporting.
- Improve tournament simulation with scoreline and bracket-specific models.
