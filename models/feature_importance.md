# CupCast AI Feature Importance

Rows used: 12000 of 49484
Note: feature importance uses the most recent 12000 rows for runtime

## random_forest

| Feature | Importance |
|---|---:|
| `avg_elo_diff` | 0.072090 |
| `elo_diff` | 0.057087 |
| `elo_external_diff` | 0.056288 |
| `competitive_goal_diff_diff` | 0.025854 |
| `elo_a` | 0.023746 |
| `elo_b` | 0.023197 |
| `max_elo_last_10_b` | 0.022885 |
| `elo_volatility_diff` | 0.022423 |
| `avg_elo_last_5_a` | 0.022368 |
| `elo_volatility_10_matches_a` | 0.022367 |
| `avg_elo_last_5_b` | 0.021841 |
| `opponent_adjusted_form_diff` | 0.021782 |
| `elo_trend_5_matches_a` | 0.021553 |
| `max_elo_last_10_a` | 0.021545 |
| `elo_volatility_10_matches_b` | 0.021323 |
| `elo_external_b` | 0.020755 |
| `elo_trend_5_matches_b` | 0.020736 |
| `elo_trend_diff` | 0.020587 |
| `goal_diff_form_diff` | 0.020527 |
| `elo_external_a` | 0.020027 |

## gradient_boosting

| Feature | Importance |
|---|---:|
| `elo_diff` | 0.703855 |
| `avg_elo_diff` | 0.097284 |
| `elo_external_diff` | 0.041014 |
| `home_advantage_flag` | 0.012023 |
| `team_a_goals_conceded_last_5` | 0.011464 |
| `team_b_goals_conceded_last_5` | 0.011183 |
| `rank_diff` | 0.010665 |
| `neutral` | 0.009521 |
| `goal_diff_form_diff` | 0.008621 |
| `elo_external_b` | 0.007342 |
| `competitive_goal_diff_diff` | 0.007238 |
| `weighted_goal_diff_last_5_a` | 0.005586 |
| `elo_trend_diff` | 0.005228 |
| `elo_external_a` | 0.004418 |
| `rest_days_diff` | 0.003801 |
| `elo_volatility_10_matches_a` | 0.003090 |
| `weighted_goal_diff_last_5_b` | 0.002905 |
| `elo_volatility_10_matches_b` | 0.002830 |
| `elo_b` | 0.002614 |
| `team_a_goals_scored_last_5` | 0.002515 |

## logistic_regression

| Feature | Importance |
|---|---:|
| `elo_diff` | 0.663980 |
| `elo_a` | 0.432126 |
| `rank_diff` | 0.394681 |
| `fifa_rank_external_diff` | 0.359731 |
| `max_elo_last_10_b` | 0.329496 |
| `fifa_rank_a` | 0.281044 |
| `avg_elo_diff` | 0.261598 |
| `fifa_rank_external_a` | 0.248245 |
| `elo_b` | 0.227263 |
| `max_elo_last_10_a` | 0.222591 |
| `fifa_rank_b` | 0.185026 |
| `fifa_rank_external_b` | 0.176759 |
| `avg_elo_last_5_b` | 0.176321 |
| `weighted_goal_diff_last_5_b` | 0.161203 |
| `team_a_recent_form` | 0.144818 |
| `elo_external_b` | 0.133325 |
| `avg_elo_last_5_a` | 0.114890 |
| `elo_external_a` | 0.114691 |
| `weighted_goal_diff_last_5_a` | 0.104739 |
| `weighted_form_last_5_b` | 0.103689 |
