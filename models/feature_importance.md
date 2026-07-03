# CupCast AI Feature Importance

Rows used: 12000 of 49484
Note: feature importance uses the most recent 12000 rows for runtime

## random_forest

| Feature | Importance |
|---|---:|
| `elo_external_diff` | 0.071616 |
| `avg_elo_diff` | 0.056942 |
| `elo_diff` | 0.051865 |
| `elo_b` | 0.026034 |
| `elo_a` | 0.025500 |
| `goal_diff_form_diff` | 0.024932 |
| `avg_elo_last_5_a` | 0.024777 |
| `max_elo_last_10_b` | 0.024635 |
| `competitive_goal_diff_diff` | 0.024467 |
| `avg_elo_last_5_b` | 0.024381 |
| `elo_trend_5_matches_a` | 0.023782 |
| `opponent_adjusted_form_diff` | 0.023542 |
| `elo_external_b` | 0.023476 |
| `elo_volatility_diff` | 0.023132 |
| `elo_volatility_10_matches_a` | 0.023104 |
| `max_elo_last_10_a` | 0.022952 |
| `elo_external_a` | 0.022655 |
| `elo_volatility_10_matches_b` | 0.022601 |
| `elo_trend_5_matches_b` | 0.021704 |
| `elo_trend_diff` | 0.021504 |

## gradient_boosting

| Feature | Importance |
|---|---:|
| `elo_diff` | 0.467576 |
| `elo_external_diff` | 0.264202 |
| `avg_elo_diff` | 0.123461 |
| `home_advantage_flag` | 0.012693 |
| `team_b_goals_conceded_last_5` | 0.011295 |
| `team_a_goals_conceded_last_5` | 0.011202 |
| `goal_diff_form_diff` | 0.008922 |
| `neutral` | 0.008474 |
| `weighted_goal_diff_last_5_a` | 0.007309 |
| `competitive_goal_diff_diff` | 0.006958 |
| `elo_b` | 0.004340 |
| `avg_elo_last_5_b` | 0.004335 |
| `weighted_goal_diff_last_5_b` | 0.004286 |
| `max_elo_last_10_b` | 0.004112 |
| `elo_trend_diff` | 0.003900 |
| `rest_days_diff` | 0.003890 |
| `elo_volatility_10_matches_a` | 0.002943 |
| `is_continental_tournament` | 0.002807 |
| `opponent_adjusted_form_diff` | 0.002789 |
| `elo_volatility_10_matches_b` | 0.002689 |

## logistic_regression

| Feature | Importance |
|---|---:|
| `elo_diff` | 0.368214 |
| `elo_external_diff` | 0.368214 |
| `avg_elo_diff` | 0.309887 |
| `elo_a` | 0.245100 |
| `elo_external_a` | 0.245100 |
| `max_elo_last_10_b` | 0.224871 |
| `avg_elo_last_5_b` | 0.175442 |
| `avg_elo_last_5_a` | 0.164144 |
| `weighted_goal_diff_last_5_b` | 0.163598 |
| `max_elo_last_10_a` | 0.161831 |
| `team_a_recent_form` | 0.137573 |
| `elo_b` | 0.120452 |
| `elo_external_b` | 0.120452 |
| `weighted_goal_diff_last_5_a` | 0.108498 |
| `weighted_form_last_5_b` | 0.106154 |
| `weighted_form_diff` | 0.095455 |
| `neutral` | 0.080315 |
| `home_advantage_flag` | 0.080315 |
| `host_country_advantage_flag` | 0.074835 |
| `opponent_adjusted_form_a` | 0.068697 |
