# CupCast AI Ablation Study

Train matches: 9000; test matches: 3000
Rows used: 12000 of 49484
Note: ablation uses the most recent 12000 rows for runtime

| Feature group | Status | Accuracy | Log loss | Brier | ECE | Reason |
|---|---:|---:|---:|---:|---:|---|
| elo_only | ok | 0.6023 | 0.8680 | 0.5111 | 0.0135 |  |
| form_only | ok | 0.5380 | 0.9704 | 0.5775 | 0.0059 |  |
| ranking_only | ok | 0.4727 | 1.0543 | 0.6360 | 0.0054 |  |
| schedule_only | ok | 0.4740 | 1.0464 | 0.6304 | 0.0137 |  |
| goals_only | ok | 0.5390 | 0.9680 | 0.5755 | 0.0111 |  |
| elo_plus_form | ok | 0.6090 | 0.8610 | 0.5062 | 0.0170 |  |
| all_features | ok | 0.6090 | 0.8601 | 0.5059 | 0.0261 |  |
| all_features_plus_external_ratings | unavailable | n/a | n/a | n/a | n/a | external rating CSVs not found |
