# CupCast AI Ablation Study

Train matches: 9000; test matches: 3000
Rows used: 12000 of 49484
Note: ablation uses the most recent 12000 rows for runtime

| Feature group | Status | Accuracy | Log loss | Brier | ECE | Reason |
|---|---:|---:|---:|---:|---:|---|
| elo_only | ok | 0.6023 | 0.8680 | 0.5111 | 0.0135 |  |
| form_only | ok | 0.5307 | 0.9771 | 0.5817 | 0.0139 |  |
| ranking_only | ok | 0.4727 | 1.0543 | 0.6360 | 0.0054 |  |
| schedule_only | ok | 0.4740 | 1.0464 | 0.6304 | 0.0137 |  |
| goals_only | ok | 0.5343 | 0.9745 | 0.5797 | 0.0138 |  |
| elo_plus_form | ok | 0.6070 | 0.8610 | 0.5062 | 0.0151 |  |
| all_features | ok | 0.6070 | 0.8599 | 0.5060 | 0.0210 |  |
| all_features_plus_external_ratings | unavailable | n/a | n/a | n/a | n/a | external rating CSVs not found |
