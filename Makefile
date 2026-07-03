PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-build-isolation

sample-data:
	$(PYTHON) scripts/create_sample_data.py

test:
	$(PYTHON) -m pytest

validate-data:
	$(PYTHON) scripts/validate_dataset.py --matches data/raw/historical_matches.csv --teams data/raw/teams.csv

validate-real-data:
	$(PYTHON) scripts/validate_dataset.py --matches data/real/results.csv --shootouts data/real/shootouts.csv --source international-results

validate-ratings:
	$(PYTHON) scripts/validate_ratings.py --elo data/real/elo_ratings.csv --fifa data/real/fifa_rankings.csv

build-real-teams:
	$(PYTHON) scripts/build_teams_from_matches.py --matches data/real/results.csv --out data/processed/teams_real.csv

train-llm:
	$(PYTHON) scripts/train_mini_llm.py --config configs/mini_llm_small.yaml --data data/raw/football_text.txt --checkpoint-dir checkpoints/mini_llm

generate:
	$(PYTHON) -m cupcast.mini_llm.generate --checkpoint checkpoints/mini_llm/best.pt --prompt "### Context:\nTeam A: France\nTeam B: Brazil\n"

evaluate-llm:
	$(PYTHON) scripts/evaluate_mini_llm.py --checkpoint checkpoints/mini_llm/best.pt --eval-data data/raw/football_text_eval.txt

train-predictor:
	$(PYTHON) scripts/train_prediction_model.py --config configs/prediction_model.yaml

backtest:
	$(PYTHON) scripts/run_backtest.py --train-before 2022 --test-tournament "World Cup 2022"

compare-models:
	$(PYTHON) scripts/compare_models.py --config configs/prediction_model.yaml --train-before 2022 --test-tournament "World Cup 2022"

backtest-world-cups:
	$(PYTHON) scripts/backtest_world_cups.py --matches data/real/results.csv --shootouts data/real/shootouts.csv --years 2014 2018 2022

real-data-pipeline:
	$(PYTHON) scripts/run_real_data_pipeline.py --matches data/real/results.csv --shootouts data/real/shootouts.csv --elo data/real/elo_ratings.csv --fifa data/real/fifa_rankings.csv --years 2014 2018 2022

feature-importance:
	$(PYTHON) scripts/analyze_feature_importance.py --config configs/prediction_model.yaml

ablation:
	$(PYTHON) scripts/run_ablation_study.py --config configs/prediction_model.yaml

portfolio-docs:
	$(PYTHON) scripts/generate_portfolio_docs.py

frontend-reports:
	$(PYTHON) scripts/export_frontend_reports.py

report-plots:
	$(PYTHON) scripts/generate_report_plots.py

simulate:
	$(PYTHON) scripts/run_simulation.py --config configs/simulation.yaml --simulations 10000

readiness:
	$(PYTHON) scripts/check_project_readiness.py

api:
	uvicorn cupcast.api.main:app --reload
