# Full Tournament Simulation

Tournament: World Cup 2026
Simulations: 3000
Model version: prediction_model_real-local

## Top Title Probabilities

| Team | Title Probability |
| --- | ---: |
| Argentina | 11.2% |
| France | 9.9% |
| Spain | 8.4% |
| Brazil | 7.6% |
| Mexico | 5.4% |
| Germany | 5.2% |
| Portugal | 5.1% |
| Netherlands | 4.9% |
| Morocco | 4.6% |
| England | 4.3% |

## Group Winners

| Group | Most Likely Winner | Win Group |
| --- | --- | ---: |
| A | Mexico | 57.1% |
| B | Canada | 43.2% |
| C | Brazil | 51.7% |
| D | United States | 35.0% |
| E | Germany | 56.1% |
| F | Netherlands | 49.1% |
| G | Belgium | 50.8% |
| H | Spain | 66.0% |
| I | France | 60.9% |
| J | Argentina | 61.8% |
| K | Portugal | 51.5% |
| L | England | 54.3% |

## Method Notes

- Group stage is simulated from scratch using match-level win/draw/loss probabilities.
- Top two teams from each group qualify, plus the eight strongest third-place teams.
- Round of 32 placement uses a deterministic approximation. It separates same-group teams where possible, but it is not the official FIFA third-place placement matrix.
- Predictions are probabilistic estimates, not guarantees.
