# Full Tournament Simulation

Tournament: World Cup 2026
Simulations: 3000
Model version: prediction_model_real-local

## Top Title Probabilities

| Team | Title Probability |
| --- | ---: |
| France | 11.6% |
| Argentina | 11.4% |
| Spain | 8.7% |
| Brazil | 7.6% |
| Germany | 5.3% |
| Portugal | 5.2% |
| Netherlands | 4.9% |
| Mexico | 4.7% |
| Belgium | 3.6% |
| Japan | 3.5% |

## Group Winners

| Group | Most Likely Winner | Win Group |
| --- | --- | ---: |
| A | Mexico | 54.3% |
| B | Canada | 41.7% |
| C | Brazil | 55.0% |
| D | United States | 39.0% |
| E | Germany | 54.7% |
| F | Netherlands | 48.9% |
| G | Belgium | 50.6% |
| H | Spain | 64.8% |
| I | France | 61.6% |
| J | Argentina | 61.6% |
| K | Portugal | 51.9% |
| L | England | 52.6% |

## Method Notes

- Group stage is simulated from scratch using match-level win/draw/loss probabilities.
- Top two teams from each group qualify, plus the eight strongest third-place teams.
- Round of 32 placement uses a deterministic approximation. It separates same-group teams where possible, but it is not the official FIFA third-place placement matrix.
- Predictions are probabilistic estimates, not guarantees.
