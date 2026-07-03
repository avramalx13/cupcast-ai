# Full Tournament Simulation

Tournament: World Cup 2026
Simulations: 3000
Model version: prediction_model_real-local

## Top Title Probabilities

| Team | Title Probability |
| --- | ---: |
| Argentina | 13.3% |
| Spain | 11.9% |
| Mexico | 11.4% |
| France | 10.2% |
| Brazil | 9.7% |
| Germany | 5.9% |
| Morocco | 4.7% |
| Portugal | 3.9% |
| Colombia | 3.6% |
| Netherlands | 3.0% |

## Group Winners

| Group | Most Likely Winner | Win Group |
| --- | --- | ---: |
| A | Mexico | 67.5% |
| B | Canada | 54.8% |
| C | Brazil | 62.4% |
| D | United States | 35.6% |
| E | Germany | 63.0% |
| F | Netherlands | 56.6% |
| G | Belgium | 64.3% |
| H | Spain | 83.5% |
| I | France | 74.9% |
| J | Argentina | 80.6% |
| K | Portugal | 60.0% |
| L | England | 67.8% |

## Method Notes

- Group stage is simulated from scratch using match-level win/draw/loss probabilities.
- Top two teams from each group qualify, plus the eight strongest third-place teams.
- Round of 32 placement uses a deterministic approximation. It separates same-group teams where possible, but it is not the official FIFA third-place placement matrix.
- Predictions are probabilistic estimates, not guarantees.
