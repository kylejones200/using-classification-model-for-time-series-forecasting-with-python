# Using Classification Model for Time Series Forecasting with Python

This project demonstrates using classification models for time series forecasting by converting regression to classification.

## Business context

Traditionally, time series forecasting focuses on predicting specific numerical values (like temperature, sales, or stock prices). However, sometimes the goal is not to predict the exact value but rather to identify trends, directions, or events. For example:

- Will the value increase or decrease? - Will a machine fail within the next hour? - Will sales cross a specific threshold?

<figcaption>Photo by <a class="markup--anchor markup--figure-anchor" rel="photo-creator noopener" target="_blank">zhang kaiyv</a> on <a class="markup--anchor markup--figure-anchor"

## Article

Medium article: [Using Classification Model for Time Series Forecasting with Python](https://medium.com/@kylejones_47003/using-classification-model-for-time-series-forecasting-with-python-d74a1021a5c4)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Classification forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Number of bins for classification
- Lag window size
- Output settings

## Classification Approach

Converting regression to classification:
- Bin continuous values into discrete classes
- Use classification models (Random Forest, etc.)
- Can handle non-linear patterns effectively

## Caveats

- By default, generates synthetic time series data.
- Number of bins affects granularity of predictions.
- Classification loses information compared to regression.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).