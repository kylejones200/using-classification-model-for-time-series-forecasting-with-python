#!/usr/bin/env python3
"""Using Classification Model for Time Series Forecasting."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from src.core import (
    create_classification_targets,
    create_lagged_features,
    plot_classification_results,
    train_classification_model,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Using Classification Model for Time Series Forecasting"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)

    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        data = df.iloc[:, 0]
    elif config["data"]["generate_synthetic"]:
        np.random.seed(config["data"]["seed"])
        data = pd.Series(
            np.sin(np.arange(config["data"]["n_periods"]) / 10)
            + np.random.normal(0, 0.1, config["data"]["n_periods"])
        )
    else:
        raise ValueError("No data source specified")

    y_class, _encoder = create_classification_targets(
        data, config["model"]["n_bins"]
    )
    X, _y = create_lagged_features(data, config["model"]["lag"])
    y_class_lagged = y_class[config["model"]["lag"] :]
    train_size = int(len(X) * config["model"]["train_size"])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_class_lagged[:train_size], y_class_lagged[train_size:]
    model = train_classification_model(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"\nClassification Accuracy: {accuracy:.4f}")
    plot_classification_results(
        y_test,
        y_pred,
        "Classification-Based Forecasting",
        output_dir / "classification_forecast.png",
        plot=True,
    )
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
