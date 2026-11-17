
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation.metrics import calculate_metrics


def evaluate_xgboost():

    print("=" * 60)
    print("XGBOOST MODEL EVALUATION")
    print("=" * 60)

    model_dir = Path("models/xgboost")

    test_file = Path("data/processed/test.csv")
    model_file = model_dir / "credit_default_model.json"
    metadata_file = model_dir / "metadata.json"

    for file in [test_file, model_file, metadata_file]:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    test_df = pd.read_csv(test_file)

    with open(metadata_file) as file:
        metadata = json.load(file)

    model = xgb.Booster()
    model.load_model(str(model_file))

    # Use exactly the features and order used during training.
    feature_names = metadata["features"]
    X_test = test_df[feature_names].copy()

    # Restore categorical data types using training categories.
    for col, categories in metadata["category_levels"].items():
        X_test[col] = pd.Categorical(
            X_test[col],
            categories=categories
        )

    y_test = test_df["default"].to_numpy(dtype=int)

    dtest = xgb.DMatrix(
        X_test,
        enable_categorical=True
    )

    probabilities = model.predict(dtest)

    metrics = calculate_metrics(
        y_true=y_test,
        probabilities=probabilities,
        threshold=0.5
    )

    print("\nEvaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    output_dir = Path("models/xgboost")
    output_file = output_dir / "evaluation_metrics.json"

    with open(output_file, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nEvaluation saved:", output_file)
    print("\nXGBoost evaluation completed!")


if __name__ == "__main__":
    evaluate_xgboost()
