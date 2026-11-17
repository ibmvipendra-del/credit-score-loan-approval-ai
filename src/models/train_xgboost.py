
from pathlib import Path
import json

import pandas as pd
import xgboost as xgb

from src.data.feature_mapping import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


TARGET = "default"

# Sensitive attributes excluded from initial prediction model.
EXCLUDED_FEATURES = [
    "personal_status_sex",
    "foreign_worker"
]


def prepare_features(df, categories=None):

    excluded = [TARGET] + EXCLUDED_FEATURES

    X = df.drop(columns=excluded, errors="ignore").copy()

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:

            if categories is None:
                X[col] = X[col].astype("category")
            else:
                X[col] = pd.Categorical(
                    X[col],
                    categories=categories[col]
                )

    return X


def train_xgboost():

    print("=" * 60)
    print("XGBOOST CREDIT DEFAULT MODEL TRAINING")
    print("=" * 60)

    train_file = Path("data/processed/train.csv")

    if not train_file.exists():
        raise FileNotFoundError(
            "Training dataset not found. Run split_data.py first."
        )

    train_df = pd.read_csv(train_file)

    y_train = train_df[TARGET].astype("int32")

    X_train = prepare_features(train_df)

    # Store training category levels for consistent inference.
    category_levels = {
        col: X_train[col].cat.categories.tolist()
        for col in CATEGORICAL_FEATURES
        if col in X_train.columns
    }

    # Native XGBoost data structure
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        enable_categorical=True
    )

    # Handle class imbalance using training data only.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42
    }

    print("\nTraining records:", len(X_train))
    print("Features used:", len(X_train.columns))
    print("Positive class weight:", round(scale_pos_weight, 3))

    print("\nTraining XGBoost...")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        verbose_eval=False
    )

    # Save trained model
    output_dir = Path("models/xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(output_dir / "credit_default_model.json")

    # Save model metadata
    metadata = {
        "model": "XGBoost",
        "target": TARGET,
        "features": X_train.columns.tolist(),
        "categorical_features": list(category_levels.keys()),
        "category_levels": category_levels,
        "excluded_sensitive_features": EXCLUDED_FEATURES,
        "training_records": len(X_train),
        "random_state": 42
    }

    with open(output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining completed successfully!")
    print("Model saved: models/xgboost/credit_default_model.json")
    print("Metadata saved: models/xgboost/metadata.json")
    print("\nModel version: XGBoost v", xgb.__version__)


if __name__ == "__main__":
    train_xgboost()
