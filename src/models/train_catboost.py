
from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostClassifier

from src.data.feature_mapping import CATEGORICAL_FEATURES

TARGET = "default"

EXCLUDED_FEATURES = [
    "personal_status_sex",
    "foreign_worker"
]


def prepare_features(df):

    X = df.drop(
        columns=[TARGET] + EXCLUDED_FEATURES
    ).copy()

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(str)

    return X


def train_catboost():

    print("=" * 60)
    print("CATBOOST CREDIT DEFAULT MODEL TRAINING")
    print("=" * 60)

    train_file = Path("data/processed/train.csv")

    if not train_file.exists():
        raise FileNotFoundError("Training dataset not found.")

    train_df = pd.read_csv(train_file)

    y_train = train_df[TARGET].astype(int)
    X_train = prepare_features(train_df)

    categorical_columns = [
        col for col in CATEGORICAL_FEATURES
        if col in X_train.columns
    ]

    categorical_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_columns
    ]

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    positive_weight = negative_count / positive_count

    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
        class_weights=[1.0, positive_weight]
    )

    print("\nTraining records:", len(X_train))
    print("Features used:", len(X_train.columns))
    print("Categorical features:", len(categorical_columns))
    print("Positive class weight:", round(positive_weight, 3))

    print("\nTraining CatBoost...")

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_indices
    )

    output_dir = Path("models/catboost")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(
        str(output_dir / "credit_default_model.cbm")
    )

    metadata = {
        "model": "CatBoost",
        "target": TARGET,
        "features": X_train.columns.tolist(),
        "categorical_features": categorical_columns,
        "excluded_sensitive_features": EXCLUDED_FEATURES,
        "training_records": len(X_train),
        "random_state": 42
    }

    with open(output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining completed successfully!")
    print("Model saved: models/catboost/credit_default_model.cbm")
    print("Metadata saved: models/catboost/metadata.json")


if __name__ == "__main__":
    train_catboost()
