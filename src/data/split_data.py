
from pathlib import Path
import numpy as np
import pandas as pd


def stratified_split(
    df,
    target="default",
    test_size=0.20,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    for class_value in sorted(df[target].unique()):

        class_indices = np.flatnonzero(
            df[target].to_numpy() == class_value
        )

        rng.shuffle(class_indices)

        test_count = round(len(class_indices) * test_size)

        test_indices.extend(class_indices[:test_count])
        train_indices.extend(class_indices[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    return train_df, test_df


def main():

    input_file = Path("data/processed/german_credit_clean.csv")

    if not input_file.exists():
        raise FileNotFoundError(
            "Run prepare_data.py before splitting."
        )

    df = pd.read_csv(input_file)

    train_df, test_df = stratified_split(df)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("=" * 55)
    print("TRAIN / TEST SPLIT COMPLETED")
    print("=" * 55)

    print("\nTraining records:", len(train_df))
    print("Testing records:", len(test_df))

    print("\nTraining distribution:")
    print(train_df["default"].value_counts())

    print("\nTesting distribution:")
    print(test_df["default"].value_counts())

    print("\nTraining percentage:",
          round(len(train_df) / len(df) * 100, 2))

    print("Testing percentage:",
          round(len(test_df) / len(df) * 100, 2))


if __name__ == "__main__":
    main()
