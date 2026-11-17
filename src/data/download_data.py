
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_dataset():

    print("=" * 60)
    print("CREDIT SCORE & LOAN APPROVAL AI")
    print("Downloading UCI German Credit Dataset")
    print("=" * 60)

    # Fetch dataset from UCI repository
    dataset = fetch_ucirepo(id=144)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    # Combine features and target
    df = features.copy()

    # Original target: 1 = Good, 2 = Bad
    original_target = pd.to_numeric(targets.iloc[:, 0])

    df["default"] = original_target.map({
        1: 0,
        2: 1
    })

    # Validate target mapping
    if df["default"].isna().any():
        raise ValueError("Unexpected target values detected")

    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "german_credit.csv"

    # Save dataset
    df.to_csv(output_file, index=False)

    print("\nDataset downloaded successfully!")
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nTarget distribution:")
    print(df["default"].value_counts())

    print("\nDefault distribution:")
    print(df["default"].value_counts(normalize=True) * 100)

    print("\nDataset saved at:", output_file)


if __name__ == "__main__":
    download_dataset()
