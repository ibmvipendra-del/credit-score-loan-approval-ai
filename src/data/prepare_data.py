
from pathlib import Path
import pandas as pd

from src.data.feature_mapping import FEATURE_MAPPING


def prepare_dataset():

    input_file = Path("data/raw/german_credit.csv")
    output_file = Path("data/processed/german_credit_clean.csv")

    if not input_file.exists():
        raise FileNotFoundError("Download dataset first.")

    df = pd.read_csv(input_file)

    # Rename original dataset columns
    df = df.rename(columns=FEATURE_MAPPING)

    # Create processed directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(output_file, index=False)

    print("=" * 55)
    print("DATA PREPARATION COMPLETED")
    print("=" * 55)

    print("\nProcessed dataset shape:", df.shape)

    print("\nFeature names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nTarget distribution:")
    print(df["default"].value_counts())

    print("\nProcessed file:", output_file)


if __name__ == "__main__":
    prepare_dataset()
