
from pathlib import Path
import pandas as pd


def validate_dataset():

    file_path = Path("data/raw/german_credit.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run download_data.py first."
        )

    df = pd.read_csv(file_path)

    print("\n" + "=" * 50)
    print("DATA QUALITY VALIDATION")
    print("=" * 50)

    print("\n1. Dataset shape:")
    print(df.shape)

    print("\n2. Column names:")
    print(df.columns.tolist())

    print("\n3. Missing values:")
    print(df.isnull().sum())

    print("\n4. Duplicate records:")
    print(df.duplicated().sum())

    print("\n5. Target distribution:")
    print(df["default"].value_counts())

    print("\n6. Data types:")
    print(df.dtypes)

    print("\n7. Statistical summary:")
    print(df.describe(include="all").T)

    print("\nValidation completed!")


if __name__ == "__main__":
    validate_dataset()
