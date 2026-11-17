
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.data.feature_mapping import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


def explore_dataset():

    df = pd.read_csv(
        "data/processed/german_credit_clean.csv"
    )

    print("\n========== CREDIT DATA ANALYSIS ==========")

    print("\n1. Numerical feature summary:")
    print(df[NUMERICAL_FEATURES].describe())

    print("\n2. Categorical feature summary:")
    print(df[CATEGORICAL_FEATURES].describe())

    print("\n3. Default distribution:")
    print(df["default"].value_counts(normalize=True))

    print("\n4. Average credit amount by default:")
    print(df.groupby("default")["credit_amount"].mean())

    print("\n5. Average age by default:")
    print(df.groupby("default")["age"].mean())

    print("\n6. Average loan duration by default:")
    print(df.groupby("default")["loan_duration_months"].mean())

    # Visualize target distribution
    plt.figure(figsize=(7, 5))

    df["default"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Good Credit vs Bad Credit")
    plt.xlabel("Default Class")
    plt.ylabel("Number of Applicants")
    plt.xticks([0, 1], ["Good Credit", "Bad Credit"],
               rotation=0)

    plt.tight_layout()

    Path("docs").mkdir(exist_ok=True)

    plt.savefig("docs/credit_distribution.png")
    plt.close()

    print("\nEDA completed successfully!")


if __name__ == "__main__":
    explore_dataset()
