import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
    TEXT_COL,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
)
from .utils import clean_text


def prepare_data():
    print(f"Reading raw data from: {RAW_DATA_PATH}")
    df = pd.read_excel(RAW_DATA_PATH)   # <<----- USING EXCEL FILE

    # Check required columns
    required_cols = ["headline", "short_description", "category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from dataset.")

    # Combine headline + short_description into one text field
    df[TEXT_COL] = (
        df["headline"].astype(str) + " " + df["short_description"].astype(str)
    )

    # Clean text
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

    # Target column
    df[TARGET_COL] = df["category"].astype(str)

    # Remove rows with missing data
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])

    # Train/test split
    train_df, test_df = train_test_split(
        df[[TEXT_COL, TARGET_COL]],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )

    # Save processed files
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    print(f"Saved processed train data → {PROCESSED_TRAIN_PATH}")
    print(f"Saved processed test data  → {PROCESSED_TEST_PATH}")
    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()
