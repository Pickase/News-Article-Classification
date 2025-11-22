import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

from .config import (
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
    TEXT_COL,
    TARGET_COL,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
)
from .pipelines import build_model_pipeline


def train():
    print("Reading processed train & test data...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    # Combine for fitting LabelEncoder on all labels (optional but safe)
    all_labels = pd.concat([train_df[TARGET_COL], test_df[TARGET_COL]])

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Transform labels
    y_train = label_encoder.transform(train_df[TARGET_COL])
    y_test = label_encoder.transform(test_df[TARGET_COL])

    X_train = train_df[TEXT_COL].astype(str)
    X_test = test_df[TEXT_COL].astype(str)

    print("Building pipeline...")
    pipeline = build_model_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print(f"Saving model to: {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Saving label encoder to: {LABEL_ENCODER_PATH}")
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Quick train/test score (rough sanity check)
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test  accuracy: {test_score:.4f}")


if __name__ == "__main__":
    train()
