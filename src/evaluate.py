import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from .config import (
    PROCESSED_TEST_PATH,
    TEXT_COL,
    TARGET_COL,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
)


def evaluate():
    print("Loading test data...")
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    print("Loading model and label encoder...")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    X_test = test_df[TEXT_COL].astype(str)
    y_true_labels = test_df[TARGET_COL]
    y_true = label_encoder.transform(y_true_labels)

    print("Running predictions...")
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
