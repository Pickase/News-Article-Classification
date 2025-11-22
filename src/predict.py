import joblib
import pandas as pd

from .config import MODEL_PATH, LABEL_ENCODER_PATH


def _load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder


def predict_text(text: str):
    """
    Predicts the class for a single news article text.
    Returns (label_string, probabilities_dict)
    """
    model, label_encoder = _load_artifacts()

    X = pd.Series([text])

    probs = model.predict_proba(X)[0]
    pred_idx = probs.argmax()
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Map class → probability
    proba_dict = {
        label: float(prob)
        for label, prob in zip(label_encoder.classes_, probs)
    }

    return pred_label, proba_dict


if __name__ == "__main__":
    demo_text = "Stock markets crashed as inflation data surprised investors."
    label, probs = predict_text(demo_text)
    print("Predicted label:", label)
    print("Probabilities:", probs)
