from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from .features import build_tfidf_vectorizer


def build_model_pipeline():
    """
    Full text → TF-IDF → Logistic Regression pipeline.

    The TF-IDF vectorizer and the classifier are saved together inside model.pkl.
    """
    tfidf = build_tfidf_vectorizer()
    clf = LogisticRegression(max_iter=1000)

    pipe = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )
    return pipe
