from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer():
    """
    Create a TF-IDF vectorizer with reasonable defaults.
    Adjust parameters based on your notebook experiments.
    """
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    return vectorizer
