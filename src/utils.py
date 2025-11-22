import re

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove urls, non-letters, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove urls
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()            # collapse spaces
    return text
