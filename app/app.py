import streamlit as st
import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict import predict_text

st.title("News Article Classification")

user_input = st.text_area("Enter news article text:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        label, probs = predict_text(user_input)
        st.success(f"Predicted category: {label}")

        st.subheader("Class probabilities:")
        for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{cls}: {p:.3f}")
