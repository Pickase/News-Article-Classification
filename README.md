# News Article Classification — Machine Learning Project

This project classifies news articles into categories using TF-IDF vectorization and Logistic Regression. It follows a clean, production-style structure with dedicated modules for preprocessing, feature extraction, model training, evaluation, and deployment via Streamlit.

---

## Features

- Modular, industry-style ML project structure  
- Text cleaning and preprocessing  
- TF-IDF vectorization  
- Logistic Regression classifier  
- Label encoding for multi-class targets  
- End-to-end training and evaluation pipeline  
- Streamlit app for real-time predictions  
- Reproducible workflow with clear separation of steps  

---

## Accuracy

- **Training Accuracy:** ~89%  
- **Test Accuracy:** ~86%  
These values indicate a strong baseline model with good generalization.

---

## Live Streamlit Demo

You can try the model live here:

**https://news-article-classification-xcq5zaje5riai6kxhrrhan.streamlit.app/**

---

## Project Structure

news-classification/  
│  
├── data/  
│   ├── raw/  
│   │   └── news.csv  
│   └── processed/  
│       ├── train.csv  
│       └── test.csv  
│  
├── models/  
│   ├── model.pkl  
│   └── label_encoder.pkl  
│  
├── src/  
│   ├── config.py  
│   ├── utils.py  
│   ├── data_prep.py  
│   ├── features.py  
│   ├── pipelines.py  
│   ├── train.py  
│   ├── evaluate.py  
│   └── predict.py  
│  
├── app/  
│   └── app.py  
│  
├── requirements.txt  
└── README.md  

---

## How to Run the Project

### 1. Install dependencies

pip install -r requirements.txt

---

### 2. Prepare the dataset

Place your raw dataset at:

data/raw/news.csv

Then run:

python -m src.data_prep

This will clean text and generate train/test splits inside `data/processed/`.

---

### 3. Train the model

python -m src.train

This creates:

models/model.pkl  
models/label_encoder.pkl

---

### 4. Evaluate the model

python -m src.evaluate

This outputs:

- accuracy  
- classification report  
- confusion matrix  

---

### 5. Run the Streamlit app

streamlit run app/app.py

Paste any news article into the textbox to get the predicted category and probability distribution.

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- Streamlit  

---

## Highlights

- Modular ML pipeline suitable for real-world deployment  
- Reusable preprocessing and scalable structure  
- Clear separation of training, evaluation, and inference  
- Interactive online demo using Streamlit  
- Strong baseline accuracy  
- Ideal starting point for advanced NLP techniques (word embeddings, transformers, etc.)  

---

## Author

Pranav Joshi
