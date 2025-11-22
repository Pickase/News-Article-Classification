# News Article Classification — Machine Learning Project

This project classifies news articles into categories using TF-IDF vectorization and Logistic Regression. It is structured like a real production machine-learning system, with separate modules for preprocessing, feature extraction, training, evaluation, and deployment via a Streamlit web application.

---

## Features

- Modular, production-style ML project structure  
- Text preprocessing and cleaning  
- TF-IDF vectorization for feature extraction  
- Logistic Regression classifier  
- Label encoding for multi-class targets  
- End-to-end training and evaluation pipeline  
- Streamlit-based UI for real-time predictions  
- Fully reproducible workflow  

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

This will clean the text, encode labels, and create train/test splits inside `data/processed/`.

---

### 3. Train the model

python -m src.train

This generates:

models/model.pkl  
models/label_encoder.pkl

---

### 4. Evaluate the model

python -m src.evaluate

Outputs include:

- Accuracy score  
- Classification report  
- Confusion matrix  

---

### 5. Run the Streamlit prediction app

streamlit run app/app.py

A browser window will open where you can paste a news article and view the predicted category along with class probabilities.

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

- Clean and modular ML design suitable for portfolio projects  
- Reusable preprocessing and feature extraction  
- Training and inference pipelines separated for clarity  
- Interactive web UI for article classification  
- Ideal template to scale with more advanced NLP techniques  

---

## Author

Pranav Joshi  
This project was converted from a notebook into a full modular ML system for learning and portfolio use.
