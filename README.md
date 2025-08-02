# News Article Classification

This repository implements NLP pipelines to classify news articles into **Sports**, **Politics**, or **Technology** categories using TF-IDF, word embeddings, and machine learning algorithms. The pipeline encompasses data loading, preprocessing, feature extraction, model training, evaluation, and prediction.

## Table of Contents

- [Dataset](#dataset-1)  
- [Preprocessing](#preprocessing-1)  
- [Feature Engineering](#feature-engineering-1)  
- [Model Implementation](#model-implementation-1)  
- [Results](#results-1)  
- [Usage](#usage-1)  
- [Project Structure](#project-structure-1)  
- [Insights & Observations](#insights--observations-1)  
- [License](#license-1)  

## Dataset

- **Source:** Custom news dataset (`data_news.csv`)  
- **Total Articles:** 20,000  
- **Categories:** `SPORTS`, `POLITICS`, `TECHNOLOGY`  

Split:  
- **Training Set:** 16,000 articles (80%)  
- **Test Set:** 4,000 articles (20%)  

## Preprocessing

1. **Text Cleaning**  
   - Remove URLs, HTML tags, punctuation  
2. **Case Normalization**  
   - Convert all text to lowercase  
3. **Tokenization**  
   - Split headlines and descriptions into tokens  
4. **Stop-Word Removal & Lemmatization**  

## Feature Engineering

- **TF-IDF:** On combined `headline + short_description + keywords`  
- **Word Embeddings:** Pretrained GloVe vectors (100d)  
- **Custom Features:**  
  - Headline length  
  - Keyword count  

## Model Implementation

| Model                  | Key Hyperparameters                        |
|------------------------|---------------------------------------------|
| Logistic Regression    | C, penalty                                  |
| Multinomial Naive Bayes| alpha                                       |
| Linear SVM             | C                                           |

All training scripts are in **src/train.py**; evaluation in **src/evaluate.py**; notebook at **notebooks/Nlp-Project-2-PranavJoshi.ipynb**.

## Results

| Metric     | Logistic Regression | Naive Bayes | SVM   |
|------------|---------------------|-------------|-------|
| Accuracy   | 0.91                | 0.89        | 0.92  |
| Precision  | 0.90                | 0.88        | 0.92  |
| Recall     | 0.89                | 0.87        | 0.91  |
| F1-Score   | 0.90                | 0.88        | 0.92  |

  
    
  Confusion matrix for the best-performing SVM model.  


## Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/news-article-classification.git
   cd news-article-classification
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**  
   Open `notebooks/Nlp-Project-2-PranavJoshi.ipynb`.  


## Project Structure

```
news-article-classification/
├── data/
│   └── data_news.csv
├── notebooks/
│   └── Nlp-Project-2-PranavJoshi.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── outputs/
│   ├── figures/
│   │   ├── category_distribution.png
│   │   ├── wordcloud_sports.png
│   │   └── confusion_matrix_svm.png
│   └── results.csv
├── requirements.txt
└── README.md
```

## Insights & Observations

- **Category Distribution:** Balanced across three classes.  
- **Headline Length:** Sports articles average 10 words; Tech articles 12 words.  
- **Top Keywords:**  
  - Sports: “game”, “tournament”  
  - Politics: “election”, “policy”  
  - Technology: “device”, “startup”  
- **Best Model:** Linear SVM with 0.92 accuracy and 0.92 macro-F1.
