import os

# ---- BASE PATHS ----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "news.xlsx")
PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ---- COLUMNS ----
TEXT_COL = "text"     # change if your notebook used a different name
TARGET_COL = "label"  # change if needed

# ---- TRAIN/TEST SPLIT ----
TEST_SIZE = 0.2
RANDOM_STATE = 42
