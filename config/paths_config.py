import os
RAW_DIR = "artifacts/raw"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "artifacts")

TRAIN_PATH = os.path.join(RAW_DIR, "heart_disease_train.csv")
TEST_PATH = os.path.join(RAW_DIR, "heart_disease_test.csv")

PROCESSED_DIR = "artifacts/processed"

