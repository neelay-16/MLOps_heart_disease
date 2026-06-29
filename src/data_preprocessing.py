import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)


class HeartDiseaseDataProcessing:
    def __init__(self, data_path, feature_store: RedisFeatureStore):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_store = feature_store
        logger.info("Heart Disease Data Processing initialized...")

    def load_data(self):
        try:
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                            'restecg', 'thalach', 'exang', 'oldpeak', 
                            'slope', 'ca', 'thal', 'target']

            self.data = pd.read_csv(self.data_path, header=None, names=column_names)
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error while reading data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def preprocess_data(self):
        try:
            # Replace '?' with NaN (as done in notebook)
            self.data = self.data.replace('?', np.nan)

            # Convert all columns to numeric
            self.data = self.data.apply(pd.to_numeric, errors='coerce')

            # Handle missing values using median (as done in notebook)
            self.data['ca'] = self.data['ca'].fillna(self.data['ca'].median())
            self.data['thal'] = self.data['thal'].fillna(self.data['thal'].median())

            # Create binary target (as done in notebook)
            self.data['target_binary'] = self.data['target'].apply(lambda x: 0 if x == 0 else 1)

            logger.info("Data preprocessing completed successfully")

        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def split_data(self):
        try:
            X = self.data.drop(['target', 'target_binary'], axis=1)
            y = self.data['target_binary']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            logger.info(f"Data split completed. Train: {self.X_train.shape}, Test: {self.X_test.shape}")

        except Exception as e:
            logger.error(f"Error while splitting data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def store_feature_in_redis(self):
        try:
            batch_data = {}
            for idx, row in self.data.iterrows():
                entity_id = str(idx)   # Using index as key (same as your test)

                features = {
                    "age": row["age"],
                    "sex": row["sex"],
                    "cp": row["cp"],
                    "trestbps": row["trestbps"],
                    "chol": row["chol"],
                    "fbs": row["fbs"],
                    "restecg": row["restecg"],
                    "thalach": row["thalach"],
                    "exang": row["exang"],
                    "oldpeak": row["oldpeak"],
                    "slope": row["slope"],
                    "ca": row["ca"],
                    "thal": row["thal"],
                    "target": int(row["target"]),
                    "target_binary": int(row["target_binary"])   # ← ADD THIS
                }
                batch_data[entity_id] = features

            self.feature_store.store_batch_features(batch_data)
            logger.info("Features stored successfully in Redis")

        except Exception as e:
            logger.error(f"Error while storing features in Redis: {e}")
            raise CustomException(str(e), sys.exc_info())

    def run(self):
        try:
            logger.info("Starting Heart Disease Data Processing Pipeline...")
            self.load_data()
            self.preprocess_data()
            self.split_data()
            self.store_feature_in_redis()
            logger.info("Heart Disease Data Processing Pipeline completed successfully")
        except Exception as e:
            raise CustomException("Failed to run Heart Disease data processing pipeline", sys.exc_info())


if __name__ == "__main__":
    feature_store = RedisFeatureStore()

    data_path = r"C:\WILP 3rd sem\MLOps\Assignment_01\heart+disease\processed.cleveland.data"

    data_processor = HeartDiseaseDataProcessing(
        data_path=data_path,
        feature_store=feature_store
    )

    data_processor.run()

    # Test retrieval
    print("\nSample retrieved from Redis:")
    print(data_processor.feature_store.get_features("train_0"))