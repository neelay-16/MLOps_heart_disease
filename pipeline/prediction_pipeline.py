import pandas as pd
import numpy as np
import pickle
import os
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import MODEL_SAVE_PATH
import sys

logger = get_logger(__name__)


class HeartDiseasePrediction:
    def __init__(self, model_dir=MODEL_SAVE_PATH):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.load_models()

    def load_models(self):
        """Load all three saved models"""
        try:
            logger.info("Loading trained models...")

            # 1. Logistic Regression + Scaler
            lr_path = os.path.join(self.model_dir, "logistic_regression.pkl")
            if os.path.exists(lr_path):
                with open(lr_path, "rb") as f:
                    self.models["LogisticRegression"] = pickle.load(f)
                logger.info("Logistic Regression model loaded.")

            # 2. Random Forest
            rf_path = os.path.join(self.model_dir, "random_forest.pkl")
            if os.path.exists(rf_path):
                with open(rf_path, "rb") as f:
                    self.models["RandomForest"] = pickle.load(f)
                logger.info("Random Forest model loaded.")

            # 3. Keras Neural Network + Scaler
            keras_path = os.path.join(self.model_dir, "keras_neural_network.keras")
            scaler_path = os.path.join(self.model_dir, "keras_scaler.pkl")

            if os.path.exists(keras_path) and os.path.exists(scaler_path):
                from tensorflow.keras.models import load_model
                self.models["KerasNN"] = load_model(keras_path)
                with open(scaler_path, "rb") as f:
                    self.scalers["KerasNN"] = pickle.load(f)
                logger.info("Keras Neural Network model loaded.")

            if not self.models:
                raise FileNotFoundError("No models found in the model directory.")

        except Exception as e:
            logger.error(f"Error while loading models: {e}")
            raise CustomException(str(e), sys.exc_info())

    def predict(self, input_data):
        """
        Make predictions using all three models.
        input_data: pandas DataFrame with same features used during training.
        """
        try:
            results = {}

            # Logistic Regression
            if "LogisticRegression" in self.models:
                lr_model = self.models["LogisticRegression"]
                lr_pred = lr_model.predict(input_data)[0]
                lr_prob = lr_model.predict_proba(input_data)[0][1]
                results["LogisticRegression"] = {
                    "prediction": int(lr_pred),
                    "probability": round(float(lr_prob), 4)
                }

            # Random Forest
            if "RandomForest" in self.models:
                rf_model = self.models["RandomForest"]
                rf_pred = rf_model.predict(input_data)[0]
                rf_prob = rf_model.predict_proba(input_data)[0][1]
                results["RandomForest"] = {
                    "prediction": int(rf_pred),
                    "probability": round(float(rf_prob), 4)
                }

            # Keras Neural Network
            if "KerasNN" in self.models:
                keras_model = self.models["KerasNN"]
                scaler = self.scalers.get("KerasNN")

                input_scaled = scaler.transform(input_data) if scaler else input_data
                keras_prob = keras_model.predict(input_scaled, verbose=0)[0][0]
                keras_pred = 1 if keras_prob > 0.5 else 0

                results["KerasNN"] = {
                    "prediction": int(keras_pred),
                    "probability": round(float(keras_prob), 4)
                }

            return results

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise CustomException(str(e), sys.exc_info())

    def predict_on_test_data(self, X_test):
        """Predict on entire test set and return DataFrame with predictions"""
        try:
            logger.info("Making predictions on test data...")

            predictions = []
            for idx in range(len(X_test)):
                input_row = X_test.iloc[[idx]]
                result = self.predict(input_row)

                row_result = {
                    "index": idx,
                    "LR_Prediction": result.get("LogisticRegression", {}).get("prediction"),
                    "LR_Probability": result.get("LogisticRegression", {}).get("probability"),
                    "RF_Prediction": result.get("RandomForest", {}).get("prediction"),
                    "RF_Probability": result.get("RandomForest", {}).get("probability"),
                    "Keras_Prediction": result.get("KerasNN", {}).get("prediction"),
                    "Keras_Probability": result.get("KerasNN", {}).get("probability"),
                }
                predictions.append(row_result)

            return pd.DataFrame(predictions)

        except Exception as e:
            logger.error(f"Error predicting on test data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def predict_custom_input(self, input_dict):
        """
        Predict on a single custom input.
        input_dict: Dictionary with feature names as keys.
        """
        try:
            input_df = pd.DataFrame([input_dict])
            result = self.predict(input_df)

            print("\n" + "="*70)
            print("HEART DISEASE PREDICTION RESULTS (Custom Input)")
            print("="*70)

            for model_name, pred in result.items():
                status = "Heart Disease" if pred["prediction"] == 1 else "No Heart Disease"
                print(f"{model_name:20} → {status:18} | Probability: {pred['probability']:.4f}")

            # Majority Vote
            votes = [pred["prediction"] for pred in result.values()]
            final_vote = 1 if sum(votes) >= 2 else 0
            final_status = "Heart Disease" if final_vote == 1 else "No Heart Disease"

            print("-" * 70)
            print(f"{'Majority Vote':20} → {final_status:18}")
            print("="*70)

            return result

        except Exception as e:
            logger.error(f"Error in custom prediction: {e}")
            raise CustomException(str(e), sys.exc_info())


# ====================== Example Usage ======================

if __name__ == "__main__":
    predictor = HeartDiseasePrediction()

    # Example 1: Predict on test data (first 5 rows)
    # from src.data_preprocessing import HeartDiseaseDataProcessing
    # from src.feature_store import RedisFeatureStore

    # feature_store = RedisFeatureStore()
    # data_path = r"C:\WILP 3rd sem\MLOps\Assignment_01\heart+disease\processed.cleveland.data"
    # processor = HeartDiseaseDataProcessing(data_path, feature_store)
    # processor.load_data()
    # processor.preprocess_data()
    # X = processor.data.drop(['target', 'target_binary'], axis=1)
    # print(predictor.predict_on_test_data(X.head()))

    # Example 2: Custom Input Prediction
    custom_input = {
        'age': 55,
        'sex': 1,
        'cp': 3,
        'trestbps': 140,
        'chol': 250,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 1,
        'oldpeak': 1.5,
        'slope': 2,
        'ca': 1,
        'thal': 6
    }

    predictor.predict_custom_input(custom_input)